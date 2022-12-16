import time
import numpy as np
import timeit
import saverloader
from nets.raftnet import Raftnet
from nets.pips import Pips
import random
from utils.basic import print_, print_stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from badjadataset import BadjaDataset
import utils.basic
import utils.improc
import utils.test
from fire import Fire
import cv2

device = 'cuda'
patch_size = 8
random.seed(125)
np.random.seed(125)

def run_pips(pips, d, sw):
    metrics = {}

    file0 = str(d['file0'])
    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    segs = d['segs'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    visibles = d['visibles'].cuda().float() # B, S, N

    # print('file0', file0)
    if 'extra_videos' in file0:
        animal = file0.split('/')[-3]
    else:
        animal = file0.split('/')[-2]
    metrics['animal'] = animal
    # print('animal', animal)
    
    B, S, C, H, W = rgbs.shape
    B, S, N, D = trajs_g.shape
    assert(D==2)

    rgbs_ = rgbs.reshape(B*S, C, H, W)
    segs_ = segs.reshape(B*S, 1, H, W)
    H_, W_ = 320, 512
    sy = H_/H
    sx = W_/W
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    segs_ = F.interpolate(segs_, (H_, W_), mode='nearest')
    rgbs = rgbs_.reshape(B, S, 3, H_, W_)
    segs = segs_.reshape(B, S, 1, H_, W_)
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    segs = (segs > 0).float()
    assert(B==1)

    xy0 = trajs_g[:,0] # B, N, 2
    assert(S >= 8)

    trajs_e = torch.zeros_like(trajs_g)
    for n in range(N):
        # print('working on keypoint %d/%d' % (n+1, N))
        cur_frame = 0
        done = False
        traj_e = torch.zeros_like(trajs_g[:,:,n]) # B, S, 2
        # xy0_n = trajs_g[:,0,n] # B, 1, 2
        traj_e[:,0] = trajs_g[:,0,n] # B, 1, 2  # set first position to gt
        feat_init = None
        while not done:
            end_frame = cur_frame + 8
            # print('cur_frame', cur_frame)
            # print('end_frame', end_frame)

            rgb_seq = rgbs[:,cur_frame:end_frame]
            S_local = rgb_seq.shape[1]
            # print('S_local', S_local)
            rgb_seq = torch.cat([rgb_seq, rgb_seq[:,-1].unsqueeze(1).repeat(1,8-S_local,1,1,1)], dim=1)
            # print('rgb_seq (%d:%d)' % (cur_frame, end_frame), rgb_seq.shape)

            outs = pips(traj_e[:,cur_frame].reshape(1, -1, 2), rgb_seq, iters=6, feat_init=feat_init, return_feat=True)
            preds = outs[0]
            vis = outs[2] # B, S, 1
            feat_init = outs[3]
            
            vis = torch.sigmoid(vis) # visibility confidence
            xys = preds[-1].reshape(1, 8, 2)
            traj_e[:,cur_frame:end_frame] = xys[:,:S_local]

            found_skip = False
            thr = 0.9
            si_last = 8-1 # last frame we are willing to take
            si_earliest = 1 # earliest frame we are willing to take
            si = si_last
            while not found_skip:
                if vis[0,si] > thr:
                    found_skip = True
                else:
                    si -= 1
                if si == si_earliest:
                    # print('decreasing thresh')
                    thr -= 0.02
                    si = si_last
            # print('found skip at frame %d, where we have' % si, vis[0,si].detach().item())

            cur_frame = cur_frame + si

            if cur_frame >= S:
                done = True
        trajs_e[:,:,n] = traj_e

    prep_rgbs = utils.improc.preprocess_color(rgbs)
    label_colors = utils.improc.get_n_colors(N)
    gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
    
    if sw is not None and sw.save_this:
        for n in range(N):
            if visibles[0,0,n] > 0:
                print('visualizing kp %d' % n)
                # sw.summ_traj2ds_on_rgbs('kp_outputs_%02d/trajs_e_on_rgbs' % n, trajs_e[0:1,:,n:n+1], gray_rgbs[0:1,:S], cmap='spring', linewidth=2)
                sw.summ_traj2ds_on_rgbs('video_%d/kp_%d_trajs_e_on_rgbs' % (sw.global_step, n), trajs_e[0:1,:,n:n+1], gray_rgbs[0:1,:S], cmap='spring', linewidth=2)
                    
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb', trajs_e[0:1], prep_rgbs[0:1,0], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb2', trajs_e[0:1], torch.mean(prep_rgbs[0:1], dim=1), cmap='spring')

        if False: # very expensive vis
            kp_vis = []
            for s in range(S):
                kp = utils.improc.draw_circles_at_xy(trajs_e[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
                kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()
                kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
                rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
                kp[kp_any==0] = rgb[kp_any==0]
                kp_vis.append(kp)
            sw.summ_rgbs('outputs/kp_vis', kp_vis)

    assert(B==1)
    accs = []
    for s1 in range(1,S): # target frame
        for n in range(N):
            vis = visibles[0,s1,n]
            if vis > 0:
                coord_e = trajs_e[0,s1,n] # 2
                coord_g = trajs_g[0,s1,n] # 2
                dist = torch.sqrt(torch.sum((coord_e-coord_g)**2, dim=0))
                # print_('dist', dist)
                area = torch.sum(segs[0,s1])
                # print_('0.2*sqrt(area)', 0.2*torch.sqrt(area))
                thr = 0.2 * torch.sqrt(area)
                correct = (dist < thr).float()
                # print_('correct', correct)
                accs.append(correct)
    # assert(len(acc) == S*(S-1))
    pck = torch.mean(torch.stack(accs)) * 100.0
    metrics['pck'] = pck.item()
    return metrics
        
def run_raft(raft, d, sw):
    metrics = {}

    file0 = str(d['file0'])
    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    segs = d['segs'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    visibles = d['visibles'].cuda().float() # B, S, N

    # print('file0', file0)
    if 'extra_videos' in file0:
        animal = file0.split('/')[-3]
    else:
        animal = file0.split('/')[-2]
    metrics['animal'] = animal
    # print('animal', animal)
    
    B, S, C, H, W = rgbs.shape
    B, S, N, D = trajs_g.shape
    assert(D==2)
    
    rgbs_ = rgbs.reshape(B*S, C, H, W)
    segs_ = segs.reshape(B*S, 1, H, W)
    H_, W_ = 320, 512
    sy = H_/H
    sx = W_/W
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    segs_ = F.interpolate(segs_, (H_, W_), mode='nearest')
    rgbs = rgbs_.reshape(B, S, 3, H_, W_)
    segs = segs_.reshape(B, S, 1, H_, W_)
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    segs = (segs > 0).float()
    assert(B==1)
    
    prep_rgbs = utils.improc.preprocess_color(rgbs)
    gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)

    flows_e = []
    for s in range(S-1):
        rgb0 = prep_rgbs[:,s]
        rgb1 = prep_rgbs[:,s+1]
        flow, _ = raft(rgb0, rgb1, iters=32)
        flows_e.append(flow)
        
    flows_e = torch.stack(flows_e, dim=1) # B, S-1, 2, H, W

    coords = []
    coord0 = trajs_g[:,0] # B, N, 2
    coords.append(coord0)
    coord = coord0.clone()
    for s in range(S-1):
        delta = utils.samp.bilinear_sample2d(
            flows_e[:,s], coord[:,:,0], coord[:,:,1]).permute(0,2,1) # B, N, 2, forward flow at the discrete points
        coord = coord + delta
        coords.append(coord)
    trajs_e = torch.stack(coords, dim=1) # B, S, N, 2

    assert(B==1)
    accs = []
    for s1 in range(1,S): # target frame
        for n in range(N):
            vis = visibles[0,s1,n]
            if vis > 0:
                coord_e = trajs_e[0,s1,n] # 2
                coord_g = trajs_g[0,s1,n] # 2
                dist = torch.sqrt(torch.sum((coord_e-coord_g)**2, dim=0))
                # print_('dist', dist)
                area = torch.sum(segs[0,s1])
                # print_('0.2*sqrt(area)', 0.2*torch.sqrt(area))
                thr = 0.2 * torch.sqrt(area)
                correct = (dist < thr).float()
                # print_('correct', correct)
                accs.append(correct)
    pck = torch.mean(torch.stack(accs)) * 100.0
    metrics['pck'] = pck.item()
    
    label_colors = utils.improc.get_n_colors(N)

    if sw is not None and sw.save_this:
        sw.summ_rgbs('inputs/rgbs', prep_rgbs.unbind(1))
        sw.summ_oneds('inputs/segs', segs.unbind(1))

        for n in range(N):
            if visibles[0,0,n] > 0:
                sw.summ_traj2ds_on_rgbs('outputs/kp%d_trajs_e_on_rgbs' % n, trajs_e[0:1,:,n:n+1], gray_rgbs[0:1,:S], cmap='spring', linewidth=2)

        if False: 
            kp_vis = []
            for s in range(S):
                kp = utils.improc.draw_circles_at_xy(trajs_g[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
                kp = kp * visibles[0:1,0].reshape(1, N, 1, 1)
                kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()

                kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
                # rgb = (torch.mean(rgbs[:,s] * 0.5, dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
                rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
                # print('rgb', rgb.shape)
                kp[kp_any==0] = rgb[kp_any==0]
                kp_vis.append(kp)
            sw.summ_rgbs('inputs/kp_vis', kp_vis)
            # sw.summ_traj2ds_on_rgbs('inputs/trajs_g_on_rgbs', trajs_g[0:1], prep_rgbs[0:1], cmap='winter', valids=visibles[0:1])
            # sw.summ_traj2ds_on_rgb('inputs/trajs_g_on_rgb', trajs_g[0:1], prep_rgbs[0:1,0], cmap='winter', valids=visibles[0:1])

            kp_vis = []
            for s in range(S):
                kp = utils.improc.draw_circles_at_xy(trajs_e[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
                kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()
                kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
                rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
                kp[kp_any==0] = rgb[kp_any==0]
                kp_vis.append(kp)
            sw.summ_rgbs('outputs/kp_vis', kp_vis)
        
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb', trajs_e[0:1], prep_rgbs[0:1,0], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb2', trajs_e[0:1], torch.mean(prep_rgbs[0:1], dim=1), cmap='spring')

    return metrics

def prep_frame_for_dino(img, scale_size=[192]):
    """
    read a single frame & preprocess
    """
    ori_h, ori_w, _ = img.shape
    if len(scale_size) == 1:
        if(ori_h > ori_w):
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 64) * 64)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 64) * 64)
    else:
        th, tw = scale_size
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()

    def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
        for t, m, s in zip(x, mean, std):
            t.sub_(m)
            t.div_(s)
        return x
    
    img = color_normalize(img)
    return img, ori_h, ori_w

def get_feats_from_dino(model, frame):
    # batch version of the other func
    B = frame.shape[0]
    h, w = int(frame.shape[2] / model.patch_embed.patch_size), int(frame.shape[3] / model.patch_embed.patch_size)
    out = model.get_intermediate_layers(frame.cuda(), n=1)[0] # B, 1+h*w, dim
    dim = out.shape[-1]
    out = out[:, 1:, :]  # discard the [CLS] token
    outmap = out.permute(0, 2, 1).reshape(B, dim, h, w)
    return out, outmap, h, w

def restrict_neighborhood(h, w):
    size_mask_neighborhood = 12
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * size_mask_neighborhood + 1):
                for q in range(2 * size_mask_neighborhood + 1):
                    if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                        continue
                    if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1

    mask = mask.reshape(h * w, h * w)
    return mask.cuda(non_blocking=True)

def label_propagation(h, w, feat_tar, list_frame_feats, list_segs, mask_neighborhood=None):
    ncontext = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats) # nmb_context x dim x h*w

    feat_tar = F.normalize(feat_tar, dim=1, p=2)
    feat_sources = F.normalize(feat_sources, dim=1, p=2)

    # print('feat_tar', feat_tar.shape)
    # print('feat_sources', feat_sources.shape)

    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1)

    size_mask_neighborhood = 0
    if size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(h, w)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        aff *= mask_neighborhood

    aff = aff.transpose(2, 1).reshape(-1, h*w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
    topk = 5
    tk_val, _ = torch.topk(aff, dim=0, k=topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)

    list_segs = [s.cuda() for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, h, w = segs.shape
    segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, h, w)
    
    return seg_tar, mask_neighborhood

def norm_mask(mask):
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt,:,:]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt/mask_cnt.max()
            mask[cnt,:,:] = mask_cnt
    return mask

def run_dino(dino, d, sw):
    import copy
    metrics = {}

    file0 = str(d['file0'])
    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    segs = d['segs'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    visibles = d['visibles'].cuda().float() # B, S, N

    B, S, C, H, W = rgbs.shape
    B, S, N, D = trajs_g.shape
    assert(D==2)

    if 'extra_videos' in file0:
        animal = file0.split('/')[-3]
    else:
        animal = file0.split('/')[-2]
    metrics['animal'] = animal

    patch_size = 8

    rgbs_ = rgbs.reshape(B*S, C, H, W)
    segs_ = segs.reshape(B*S, 1, H, W)
    H_, W_ = 320, 512
    sy = H_/H
    sx = W_/W
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    segs_ = F.interpolate(segs_, (H_, W_), mode='nearest')
    rgbs = rgbs_.reshape(B, S, 3, H_, W_)
    segs = segs_.reshape(B, S, 1, H_, W_)
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    segs = (segs > 0).float()
    assert(B==1)
    xy0 = trajs_g[:,0] # B, N, 2

    # The queue stores the n preceeding frames
    import queue
    import copy
    n_last_frames = 7
    que = queue.Queue(n_last_frames)

    # run dino
    prep_rgbs = []
    for s in range(S):
        prep_rgb, ori_h, ori_w = prep_frame_for_dino(rgbs[0, s].permute(1,2,0).detach().cpu().numpy(), scale_size=[H_])
        prep_rgbs.append(prep_rgb)
    prep_rgbs = torch.stack(prep_rgbs, dim=0) # S, 3, H, W
    with torch.no_grad():
        bs = 8
        idx = 0 
        featmaps = []
        while idx < S:
            end_id = min(S, idx+bs)
            _, featmaps_cur, h, w = get_feats_from_dino(dino, prep_rgbs[idx:end_id]) # S, C, h, w
            idx = end_id
            featmaps.append(featmaps_cur)
        featmaps = torch.cat(featmaps, dim=0)
    C = featmaps.shape[1]
    featmaps = featmaps.unsqueeze(0) # 1, S, C, h, w
    # featmaps = F.normalize(featmaps, dim=2, p=2)

    first_seg = torch.zeros((1, N, H_//patch_size, W_//patch_size))
    for n in range(N):
        first_seg[0, n, (xy0[0, n, 1]/patch_size).long(), (xy0[0, n, 0]/patch_size).long()] = 1

    frame1_feat = featmaps[0, 0].reshape(C, h*w) # dim x h*w
    mask_neighborhood = None
    accs = []
    trajs_e = torch.zeros_like(trajs_g).to(device)
    trajs_e[0,0] = trajs_g[0,0]
    for cnt in range(1, S):
        used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

        feat_tar = featmaps[0, cnt].reshape(C, h*w)

        frame_tar_avg, mask_neighborhood = label_propagation(h, w, feat_tar.T, used_frame_feats, used_segs, mask_neighborhood)

        # pop out oldest frame if neccessary
        if que.qsize() == n_last_frames:
            que.get()
        # push current results into queue
        seg = copy.deepcopy(frame_tar_avg)
        que.put([feat_tar, seg])

        # upsampling & argmax
        frame_tar_avg = F.interpolate(frame_tar_avg, scale_factor=patch_size, mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
        frame_tar_avg = norm_mask(frame_tar_avg)
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

        for n in range(N):
            vis = visibles[0,cnt,n]
            if len(torch.nonzero(frame_tar_avg[n])) > 0:
                # weighted average
                nz = torch.nonzero(frame_tar_avg[n])
                coord_e = torch.sum(frame_tar_avg[n][nz[:,0], nz[:,1]].reshape(-1,1) * nz.float(), 0) / frame_tar_avg[n][nz[:,0], nz[:,1]].sum() # 2
                coord_e = coord_e[[1,0]]
            else:
                # stay where it was
                # coord_e = trajs_g[0,0,n]
                coord_e = trajs_e[0,cnt-1,n]
            trajs_e[0, cnt, n] = coord_e
            if vis > 0:
                coord_g = trajs_g[0,cnt,n] # 2
                dist = torch.sqrt(torch.sum((coord_e-coord_g)**2, dim=0))
                # print_('dist', dist)
                area = torch.sum(segs[0,cnt])
                # print_('0.2*sqrt(area)', 0.2*torch.sqrt(area))
                thr = 0.2 * torch.sqrt(area)
                correct = (dist < thr).float()
                accs.append(correct)

    pck = torch.mean(torch.stack(accs)) * 100.0
    metrics['pck'] = pck.item()

    prep_rgbs = utils.improc.preprocess_color(rgbs)
    gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
    if sw is not None and sw.save_this:
        for n in range(N):
            if visibles[0,0,n] > 0:
                sw.summ_traj2ds_on_rgbs('outputs/kp%d_trajs_e_on_rgbs' % n, trajs_e[0:1,:,n:n+1], gray_rgbs[0:1,:S], cmap='spring', linewidth=2)
        sw.summ_rgbs('inputs/rgbs', prep_rgbs.unbind(1))
        sw.summ_oneds('inputs/segs', segs.unbind(1))
        label_colors = utils.improc.get_n_colors(N)

        if False:
            kp_vis = []
            for s in range(S):
                kp = utils.improc.draw_circles_at_xy(trajs_g[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
                kp = kp * visibles[0:1,0].reshape(1, N, 1, 1)
                kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()
                kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
                rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
                # print('rgb', rgb.shape)
                kp[kp_any==0] = rgb[kp_any==0]
                kp_vis.append(kp)
            sw.summ_rgbs('inputs/kp_vis', kp_vis)

            kp_vis = []
            for s in range(S):
                kp = utils.improc.draw_circles_at_xy(trajs_e[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
                kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()
                kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
                rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
                kp[kp_any==0] = rgb[kp_any==0]
                kp_vis.append(kp)
            sw.summ_rgbs('outputs/kp_vis', kp_vis)
            
        sw.summ_traj2ds_on_rgbs('outputs/trajs_e_on_rgbs', trajs_e[0:1], prep_rgbs[0:1,:S], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb', trajs_e[0:1], prep_rgbs[0:1,0], cmap='spring')
    
    return metrics

def main(
        exp_name='badja',
        B=1,
        S=8,
        modeltype='pips',
        init_dir='reference_model',
        log_dir='logs_test_on_badja',
        data_dir='/data/badja_data',
        stride=4,
        max_iters=7,
        log_freq=99, # vis is very slow here
        shuffle=False,
):
    # the idea in this file is to evaluate on keypoint propagation in BADJA

    init_dir = './reference_model'
    
    assert(modeltype=='pips' or modeltype=='raft' or modeltype=='dino')
    
    ## autogen a name
    model_name = "%d_%d_%s" % (B, S, modeltype)
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    test_dataset = BadjaDataset(data_dir)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=1)
    test_iterloader = iter(test_dataloader)
    
    global_step = 0

    if modeltype=='pips':
        model = Pips(S=S, stride=stride).cuda()
        _ = saverloader.load(init_dir, model)
        model.eval()
    elif modeltype=='raft':
        model = Raftnet(ckpt_name='../RAFT/models/raft-things.pth').cuda()
        model.eval()
    elif modeltype=='dino':
        patch_size = 8
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits%d' % patch_size).cuda()
        model.eval()
    else:
        assert(False) # need to choose a valid modeltype
        
    results = []
    while global_step < max_iters:
        
        read_start_time = time.time()
        
        global_step += 1

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=24,
            scalar_freq=int(log_freq/2),
            just_gif=True)

        try:
            sample = next(test_iterloader)
        except StopIteration:
            test_iterloader = iter(test_dataloader)
            sample = next(test_iterloader)

        read_time = time.time()-read_start_time
        iter_start_time = time.time()

        with torch.no_grad():
            if modeltype=='pips':
                metrics = run_pips(model, sample, sw_t)
            elif modeltype=='raft':
                metrics = run_raft(model, sample, sw_t)
            elif modeltype=='dino':
                metrics = run_dino(model, sample, sw_t)
            else:
                assert(False) # need to choose a valid modeltype

        results.append(metrics['pck'])

        iter_time = time.time()-iter_start_time
        print('%s; step %06d/%d; rtime %.2f; itime %.2f; %s; pck %.1f' % (
            model_name, global_step, max_iters, read_time, iter_time,
            metrics['animal'], metrics['pck']))

        rp = []
        for result in results:
            rp.append('%.1f' % (result))
        rp.append('avg %.1f' % (np.mean(results)))
        print('results', rp)
            
    writer_t.close()
    
if __name__ == '__main__':
    Fire(main)
