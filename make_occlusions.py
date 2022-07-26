import time
import numpy as np
import timeit
import imageio
import io
import os
import math
from PIL import Image
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn.functional as F
import utils.improc
from utils.basic import readPFM, print_stats
import random
import glob
import scipy.spatial
from filter_trajs import filter_trajs
from tensorboardX import SummaryWriter

flt3d_path = "../flyingthings"
dsets = ["TRAIN", "TEST"]
subsets = ["A", "B", "C"]

device = 'cuda'

min_lifespan = 8

mod = 'aa' # start
mod = 'ab' # float16
mod = 'ac' # drop hull failures
mod = 'ad' # keep hull failures, but only save if valid
mod = 'ae' # come back to the old method, but export more
# somehow a set of headphones with holes all over made it through
mod = 'af' # drop hull failures
mod = 'ag' # export a single file for all ids in a vid; reject if len(singles_sums_)<3 (instead of 2)
mod = 'ah' # use empty arrays instead of Nones
mod = 'ai' # properly drop hull failures (return None early)
mod = 'aj' # hu_thr=0.95 instead of 0.9
mod = 'ak' # fix bug in filter_trajs, for visibility on last frame
mod = 'al' # hu_thr=0.98

min_size = 32*32

def readImage(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        data = readPFM(name)
        if len(data.shape)==3:
            return data[:,:,0:3]
        else:
            return data
    return imageio.imread(name)

def flood_fill_hull(image):
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img, hull

def consider_id(id_, all_rgbs, all_masks, all_flows_f, all_flows_b,
                fw_thr=0.95, bw_thr=0.95, hu_thr=0.98):
                # fw_thr=0.5, bw_thr=0.5, hu_thr=0.3):
                # fw_thr=0.6, bw_thr=0.6, hu_thr=0.0):
    B, S, C, H, W = all_rgbs.shape
    
    singles = (all_masks==id_).float()

    singles_flat = singles.reshape(S, -1)
    singles_sums = torch.sum(singles_flat, dim=1)
    singles_sums_ = singles_sums[singles_sums > 0]
    mean_nonzero_size = torch.mean(singles_sums[singles_sums > 0])

    if mean_nonzero_size < min_size*2:
        return None, None, None
    
    # min presence
    if len(singles_sums_) < 3:
        return None, None, None

    singles_fat = singles.clone()
    
    hu_match_amounts = []
    for s in range(S):
        single = singles[:,s]
        if torch.sum(single) > 4:
            try:
                single_py = single.reshape(H,W).cpu().long().numpy()
                close, _ = flood_fill_hull(single_py)

                close = torch.from_numpy(close).float().cuda()
                inter = close*single.reshape(H,W).float()
                union = (close+single.reshape(H,W).float()).clamp(0,1)

                hu_match_amount = torch.sum(inter)/torch.sum(union)

                singles_fat[0,s,0] = singles_fat[0,s,0]*0.5 + close*0.5
                
                if hu_match_amount < hu_thr:
                    return None, None, None
            except Exception as e:
                # if the shape broke my convex hull function, just drop it
                return None, None, None
        else:
            hu_match_amount = torch.tensor(1.0, device='cuda')
        hu_match_amounts.append(hu_match_amount)

    fw_match_amounts = []
    for s in range(S-1):
        single = singles[:,s]
        single_next = singles[:,s+1]
        if torch.sum(single) > min_size:
            ys, xs = utils.basic.meshgrid2d(1, H, W)
            xs = xs.reshape(-1).long()
            ys = ys.reshape(-1).long()
            ys = ys[single.reshape(-1) > 0]
            xs = xs[single.reshape(-1) > 0]

            delta = all_flows_f[0,s,:,ys,xs] # 2, N

            xs_ = (xs + delta[0]).round().long()
            ys_ = (ys + delta[1]).round().long()

            inds_ok = (xs_ >= 0) & (xs_ <= W-1) & (ys_ >= 0) & (ys_ <= H-1)
            xs_ = xs_[inds_ok]
            ys_ = ys_[inds_ok]

            if len(xs_) > min_size:
                match_next = single_next[0,0,ys_,xs_]
                fw_match_amount = torch.mean(match_next)
                if fw_match_amount < fw_thr:
                    return None, None, None
            else:
                fw_match_amount = torch.tensor(1.0, device='cuda')
        else:
            fw_match_amount = torch.tensor(1.0, device='cuda')
        fw_match_amounts.append(fw_match_amount)
    fw_match_amounts.append(torch.tensor(1.0, device='cuda'))
    
    bw_match_amounts = []
    bw_match_amounts.append(torch.tensor(1.0, device='cuda'))
    for s in range(S-1):
        single = singles[:,s]
        single_next = singles[:,s+1]
        if torch.sum(single) > min_size:
            ys, xs = utils.basic.meshgrid2d(1, H, W)
            xs = xs.reshape(-1).long()
            ys = ys.reshape(-1).long()
            ys = ys[single_next.reshape(-1) > 0]
            xs = xs[single_next.reshape(-1) > 0]

            delta = all_flows_b[0,s,:,ys,xs]
            xs_ = (xs + delta[0]).round().long()
            ys_ = (ys + delta[1]).round().long()
            inds_ok = (xs_ >= 0) & (xs_ <= W-1) & (ys_ >= 0) & (ys_ <= H-1)
            xs_ = xs_[inds_ok]
            ys_ = ys_[inds_ok]
            if len(xs_) > min_size:
                match_prev = single[0,0,ys_,xs_]
                bw_match_amount = torch.mean(match_prev)
                # print('match_amount on frame %d' % s, match_amount)
                if bw_match_amount < bw_thr:
                    return None, None, None
            else:
                bw_match_amount = torch.tensor(1.0, device='cuda')
        else:
            bw_match_amount = torch.tensor(1.0, device='cuda')
        bw_match_amounts.append(bw_match_amount)
    # print('hu match_amounts', torch.stack(hu_match_amounts).cpu().numpy())
    # print('fw match_amounts', torch.stack(fw_match_amounts).cpu().numpy())
    # print('bw match_amounts', torch.stack(bw_match_amounts).cpu().numpy())

    ys, xs = utils.basic.meshgrid2d(1, H, W)
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    xs = xs[singles[:,0].reshape(-1)>0]
    ys = ys[singles[:,0].reshape(-1)>0]
    if len(xs):
        # print('assembling trajs...'
        coords = []
        coord = torch.stack([xs, ys], dim=1) # N, 2
        coords.append(coord)
        for s in range(S-1):
            # delta = utils.samp.bilinear_sample2d(all_flows_f[:,s], coord[:,:,0].round(), coord[:,:,1].round()).permute(0,2,1) # 1,N,2: forward flow at the discrete points
            x_ = coord[:,0].round().long()
            y_ = coord[:,1].round().long()
            delta = all_flows_f[0,s,:,y_.clamp(0,H-1),x_.clamp(0,W-1)].permute(1,0) # 1,N,2: forward flow at the discrete points
            # print('delta', delta.shape)
            coord = coord + delta
            coords.append(coord)
        trajs_e = torch.stack(coords, dim=0).unsqueeze(0) # 1,S,N,2

        trajs_e = filter_trajs(trajs_e, all_masks, all_flows_f, all_flows_b)
    else:
        trajs_e = torch.zeros((1,S,0,2), dtype=torch.float32, device='cuda')

    return singles_fat, trajs_e, fw_match_amounts

def helper(rgb_path, mask_path, flow_path, out_dir, folder_name, lr, start_ind, sw=None, include_vis=False):
    
    cur_out_dir = os.path.join(out_dir, folder_name, lr)
    out_f = os.path.join(cur_out_dir, 'occluder_at_%d.npy' % (start_ind))
    if os.path.isfile(out_f):
        sys.stdout.write(':')
        return
    if not os.path.exists(cur_out_dir):
        os.makedirs(cur_out_dir)

    cur_rgb_path = os.path.join(rgb_path, folder_name, lr)
    cur_mask_path = os.path.join(mask_path, folder_name, lr)
    cur_flow_f_path = os.path.join(flow_path, folder_name, "into_future", lr)
    cur_flow_b_path = os.path.join(flow_path, folder_name, "into_past", lr)
    
    img_names = [folder.split('/')[-1].split('.')[0] for folder in glob.glob(os.path.join(cur_rgb_path, "*"))]
    img_names = sorted(img_names)
    masks = []
    for img_name in img_names:
        masks.append(readImage(os.path.join(cur_mask_path, '{0}.pfm'.format(img_name))))
    bak_all_masks = torch.from_numpy(np.stack(masks, 0)).to(device).unsqueeze(0).unsqueeze(2) # 1, S, 1, H, W

    # read rgbs and flows
    rgbs = []
    flows_f = []
    flows_b = []
    for img_name in img_names:
        rgbs.append(np.array(Image.open(os.path.join(cur_rgb_path, '{0}.webp'.format(img_name)))))
        try:
            if lr == "left":
                flows_f.append(readPFM(os.path.join(cur_flow_f_path, 'OpticalFlowIntoFuture_{0}_L.pfm'.format(img_name)))[:,:,:2])
                flows_b.append(readPFM(os.path.join(cur_flow_b_path, 'OpticalFlowIntoPast_{0}_L.pfm'.format(img_name)))[:,:,:2])
            else:
                flows_f.append(readPFM(os.path.join(cur_flow_f_path, 'OpticalFlowIntoFuture_{0}_R.pfm'.format(img_name)))[:,:,:2])
                flows_b.append(readPFM(os.path.join(cur_flow_b_path, 'OpticalFlowIntoPast_{0}_R.pfm'.format(img_name)))[:,:,:2])

        except FileNotFoundError:
            sys.stdout.write('!')
            sys.stdout.flush()
            return
    bak_all_rgbs = utils.improc.preprocess_color(torch.from_numpy(np.stack(rgbs, 0)).to(device)).permute(0,3,1,2).unsqueeze(0)
    bak_all_flows_f = torch.from_numpy(np.stack(flows_f, 0)).to(device).permute(0,3,1,2).unsqueeze(0)
    bak_all_flows_b = torch.from_numpy(np.stack(flows_b, 0)).to(device).permute(0,3,1,2).unsqueeze(0)

    _, bak_S, _, H, W = bak_all_rgbs.shape

    all_masks = bak_all_masks[:,start_ind:start_ind+min_lifespan]
    all_rgbs = bak_all_rgbs[:,start_ind:start_ind+min_lifespan]
    all_flows_f = bak_all_flows_f[:,start_ind:start_ind+min_lifespan-1]
    all_flows_b = bak_all_flows_b[:,start_ind+1:start_ind+min_lifespan+1]
    
    S = min_lifespan
    ids, counts = torch.unique(all_masks, return_counts=True)
    
    all_ids = []
    all_trajs = []

    save_d = {}
    
    for ii in range(len(ids)):
        id_ = ids[ii]

        if include_vis:
            sw.summ_rgbs('inputs_%d/rgbs' % start_ind, all_rgbs.unbind(1))
            sw.summ_oneds('inputs_%d/masks' % start_ind, all_masks.unbind(1))

        singles, trajs, stats = consider_id(id_, all_rgbs, all_masks, all_flows_f, all_flows_b)

        if singles is not None:
            N = trajs.shape[2]
            if include_vis:
                sw.summ_rgbs('inputs_%d/singles_rgb_%d' % (start_ind, ii), ((singles*(all_rgbs+0.5))-0.5).unbind(1), frame_ids=stats)
                max_show = 100
                if trajs.shape[2] > max_show:
                    inds = utils.geom.farthest_point_sample(trajs[:,0], max_show, deterministic=False)
                    trajs_ = trajs[:,:,inds.reshape(-1)]
                else:
                    trajs_ = trajs.clone()
                if trajs_.shape[2] > 0:
                    sw.summ_traj2ds_on_rgbs('outputs_%d/trajs_on_single_%d' % (start_ind, ii), trajs_, (singles*(all_rgbs+0.5))-0.5, cmap='winter', frame_ids=list(range(S)))
            trajs = trajs[0].detach().cpu().numpy().astype(np.float16)
            id_ = id_.detach().cpu().numpy()

            save_d['%d' % int(id_)] = trajs
            all_trajs.append(trajs)
            all_ids.append(id_)
        # endif not None
    # end loop over ids

    np.save(out_f, save_d)
    sys.stdout.write('.')
    sys.stdout.flush()
    

def go():
    ## choose hyps
    log_freq = 1
    include_vis = False

    log_dir = 'logs_make_occlusions'
    import datetime
    exp_date = datetime.datetime.now().strftime('%H:%M:%S')
    exp_name = '%s' % (exp_date)
    print(exp_name)
    writer = SummaryWriter(log_dir + '/' + exp_name, max_queue=10, flush_secs=60)

    global_step = 0
    for dset in dsets:
        for subset in subsets:
            rgb_path = os.path.join(flt3d_path, "frames_cleanpass_webp", dset, subset)
            flow_path = os.path.join(flt3d_path, "optical_flow", dset, subset)
            mask_path = os.path.join(flt3d_path, "object_index", dset, subset)
            folder_names = [folder.split('/')[-1] for folder in glob.glob(os.path.join(rgb_path, "*"))]
            print(flt3d_path, dset, subset, mod)

            out_dir = os.path.join(flt3d_path, "occluders_%s" % mod, dset, subset)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            random.shuffle(folder_names)

            for folder_name in folder_names:
                for lr in ['left', 'right']:
                    for start_ind in [0,1,2]:
                        global_step += 1
                        sw = utils.improc.Summ_writer(
                            writer=writer,
                            global_step=global_step,
                            log_freq=log_freq,
                            fps=5,
                            scalar_freq=100,
                            just_gif=True)
                        helper(rgb_path, mask_path, flow_path, out_dir, folder_name, lr, start_ind, sw=sw, include_vis=include_vis)
                        sys.stdout.flush()
            print('done')

if __name__ == "__main__":
    go()
