import time
import argparse
import numpy as np
import timeit
import matplotlib
# import tensorflow as tf
# import scipy.misc
import io
import os
import math
from PIL import Image
matplotlib.use('Agg') # suppress plot showing

import matplotlib.pyplot as plt

import matplotlib.animation as animation
import cv2
import saverloader
from nets.singlepoint import Singlepoint

import utils.py
# import utils.box
import utils.misc
import utils.improc
# import utils.vox
import utils.grouping
from tqdm import tqdm
import random
import glob
# import color2d

from utils.basic import print_, print_stats

# import datasets
import flyingthingsdataset
# import cater_pointtraj_dataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

import torch.nn.functional as F

# import inputs

device = 'cuda'
patch_size = 8
random.seed(125)
np.random.seed(125)

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    """ Create the optimizer and learning rate scheduler """
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler




def run_model(model, d, I=6, horz_flip=False, vert_flip=False, sw=None):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    # metrics = {
    #     'epe_all': 0,
    #     'epe_vis': 0,
    #     'epe_occ': 0,
    # }

    # flow = d['flow'].cuda().permute(0, 3, 1, 2)
    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    occs = d['occs'].cuda().float() # B, S, 1, H, W
    masks = d['masks'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    vis_g = d['visibles'].cuda().float() # B, S, N
    valids = d['valids'].cuda().float() # B, S, N

    B, S, C, H, W = rgbs.shape
    assert(C==3)
    B, S, N, D = trajs_g.shape

    assert(torch.sum(valids)==B*S*N)

    # lengths = torch.norm(trajs_g[:,-1]-trajs_g[:,0], dim=-1) # B, N
    # # valids = valids * (lengths < 1000).float().reshape(B, 1, N)
    # valids = valids * (lengths < W*2).float().reshape(B, 1, N)

    # print_stats('rgbs', rgbs)
    # print_stats('trajs_g', trajs_g)
    # print_stats('vis_g', vis_g)
    # print_stats('valids', valids)
    # print_stats('masks', masks)
    # print_stats('occs', occs)

    if horz_flip:
        rgbs_flip = torch.flip(rgbs, [4])
        occs_flip = torch.flip(occs, [4])
        masks_flip = torch.flip(masks, [4])
        trajs_g_flip = trajs_g.clone()
        trajs_g_flip[:,:,:,0] = W-1 - trajs_g_flip[:,:,:,0]
        vis_g_flip = vis_g.clone()
        valids_flip = valids.clone()
        trajs_g = torch.cat([trajs_g, trajs_g_flip], dim=0)
        vis_g = torch.cat([vis_g, vis_g_flip], dim=0)
        valids = torch.cat([valids, valids_flip], dim=0)
        rgbs = torch.cat([rgbs, rgbs_flip], dim=0)
        occs = torch.cat([occs, occs_flip], dim=0)
        masks = torch.cat([masks, masks_flip], dim=0)
        B = B * 2

    if vert_flip:
        rgbs_flip = torch.flip(rgbs, [3])
        occs_flip = torch.flip(occs, [3])
        masks_flip = torch.flip(masks, [3])
        trajs_g_flip = trajs_g.clone()
        trajs_g_flip[:,:,:,1] = H-1 - trajs_g_flip[:,:,:,1]
        vis_g_flip = vis_g.clone()
        valids_flip = valids.clone()
        trajs_g = torch.cat([trajs_g, trajs_g_flip], dim=0)
        vis_g = torch.cat([vis_g, vis_g_flip], dim=0)
        valids = torch.cat([valids, valids_flip], dim=0)
        rgbs = torch.cat([rgbs, rgbs_flip], dim=0)
        occs = torch.cat([occs, occs_flip], dim=0)
        masks = torch.cat([masks, masks_flip], dim=0)
        B = B * 2

    # print('rgbs out', rgbs.shape)
    # preds, preds2, fcps, ccps, vis_e = model(trajs_g[:,0], rgbs, coords_init=None, iters=I, coords_g=trajs_g, vis_g=vis_g, valids=valids, sw=sw)
    # preds, preds2, vis_e, seq_loss, vis_loss, ce_loss = model(trajs_g[:,0], rgbs, coords_init=None, iters=I, trajs_g=trajs_g, vis_g=vis_g, valids=valids, sw=sw)
    preds, preds2, vis_e, stats = model(trajs_g[:,0], rgbs, coords_init=None, iters=I, trajs_g=trajs_g, vis_g=vis_g, valids=valids, sw=sw)
    # preds is a list of B,S,N,2 elements
    seq_loss, vis_loss, ce_loss = stats
    
    total_loss += seq_loss.mean()
    total_loss += vis_loss.mean()
    total_loss += ce_loss.mean()

    ate = torch.norm(preds[-1] - trajs_g, dim=-1) # B, S, N
    ate_all = utils.basic.reduce_masked_mean(ate, valids)
    ate_vis = utils.basic.reduce_masked_mean(ate, valids*vis_g)
    ate_occ = utils.basic.reduce_masked_mean(ate, valids*(1.0-vis_g))
    
    metrics = {
        'ate_all': ate_all.item(),
        'ate_vis': ate_vis.item(),
        'ate_occ': ate_occ.item(),
        'seq': seq_loss.mean().item(),
        'vis': vis_loss.mean().item(),
        'ce': ce_loss.mean().item()
    }
    # print_stats('preds[0]', preds[0])
    # print_stats('fcps', fcps)

    # vis_loss, _ = balanced_ce_loss(vis_e, vis_g)
    # total_loss += vis_loss

    # use_ce = True
    # if use_ce:
    #     # compute cross entropy loss on the heatmaps
    #     stride = model.module.stride
    #     H8, W8 = H//stride, W//stride
    #     # fcps is B,S,I,N,H8,W8

    # def score_map_loss(fcps, trajs_g, vis_g, valids):
    #     fcp_ = fcps.permute(0,1,3,2,4,5).reshape(B*S*N,I,H8,W8) # BSN,I,H8,W8
    #     # print('fcp_', fcp_.shape)
    #     xy_ = (trajs_g.reshape(B*S*N,2)/stride).round().long() # BSN,2
    #     vis_ = vis_g.reshape(B*S*N) # BSN
    #     valid_ = valids.reshape(B*S*N) # BSN
    #     x_, y_ = xy_[:,0], xy_[:,1] # BSN
    #     print('x_', x_.shape)
    #     print('y_', y_.shape)
    #     print('vis_', vis_.shape)
    #     print('valid_', valid_.shape)
    #     ind = (x_ >= 0) & (x_ <= (W8-1)) & (y_ >= 0) & (y_ <= (H8-1)) & (valid_ > 0) & (vis_ > 0) # BSN
    #     print('ind', ind.shape, torch.sum(ind))
    #     # if torch.sum(ind) > 0:
    #     fcp_ = fcp_[ind] # N_,I,H8,W8
    #     xy_ = xy_[ind] # N_
    #     N_ = fcp_.shape[0]
    #     print('fcp_', fcp_.shape)
    #     print('xy_', xy_.shape)

    #     # N_ is the number of heatmaps with valid targets
        
    #     # make gt with ones at the rounded spatial inds in here
    #     gt_ = torch.zeros_like(fcp_) # N_,I,H8,W8
    #     gt_[:,:,xy_[:,1],xy_[:,0]] = 1 # N_,I,H8,W8 with a 1 in the right spot
    #     # fcp_ = fcp_.reshape(N_*I,H8*W8)
    #     # gt_ = gt_.reshape(N_*I,H8*W8)
    #     # argm = torch.argmax(gt_, dim=1)
    #     # ce_loss = F.cross_entropy(fcp_, argm, reduction='mean')

    #     fcp_ = fcp_.reshape(N_*I*H8*W8)
    #     gt_ = gt_.reshape(N_*I*H8*W8)
    #     # ce_loss = F.binary_cross_entropy_with_logits(fcp_, gt_, reduction='mean')
    #     ce_loss, _ = balanced_ce_loss(fcp_, gt_)
    #     print('ce_loss', ce_loss)
    #     total_loss += ce_loss
    #     metrics['ce'] = ce_loss.item()
    # else:
    #     metrics['ce'] = 0
    
    # fcp_ = fcp_.reshape(H8*W8)
    
    # # cp_e = utils.samp.bilinear_sample2d(__p(fcps[:,:,-1]), __p(trajs_e)[:,0:1,0]/stride, __p(trajs_e)[:,0:1,1]/stride))
    # if False:
    #     cp_e = []
    #     argm = []
    #     for b in range(B):
    #         for s in range(S):
    #             for n in range(N):
    #                 cp = fcps[b,s,-1,n] # H8, W8
    #                 xy = (trajs_g[b,s,n]/stride).round().long() # 2
    #                 x, y = xy[0], xy[1]
    #                 if (x >= 0 and
    #                     x <= W8-1 and
    #                     y >= 0 and
    #                     y <= H8-1 and
    #                     vis_g[b,s,n] > 0 and
    #                     valids[b,s,n] > 0 
    #                 ):
    #                     heatmap_g = torch.zeros_like(cp)
    #                     heatmap_g[y,x] = 1
    #                     cp_e.append(cp.reshape(1, -1))
    #                     cp_g = heatmap_g.reshape(1, -1)
    #                     argm.append(torch.argmax(cp_g, dim=1))
    #     if len(cp_e):
    #         cp_e = torch.cat(cp_e, dim=0)
    #         argm = torch.cat(argm, dim=0)
    #         ce_loss = F.cross_entropy(cp_e, argm, reduction='mean')
    #         total_loss += ce_loss
    #     else:
    #         ce_loss = total_loss*0
    #     metrics['ce'] = ce_loss.item()
    
    if sw is not None and sw.save_this:
        trajs_e = preds[-1]

        # pad_x0 = int((-torch.min(trajs_g[:,:,:,0])).clamp(min=0).round().item())
        # pad_x1 = int((torch.max(trajs_g[:,:,:,0]) - W).clamp(min=0).round().item())
        # pad_y0 = int((-torch.min(trajs_g[:,:,:,1])).clamp(min=0).round().item())
        # pad_y1 = int((torch.max(trajs_g[:,:,:,1]) - H).clamp(min=0).round().item())
        # pad_x0 = 50
        # pad_y0 = 50
        # pad_x1 = 50
        # pad_y1 = 50

        pad = 50
        rgbs = F.pad(rgbs.reshape(B*S, 3, H, W), (pad, pad, pad, pad), 'constant', 0).reshape(B, S, 3, H+pad*2, W+pad*2)
        occs = F.pad(occs.reshape(B*S, 1, H, W), (pad, pad, pad, pad), 'constant', 0).reshape(B, S, 1, H+pad*2, W+pad*2)
        masks = F.pad(masks.reshape(B*S, 1, H, W), (pad, pad, pad, pad), 'constant', 0).reshape(B, S, 1, H+pad*2, W+pad*2)
        trajs_e = trajs_e + pad
        trajs_g = trajs_g + pad
        
        # rgbs = F.pad(rgbs.reshape(B*S, 3, H, W), (pad_x0, pad_x1, pad_y0, pad_y1), 'constant', 0).reshape(B, S, 3, H+pad_y0+pad_y1, W+pad_x0+pad_x1)
        # trajs_e[:,:,0] += pad_x0
        # trajs_e[:,:,1] += pad_y0
        # trajs_g[:,:,0] += pad_x0
        # trajs_g[:,:,1] += pad_y0

        occs_ = occs[0].reshape(S, -1)
        counts_ = torch.max(occs_, dim=1)[0]
        # print('counts_', counts_)
        sw.summ_rgbs('inputs_0/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1))
        sw.summ_oneds('inputs_0/occs', occs.unbind(1), frame_ids=counts_)
        sw.summ_oneds('inputs_0/masks', masks.unbind(1), frame_ids=counts_)
        sw.summ_traj2ds_on_rgbs2('inputs_0/trajs_g_on_rgbs', trajs_g[0:1], vis_g[0:1], utils.improc.preprocess_color(rgbs[0:1]), valids=valids[0:1], cmap='winter')
        sw.summ_traj2ds_on_rgb('inputs_0/trajs_g_on_rgb', trajs_g[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1), cmap='winter')

        for b in range(B):
            sw.summ_traj2ds_on_rgb('batch_inputs_0/trajs_g_on_rgb_%d' % b, trajs_g[b:b+1], torch.mean(utils.improc.preprocess_color(rgbs[b:b+1]), dim=1), cmap='winter')

        sw.summ_traj2ds_on_rgbs2('outputs/trajs_e_on_rgbs', trajs_e[0:1], torch.sigmoid(vis_e[0:1]), utils.improc.preprocess_color(rgbs[0:1]), cmap='spring')
        # sw.summ_traj2ds_on_rgbs('outputs/trajs_on_black', trajs_e[0:1], torch.ones_like(rgbs[0:1])*-0.5, cmap='spring')

        gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('', trajs_g[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1), valids=valids[0:1], cmap='winter', frame_id=metrics['ate_all'], only_return=True))
        gt_black = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('', trajs_g[0:1], torch.ones_like(rgbs[0:1,0])*-0.5, valids=valids[0:1], cmap='winter', frame_id=metrics['ate_all'], only_return=True))
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_rgb', trajs_e[0:1], gt_rgb[0:1], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_black', trajs_e[0:1], gt_black[0:1], cmap='spring')

        if False:
            rgb_vis = []
            black_vis = []
            for trajs_e in preds2:
                rgb_vis.append(sw.summ_traj2ds_on_rgb('', trajs_e[0:1], gt_rgb, only_return=True, cmap='spring'))
                black_vis.append(sw.summ_traj2ds_on_rgb('', trajs_e[0:1], gt_black, only_return=True, cmap='spring'))
            sw.summ_rgbs('outputs/animated_trajs_on_black', black_vis)
            sw.summ_rgbs('outputs/animated_trajs_on_rgb', rgb_vis)

    return total_loss, metrics
    
def train():

    # default coeffs (don't touch)
    init_dir = ''
    coeff_prob = 0.0
    use_augs = False
    load_optimizer = False
    load_step = False
    ignore_load = None

    # the idea here is to train a basic model on the upgraded flt
    exp_name = 'tb00' # I=6 < ok but still slowing
    exp_name = 'tb01' # N=512 < OOM
    exp_name = 'tb02' # I=4
    exp_name = 'tb03' # B1, but double the data inside the runner; ensure it's being split < yes. but still some hiccups
    exp_name = 'tb04' # I=8
    exp_name = 'tb05' # I=12, N128
    exp_name = 'tb06' # I=8, N128, higher resolution input, to slow down the cnn < 0.6s
    exp_name = 'tb07' # 256 < OOM
    exp_name = 'tb08' # 384,384; 256 < 0.9s; let's go with this. 
    exp_name = 'tb09' # sb
    exp_name = 'tb10' # one more flip; 4gpu
    exp_name = 'tb11' # 4gpu proper; sb
    exp_name = 'tb12' # interactive, chasing (but maybe dying in 8h) < ah, oom. and the other is too probably
    exp_name = 'tb13' # N=128, to not go oom < 0.9
    exp_name = 'tb14' # N=256, crop 256,384 < 1.0
    exp_name = 'tb15' # do lots of val steps, to tempt oom
    # looks great
    exp_name = 'tb16' # val_freq 100; sb
    exp_name = 'tb17' # interactive, chase
    exp_name = 'tb18' # val_freq 50; radius=3, levels=4 (instead of 4,3)
    # next i want to investigate using bilinear samples from the featmap, to help the model judge visibility and so on
    exp_name = 'tb19' # quick; no flips; b1; 1gpu
    exp_name = 'tb20' # stride 1/4 instead of 1/8
    exp_name = 'tb21' # quick vis
    exp_name = 'tb22' # cat finer features too
    exp_name = 'tb23' # conv2 3x3, then norm, relu, conv3 
    exp_name = 'tb24' # show gt in the tff vis; 10 itesr
    exp_name = 'tb25' # go 1k, log100
    exp_name = 'tb26' # only show kp version
    exp_name = 'tb27' # radius=3,levels=4, to match tb18 < killed
    # looks fine
    # the boundary effects (in both large moa are a bit bothersome
    # though honestly they did reduce a bit, with the extra scale added
    exp_name = 'tb28' # restore flips for B4; train on sbatch, to see the effects of the net upgrade
    # for this i need to kill something
    # ok... oom on g1
    exp_name = 'tb29' # crop 384,384 since i'm using bigger net now
    # still oom. why?
    exp_name = 'tb30' # I=6
    exp_name = 'tb31' # inspect the elements
    exp_name = 'tb32' # sb < running
    # ok, so this version has maybe a more difficult job, since it's 384x384 and I=6, but if it wins then i'm happy
    # < heavy boundary effect!
    exp_name = 'tb33' # sb; reflect padding instead of zero
    exp_name = 'tb34' # sb; b1, 1gpu (baby version of tb33)
    exp_name = 'tb35' # cat the corr valids; quick
    exp_name = 'tb36' # sb; b1; 1gpu; quick=False
    # both of these are having some trouble <5k
    # but i can relax and let them run
    # actually, both are tb36
    # so let's kill and try b2, to see if we approach a b4 model at all
    exp_name = 'tb37' # sb; b2 (horz); 2gpu; quick=False
    exp_name = 'tb38' # sb; b4; 4gpu; quick=False < nan
    # so tb38 and tb32 are running
    # i am interested to see if boundary effects persist
    # esp. in tb38, which uses mirror. 
    # and, if tb38 wins, maybe those oob indicators are helping
    # and if either of these go under the tb16 model, it means my architecture change helped
    exp_name = 'tb39' # sb; b4; 4gpu; quick=False; fixed bug with nan (added an else in the pca func) < died later, due to loss/params being nan
    exp_name = 'tb40' # align_corners=True for the fcp_ interp; quick; in another model, disable ce
    exp_name = 'tb41' # back to normal, for a baseline
    # ok none of these are producing a clear top boundary effect
    exp_name = 'tb42' # padding_mode=zeros
    exp_name = 'tb43' # align_corners=True for fcp
    exp_name = 'tb44' # padding_mode='reflect'; print about the ce, to maybe inform about the nan that killed tb38 and tb39
    # but why didn't tb32 fail?
    # what are the diffs?
    # < reflect padding
    # < corr valid concat
    # < 
    exp_name = 'tb45' # quick=False; padding_mode='reflect'; print about the ce, to maybe inform about the nan that killed tb38 and tb39
    exp_name = 'tb46' # bce instead of ce
    exp_name = 'tb47' # balanced ce instead of bce
    # OOM
    exp_name = 'tb48' # I = 5 instead of 6
    exp_name = 'tb49' # I = 4
    # why are all these oom? could it be that the fcp tensor is huge, and i'm making two of them now, so that's why?
    # no wait, not oom. it's printing an error
    exp_name = 'tb50' # I = 5
    exp_name = 'tb51' # I = 4,3
    # oooh now i know:
    # the issue is: we are computing the cost volume stuff outside,
    # so it's all being piled onto the zeroth gpu
    # let me fix this
    exp_name = 'tb52' # everything inside; 1gpu; quick
    exp_name = 'tb53' # I = 8
    exp_name = 'tb54' # 4gpu
    exp_name = 'tb55' # I = 6
    exp_name = 'tb56' # quick=False; sb
    exp_name = 'tb57' # no ce
    # ok i think with the balanced loss, i fixed the nan. somehow i'm running into some instability with the native functions, maybe to do with my indexing
    # no! not fixed. everything died
    # so it must be the corr valid concat... let's disable it and see
    exp_name = 'tb58' # disable corr valid concat
    exp_name = 'tb59' # detach corr valid concat
    exp_name = 'tb60' # no corr valid
    exp_name = 'tb61' # use_ones with detach
    exp_name = 'tb62' # use_ones=False
    exp_name = 'tb63' # use_ones=True (with detach)
    # wow tb62 nan
    # and note everything since tb57 has no ce
    # it's really shocking for tb62 to fail
    # tb63 nan by ~12k
    exp_name = 'tb64' # print all three losses, and print stats on coords, so i can see which one gives nan first < first coord pred is the first nan
    exp_name = 'tb65' # 1gpu to chase the nan < nan occured after an iter with high loss, so maybe discarding huge trajs will help
    exp_name = 'tb66' # padding_mode='zeros' < no more nan... could it really be the padding?
    exp_name = 'tb67' # grad_acc = 4
    exp_name = 'tb68' # clip 5 instead of 1
    exp_name = 'tb69' # valid = length<1k < killed by typo
    exp_name = 'tb70' # valid = length<W*2; divide flow loss by n_predictions
    exp_name = 'tb71' # valid = length<W*2; padding='reflect'
    exp_name = 'tb72' # valid = length<W*2; padding='replicate'
    exp_name = 'tb73' # valid=ones
    exp_name = 'tb74' # 4gpu; padding='replicate'
    # ok, everything without zero-padding arrived to nan
    exp_name = 'tb75' # 4gpu; padding='zeros'; init from tb70; go 200k < killed by accidentally deleting the slurm file 
    exp_name = 'tb76' # same but higher res: 384,512 < killed since tb77 is similar
    exp_name = 'tb77' # lr 4e-4 < killed bc tb78 is similar 
    exp_name = 'tb78' # args for flips and I; grad_acc=1 < queued as 3858631 < actually, 4e-4 never took effect bc of init
    exp_name = 'tb79' # stride=8 instead of stride=4, no init  
    exp_name = 'tb80' # elim layer4 and conv3; back to 3e-4; fix bug in animated coords-on-heat vis
    exp_name = 'tb81' # temp; just tell me some stats
    exp_name = 'tb82' # .train() at end of val step
    # probably something needs updating, since i updated in the tester
    exp_name = 'tb83' # train interactive, stride8, shallow=True, 100k
    exp_name = 'tb84' # init from tb83; load step, optimizer, to maybe start from 6k
    exp_name = 'tb85' # init from tb83; load step, optimizer, to maybe start from 6k; shallow=True < 3860500
    exp_name = 'tb86' # fix bug in fcp vis
    exp_name = 'tb87' # re-run with quick=False on 4 gpus
    exp_name = 'tb88' # no flips, so i can run quickly  

    # init_dir = 'checkpoints/01_8_128_3e-4_p1_A_tb70_23:16:18'
    init_dir = ''
    # init_dir = 'checkpoints/4hv_8_128_I6_3e-4_p1_A_tb83_12:42:52'
    # load_optimizer = True
    # load_step = True
    # ignore_load = None

    ## choose hyps
    B = 1
    S = 8
    N = 128
    lr = 3e-4
    grad_acc = 1
    horz_flip = False
    vert_flip = False
    stride = 8
    # horz_flip = False
    # vert_flip = False
    I = 6
    
    # crop_size = (384,384) # the raw data is 540,960
    crop_size = (384,512) # the raw data is 540,960
    # crop_size = (384,512) # the raw data is 540,960
    # crop_size = (384,512) # the raw data is 540,960
    # crop_size = (256,384) # the raw data is 540,960
    assert(crop_size[0] % 128 == 0)
    assert(crop_size[1] % 128 == 0)

    quick = True
    quick = False
    if quick:
        # max_iters = 10000
        # log_freq = 500
        max_iters = 1000
        log_freq = 100
        save_freq = 999999
        shuffle = False
        do_val = False
        val_freq = 4
        cache_len = 11
        cache_freq = 99999999
        subset = 'A'
        use_augs = True
    else:
        max_iters = 100000
        log_freq = 500
        val_freq = 50
        save_freq = 1000
        shuffle = True
        do_val = True
        cache_len = 0
        cache_freq = 99999999
        subset = 'all'
        use_augs = True
    
    # actual coeffs
    coeff_prob = 1.0

    ## autogen a name
    if horz_flip and vert_flip:
        model_name = "%dhv" % (B*4)
    elif horz_flip:
        model_name = "%dh" % (B*2)
    elif vert_flip:
        model_name = "%dv" % (B*2)
    else:
        model_name = "%d" % (B)
    model_name += "_%d_%d" % (S, N)
    model_name += "_I%d" % (I)
    
    lrn = "%.1e" % lr # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1] # e.g., 5e-4
    model_name += "_%s" % lrn
    if cache_len:
        model_name += "_cache%d" % cache_len
    all_coeffs = [
        coeff_prob,
    ]
    all_prefixes = [
        "p",
    ]
    for l_, l in enumerate(all_coeffs):
        if l > 0:
            model_name += "_%s%s" % (all_prefixes[l_], utils.basic.strnum(l))
    if use_augs:
        model_name += "_A"
    model_name += "_%s" % exp_name
    
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    ckpt_dir = 'checkpoints/%s' % model_name
    log_dir = 'logs_train_basic'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)
    if do_val:
        writer_v = SummaryWriter(log_dir + '/' + model_name + '/v', max_queue=10, flush_secs=60)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    
    train_dataset = flyingthingsdataset.FlyingThingsDataset(
        dset='TRAIN', subset=subset,
        use_augs=use_augs,
        N=N, S=S,
        crop_size=crop_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=12,
        worker_init_fn=worker_init_fn,
        drop_last=True)
    train_iterloader = iter(train_dataloader)
    
    if cache_len:
        print('we will cache %d' % cache_len)
        sample_pool = utils.misc.SimplePool(cache_len, version='np')
    
    if do_val:
        print('not using augs in val')
        val_dataset = flyingthingsdataset.FlyingThingsDataset(
            dset='TEST', subset='all',
            use_augs=use_augs,
            N=N, S=S,
            crop_size=crop_size)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=B,
            shuffle=shuffle,
            num_workers=4,
            drop_last=False)
        val_iterloader = iter(val_dataloader)
    
    
    model = Singlepoint(stride=stride).cuda()
    model = torch.nn.DataParallel(model)
    parameters = list(model.parameters())
    # optimizer, scheduler = fetch_optimizer(lr, 0.0001, 1e-8, max_iters//grad_acc, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=1e-7)

    global_step = 0
    if init_dir:
        if load_step and load_optimizer:
            global_step = saverloader.load(init_dir, model.module, optimizer, ignore_load=ignore_load)
        elif load_step:
            global_step = saverloader.load(init_dir, model.module, ignore_load=ignore_load)
        else:
            _ = saverloader.load(init_dir, model.module, ignore_load=ignore_load)
            global_step = 0
    requires_grad(parameters, True)
    model.train()
    

    n_pool = 100
    loss_pool_t = utils.misc.SimplePool(n_pool, version='np')
    ce_pool_t = utils.misc.SimplePool(n_pool, version='np')
    vis_pool_t = utils.misc.SimplePool(n_pool, version='np')
    seq_pool_t = utils.misc.SimplePool(n_pool, version='np')
    ate_all_pool_t = utils.misc.SimplePool(n_pool, version='np')
    ate_vis_pool_t = utils.misc.SimplePool(n_pool, version='np')
    ate_occ_pool_t = utils.misc.SimplePool(n_pool, version='np')
    if do_val:
        loss_pool_v = utils.misc.SimplePool(n_pool, version='np')
        ce_pool_v = utils.misc.SimplePool(n_pool, version='np')
        vis_pool_v = utils.misc.SimplePool(n_pool, version='np')
        seq_pool_v = utils.misc.SimplePool(n_pool, version='np')
        ate_all_pool_v = utils.misc.SimplePool(n_pool, version='np')
        ate_vis_pool_v = utils.misc.SimplePool(n_pool, version='np')
        ate_occ_pool_v = utils.misc.SimplePool(n_pool, version='np')
    
    while global_step < max_iters:
        
        read_start_time = time.time()
        
        global_step += 1
        total_loss = torch.tensor(0.0, requires_grad=True).to(device)

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=5,
            scalar_freq=int(log_freq/2),
            just_gif=True)

        if cache_len:
            if (global_step) % cache_freq == 0:
                sample_pool.empty()
            
            if len(sample_pool) < cache_len:
                print('caching a new sample')
                try:
                    sample = next(train_iterloader)
                except StopIteration:
                    train_iterloader = iter(train_dataloader)
                    sample = next(train_iterloader)
                # sample['rgbs'] = sample['rgbs'].cpu().detach().numpy()
                # sample['masks'] = sample['masks'].cpu().detach().numpy()
                sample_pool.update([sample])
            else:
                sample = sample_pool.sample()
        else:
            try:
                sample = next(train_iterloader)
            except StopIteration:
                train_iterloader = iter(train_dataloader)
                sample = next(train_iterloader)
            # sample['rgbs'] = sample['rgbs'].cpu().detach().numpy()
            # sample['masks'] = sample['masks'].cpu().detach().numpy()

        read_time = time.time()-read_start_time
        iter_start_time = time.time()
            
        total_loss, metrics = run_model(model, sample, I, horz_flip, vert_flip, sw_t)

        sw_t.summ_scalar('total_loss', total_loss)
        loss_pool_t.update([total_loss.detach().cpu().numpy()])
        sw_t.summ_scalar('pooled/total_loss', loss_pool_t.mean())

        if metrics['ate_all'] > 0:
            ate_all_pool_t.update([metrics['ate_all']])
        if metrics['ate_vis'] > 0:
            ate_vis_pool_t.update([metrics['ate_vis']])
        if metrics['ate_occ'] > 0:
            ate_occ_pool_t.update([metrics['ate_occ']])
        if metrics['ce'] > 0:
            ce_pool_t.update([metrics['ce']])
        if metrics['vis'] > 0:
            vis_pool_t.update([metrics['vis']])
        if metrics['seq'] > 0:
            seq_pool_t.update([metrics['seq']])
        sw_t.summ_scalar('pooled/ate_all', ate_all_pool_t.mean())
        sw_t.summ_scalar('pooled/ate_vis', ate_vis_pool_t.mean())
        sw_t.summ_scalar('pooled/ate_occ', ate_occ_pool_t.mean())
        sw_t.summ_scalar('pooled/ce', ce_pool_t.mean())
        sw_t.summ_scalar('pooled/vis', vis_pool_t.mean())
        sw_t.summ_scalar('pooled/seq', seq_pool_t.mean())

        total_loss.backward()
        
        if (global_step) % grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

        if do_val and (global_step) % val_freq == 0:
            torch.cuda.empty_cache()
            model.eval()
            sw_v = utils.improc.Summ_writer(
                writer=writer_v,
                global_step=global_step,
                log_freq=log_freq,
                fps=5,
                scalar_freq=int(log_freq/2),
                just_gif=True)
            try:
                sample = next(val_iterloader)
            except StopIteration:
                val_iterloader = iter(val_dataloader)
                sample = next(val_iterloader)

            with torch.no_grad():
                total_loss, metrics = run_model(model, sample, I, horz_flip, vert_flip, sw_v)

            sw_v.summ_scalar('total_loss', total_loss)
            loss_pool_v.update([total_loss.detach().cpu().numpy()])
            sw_v.summ_scalar('pooled/total_loss', loss_pool_v.mean())

            if metrics['ate_all'] > 0:
                ate_all_pool_v.update([metrics['ate_all']])
            if metrics['ate_vis'] > 0:
                ate_vis_pool_v.update([metrics['ate_vis']])
            if metrics['ate_occ'] > 0:
                ate_occ_pool_v.update([metrics['ate_occ']])
            if metrics['ce'] > 0:
                ce_pool_v.update([metrics['ce']])
            if metrics['vis'] > 0:
                vis_pool_v.update([metrics['vis']])
            if metrics['seq'] > 0:
                seq_pool_v.update([metrics['seq']])
            sw_v.summ_scalar('pooled/ate_all', ate_all_pool_v.mean())
            sw_v.summ_scalar('pooled/ate_vis', ate_vis_pool_v.mean())
            sw_v.summ_scalar('pooled/ate_occ', ate_occ_pool_v.mean())
            sw_v.summ_scalar('pooled/ce', ce_pool_v.mean())
            sw_v.summ_scalar('pooled/vis', vis_pool_v.mean())
            sw_v.summ_scalar('pooled/seq', seq_pool_v.mean())
            model.train()

        if np.mod(global_step, save_freq)==0:
            saverloader.save(ckpt_dir, optimizer, model.module, global_step, keep_latest=1)

        current_lr = optimizer.param_groups[0]['lr']
        sw_t.summ_scalar('_/current_lr', current_lr)
        
        iter_time = time.time()-iter_start_time
        print('%s; step %06d/%d; rtime %.2f; itime %.2f; loss = %.5f' % (
            model_name, global_step, max_iters, read_time, iter_time,
            total_loss.item()))
            
    writer_t.close()
    if do_val:
        writer_v.close()
            

if __name__ == '__main__':
    train()
