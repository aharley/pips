import time
import numpy as np
import timeit
import imageio
import matplotlib
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
from utils.basic import readPFM
import random
import glob
from filter_trajs import filter_trajs
from tensorboardX import SummaryWriter

flt3d_path = "../flyingthings"

dsets = ["TRAIN", "TEST"]
subsets = ["A", "B", "C"]

device = 'cuda'

min_lifespan = 8

mod = 'aa' # start export on orion
mod = 'ab' # float16 instead of float32, to save space
mod = 'ac' # req 256 trajs
mod = 'ad' # fix bug in filter_trajs, with visibility on last frame

def readImage(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        data = readPFM(name)
        if len(data.shape)==3:
            return data[:,:,0:3]
        else:
            return data
    return imageio.imread(name)


def helper(rgb_path, mask_path, flow_path, out_dir, folder_name, lr, start_ind, sw=None, include_vis=False):
    
    cur_out_dir = os.path.join(out_dir, folder_name, lr)
    out_f = os.path.join(cur_out_dir, 'trajs_at_%d.npz' % start_ind)
    # print('out_f', out_f)
    if os.path.isfile(out_f):
        sys.stdout.write(':')
        return

    cur_rgb_path = os.path.join(rgb_path, folder_name, lr)
    cur_mask_path = os.path.join(mask_path, folder_name, lr)
    cur_flow_f_path = os.path.join(flow_path, folder_name, "into_future", lr)
    cur_flow_b_path = os.path.join(flow_path, folder_name, "into_past", lr)

    img_names = [folder.split('/')[-1].split('.')[0] for folder in glob.glob(os.path.join(cur_rgb_path, "*"))]
    img_names = sorted(img_names)

    # read rgbs and flows
    rgbs = []
    masks = []
    flows_f = []
    flows_b = []
    # segs = []
    for img_name in img_names:
        rgbs.append(np.array(Image.open(os.path.join(cur_rgb_path, '{0}.webp'.format(img_name)))))
        masks.append(readImage(os.path.join(cur_mask_path, '{0}.pfm'.format(img_name))))
        try:
            if lr == "left":
                flows_f.append(readPFM(os.path.join(cur_flow_f_path, 'OpticalFlowIntoFuture_{0}_L.pfm'.format(img_name)))[:,:,:2])
                flows_b.append(readPFM(os.path.join(cur_flow_b_path, 'OpticalFlowIntoPast_{0}_L.pfm'.format(img_name)))[:,:,:2])
            else:
                flows_f.append(readPFM(os.path.join(cur_flow_f_path, 'OpticalFlowIntoFuture_{0}_R.pfm'.format(img_name)))[:,:,:2])
                flows_b.append(readPFM(os.path.join(cur_flow_b_path, 'OpticalFlowIntoPast_{0}_R.pfm'.format(img_name)))[:,:,:2])

        except FileNotFoundError:
            sys.stdout.write('!')
            return

    bak_all_rgbs = utils.improc.preprocess_color(torch.from_numpy(np.stack(rgbs, 0)).to(device)).permute(0,3,1,2).unsqueeze(0)
    bak_all_masks = torch.from_numpy(np.stack(masks, 0)).to(device).unsqueeze(0).unsqueeze(2) # 1, S, 1, H, W
    bak_all_flows_f = torch.from_numpy(np.stack(flows_f, 0)).to(device).permute(0,3,1,2).unsqueeze(0)
    bak_all_flows_b = torch.from_numpy(np.stack(flows_b, 0)).to(device).permute(0,3,1,2).unsqueeze(0)

    _, bak_S, _, H, W = bak_all_rgbs.shape

    all_rgbs = bak_all_rgbs[:,start_ind:start_ind+min_lifespan]
    all_masks = bak_all_masks[:,start_ind:start_ind+min_lifespan]
    all_flows_f = bak_all_flows_f[:,start_ind:start_ind+min_lifespan-1]
    all_flows_b = bak_all_flows_b[:,start_ind+1:start_ind+min_lifespan+1]
    S = min_lifespan

    all_masks = all_masks.float()

    if include_vis:
        flows_f_vis = [sw.summ_flow('', all_flows_f[:, idx], clip=300, only_return=True) for idx in range(S-1)]
        flows_b_vis = [sw.summ_flow('', all_flows_b[:, idx], clip=300, only_return=True) for idx in range(S-1)]
        sw.summ_rgbs('inputs_%d/flows_f' % start_ind, flows_f_vis)
        sw.summ_rgbs('inputs_%d/flows_b' % start_ind, flows_b_vis)
        sw.summ_rgbs('inputs_%d/rgbs' % start_ind, all_rgbs.unbind(1))
        sw.summ_oneds('inputs_%d/masks' % start_ind, all_masks.unbind(1))

    ys, xs = utils.basic.meshgrid2d(1, H, W)
    xs = xs.reshape(1, -1)
    ys = ys.reshape(1, -1)

    coords = []
    coord = torch.stack([xs, ys], dim=2) # B, N, 2
    coords.append(coord)
    for s in range(S-1):
        delta = utils.samp.bilinear_sample2d(all_flows_f[:,s], coord[:,:,0].round(), coord[:,:,1].round()).permute(0,2,1) # B,N,2: forward flow at the discrete points
        coord = coord + delta
        coords.append(coord)
    trajs = torch.stack(coords, dim=1) # B, S, N, 2
    # N == 540*960 == 518400 

    trajs = filter_trajs(trajs, all_masks, all_flows_f, all_flows_b)

    if include_vis:

        max_show = 500
        if not trajs.shape[2] < max_show:
            inds = utils.geom.farthest_point_sample(trajs[:,0], max_show, deterministic=False)
            trajs_vis = trajs[:,:,inds.reshape(-1)]
        else:
            trajs_vis = trajs.clone()
        print('trajs vis', trajs_vis.shape)

        pad = 0
        if pad > 0:
            all_rgbs_ = F.pad(all_rgbs[0,:min_lifespan], (pad, pad, pad, pad), 'constant', 0).unsqueeze(0)
            trajs_vis = trajs_vis + pad
            sw.summ_traj2ds_on_rgbs2('outputs_%d/trajs_on_rgbs' % start_ind, trajs_vis, torch.ones_like(trajs_vis[:,:,:,0]), all_rgbs_)
        else:
            sw.summ_traj2ds_on_rgbs2('outputs_%d/trajs_on_rgbs' % start_ind, trajs_vis, torch.ones_like(trajs_vis[:,:,:,0]), all_rgbs)

    trajs = trajs[0].detach().cpu().numpy() # S, N, 2
    trajs = trajs.astype(np.float16) # save space
    
    # if there aren't 256, make it empty.
    # this lets us discard them later, but still write in parallel jobs
    N = trajs.shape[1]
    if N < 256:
        trajs = None
        sys.stdout.write('%d ' % N)
    else:
        sys.stdout.write('.')
        
    if not os.path.exists(cur_out_dir):
        os.makedirs(cur_out_dir)
    np.savez(out_f, trajs=trajs)#, visibles=visibles)


def go():
    log_freq = 1
    include_vis = False

    log_dir = 'logs_make_trajs'
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

            out_dir = os.path.join(flt3d_path, "trajs_%s" % mod, dset, subset)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            random.shuffle(folder_names)

            for folder_name in folder_names:
                for lr in ['left', 'right']:
                    for start_ind in [0,1,2]:
                        global_step += 1
                        if include_vis:
                            sw = utils.improc.Summ_writer(
                                writer=writer,
                                global_step=global_step,
                                log_freq=log_freq,
                                fps=5,
                                scalar_freq=100,
                                just_gif=True)
                        else:
                            sw = None
                        helper(rgb_path, mask_path, flow_path, out_dir, folder_name, lr, start_ind, sw=sw, include_vis=include_vis)
                        sys.stdout.flush()
            print('done')
            
if __name__ == "__main__":
    go()
