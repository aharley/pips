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
import imageio.v2 as imageio

from nets.singlepoint import Singlepoint
import nets.raftnet

import utils.py
import utils.misc
import utils.improc
import random
import glob
from utils.basic import print_, print_stats

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

import torch.nn.functional as F

device = 'cuda'
random.seed(125)
np.random.seed(125)

def run_model(model, rgbs, N, sw):
    rgbs = rgbs.cuda().float() # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    rgbs_ = rgbs.reshape(B*S, C, H, W)
    H_, W_ = 360, 640
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    H, W = H_, W_
    rgbs = rgbs_.reshape(B, S, C, H, W)

    # pick N points to track; we'll use a uniform grid
    N_ = np.sqrt(N).round().astype(np.int32)
    grid_y, grid_x = utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device='cuda')
    grid_y = 8 + grid_y.reshape(B, -1)/float(N_-1) * (H-16)
    grid_x = 8 + grid_x.reshape(B, -1)/float(N_-1) * (W-16)
    xy = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2
    _, S, C, H, W = rgbs.shape

    print_stats('rgbs', rgbs)
    preds, preds_anim, vis_e, stats = model(xy, rgbs, iters=6)
    trajs_e = preds[-1]
    print_stats('trajs_e', trajs_e)
    
    pad = 50
    rgbs = F.pad(rgbs.reshape(B*S, 3, H, W), (pad, pad, pad, pad), 'constant', 0).reshape(B, S, 3, H+pad*2, W+pad*2)
    trajs_e = trajs_e + pad
    
    if sw is not None and sw.save_this:
        linewidth = 2

        o1 = sw.summ_rgbs('inputs_0/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1))
        o2 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='spring', linewidth=linewidth)
        o3 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_black', trajs_e[0:1], torch.ones_like(rgbs[0:1])*-0.5, cmap='spring', linewidth=linewidth)
        sw.summ_traj2ds_on_rgbs2('outputs/trajs_on_rgbs2', trajs_e[0:1], vis_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='spring')
        wide = torch.cat([o1, o2, o3], dim=-1)
        sw.summ_rgbs('outputs/wide', wide.unbind(1))

        # alternate vis
        sw.summ_traj2ds_on_rgbs2('outputs/trajs_on_rgbs2', trajs_e[0:1], vis_e[0:1], utils.improc.preprocess_color(rgbs[0:1]))
        
        # animation of inference iterations
        rgb_vis = []
        for trajs_e_ in preds_anim:
            trajs_e_ = trajs_e_ + pad
            rgb_vis.append(sw.summ_traj2ds_on_rgb('', trajs_e_[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1), cmap='spring', linewidth=linewidth, only_return=True))
        sw.summ_rgbs('outputs/animated_trajs_on_rgb', rgb_vis)

    return trajs_e-pad
    
def train():

    # test fast praft on puppy video
    exp_name = 'pu00' # copy from test_fast_praft_point_track.py
    exp_name = 'pu01' # load jpg and vis
    exp_name = 'pu02' # N2 = 16 instead of 4
    exp_name = 'pu03' # N2 = 32
    exp_name = 'pu04' # larger video stride
    exp_name = 'pu05' # N2 = 64
    exp_name = 'pu06' # start on frame 100
    exp_name = 'pu07' # use meshgrid instead of random
    exp_name = 'pu08' # -1
    exp_name = 'pu09' # bugfix
    exp_name = 'pu10' # squish inside
    exp_name = 'pu11' # N2 = 128
    exp_name = 'pu12' # 144; pad
    exp_name = 'pu13' # 16**2
    exp_name = 'pu14' # bugfix on consecutiveness
    exp_name = 'pu15' # log10, to see how fast the nonlog iters are < 0.26
    exp_name = 'pu16' # new vis, where color is constant and we adjust opacity < cv2 doesn't have opacity; Reds looks worse than coolwarm
    exp_name = 'pu17' # repeat
    exp_name = 'pu18' # au14
    exp_name = 'pu19' # au31 
    exp_name = 'pu20' # 8**2 instead of 16
    exp_name = 'pu21' # 105k au48
    exp_name = 'pu22' # same 
    exp_name = 'pu23' # au31
    exp_name = 'pu24' # 130k au31
    exp_name = 'pu25' # 155k au53
    exp_name = 'pu26' # 16**2
    exp_name = 'pu27' # run on orion; tb79
    exp_name = 'pu28' # stride4
    exp_name = 'pu29' # actually load please
    exp_name = 'pu30' # wide vis
    exp_name = 'pu31' # 32**2
    exp_name = 'pu32' # linewidth=1
    exp_name = 'pu33' # just wide vis
    exp_name = 'pu33' # linewidth=3, N2=8**2
    exp_name = 'pu34' # re-test; show anim
    exp_name = 'pu35' # 

    init_dir = 'checkpoints/04_8_256_1e-4_p1_A_big08_01:08:24'
    init_dir = 'checkpoints/12_8_256_1e-4_p1_A_big14_22:20:05'
    init_dir = 'checkpoints/12_8_256_1e-4_p1_A_big13_20:19:56'
    init_dir = 'checkpoints/12_8_256_1e-4_p1_A_big11_17:31:32'
    init_dir = 'checkpoints/01_8_64_10_1e-4_p1_au14_13:39:34'
    init_dir = 'checkpoints/02_8_128_96_1e-4_p1_au31_12:38:03'
    init_dir = 'checkpoints/04_8_64_10_2e-4_p1_au48_17:32:27'
    init_dir = 'checkpoints/02_8_128_96_1e-4_p1_au31_12:38:03'
    init_dir = 'checkpoints/04_8_64_10_2e-4_p1_A_au53_19:19:06'
    init_dir = 'saved_checkpoints/4hv_8_128_I6_4e-4_p1_A_tb79_09:11:39' # new best

    ## choose hyps
    B = 1
    S = 8
    N = 8**2 # number of points to track
    N = 16**2 # number of points to track
    N = 32**2 # number of points to track
    
    max_iters = 40
    log_freq = 4
    
    ## autogen a name
    model_name = "%02d_%d_%d" % (B, S, N)
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    log_dir = 'logs_test_on_puppy'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    global_step = 0

    model = Singlepoint(stride=4).cuda()
    # model = torch.nn.DataParallel(model).cuda()
    parameters = list(model.parameters())
    if init_dir:
        _ = saverloader.load(init_dir, model)
    global_step = 0
    model.eval()

    filenames = glob.glob('./demo_images/*.jpg')
    filenames = sorted(filenames)
    print('filenames', filenames)
    max_iters = len(filenames)//S
    
    while global_step < max_iters:
        
        read_start_time = time.time()
        
        global_step += 1

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=5,
            scalar_freq=int(log_freq/2),
            just_gif=True)

        try:
            rgbs = []
            for s in range(S):
                fn = filenames[(global_step-1)*S+s]
                if s==0:
                    print('fn', fn)
                # fn = 'demo_images/%06d.jpg' % ((global_step-1)*S + 1 + s)
                im = imageio.imread(fn)
                im = im.astype(np.uint8)
                rgbs.append(torch.from_numpy(im).permute(2,0,1))
            rgbs = torch.stack(rgbs, dim=0).unsqueeze(0) # 1, S, C, H, W

            read_time = time.time()-read_start_time
            iter_start_time = time.time()

            with torch.no_grad():
                trajs_e = run_model(model, rgbs, N, sw_t)

            iter_time = time.time()-iter_start_time
            print('%s; step %06d/%d; rtime %.2f; itime %.2f' % (
                model_name, global_step, max_iters, read_time, iter_time))
        except FileNotFoundError as e:
            print('error', e)
            
    writer_t.close()

if __name__ == '__main__':
    train()
