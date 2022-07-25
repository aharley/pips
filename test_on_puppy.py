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
import imageio

#from raft_core.raft import RAFT
# from relative_perceiver import RelativePerceiver
# from sparse_relative_perceiver import SparseRelativePerceiver
# from nets.graph_raft import GraphRaft
#from nets.st_graph_raft import StGraphRaft
# from nets.st_spraft import StSpRaft
# from nets.st_graph_raft import StGraphRaft
# from nets.mraft import Mraft
# from nets.praft import Praft
# from nets.mpraft import Mpraft
from nets.singlepoint import Singlepoint
# from perceiver_graph import PerceiverGraph
# from relative_mlp import RelativeMlp
import nets.raftnet

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    """ Create the optimizer and learning rate scheduler """
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

# import nets.relation_ebm as relation_ebm
# import nets.transformer_forecaster as transformer_forecaster
# import transformer_forecaster

# import nets.transformer_ebm as transformer_ebm
# # from nets.improved_cd_models import CelebAModel, ResNetModel
# import nets.conv1d_ebm
# import nets.traj_mlp_ebm
# import nets.encoder2d
# import nets.sparse_invar_encoder2d
# # import nets.improved_cd_models
# import nets.raftnet
# import nets.seg2dnet
# import nets.segpointnet

import utils.py
# import utils.box
import utils.misc
import utils.improc
# import utils.vox
# import utils.grouping
# from tqdm import tqdm
import random
import glob
# import color2d

# import detectron2
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.modeling import build_model

from utils.basic import print_, print_stats


# import relation_model

# import datasets
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


def run_model(model, rgbs, N2, sw):
    rgbs = rgbs.cuda().float() # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    rgbs_ = rgbs.reshape(B*S, C, H, W)
    H_, W_ = 360, 640
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    H, W = H_, W_
    rgbs = rgbs_.reshape(B, S, C, H, W)

    # pick N2 points to track; we'll use a uniform grid
    N_ = np.sqrt(N2).round().astype(np.int32)
    grid_y, grid_x = utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device='cuda')
    grid_y = 8 + grid_y.reshape(B, -1)/float(N_-1) * (H-16)
    grid_x = 8 + grid_x.reshape(B, -1)/float(N_-1) * (W-16)
    xy = torch.stack([grid_x, grid_y], dim=-1) # B, N2*N2, 2
    _, S, C, H, W = rgbs.shape

    preds, preds_anim, vis_e, stats = model(xy, rgbs, iters=6)
    trajs_e = preds[-1]
    print_stats('trajs_e', trajs_e)

    print(B, S, H, W)
    pad = 50
    rgbs = F.pad(rgbs.reshape(B*S, 3, H, W), (pad, pad, pad, pad), 'constant', 0).reshape(B, S, 3, H+pad*2, W+pad*2)
    trajs_e = trajs_e + pad
    
    if sw is not None and sw.save_this:

        # sw.summ_rgbs('inputs_0/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1))
        # sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='spring', linewidth=2)
        # sw.summ_traj2ds_on_rgbs('outputs/trajs_on_black', trajs_e[0:1], torch.ones_like(rgbs[0:1])*-0.5, cmap='spring', linewidth=2)
        # sw.summ_traj2ds_on_rgbs2('outputs/trajs_on_rgbs2', trajs_e[0:1], vis_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='spring')

        linewidth = 2
        o1 = sw.summ_rgbs('inputs_0/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1), only_return=True)
        o2 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='spring', linewidth=linewidth, only_return=True)
        o3 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_black', trajs_e[0:1], torch.ones_like(rgbs[0:1])*-0.5, cmap='spring', linewidth=linewidth, only_return=True)
        print('o1', o1.shape)
        wide = torch.cat([o1, o2, o3], dim=-1)
        print('wide', wide.shape)
        sw.summ_rgbs('outputs/wide', wide.unbind(1))
        
        # sw.summ_traj2ds_on_rgbs2('outputs/trajs_on_rgbs2', trajs_e[0:1], vis_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='spring')
        # gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('inputs_0_all/single_trajs_on_rgb', target_trajs[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1), cmap='winter', frame_id=metrics['epe'], only_return=True))
        # gt_black = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('inputs_0_all/single_trajs_on_rgb', target_trajs[0:1], torch.ones_like(rgbs[0:1,0])*-0.5, cmap='winter', frame_id=metrics['epe'], only_return=True))
        # # gt_rgb = torch.mean(utils.improc.preprocess_color(rgbs), dim=1)
        # # gt_black = torch.ones_like(rgbs[:,0])*-0.5

        # # animate_traj2ds_on_rgbs
        # sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_rgb', trajs_e[0:1], gt_rgb[0:1], cmap='coolwarm', linewidth=2)
        # sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_black', trajs_e[0:1], gt_black[0:1], cmap='coolwarm', linewidth=2)

        rgb_vis = []
        for trajs_e_ in preds_anim:
            trajs_e_ = trajs_e_ + pad
            rgb_vis.append(sw.summ_traj2ds_on_rgb('', trajs_e_[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1), only_return=True, cmap='spring', linewidth=linewidth))
        sw.summ_rgbs('outputs/animated_trajs_on_rgb', rgb_vis)

    return trajs_e-pad
    
def train():

    # default coeffs (don't touch)
    init_dir = ''
    coeff_prob = 0.0
    use_augs = False

    # device = 'cpu:0'
    # device = 'cuda'

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
    N = 256 # we need to load at least 4 i think
    N2 = 8**2 # number of points to track
    N2 = 16**2 # number of points to track
    # N2 = 32**2 # number of points to track
    lr = 1e-4
    grad_acc = 1
    
    max_iters = 40
    log_freq = 2
    
    ## autogen a name
    model_name = "%02d_%d_%d_%d" % (B, S, N, N2)
    if use_augs:
        model_name += "_A"
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
    requires_grad(parameters, False)
    model.eval()

    

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
                fn = '../puppy/%06d.jpg' % ((global_step-1)*S + 1 + s + 100)
                # print('fn', fn)
                im = imageio.imread(fn)
                im = im.astype(np.uint8)
                # print('im', im.shape)
                rgbs.append(torch.from_numpy(im).permute(2,0,1))
                # rgb = np.asarray(Image.open(os.path.join(self.image_location, img_record)).convert('RGB'))
                # rgb = rgb[np.newaxis, :, :, :]
            rgbs = torch.stack(rgbs, dim=0).unsqueeze(0) # 1, S, C, H, W

            read_time = time.time()-read_start_time
            iter_start_time = time.time()

            with torch.no_grad():
                trajs_e = run_model(model, rgbs, N2, sw_t)


            iter_time = time.time()-iter_start_time
            print('%s; step %06d/%d; rtime %.2f; itime %.2f' % (
                model_name, global_step, max_iters, read_time, iter_time))
        except FileNotFoundError as e:
            print('error', e)
            

    writer_t.close()
            

if __name__ == '__main__':
    train()
