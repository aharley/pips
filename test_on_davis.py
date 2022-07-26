import time
import argparse
import numpy as np
import timeit
import cv2
import saverloader
from nets.pips import Pips
import utils.basic
import utils.improc
import random
from utils.basic import print_, print_stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import os
import glob
from PIL import Image

def read_frame_list(video_dir):
    frame_list = [img for img in glob.glob(os.path.join(video_dir,"*.jpg"))]
    frame_list = sorted(frame_list)
    return frame_list

def read_frame(frame_dir, scale_size=[480]):
    """
    read a single frame & preprocess
    """
    img = cv2.imread(frame_dir)
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
    # img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    return img, ori_h, ori_w

def read_seg(seg_dir, factor, scale_size=[480]):
    seg = Image.open(seg_dir)
    _w, _h = seg.size # note PIL.Image.Image's size is (w, h)
    if len(scale_size) == 1:
        if(_w > _h):
            _th = scale_size[0]
            _tw = (_th * _w) / _h
            _tw = int((_tw // 64) * 64)
        else:
            _tw = scale_size[0]
            _th = (_tw * _h) / _w
            _th = int((_th // 64) * 64)
    else:
        _th = scale_size[1]
        _tw = scale_size[0]
    small_seg = np.array(seg.resize((_tw // factor, _th // factor), 0))
    small_seg = torch.from_numpy(small_seg.copy()).contiguous().float().unsqueeze(0)
    return small_seg, np.asarray(seg)


def run_model(model, frame_list, video_dir, first_seg, seg_ori, sw, use_mask=False):
    T = len(frame_list)

    S = 8
    rgbs = []
    segs = []
    for s in range(0, S):
        rgb, _, _ = read_frame(frame_list[s])
        rgbs.append(rgb)
        seg_path = frame_list[s].replace("JPEGImages", "Annotations").replace("jpg", "png")
        seg, _ = read_seg(seg_path, 1)
        segs.append(seg)
    rgbs = torch.stack(rgbs, dim=0).float().cuda()
    segs = torch.stack(segs, dim=0).float().cuda()

    S, C, H, W = rgbs.shape
    H_, W_ = 480, 1024
    sy = H_/H
    sx = W_/W
    rgbs = F.interpolate(rgbs, (H_, W_), mode='bilinear')
    segs = F.interpolate(segs, (H_, W_), mode='nearest')
    H, W = H_, W_
    rgbs = rgbs.unsqueeze(0) # B, S, C, H, W
    segs = segs.unsqueeze(0) # B, S, 1, H, W
    
    B, S, C, H, W = rgbs.shape

    segs = (segs==1).float()
    
    seg0 = segs[:,0]
    seg0_safe = utils.improc.erode2d(seg0, times=3)
    
    point_stride = 8
    H2, W2 = int(H/point_stride), int(W/point_stride)
    xy = utils.basic.gridcloud2d(1, H2, W2).reshape(H2*W2, 2)*point_stride
    if use_mask:
        seg0_ = F.interpolate(seg0_safe, (H2, W2), mode='nearest')
        xy = xy[seg0_.reshape(H2*W2) > 0]

    print('xy', xy.shape)
    xy_list = torch.split(xy, 256, dim=0)
    trajs_e = []
    vis_e = []
    full_start_time = time.time()
    step_times = []
    for xy0 in xy_list:
        step_start_time = time.time()
        outs = model(xy0.reshape(1, -1, 2), rgbs, iters=6)
        preds = outs[0]
        xys = preds[-1]
        vis = outs[2]
        trajs_e.append(xys)
        vis_e.append(vis)
        step_time = time.time()-step_start_time
        step_times.append(step_time)
    full_time = time.time()-full_start_time
    print('our full_time', full_time)
    print('our FPS', full_time/len(xy))
    print('our step time', np.stack(step_times).mean())
    print('our TPS', len(xy)/full_time)
    trajs_e = torch.cat(trajs_e, dim=2)
    vis_e = torch.sigmoid(torch.cat(vis_e, dim=2))

    print('trajs_e', trajs_e.shape)
    print('vis_e', vis_e.shape)

    sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs_plasma', trajs_e[0:1], utils.improc.preprocess_color(rgbs), cmap='plasma')
    sw.summ_traj2ds_on_rgbs2('outputs/trajs_on_rgbs2', trajs_e[0:1], vis_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='spring')
    sw.summ_rgb('outputs/rgb0', utils.improc.preprocess_color(rgbs[:,0]))
    sw.summ_rgb('outputs/mean_rgb', utils.improc.preprocess_color(torch.mean(rgbs, dim=1)))
    sw.summ_traj2ds_on_rgb('outputs/trajs_on_rgb', trajs_e[0:1], utils.improc.preprocess_color(rgbs[:,0]), cmap='spring')
    sw.summ_traj2ds_on_rgb('outputs/trajs_on_black', trajs_e[0:1], utils.improc.preprocess_color(rgbs[:,0]*0), cmap='spring')
    sw.summ_traj2ds_on_rgb('outputs/trajs_on_white', trajs_e[0:1], 0.5*torch.ones_like(rgbs[:,0]), cmap='spring')
    sw.summ_traj2ds_on_rgb('outputs/trajs_on_white6', trajs_e[0:1], 0.5*torch.ones_like(rgbs[:,0]), cmap='plasma')
    sw.summ_traj2ds_on_rgb('outputs/trajs_on_rgb6', trajs_e[0:1], utils.improc.preprocess_color(rgbs[:,0]), cmap='plasma')
    sw.summ_traj2ds_on_rgb('outputs/trajs_on_white7', trajs_e[0:1], 0.5*torch.ones_like(rgbs[:,0]), cmap='plasma_r')
    sw.summ_traj2ds_on_rgb('outputs/trajs_on_white8', trajs_e[0:1], 0.5*torch.ones_like(rgbs[:,0]), cmap='hot')
    return 


if __name__ == '__main__':

    # the idea in this file is to visualize results in DAVIS
    
    exp_name = '00' # (exp_name is used for logging notes that correspond to different runs)

    init_dir = 'reference_model'
    
    data_path = '../badja_data/DAVIS'

    stride = 8
    model = Pips(stride=stride).cuda()
    _ = saverloader.load(init_dir, model)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    import datetime
    exp_date = datetime.datetime.now().strftime('%H:%M:%S')
    exp_name = exp_name + '_' + exp_date
    print('exp_name', exp_name)

    log_dir = 'logs_test_on_davis'
    writer = SummaryWriter(log_dir + '/' + exp_name + '/t', max_queue=10, flush_secs=60)

    video_list = open(os.path.join(data_path, "ImageSets/2017/val.txt")).readlines()

    video_list = video_list[:20]
    for i, video_name in enumerate(video_list):
        video_name = video_name.strip()

        global_step = i
        sw = utils.improc.Summ_writer(
            writer=writer,
            global_step=global_step,
            log_freq=99999,
            fps=4,
            scalar_freq=1,
            just_gif=True)
        
        print(f'{exp_name} [{i}/{len(video_list)}] starting {video_name}.')
        video_dir = os.path.join(data_path, "JPEGImages/480p/", video_name)
        frame_list = read_frame_list(video_dir)
        seg_path = frame_list[0].replace("JPEGImages", "Annotations").replace("jpg", "png")
        first_seg, seg_ori = read_seg(seg_path, stride)
        run_model(model, frame_list, video_dir, first_seg, seg_ori, sw)
        
    writer.close()
