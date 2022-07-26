import time
import numpy as np
import io
import os
from PIL import Image
import cv2
import saverloader
import imageio.v2 as imageio
from nets.pips import Pips
import utils.improc
import random
import glob
from utils.basic import print_, print_stats
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F

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

        # visualize the input
        o1 = sw.summ_rgbs('inputs/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1))
        # visualize the trajs overlaid on the rgbs
        o2 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='spring', linewidth=linewidth)
        # visualize the trajs alone
        o3 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_black', trajs_e[0:1], torch.ones_like(rgbs[0:1])*-0.5, cmap='spring', linewidth=linewidth)
        # concat these for a synced wide vis
        wide_cat = torch.cat([o1, o2, o3], dim=-1)
        sw.summ_rgbs('outputs/wide_cat', wide_cat.unbind(1))

        # write to disk, in case that's more convenient
        wide_list = list(wide_cat.unbind(1))
        wide_list = [wide[0].permute(1,2,0).cpu().numpy() for wide in wide_list]
        wide_list = [Image.fromarray(wide) for wide in wide_list]
        out_fn = './out_%d.gif' % sw.global_step
        wide_list[0].save(out_fn, save_all=True, append_images=wide_list[1:])
        print('saved %s' % out_fn)

        # alternate vis
        sw.summ_traj2ds_on_rgbs2('outputs/trajs_on_rgbs2', trajs_e[0:1], vis_e[0:1], utils.improc.preprocess_color(rgbs[0:1]))
        
        # animation of inference iterations
        rgb_vis = []
        for trajs_e_ in preds_anim:
            trajs_e_ = trajs_e_ + pad
            rgb_vis.append(sw.summ_traj2ds_on_rgb('', trajs_e_[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1), cmap='spring', linewidth=linewidth, only_return=True))
        sw.summ_rgbs('outputs/animated_trajs_on_rgb', rgb_vis)

    return trajs_e-pad
    
def main():

    # the idea in this file is to run the model on some demo images, and return some visualizations
    
    exp_name = '00' # (exp_name is used for logging notes that correspond to different runs)

    init_dir = 'reference_model'

    ## choose hyps
    B = 1
    S = 8
    N = 16**2 # number of points to track

    filenames = glob.glob('./demo_images/*.jpg')
    filenames = sorted(filenames)
    print('filenames', filenames)
    max_iters = len(filenames)//S # run each unique subsequence

    log_freq = 2 # when to produce visualizations 
    
    ## autogen a name
    model_name = "%02d_%d_%d" % (B, S, N)
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    log_dir = 'logs_demo'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    global_step = 0

    model = Pips(stride=4).cuda()
    parameters = list(model.parameters())
    if init_dir:
        _ = saverloader.load(init_dir, model)
    global_step = 0
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
                fn = filenames[(global_step-1)*S+s]
                if s==0:
                    print('start frame', fn)
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
    main()
