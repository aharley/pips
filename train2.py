import time
import numpy as np
import timeit
import saverloader
from nets.pips import Pips
import utils.improc
import random
from utils.basic import print_, print_stats
# import flyingthingsdataset
import pointodysseydataset
import fltdataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from fire import Fire

random.seed(125)
np.random.seed(125)

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def run_model(model, d, device, I=6, horz_flip=False, vert_flip=False, sw=None, is_train=True):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    rgbs = d['rgbs'].to(device).float() # B, S, C, H, W
    trajs_g = d['trajs'].to(device).float() # B, S, N, 2
    vis_g = d['visibs'].to(device).float() # B, S, N
    valids = d['valids'].to(device).float() # B, S, N

    # print_stats('trajs_g', trajs_g)
    # print_stats('vis_g', vis_g)
    # print_stats('valids', valids)

    B, S, C, H, W = rgbs.shape
    assert(C==3)
    B, S, N, D = trajs_g.shape

    # print('torch.sum(valids)', torch.sum(valids))
    # assert(torch.sum(valids)==B*S*N)

    if horz_flip: # increase the batchsize by horizontal flipping
        rgbs_flip = torch.flip(rgbs, [4])
        trajs_g_flip = trajs_g.clone()
        trajs_g_flip[:,:,:,0] = W-1 - trajs_g_flip[:,:,:,0]
        vis_g_flip = vis_g.clone()
        valids_flip = valids.clone()
        trajs_g = torch.cat([trajs_g, trajs_g_flip], dim=0)
        vis_g = torch.cat([vis_g, vis_g_flip], dim=0)
        valids = torch.cat([valids, valids_flip], dim=0)
        rgbs = torch.cat([rgbs, rgbs_flip], dim=0)
        B = B * 2

    if vert_flip: # increase the batchsize by vertical flipping
        rgbs_flip = torch.flip(rgbs, [3])
        trajs_g_flip = trajs_g.clone()
        trajs_g_flip[:,:,:,1] = H-1 - trajs_g_flip[:,:,:,1]
        vis_g_flip = vis_g.clone()
        valids_flip = valids.clone()
        trajs_g = torch.cat([trajs_g, trajs_g_flip], dim=0)
        vis_g = torch.cat([vis_g, vis_g_flip], dim=0)
        valids = torch.cat([valids, valids_flip], dim=0)
        rgbs = torch.cat([rgbs, rgbs_flip], dim=0)
        B = B * 2

    preds, preds_anim, vis_e, stats = model(trajs_g[:,0], rgbs, coords_init=None, iters=I, trajs_g=trajs_g, vis_g=vis_g, valids=valids, sw=sw, is_train=is_train)
    seq_loss, vis_loss = stats
    
    total_loss += seq_loss.mean()
    total_loss += vis_loss.mean()*10.0
    # total_loss += ce_loss.mean()

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
        # 'ce': ce_loss.mean().item()
    }
    
    if sw is not None and sw.save_this:
        trajs_e = preds[-1]

        pad = 50
        rgbs = F.pad(rgbs.reshape(B*S, 3, H, W), (pad, pad, pad, pad), 'constant', 0).reshape(B, S, 3, H+pad*2, W+pad*2)
        trajs_e = trajs_e + pad
        trajs_g = trajs_g + pad
        
        sw.summ_traj2ds_on_rgbs2('0_inputs/trajs_on_rgbs2', trajs_g[0:1], vis_g[0:1], utils.improc.preprocess_color(rgbs[0:1]), valids=valids[0:1])
        sw.summ_traj2ds_on_rgb('0_inputs/trajs_g_on_rgb', trajs_g[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1), valids=valids[0:1], cmap='winter')

        for b in range(B):
            sw.summ_traj2ds_on_rgb('0_batch_inputs/trajs_g_on_rgb_%d' % b, trajs_g[b:b+1], torch.mean(utils.improc.preprocess_color(rgbs[b:b+1]), dim=1), valids=valids[b:b+1], cmap='winter', frame_id=torch.sum(valids[b,0]).item())

        gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('', trajs_g[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1), valids=valids[0:1], cmap='winter', frame_id=metrics['ate_all'], only_return=True))
        sw.summ_traj2ds_on_rgb('2_outputs/single_trajs_on_gt_rgb', trajs_e[0:1], gt_rgb[0:1], valids=valids[0:1], cmap='spring')

        if True: # this works but it's a bit expensive
            rgb_vis = []
            for trajs_e in preds_anim:
                rgb_vis.append(sw.summ_traj2ds_on_rgb('', trajs_e[0:1]+pad, gt_rgb, valids=valids[0:1], only_return=True, cmap='spring'))
            sw.summ_rgbs('2_outputs/animated_trajs_on_rgb', rgb_vis)

    return total_loss, metrics
    
def main(
        exp_name='debug',
        # training
        B=1, # batchsize 
        S=8, # seqlen of the data/model
        N=64, # number of particles to sample from the data
        horz_flip=True, # this causes B*=2
        vert_flip=True, # this causes B*=2
        stride=8, # spatial stride of the model 
        I=4, # inference iters of the model
        crop_size=(384,512), # the raw data is 540,960
        # crop_size=(256,384), # the raw data is 540,960
        use_augs=True, # resizing/jittering/color/blur augs
        # dataset
        dataset_location='../datasets/flyingthings',
        subset='all', # dataset subset
        shuffle=True, # dataset shuffling
        # optimization
        lr=5e-5,
        grad_acc=1,
        max_iters=100000,
        use_scheduler=True,
        # summaries
        log_dir='./logs_train2',
        log_freq=1000,
        val_freq=0,
        # saving/loading
        ckpt_dir='./checkpoints',
        save_freq=1000,
        keep_latest=1,
        init_dir='',
        load_optimizer=False,
        load_step=False,
        ignore_load=None,
        # cuda
        device_ids=[0],
        quick=False,
):
    device = 'cuda:%d' % device_ids[0]
    
    # the idea in this file is to train a PIPs++ model (nets/pips2.py) 
    
    exp_name = '00' # copy from train.py
    exp_name = '01' # quick
    # i think the next step is to set up pointodyssey here
    exp_name = '02' # pointodyssey
    exp_name = '03' # fps sampling
    exp_name = '04' # shuffle = False
    exp_name = '05' # use fps
    exp_name = '06' # shuffle
    exp_name = '07' # no shuffle
    exp_name = '08' # ensure 3-step vis
    exp_name = '09' # 
    

    init_dir = 'reference_model'

    if quick:
        B = 1
        horz_flip = False
        vert_flip = False
        use_augs = False
        shuffle = False
        log_freq = 10
        max_iters = 100
        # log_freq = 50
        # max_iters = 1000
        save_freq = 9999999
    

    assert(crop_size[0] % 128 == 0)
    assert(crop_size[1] % 128 == 0)
    
    ## autogen a descriptive name
    if horz_flip and vert_flip:
        model_name = "%dhv" % (B*4)
    elif horz_flip:
        model_name = "%dh" % (B*2)
    elif vert_flip:
        model_name = "%dv" % (B*2)
    else:
        model_name = "%d" % (B)
    if grad_acc > 1:
        model_name += "x%d" % grad_acc
    model_name += "_%d_%d" % (S, N)
    model_name += "_I%d" % (I)
    lrn = "%.1e" % lr # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1] # e.g., 5e-4
    model_name += "_%s" % lrn
    if use_augs:
        model_name += "_A"
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    ckpt_dir = '%s/%s' % (ckpt_dir, model_name)
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)
    if val_freq > 0:
        writer_v = SummaryWriter(log_dir + '/' + model_name + '/v', max_queue=10, flush_secs=60)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    train_dataset = pointodysseydataset.PointOdysseyDataset(
        dset='TRAIN',
        S=S,
        N=N,
        use_augs=use_augs,
        crop_size=crop_size,
    )
    # # train_dataset = flyingthingsdataset.FlyingThingsDataset(
    # train_dataset = fltdataset.FltDataset(
    #     use_augs=use_augs,
    #     N=N, S=S,
    #     crop_size=crop_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=0,#16*len(device_ids),
        worker_init_fn=worker_init_fn,
        drop_last=True)
    train_iterloader = iter(train_dataloader)
    
    if val_freq > 0:
        print('not using augs in val')
        val_dataset = flyingthingsdataset.FlyingThingsDataset(
            dataset_location=dataset_location,
            dset='TEST', subset='all',
            use_augs=use_augs,
            N=N, S=S,
            crop_size=crop_size)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=B,
            shuffle=shuffle,
            num_workers=1,
            drop_last=False)
        val_iterloader = iter(val_dataloader)
        
    model = Pips(stride=stride).to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    parameters = list(model.parameters())
    if use_scheduler:
        optimizer, scheduler = fetch_optimizer(lr, 0.0001, 1e-8, max_iters//grad_acc, model.parameters())
    else:
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
    if val_freq > 0:
        loss_pool_v = utils.misc.SimplePool(n_pool, version='np')
        ce_pool_v = utils.misc.SimplePool(n_pool, version='np')
        vis_pool_v = utils.misc.SimplePool(n_pool, version='np')
        seq_pool_v = utils.misc.SimplePool(n_pool, version='np')
        ate_all_pool_v = utils.misc.SimplePool(n_pool, version='np')
        ate_vis_pool_v = utils.misc.SimplePool(n_pool, version='np')
        ate_occ_pool_v = utils.misc.SimplePool(n_pool, version='np')
    
    while global_step < max_iters:

        global_step += 1

        iter_start_time = time.time()
        iter_read_time = 0.0
        
        for internal_step in range(grad_acc):
            # read sample
            read_start_time = time.time()

            if internal_step==grad_acc-1:
                sw_t = utils.improc.Summ_writer(
                    writer=writer_t,
                    global_step=global_step,
                    log_freq=log_freq,
                    fps=5,
                    scalar_freq=int(log_freq/2),
                    just_gif=True)
            else:
                sw_t = None

            gotit = (False,False)
            while not all(gotit):
                try:
                    sample, gotit = next(train_iterloader)
                except StopIteration:
                    train_iterloader = iter(train_dataloader)
                    sample, gotit = next(train_iterloader)
            # try:
            #     sample = next(train_iterloader)
            # except StopIteration:
            #     train_iterloader = iter(train_dataloader)
            #     sample = next(train_iterloader)
                
            read_time = time.time()-read_start_time
            iter_read_time += read_time
            
            total_loss, metrics = run_model(model, sample, device, I, horz_flip, vert_flip, sw_t, is_train=True)
            total_loss.backward()

        iter_time = time.time()-iter_start_time

        sw_t.summ_scalar('total_loss', total_loss)
        loss_pool_t.update([total_loss.detach().cpu().numpy()])
        sw_t.summ_scalar('pooled/total_loss', loss_pool_t.mean())

        if metrics['ate_all'] > 0:
            ate_all_pool_t.update([metrics['ate_all']])
        if metrics['ate_vis'] > 0:
            ate_vis_pool_t.update([metrics['ate_vis']])
        if metrics['ate_occ'] > 0:
            ate_occ_pool_t.update([metrics['ate_occ']])
        # if metrics['ce'] > 0:
        #     ce_pool_t.update([metrics['ce']])
        if metrics['vis'] > 0:
            vis_pool_t.update([metrics['vis']])
        if metrics['seq'] > 0:
            seq_pool_t.update([metrics['seq']])
        sw_t.summ_scalar('pooled/ate_all', ate_all_pool_t.mean())
        sw_t.summ_scalar('pooled/ate_vis', ate_vis_pool_t.mean())
        sw_t.summ_scalar('pooled/ate_occ', ate_occ_pool_t.mean())
        # sw_t.summ_scalar('pooled/ce', ce_pool_t.mean())
        sw_t.summ_scalar('pooled/vis', vis_pool_t.mean())
        sw_t.summ_scalar('pooled/seq', seq_pool_t.mean())
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if use_scheduler:
            scheduler.step()
        optimizer.zero_grad()

        if val_freq > 0 and (global_step) % val_freq == 0:
            torch.cuda.empty_cache()
            model.eval()
            sw_v = utils.improc.Summ_writer(
                writer=writer_v,
                global_step=global_step,
                log_freq=log_freq,
                fps=5,
                scalar_freq=int(log_freq/2),
                just_gif=True)

            gotit = (False,False)
            while not all(gotit):
                try:
                    sample, gotit = next(val_iterloader)
                except StopIteration:
                    val_iterloader = iter(val_dataloader)
                    sample, gotit = next(val_iterloader)

            with torch.no_grad():
                total_loss, metrics = run_model(model, sample, device, I, horz_flip, vert_flip, sw_v, is_train=False)

            sw_v.summ_scalar('total_loss', total_loss)
            loss_pool_v.update([total_loss.detach().cpu().numpy()])
            sw_v.summ_scalar('pooled/total_loss', loss_pool_v.mean())

            if metrics['ate_all'] > 0:
                ate_all_pool_v.update([metrics['ate_all']])
            if metrics['ate_vis'] > 0:
                ate_vis_pool_v.update([metrics['ate_vis']])
            if metrics['ate_occ'] > 0:
                ate_occ_pool_v.update([metrics['ate_occ']])
            # if metrics['ce'] > 0:
            #     ce_pool_v.update([metrics['ce']])
            if metrics['vis'] > 0:
                vis_pool_v.update([metrics['vis']])
            if metrics['seq'] > 0:
                seq_pool_v.update([metrics['seq']])
            sw_v.summ_scalar('pooled/ate_all', ate_all_pool_v.mean())
            sw_v.summ_scalar('pooled/ate_vis', ate_vis_pool_v.mean())
            sw_v.summ_scalar('pooled/ate_occ', ate_occ_pool_v.mean())
            # sw_v.summ_scalar('pooled/ce', ce_pool_v.mean())
            sw_v.summ_scalar('pooled/vis', vis_pool_v.mean())
            sw_v.summ_scalar('pooled/seq', seq_pool_v.mean())
            model.train()

        if np.mod(global_step, save_freq)==0:
            saverloader.save(ckpt_dir, optimizer, model.module, global_step, keep_latest=keep_latest)

        current_lr = optimizer.param_groups[0]['lr']
        sw_t.summ_scalar('_/current_lr', current_lr)
        
        iter_time = time.time()-iter_start_time
        print('%s; step %06d/%d; rtime %.2f; itime %.2f; loss = %.5f' % (
            model_name, global_step, max_iters, read_time, iter_time,
            total_loss.item()))
            
    writer_t.close()
    if val_freq > 0:
        writer_v.close()
            

if __name__ == '__main__':
    Fire(main)
