import time
from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import os
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
from torch._C import dtype, set_flush_denormal
import utils.basic
import utils.improc
import glob
import json
import imageio
import cv2
import re
import sys
from torchvision.transforms import ColorJitter, GaussianBlur

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data

def readImage(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        data = readPFM(name)
        if len(data.shape)==3:
            return data[:,:,0:3]
        else:
            return data
    return imageio.imread(name)


class FlyingThingsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_location='../flyingthings',
                 dset='TRAIN',
                 subset='all',
                 use_augs=False,
                 N=0,
                 S_load=8,
                 S=12,
                 crop_size=(368, 496),
                 version='ad',
                 occ_version='al',
                 force_twice_vis=True,
                 force_last_vis=False,
                 force_all_inb=False):

        print('loading FlyingThingsDataset...')
        
        self.S_load = S_load
        self.S = S
        self.N = N
            
        self.use_augs = use_augs
        
        self.rgb_paths = []
        self.traj_paths = []
        self.mask_paths = []
        self.flow_f_paths = []
        self.flow_b_paths = []
        self.start_inds = []
        self.load_fails = []

        self.force_twice_vis = force_twice_vis
        self.force_last_vis = force_last_vis
        self.force_all_inb = force_all_inb

        self.subset = subset

        if self.subset=='all':
            subsets = ['A', 'B', 'C']
        else:
            subsets = [subset]

        for subset in subsets:
            rgb_root_path = os.path.join(dataset_location, "frames_cleanpass_webp", dset, subset)
            flow_root_path = os.path.join(dataset_location, "optical_flow", dset, subset)
            traj_root_path = os.path.join(dataset_location, "trajs_%s" % version, dset, subset)
            mask_root_path = os.path.join(dataset_location, "object_index", dset, subset)

            folder_names = [folder.split('/')[-1] for folder in glob.glob(os.path.join(traj_root_path, "*"))]
            folder_names = sorted(folder_names)

            for ii, folder_name in enumerate(folder_names):
                for lr in ['left', 'right']:
                    cur_rgb_path = os.path.join(rgb_root_path, folder_name, lr)
                    cur_traj_path = os.path.join(traj_root_path, folder_name, lr)
                    cur_mask_path = os.path.join(mask_root_path, folder_name, lr)

                    for start_ind in [0,1,2,3]:
                        traj_fn = cur_traj_path + '/trajs_at_%d.npz' % start_ind
                        if os.path.isfile(traj_fn):
                            file_size = os.path.getsize(traj_fn)
                            if file_size > 1000: # the empty ones are 264 bytes
                                self.rgb_paths.append(cur_rgb_path)
                                self.traj_paths.append(cur_traj_path)
                                self.mask_paths.append(cur_mask_path)
                                self.start_inds.append(start_ind)
                                self.load_fails.append(0)
                                if start_ind==0 and lr=='left': # reduce the total number of prints
                                    sys.stdout.write('.')
                                    sys.stdout.flush()
        print('found %d samples in %s (dset=%s, subset=%s, version=%s)' % (len(self.rgb_paths), dataset_location, dset, self.subset, version))


        # we also need to step through and collect ooccluder info
        print('loading occluders...')

        self.occ_rgb_paths = []
        self.occ_mask_paths = []
        self.occ_start_inds = []
        self.occ_traj_paths = []

        for subset in subsets:
            rgb_root_path = os.path.join(dataset_location, "frames_cleanpass_webp", dset, subset)
            flow_root_path = os.path.join(dataset_location, "optical_flow", dset, subset)
            mask_root_path = os.path.join(dataset_location, "object_index", dset, subset)
            occ_root_path = os.path.join(dataset_location, "occluders_%s" % occ_version, dset, subset)

            folder_names = [folder.split('/')[-1] for folder in glob.glob(os.path.join(occ_root_path, "*"))]
            folder_names = sorted(folder_names)

            for folder_name in folder_names:
                
                for lr in ['left', 'right']:

                    cur_rgb_path = os.path.join(rgb_root_path, folder_name, lr)
                    cur_mask_path = os.path.join(mask_root_path, folder_name, lr)
                    cur_occ_path = os.path.join(occ_root_path, folder_name, lr)
                    
                    # start_ind = 0
                    # if True:
                    for start_ind in [0,1,2]:
                        occ_fn = cur_occ_path + '/occluder_at_%d.npy' % (start_ind)

                        if os.path.isfile(occ_fn):
                            file_size = os.path.getsize(occ_fn)
                            if file_size > 1000: # the empty ones are 10 bytes
                                self.occ_rgb_paths.append(cur_rgb_path)
                                self.occ_mask_paths.append(cur_mask_path)
                                self.occ_start_inds.append(start_ind)
                                self.occ_traj_paths.append(occ_fn)

                                if start_ind==0 and lr=='left': # reduce the total number of prints
                                    sys.stdout.write('.')
                                    sys.stdout.flush()
        print('found %d occluders in %s (dset=%s, subset=%s, version=%s)' % (len(self.occ_rgb_paths), dataset_location, dset, self.subset, occ_version))

        # photometric augmentation
        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25/3.14)
        self.blur_aug = GaussianBlur(11, sigma=(0.1, 2.0))
        
        self.blur_aug_prob = 0.5
        self.color_aug_prob = 0.5

        # occlusion augmentation
        self.eraser_aug_prob = 0.9
        self.eraser_bounds = [2, 100]
        self.eraser_max = 10

        # occlusion augmentation
        self.replace_aug_prob = 0.9
        self.replace_bounds = [2, 100]
        self.replace_max = 20

        # spatial augmentations
        self.pad_bounds = [0, 100]
        self.crop_size = crop_size
        self.resize_lim = [0.25, 2.0] # sample resizes from here
        self.resize_delta = 0.2
        self.max_crop_offset = 100
        
        self.do_flip = True
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5


    def getitem_helper(self, index, print_timings=False):
        sample = None
        gotit = False

        if print_timings:
            step_start_time = time.time()

        cur_rgb_path = self.rgb_paths[index]
        cur_traj_path = self.traj_paths[index]
        cur_mask_path = self.mask_paths[index]

        start_ind = self.start_inds[index]

        img_names = [folder.split('/')[-1].split('.')[0] for folder in glob.glob(os.path.join(cur_rgb_path, "*"))]
        img_names = sorted(img_names)
        img_names = img_names[start_ind:start_ind+self.S_load]

        trajs = np.load(os.path.join(cur_traj_path, 'trajs_at_%d.npz' % start_ind), allow_pickle=True)
        trajs = dict(trajs)['trajs'] # S,N,2; S=8 probably
        trajs = trajs.astype(np.float32)
        S_load, N, D = trajs.shape
        # shuffle
        perm = np.random.permutation(N)
        trajs = trajs[:,perm]
        assert(S_load==self.S_load)
        valids = np.ones((S_load, N)).astype(np.float32)
        # the data we loaded is all visible
        visibles = np.ones((S_load, N))

        if print_timings:
            step_time = time.time()-step_start_time
            print('reading paths and npy %.2f' % step_time)
            step_start_time = time.time()
            ###

        if N < self.N:
            return None, False

        rgbs = []
        masks = []
        flows_f = []
        flows_b = []

        for img_name in img_names:
            with Image.open(os.path.join(cur_rgb_path, '{0}.webp'.format(img_name))) as im:
                rgbs.append(np.array(im))
            mask = readImage(os.path.join(cur_mask_path, '{0}.pfm'.format(img_name)))
            masks.append(mask)


        if print_timings:
            #### TIMING
            step_time = time.time()-step_start_time
            print('reading rgbs and masks %.2f' % step_time)
            step_start_time = time.time()
            ###

        # print('len(rgbs), S, S_load', len(rgbs), self.S, self.S_load)
        # print('len(rgbs)', len(rgbs))
        if self.S < self.S_load:
            s_ind = np.random.randint(0, self.S_load-self.S)
            rgbs = rgbs[s_ind:s_ind+self.S]
            masks = masks[s_ind:s_ind+self.S]
            trajs = trajs[s_ind:s_ind+self.S]
            visibles = visibles[s_ind:s_ind+self.S]
            valids = valids[s_ind:s_ind+self.S]
        # now everything should be length S

        rgbs, occs, masks, trajs, visibles, valids = self.add_occluders(rgbs, masks, trajs, visibles, valids)

        if print_timings:
            #### TIMING
            step_time = time.time()-step_start_time
            print('add occ %.2f' % step_time)
            step_start_time = time.time()
            ###

        # print('occ rgbs[0]', rgbs[0].shape)
        if self.use_augs:
            rgbs, trajs, visibles = self.add_photometric_augs(rgbs, trajs, visibles)
            rgbs, occs, masks, trajs = self.add_spatial_augs(rgbs, occs, masks, trajs, visibles)
        else:
            rgbs, occs, masks, trajs = self.just_crop(rgbs, occs, masks, trajs, visibles)


        if print_timings:
            #### TIMING
            step_time = time.time()-step_start_time
            print('other augs %.2f' % step_time)
            step_start_time = time.time()
            ###
 
        H, W = rgbs[0].shape[:2]
        assert(H==self.crop_size[0])
        assert(W==self.crop_size[1])
        # mark any traj where occ=255 as invisible, since this indicates padding
        for s in range(self.S):
            xy = trajs[s].round().astype(np.int32) # N, 2
            x, y = xy[:,0], xy[:,1] # N
            x_ = x.clip(0, W-1)
            y_ = y.clip(0, H-1)
            inds = (occs[s][y_,x_] == 255) & (x >= 0) & (x <= W-1) & (y >= 0) & (y <= H-1)
            # inds = np.logical_and(np.logical_and( >= x0, trajs[i,:,0] < x1), np.logical_and(trajs[i,:,1] >= y0, trajs[i,:,1] < y1))
            visibles[s,inds] = 0

        # mark oob points as invisible
        for s in range(self.S):
            oob_inds = np.logical_or(np.logical_or(np.logical_or(trajs[s,:,0] < 0, trajs[s,:,0] > W-1), trajs[s,:,1] < 0), trajs[s,:,1] > H-1)
            visibles[s,oob_inds] = 0

        if self.force_twice_vis:
            # ensure that the point is visible at frame0 and at least one other frame
            vis0 = visibles[0] > 0
            inbound0 = (trajs[0,:,0] >= 0) & (trajs[0,:,0] <= W-1) & (trajs[0,:,1] >= 0) & (trajs[0,:,1] <= H-1)
            inbound_other = (trajs[1,:,0] >= 0) & (trajs[1,:,0] <= W-1) & (trajs[1,:,1] >= 0) & (trajs[1,:,1] <= H-1)
            vis_other = visibles[1] > 0
            for s in range(2,self.S):
                inbound_i = (trajs[s,:,0] >= 0) & (trajs[s,:,0] <= W-1) & (trajs[s,:,1] >= 0) & (trajs[s,:,1] <= H-1)
                inbound_other = inbound_other | inbound_i
                vis_i = visibles[s] > 0
                vis_other = vis_other | vis_i
            inbound_ok = inbound0 & inbound_other
            vis_ok = vis0 & vis_other
        else:
            assert(False) # only twice inbound is supported right now

        inb_and_vis = inbound_ok & vis_ok
        trajs = trajs[:,inb_and_vis]
        visibles = visibles[:,inb_and_vis]
        valids = valids[:,inb_and_vis]

        if self.force_last_vis:
            # ensure that the point is visible at the last frame
            visI = visibles[-1] > 0
            inboundI = (trajs[-1,:,0] >= 0) & (trajs[-1,:,0] <= W-1) & (trajs[-1,:,1] >= 0) & (trajs[-1,:,1] <= H-1)
            inb_and_vis = inboundI & visI
            trajs = trajs[:,inb_and_vis]
            visibles = visibles[:,inb_and_vis]
            valids = valids[:,inb_and_vis]

        if self.force_all_inb:
            inbound = (trajs[0,:,0] >= 0) & (trajs[0,:,0] <= W-1) & (trajs[0,:,1] >= 0) & (trajs[0,:,1] <= H-1)
            for s in range(1,self.S):
                inbound_i = (trajs[s,:,0] >= 0) & (trajs[s,:,0] <= W-1) & (trajs[s,:,1] >= 0) & (trajs[s,:,1] <= H-1)
                inbound = inbound & inbound_i
            trajs = trajs[:,inbound]
            visibles = visibles[:,inbound]
            valids = valids[:,inbound]

        if trajs.shape[1] <= self.N:
            # print('warning: too few trajs; returning None')
            return None, False

        # favor trajectories that are visible in the last quarter
        favor = False
        if favor:
            vis_sta = np.mean(visibles[:self.S//4]*valids[:self.S//4], axis=0)
            inv_mid = np.mean((1-visibles[self.S//4:-self.S//4])*valids[self.S//4:-self.S//4], axis=0)
            vis_end = np.mean(visibles[-self.S//4:]*valids[-self.S//4:], axis=0)
            # inds = np.argsort(-(np.mean(visibles[-self.S//4:]*valids[-self.S//4:], axis=0))

            # argsort gives us ascending
            # vis_sta 
            # inds = np.argsort(-1*(vis_sta+inv_mid+2*vis_end))
            inds = np.argsort(-1*(vis_sta+2*vis_end))
            N_ = min(trajs.shape[1], self.N*32)
            inds = inds[:N_]
            trajs = trajs[:,inds]
            visibles = visibles[:,inds]
            valids = valids[:,inds]
            # trajs = trajs[:,:N_]
            # visibles = visibles[:,:N_]
            # valids = valids[:,:N_]
        
        N_ = min(trajs.shape[1], self.N)
        inds = np.random.choice(trajs.shape[1], N_, replace=False)

        trajs_full = np.zeros((self.S, self.N, 2)).astype(np.float32)
        visibles_full = np.zeros((self.S, self.N)).astype(np.float32)
        valids_full = np.zeros((self.S, self.N)).astype(np.float32)
        # valids = np.zeros((self.N)).astype(np.float32)
        trajs_full[:,:N_] = trajs[:,inds]
        visibles_full[:,:N_] = visibles[:,inds]
        valids_full[:,:N_] = valids[:,inds]

        rgbs = torch.from_numpy(np.stack(rgbs, 0)).permute(0, 3, 1, 2) # S, C, H, W
        occs = torch.from_numpy(np.stack(occs, 0)).unsqueeze(1) # S, 1, H, W
        masks = torch.from_numpy(np.stack(masks, 0)).unsqueeze(1) # S, 1, H, W
        trajs = torch.from_numpy(trajs_full) # S, N, 2
        visibles = torch.from_numpy(visibles_full) # S, N
        valids = torch.from_numpy(valids_full) # S, N

        if torch.sum(valids[0,:]) < self.N:
            return None, False

        if print_timings:
            #### TIMING
            step_time = time.time()-step_start_time
            print('inb and vis %.2f' % step_time)
            step_start_time = time.time()
            ###

        sample = {
            # 'cur_rgb_path': cur_rgb_path,
            # 'img_names': img_names,
            'rgbs': rgbs,
            'occs': occs,
            'masks': masks,
            'trajs': trajs,
            'visibles': visibles,
            'valids': valids,
        }
        return sample, True
    
    def __getitem__(self, index):
        gotit = False
        fail_count = 0
        
        sample, gotit = self.getitem_helper(index)
        if not gotit:
            print('warning: sampling failed')
            # fake sample, so we can still collate
            sample = {
                'rgbs': torch.zeros((self.S, 3, self.crop_size[0], self.crop_size[1])),
                'occs': torch.zeros((self.S, 1, self.crop_size[0], self.crop_size[1])),
                'masks': torch.zeros((self.S, 1, self.crop_size[0], self.crop_size[1])),
                'trajs': torch.zeros((self.S, self.N, 2)),
                'visibles': torch.zeros((self.S, self.N)),
                'valids': torch.zeros((self.S, self.N)),
            }
            
        return sample, gotit
    
    def add_occluders(self, rgbs, masks, trajs, visibles, valids, print_timings=False):
        '''
        Input:
            rgbs --- list of len S, each = np.array (H, W, 3)
            trajs --- np.array (S, N, 2)
        Output:
            rgbs_aug --- np.array (S, H, W, 3)
            trajs_aug --- np.array (S, N_new, 2)
            visibles_aug --- np.array (S, N_new)
        '''

        T, N, _ = trajs.shape

        # print('trajs', trajs.shape)
        # print('len(rgbs)', len(rgbs))
        
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]

        assert(S==T)

        if print_timings:
            step_start_time = time.time()
        

        # rgbs = [0.1*rgb.astype(np.float32) for rgb in rgbs]
        rgbs = [rgb.astype(np.float32) for rgb in rgbs]
        occs = [np.zeros_like(rgb[:,:,0]) for rgb in rgbs]

        max_occ = 12
        alt_inds = np.random.choice(len(self.occ_rgb_paths), max_occ, replace=False)

        if print_timings:
            step_time = time.time()-step_start_time
            print('  occ init %.2f' % step_time)
            step_start_time = time.time()
        
        
        ############ occluders from other videos ############
        for oi in range(max_occ): # number of occluders:
            # alt_ind = np.random.choice(len(self.occ_rgb_paths))
            alt_ind = alt_inds[oi]
            occ_rgb_path = self.occ_rgb_paths[alt_ind]
            occ_mask_path = self.occ_mask_paths[alt_ind]
            occ_start_ind = self.occ_start_inds[alt_ind]
            occ_traj_path = self.occ_traj_paths[alt_ind]

            # print('occ_rgb_path', occ_rgb_path)
            # print('occ_start_ind', occ_start_ind)
            # print('occ_traj_path', occ_traj_path)
            
            img_names = [folder.split('/')[-1].split('.')[0] for folder in glob.glob(os.path.join(occ_rgb_path, "*"))]
            img_names = sorted(img_names)
            img_names = img_names[occ_start_ind:occ_start_ind+self.S_load]

            occ_info = np.load(occ_traj_path, allow_pickle=True).item()
            id_str = list(occ_info.keys())[np.random.choice(len(occ_info))]
            alt_trajs = occ_info[id_str] # S,N,2, with often N==0

            occ_id = int(id_str)

            if print_timings:
                step_time = time.time()-step_start_time
                print('  load images and alt %.2f' % step_time)
                step_start_time = time.time()

            alt_rgbs = []
            alt_masks = []
            alt_masks_blur = []

            for img_name in img_names:
                with Image.open(os.path.join(occ_rgb_path, '{0}.webp'.format(img_name))) as im:
                    alt_rgbs.append(np.array(im))
                mask = readImage(os.path.join(occ_mask_path, '{0}.pfm'.format(img_name)))
                mask = (mask==occ_id).astype(np.float32)
                # mask_  = np.clip(cv2.GaussianBlur(mask,(3,3),0) + mask, 0,1).reshape(H, W, 1) # widen slightly, but keep all the important pixels
                mask_blur = np.clip(cv2.GaussianBlur(mask,(3,3),0), 0,1).reshape(H, W, 1)
                # mask_blur = mask
                alt_masks.append(mask)#.reshape(H, W, 1))
                alt_masks_blur.append(mask_blur)#.reshape(H, W, 1))

            if print_timings:
                step_time = time.time()-step_start_time
                print('  get masks and blur %.2f' %  step_time)
                step_start_time = time.time()
            
            alt_visibles = np.ones((self.S, alt_trajs.shape[1]))
            alt_valids = np.ones((self.S, alt_trajs.shape[1]))

            # random photometric aug on this occluder
            alt_rgbs, alt_trajs, alt_visibles = self.add_photometric_augs(alt_rgbs, alt_trajs, alt_visibles, eraser=False, replace=False)
            # if print_timings:
            #     step_time = time.time()-step_start_time
            #     print('  get photo aug %.2f' %  step_time)
            #     step_start_time = time.time()

            alt_masks_blur = [alt_mask.reshape(H,W,1) for alt_mask in alt_masks_blur]
            rgbs = [rgb*(1.0-alt_mask)+alt_rgb*alt_mask for (rgb,alt_rgb,alt_mask) in zip(rgbs,alt_rgbs,alt_masks_blur)]
            if print_timings:
                step_time = time.time()-step_start_time
                print('  apply masks %.2f' % step_time)
                step_start_time = time.time()
                
            occs = [occ+alt_mask for (occ, alt_mask) in zip(occs, alt_masks)]
            
            if print_timings:
                step_time = time.time()-step_start_time
                print('  update occs %.2f' % step_time)
                step_start_time = time.time()

            
            # # darken the non-occluder, for debug
            # rgbs = [rgb*(1.0-(alt_mask*0.5)) for (rgb,alt_rgb,alt_mask) in zip(rgbs,alt_rgbs,alt_masks)]

            # any prev traj in the new masks should be marked invisible
            for s in range(S):
                xy = trajs[s].round().astype(np.int32) # N, 2
                x, y = xy[:,0], xy[:,1] # N
                # cond1 = (x >= 0) & (x <= W-1) & (y >= 0) & (y <= H-1)
                # x = x[inds]
                # y = [inds]
                x_ = x.clip(0, W-1)
                y_ = y.clip(0, H-1)
                inds = (alt_masks[s][y_,x_] == 1) & (x >= 0) & (x <= W-1) & (y >= 0) & (y <= H-1)
                # inds = np.logical_and(np.logical_and( >= x0, trajs[i,:,0] < x1), np.logical_and(trajs[i,:,1] >= y0, trajs[i,:,1] < y1))
                visibles[s, inds] = 0


            trajs = np.concatenate([trajs, alt_trajs], axis=1)
            valids = np.concatenate([valids, alt_valids], axis=1)
            visibles = np.concatenate([visibles, alt_visibles], axis=1)

            if print_timings:
                step_time = time.time()-step_start_time
                print('  update info %.2f'%  step_time)
                step_start_time = time.time()
            
        rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        return rgbs, occs, masks, trajs, visibles, valids

    def add_photometric_augs(self, rgbs, trajs, visibles, eraser=True, replace=True):
        T, N, _ = trajs.shape

        # print('trajs', trajs.shape)
        # print('len(rgbs)', len(rgbs))
        
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert(S==T)

        # rgbs = [0.1*rgb.astype(np.float32) for rgb in rgbs]
        
        if eraser:
            ############ eraser transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            for i in range(1, S):
                if np.random.rand() < self.eraser_aug_prob:
                    # mean_color = np.mean(rgbs[i].reshape(-1, 3), axis=0)
                    for _ in range(np.random.randint(1, self.eraser_max+1)): # number of times to occlude
                        # mean_color = np.mean(rgbs[i].reshape(-1, 3), axis=0)
                        
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                        dy = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                        x0 = np.clip(xc - dx/2, 0, W-1).round().astype(np.int32)
                        x1 = np.clip(xc + dx/2, 0, W-1).round().astype(np.int32)
                        y0 = np.clip(yc - dy/2, 0, H-1).round().astype(np.int32)
                        y1 = np.clip(yc + dy/2, 0, H-1).round().astype(np.int32)
                        # print(x0, x1, y0, y1)
                        mean_color = np.mean(rgbs[i][y0:y1, x0:x1, :].reshape(-1,3), axis=0)
                        rgbs[i][y0:y1, x0:x1, :] = mean_color

                        occ_inds = np.logical_and(np.logical_and(trajs[i,:,0] >= x0, trajs[i,:,0] < x1), np.logical_and(trajs[i,:,1] >= y0, trajs[i,:,1] < y1))
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        if replace:

            rgbs_alt = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]
            rgbs_alt = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs_alt]
            
            ############ replace transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            rgbs_alt = [rgb.astype(np.float32) for rgb in rgbs_alt]
            for i in range(1, S):
                if np.random.rand() < self.replace_aug_prob:
                    for _ in range(np.random.randint(1, self.replace_max+1)): # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(self.replace_bounds[0], self.replace_bounds[1])
                        dy = np.random.randint(self.replace_bounds[0], self.replace_bounds[1])
                        x0 = np.clip(xc - dx/2, 0, W-1).round().astype(np.int32)
                        x1 = np.clip(xc + dx/2, 0, W-1).round().astype(np.int32)
                        y0 = np.clip(yc - dy/2, 0, H-1).round().astype(np.int32)
                        y1 = np.clip(yc + dy/2, 0, H-1).round().astype(np.int32)

                        wid = x1-x0
                        hei = y1-y0
                        y00 = np.random.randint(0, H-hei)
                        x00 = np.random.randint(0, W-wid)
                        fr = np.random.randint(0, S)
                        rep = rgbs_alt[fr][y00:y00+hei, x00:x00+wid, :]
                        rgbs[i][y0:y1, x0:x1, :] = rep
                                                
                        # print(x0, x1, y0, y1)
                        # mean_color = np.mean(rgbs[i][y0:y1, x0:x1, :].reshape(-1,3), axis=0)
                        # rgbs[i][y0:y1, x0:x1, :] = mean_color

                        
                        # mean_color = np.mean(rgbs[i][y0:y1, x0:x1, :].reshape(-1,3), axis=0)
                        # rgbs[i][y0:y1, x0:x1, :] = mean_color
                        
                        occ_inds = np.logical_and(np.logical_and(trajs[i,:,0] >= x0, trajs[i,:,0] < x1), np.logical_and(trajs[i,:,1] >= y0, trajs[i,:,1] < y1))
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]
            

        ############ photometric augmentation ############
        if np.random.rand() < self.color_aug_prob:
            # random per-frame amount of aug
            rgbs = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        if np.random.rand() < self.blur_aug_prob:
            # random per-frame amount of blur
            rgbs = [np.array(self.blur_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        return rgbs, trajs, visibles

    def add_spatial_augs(self, rgbs, occs, masks, trajs, visibles):
        T, N, _ = trajs.shape

        # print('trajs', trajs.shape)
        # print('len(rgbs)', len(rgbs))
        
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert(S==T)

        rgbs = [rgb.astype(np.float32) for rgb in rgbs]
        
        ############ spatial transform ############

        # padding
        pad_x0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_x1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_y0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_y1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])

        # print('rgb', rgbs[0].shape)
        # print('mask', masks[0].shape)
        # print('coc', occs[0].shape)
        rgbs = [np.pad(rgb, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for rgb in rgbs]
        occs = [np.pad(occ, ((pad_y0, pad_y1), (pad_x0, pad_x1)), constant_values=255) for occ in occs]
        masks = [np.pad(mask, ((pad_y0, pad_y1), (pad_x0, pad_x1))) for mask in masks]
        trajs[:,:,0] += pad_x0
        trajs[:,:,1] += pad_y0
        H, W = rgbs[0].shape[:2]
        

        # scaling + stretching
        scale = np.random.uniform(self.resize_lim[0], self.resize_lim[1])
        scale_x = scale
        scale_y = scale
        H_new = H
        W_new = W

        scale_delta_x = 0.0
        scale_delta_y = 0.0

        rgbs_scaled = []
        occs_scaled = []
        masks_scaled = []
        trajs_scaled = []
        
        scales_x = []
        scales_y = []
        for s in range(S):
            if s==1:
                scale_delta_x = np.random.uniform(-self.resize_delta, self.resize_delta)
                scale_delta_y = np.random.uniform(-self.resize_delta, self.resize_delta)
            elif s > 1:
                scale_delta_x = scale_delta_x*0.8 + np.random.uniform(-self.resize_delta, self.resize_delta)*0.2
                scale_delta_y = scale_delta_y*0.8 + np.random.uniform(-self.resize_delta, self.resize_delta)*0.2
            scale_x = scale_x + scale_delta_x
            scale_y = scale_y + scale_delta_y

            # bring h/w closer
            scale_xy = (scale_x + scale_y)*0.5
            scale_x = scale_x*0.5 + scale_xy*0.5
            scale_y = scale_y*0.5 + scale_xy*0.5
            
            # don't get too crazy
            scale_x = np.clip(scale_x, 0.2, 2.0)
            scale_y = np.clip(scale_y, 0.2, 2.0)
            
            H_new = int(H * scale_y)
            W_new = int(W * scale_x)

            # make it at least slightly bigger than the crop area,
            # so that the random cropping can add diversity
            H_new = np.clip(H_new, self.crop_size[0]+10, None)
            W_new = np.clip(W_new, self.crop_size[1]+10, None)
            # recompute scale in case we clipped
            scale_x = W_new/float(W)
            scale_y = H_new/float(H)

            # print('H_new, W_new', H_new, W_new)
            # dim_resize = (W_new, H_new * S)
            rgbs_scaled.append(cv2.resize(rgbs[s], (W_new, H_new), interpolation=cv2.INTER_LINEAR))
            occs_scaled.append(cv2.resize(occs[s], (W_new, H_new), interpolation=cv2.INTER_LINEAR))
            masks_scaled.append(cv2.resize(masks[s], (W_new, H_new), interpolation=cv2.INTER_LINEAR))
            trajs[s,:,0] *= scale_x
            trajs[s,:,1] *= scale_y
        rgbs = rgbs_scaled
        occs = occs_scaled
        masks = masks_scaled
        
        ok_inds = visibles[0,:] > 0
        vis_trajs = trajs[:,ok_inds] # S,?,2

        if vis_trajs.shape[1] > 0:
            mid_x = np.mean(vis_trajs[0,:,0])
            mid_y = np.mean(vis_trajs[0,:,1])
        else:
            mid_y = self.crop_size[0]
            mid_x = self.crop_size[1]
            
        x0 = int(mid_x - self.crop_size[1]//2)
        y0 = int(mid_y - self.crop_size[0]//2)
        
        offset_x = 0
        offset_y = 0
        
        for s in range(S):
            # on each frame, shift a bit more 
            if s==1:
                offset_x = np.random.randint(-self.max_crop_offset, self.max_crop_offset)
                offset_y = np.random.randint(-self.max_crop_offset, self.max_crop_offset)
            elif s > 1:
                offset_x = int(offset_x*0.8 + np.random.randint(-self.max_crop_offset, self.max_crop_offset+1)*0.2)
                offset_y = int(offset_y*0.8 + np.random.randint(-self.max_crop_offset, self.max_crop_offset+1)*0.2)
            x0 = x0 + offset_x
            y0 = y0 + offset_y

            H_new, W_new = rgbs[s].shape[:2]
            if H_new==self.crop_size[0]:
                y0 = 0
            else:
                y0 = min(max(0, y0), H_new - self.crop_size[0] - 1)
                
            if W_new==self.crop_size[1]:
                x0 = 0
            else:
                x0 = min(max(0, x0), W_new - self.crop_size[1] - 1)
            # print('rgbs[%d]' % s, rgbs[s].shape)
            # print('self.crop_size', self.crop_size)
            # print('x0, y0', x0, y0)
            rgbs[s] = rgbs[s][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            occs[s] = occs[s][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            masks[s] = masks[s][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            trajs[s,:,0] -= x0
            trajs[s,:,1] -= y0

            
        H_new = self.crop_size[0]
        W_new = self.crop_size[1]

        # flip
        h_flipped = False
        v_flipped = False
        if self.do_flip:
            # h flip
            if np.random.rand() < self.h_flip_prob:
                # print('h flip')
                h_flipped = True
                rgbs = [rgb[:,::-1] for rgb in rgbs]
                occs = [occ[:,::-1] for occ in occs]
                masks = [mask[:,::-1] for mask in masks]
            # v flip
            if np.random.rand() < self.v_flip_prob:
                # print('v flip')
                v_flipped = True
                rgbs = [rgb[::-1] for rgb in rgbs]
                occs = [occ[::-1] for occ in occs]
                masks = [mask[::-1] for mask in masks]
        if h_flipped:
            trajs[:,:,0] = W_new - trajs[:,:,0]
        if v_flipped:
            trajs[:,:,1] = H_new - trajs[:,:,1]
            
        return rgbs, occs, masks, trajs

    def just_crop(self, rgbs, occs, masks, trajs, visibles):
        T, N, _ = trajs.shape
        
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert(S==T)

        ############ spatial transform ############

        H_new = H
        W_new = W

        # simple random crop
        y0 = np.random.randint(0, H_new - self.crop_size[0])
        x0 = np.random.randint(0, W_new - self.crop_size[1])
        rgbs = [rgb[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for rgb in rgbs]
        occs = [occ[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for occ in occs]
        masks = [mask[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for mask in masks]
        trajs[:,:,0] -= x0
        trajs[:,:,1] -= y0
            
        return rgbs, occs, masks, trajs
    
    def __len__(self):
        # return 10
        return len(self.rgb_paths)


