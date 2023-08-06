from numpy import random
import torch
import numpy as np
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
from torch._C import dtype, set_flush_denormal
import utils.basic
import utils.improc
import glob
import cv2
from torchvision.transforms import ColorJitter, GaussianBlur

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')


class PointOdysseyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_location='/orion/group/point_odyssey',
                 dset='TRAIN',
                 use_augs=False,
                 S=8,
                 N=32,
                 crop_size=(368, 496),
    ):
        print('loading pointodyssey dataset...')

        self.S = S
        self.N = N

        self.use_augs = use_augs
        self.dset = dset

        self.rgb_paths = []
        self.traj_paths = []
        self.annotation_paths = []
        self.start_idx = []

        self.subdirs = []
        self.sequences = []
        self.seq_names = []
        if dset == "TRAIN":
            self.subdirs.append(os.path.join(dataset_location, 'train'))
        elif dset == "VAL":
            self.subdirs.append(os.path.join(dataset_location, 'val'))
        elif dset == "TEST":
            self.subdirs.append(os.path.join(dataset_location, 'test_clean'))

        for subdir in self.subdirs:
            for seq in glob.glob(os.path.join(subdir, "*")):
                seq_name = seq.split('/')[-1]
                self.sequences.append(seq)
                self.seq_names.append(seq_name)

        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))
        
        ## load trajectories
        print('loading trajectories...')
        for seq in self.sequences:
            dir_path = dataset_location
            rgb_path = os.path.join(seq, 'rgbs')

            for ii in range(len(os.listdir(rgb_path)) - self.S):
                self.rgb_paths.append([os.path.join(dir_path, seq, 'rgbs', 'rgb_%05d.jpg' % (ii + jj + 1)) for jj in range(self.S)])
                self.annotation_paths.append(os.path.join(seq, 'annotations.npz'))
                self.start_idx.append(ii)

        print('collected %d clips of length %d in %s (dset=%s)' % (
            len(self.rgb_paths), self.S, dataset_location, dset))

        # photometric augmentation
        # self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25 / 3.14)
        self.blur_aug = GaussianBlur(11, sigma=(0.1, 2.0))

        self.blur_aug_prob = 0.2
        self.color_aug_prob = 0.5

        # occlusion augmentation
        self.eraser_aug_prob = 0.25
        self.eraser_bounds = [20, 300]

        # spatial augmentations
        self.crop_size = crop_size
        self.min_scale = -0.1  # 2^this
        self.max_scale = 1.0  # 2^this
        # self.resize_lim = [0.8, 1.2]
        self.resize_aug_prob = 0.8

        self.crop_aug_prob = 0.5
        self.max_crop_offset = 10

        self.stretch_prob = 0.8
        self.max_stretch = 0.2
        self.do_flip = True
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5

    def getitem_helper(self, index):
        sample = None
        gotit = False

        rgb_paths = self.rgb_paths[index]
        # print('rgb_paths', len(rgb_paths))

        full_idx = self.start_idx[index] + np.arange(self.S)
        annotations_path = self.annotation_paths[index]
        annotations = np.load(annotations_path, allow_pickle=True)
        # print(annotations.files)
        trajs = annotations['trajs_2d'][full_idx].astype(np.float32)
        visibs = annotations['visibilities'][full_idx].astype(np.float32)
        visibs = (visibs==1).astype(np.float32)

        S,N,D = trajs.shape
        assert(D==2)
        assert(S==self.S)

        if N < self.N:
            print('returning before cropping: N=%d; need N=%d' % (N, self.N))
            return None, False

        rgbs = []
        for rgb_path in rgb_paths:
            with Image.open(rgb_path) as im:
                rgbs.append(np.array(im)[:, :, :3])
                
        if self.use_augs:
            assert(False)
            rgbs, trajs, visibs = self.add_photometric_augs(rgbs, trajs, visibs)
            rgbs, trajs = self.add_spatial_augs(rgbs, trajs)
        else:
            rgbs, trajs = self.just_crop(rgbs, trajs)

        H,W,C = rgbs[0].shape
        assert(C==3)
        
        # update visibility annotations
        for s in range(S):
            # avoid 1px edge
            oob_inds = np.logical_or(
                np.logical_or(trajs[s,:,0] < 1, trajs[s,:,0] > W-2),
                np.logical_or(trajs[s,:,1] < 1, trajs[s,:,1] > H-2))
            visibs[s,oob_inds] = 0

        # ensure that the point is visible at frame0
        vis0 = visibs[0] > 0
        trajs = trajs[:,vis0]
        visibs = visibs[:,vis0]

        # ensure that the point is visible in at least 3 frames total
        vis_ok = np.sum(visibs, axis=0) >= 3
        trajs = trajs[:,vis_ok]
        visibs = visibs[:,vis_ok]
        
        N = trajs.shape[1]
        
        # if N <= self.N:
        #     print('N=%d; ideally we want N=%d, but we will pad' % (N, self.N))

        N_ = min(N, self.N)

        # prep for batching, by fixing N
        valids = np.ones_like(visibs)

        if N > self.N:
            inds = utils.misc.farthest_point_sample_py(trajs[0], N_)
        else:
            inds = np.random.choice(trajs.shape[1], N_, replace=False)

        trajs_full = np.zeros((self.S, self.N, 2)).astype(np.float32)
        visibs_full = np.zeros((self.S, self.N)).astype(np.float32)
        valids_full = np.zeros((self.S, self.N)).astype(np.float32)

        trajs_full[:,:N_] = trajs[:,inds]
        visibs_full[:,:N_] = visibs[:,inds]
        valids_full[:,:N_] = valids[:,inds]


        rgbs = torch.from_numpy(np.stack(rgbs, 0)).permute(0,3,1,2)  # S, C, H, W
        trajs = torch.from_numpy(trajs_full)  # S, N, 2
        visibs = torch.from_numpy(visibs_full)  # S, N
        valids = torch.from_numpy(valids_full)  # S, N

        sample = {
            'rgbs': rgbs,
            'trajs': trajs,
            'visibs': visibs,
            'valids': valids,
        }
        return sample, True

    def __getitem__(self, index):
        gotit = False
        sample, gotit = self.getitem_helper(index)
        if not gotit:
            print('warning: sampling failed')
            # fake sample, so we can still collate
            sample = {
                'rgbs': torch.zeros((self.S, 3, self.crop_size[0], self.crop_size[1])),
                'trajs': torch.zeros((self.S, self.N, 2)),
                'visibs': torch.zeros((self.S, self.N)),
                'valids': torch.zeros((self.S, self.N)),
            }
        return sample, gotit
    

    def add_photometric_augs(self, rgbs, trajs, visibs):
        T, N, _ = trajs.shape

        # print('trajs', trajs.shape)
        # print('len(rgbs)', len(rgbs))

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert (S == T)

        # rgbs = [0.1*rgb.astype(np.float32) for rgb in rgbs]

        ############ eraser transform (per image after the first) ############
        rgbs = [rgb.astype(np.float32) for rgb in rgbs]
        for i in range(1, S):
            if np.random.rand() < self.eraser_aug_prob:
                mean_color = np.mean(rgbs[i].reshape(-1, 3), axis=0)
                for _ in range(np.random.randint(1, 3)):  # number of times to occlude
                    xc = np.random.randint(0, W)
                    yc = np.random.randint(0, H)
                    dx = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                    dy = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                    x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                    x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                    y0 = np.clip(yc - dy / 2, 0, W - 1).round().astype(np.int32)
                    y1 = np.clip(yc + dy / 2, 0, W - 1).round().astype(np.int32)
                    # print(x0, x1, y0, y1)
                    rgbs[i][y0:y1, x0:x1, :] = mean_color

                    occ_inds = np.logical_and(np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                                              np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1))
                    visibs[i, occ_inds] = 0
        rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        ############ photometric augmentation ############
        if np.random.rand() < self.color_aug_prob:
            # random per-frame amount of aug
            rgbs = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        if np.random.rand() < self.blur_aug_prob:
            # random per-frame amount of blur
            rgbs = [np.array(self.blur_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        return rgbs, trajs, visibs

    def add_spatial_augs(self, rgbs, trajs):
        T, N, _ = trajs.shape

        # print('trajs', trajs.shape)
        # print('len(rgbs)', len(rgbs))

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert (S == T)

        rgbs = [rgb.astype(np.float32) for rgb in rgbs]

        ############ spatial transform ############

        # scaling + stretching
        scale_x = 1.0
        scale_y = 1.0
        H_new = H
        W_new = W
        if np.random.rand() < self.resize_aug_prob:
            # print('spat')
            min_scale = np.maximum(
                (self.crop_size[0] + 8) / float(H),
                (self.crop_size[1] + 8) / float(W))

            scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
            scale_x = scale
            scale_y = scale
            # print('scale', scale)

            if np.random.rand() < self.stretch_prob:
                # print('stretch')
                scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
                scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

            scale_x = np.clip(scale_x, min_scale, None)
            scale_y = np.clip(scale_y, min_scale, None)

            # print('scale_x,y', scale_x, scale_y)

            H_new = int(H * scale_y)
            W_new = int(W * scale_x)

            # print('H_new, W_new', H_new, W_new)
            # dim_resize = (W_new, H_new * S)
            rgbs = [cv2.resize(rgb, (W_new, H_new), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]

        trajs[:, :, 0] *= scale_x
        trajs[:, :, 1] *= scale_y

        if np.random.rand() < self.crop_aug_prob:
            # per-timestep crop
            y0 = np.random.randint(0, H_new - self.crop_size[0])
            x0 = np.random.randint(0, W_new - self.crop_size[1])
            for s in range(S):
                # on each frame, maybe shift a bit more
                if s > 0 and np.random.rand() < self.crop_aug_prob:
                    x0 = x0 + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1)
                    y0 = y0 + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1)
                y0 = min(max(0, y0), H_new - self.crop_size[0] - 1)
                x0 = min(max(0, x0), W_new - self.crop_size[1] - 1)
                rgbs[s] = rgbs[s][y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

                trajs[s, :, 0] -= x0
                trajs[s, :, 1] -= y0
        else:
            # simple crop
            y0 = np.random.randint(0, H_new - self.crop_size[0])
            x0 = np.random.randint(0, W_new - self.crop_size[1])
            rgbs = [rgb[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]] for rgb in rgbs]
            trajs[:, :, 0] -= x0
            trajs[:, :, 1] -= y0

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
                rgbs = [rgb[:, ::-1] for rgb in rgbs]
            # v flip
            if np.random.rand() < self.v_flip_prob:
                # print('v flip')
                v_flipped = True
                rgbs = [rgb[::-1] for rgb in rgbs]
        if h_flipped:
            trajs[:, :, 0] = W_new - trajs[:, :, 0]
        if v_flipped:
            trajs[:, :, 1] = H_new - trajs[:, :, 1]

        return rgbs, trajs

    def just_crop(self, rgbs, trajs):
        T, N, _ = trajs.shape
        
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert(S==T)

        # simple random crop
        y0 = np.random.randint(0, H - self.crop_size[0])
        x0 = np.random.randint(0, W - self.crop_size[1])
        rgbs = [rgb[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for rgb in rgbs]
        trajs[:,:,0] -= x0
        trajs[:,:,1] -= y0
            
        return rgbs, trajs

    def __len__(self):
        return len(self.rgb_paths)
