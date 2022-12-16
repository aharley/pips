import torch
import numpy as np
import os
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
import glob
import json
import imageio
import cv2

class CrohdDataset(torch.utils.data.Dataset):
    def __init__(self, seqlen=8, dset='t', dataset_root='../head_tracking'):
        dataset_location = '%s/HT21' % dataset_root
        label_location = '%s/HT21Labels' % dataset_root
        subfolders = []

        if dset == 't':
            dataset_location = os.path.join(dataset_location, "train")
            label_location = os.path.join(label_location, "train")
            subfolders = ['HT21-01', 'HT21-02', 'HT21-03', 'HT21-04']
        elif dset == 'v':
            dataset_location = os.path.join(dataset_location, "val")
            label_location = os.path.join(label_location, "val")
            subfolders = ['HT21-11', 'HT21-12', 'HT21-13', 'HT21-14', 'HT21-15']
        else:
            raise Exception("unexpceted dset. Choose between t and v.")

        print('dataset_location', dataset_location)
        print('label_location', label_location)
        
        # read gt for subfolders
        self.dataset_location = dataset_location
        self.label_location = label_location
        self.seqlen = seqlen
        self.subfolders = subfolders
        self.folder_to_gt = {} # key: folder name, value: dict with fields boxlist, scorelist, vislist
        self.subfolder_lens = []
        for fid, subfolder in enumerate(subfolders):
            print("loading labels for folder {0}/{1}".format(fid+1, len(subfolders)))
            label_path = os.path.join(label_location, subfolder, 'gt/gt.txt')
            labels = np.loadtxt(label_path, delimiter=',')

            n_frames = int(labels[-1,0])
            self.subfolder_lens.append(n_frames // seqlen)
            n_heads = int(labels[:,1].max())

            # unlike our data, those lists are already aligned
            # indexing the second dimension gives the information of the head throughout the seq
            boxlist = np.zeros((n_frames, n_heads, 4))
            scorelist = -1 * np.ones((n_frames, n_heads))
            vislist = np.zeros((n_frames, n_heads))

            for i in range(labels.shape[0]):
                frame_id, head_id, bb_left, bb_top, bb_width, bb_height, conf, cid, vis = labels[i]
                frame_id = int(frame_id) - 1 # convert 1 indexed to 0 indexed
                head_id = int(head_id) - 1 # convert 1 indexed to 0 indexed

                scorelist[frame_id, head_id] = 1
                vislist[frame_id, head_id] = vis
                box_cur = np.array([bb_left, bb_top, bb_left+bb_width, bb_top+bb_height]) # convert xywh to x1, y1, x2, y2
                boxlist[frame_id, head_id] = box_cur

            self.folder_to_gt[subfolder] = {
                'boxlist': np.copy(boxlist),
                'scorelist': np.copy(scorelist),
                'vislist': np.copy(vislist)
            }

    def __getitem__(self, index):
        # identify which sample and which starting frame it is
        subfolder_id = 0
        while index >= self.subfolder_lens[subfolder_id]:
            index -= self.subfolder_lens[subfolder_id]
            subfolder_id += 1

        # start from subfolder and the frame
        subfolder = self.subfolders[subfolder_id]
        start_frame = index * self.seqlen

        # get gt
        S = self.seqlen
        boxlist = self.folder_to_gt[subfolder]['boxlist'][start_frame:start_frame+S] # S, n_head, 4
        scorelist = self.folder_to_gt[subfolder]['scorelist'][start_frame:start_frame+S] # S, n_head
        vislist = self.folder_to_gt[subfolder]['vislist'][start_frame:start_frame+S] # S, n_head

        rgbs = []
        for i in range(S):
            # read image
            image_name = os.path.join(self.dataset_location, subfolder, 'img1', str(start_frame+i+1).zfill(6)+'.jpg')
            rgb = np.array(Image.open(image_name))
            rgbs.append(rgb)

        rgbs = np.stack(rgbs, axis=0)
        xylist = np.stack([boxlist[:, :, [0,2]].mean(2), boxlist[:, :, [1,3]].mean(2)], axis=2) # center of the box

        sample = {
            'rgbs': rgbs, # (S, H, W, 3) in 0-255
            'boxlist': boxlist, # (S, N, 4), N = n heads
            'xylist': xylist, # (S, N, 2)
            'scorelist': scorelist, # (S, N)
            'vislist': vislist # (S, N)
        }

        return sample

    def __len__(self):
        return sum(self.subfolder_lens)

if __name__ == "__main__":
    B = 1
    S = 8
    shuffle=False
    dataset = HeadTrackingDataset(seqlen=S)

    from torch.utils.data import Dataset, DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=0)

    train_iterloader = iter(train_dataloader)

    sample = next(train_iterloader)

