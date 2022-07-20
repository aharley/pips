import random
import tensorboardX
import torch
import torch.nn as nn
import numpy as np
import utils.vox
import utils.improc
import utils.geom
import utils.basic
import utils.samp
import utils.py
import utils.track
import imageio
import cv2
import random

import scipy.sparse as sp
import torch.nn.functional as F
from itertools import combinations
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

from utils.basic import print_

def add_loss(name, total_loss, loss, coeff, summ_writer=None):
    if summ_writer is not None:
        # summ_writer should be Summ_writer object in utils.improc
        summ_writer.summ_scalar('unscaled_%s' % name, loss)
        summ_writer.summ_scalar('scaled_%s' % name, coeff*loss)
    total_loss = total_loss + coeff*loss
    return total_loss

# some code from: https://github.com/suruoxi/DistanceWeightedSampling
class MarginLoss(nn.Module):
    def __init__(self, margin=0.2, nu=0.0, weight=None, batch_axis=0, **kwargs):
        super(MarginLoss, self).__init__()
        self._margin = margin
        self._nu = nu

    def forward(self, anchors, positives, negatives, beta, a_indices=None):
        d_ap = torch.sqrt(torch.sum((positives - anchors)**2, dim=1) + 1e-8)
        d_an = torch.sqrt(torch.sum((negatives - anchors)**2, dim=1) + 1e-8)

        pos_loss = torch.clamp(d_ap - beta + self._margin, min=0.0)
        neg_loss = torch.clamp(beta - d_an + self._margin, min=0.0)

        pair_cnt = int(torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)))

        loss = torch.sum(pos_loss + neg_loss) / (1e-4 + pair_cnt)
        return loss, pair_cnt

class DistanceWeightedSampling(nn.Module):
    '''
    parameters
    ----------
    batch_k: int
        number of images per class

    Inputs:
        data: input tensor with shape (batch_size, edbed_dim)
            Here we assume the consecutive batch_k examples are of the same class.
            For example, if batch_k = 5, the first 5 examples belong to the same class,
            6th-10th examples belong to another class, etc.
    Outputs:
        a_indices: indicess of anchors
        x[a_indices]
        x[p_indices]
        x[n_indices]
        xxx

    '''

    def __init__(self, batch_k, cutoff=0.5, nonzero_loss_cutoff=1.4, normalize=False, **kwargs):
        super(DistanceWeightedSampling,self).__init__()
        self.batch_k = batch_k
        self.cutoff = cutoff
        self.nonzero_loss_cutoff = nonzero_loss_cutoff
        self.normalize = normalize
        
    def get_distance(self, x):
        square = torch.sum(x**2, dim=1, keepdims=True)
        distance_square = square + square.t() - (2.0 * torch.matmul(x, x.t()))
        return torch.sqrt(distance_square + torch.eye(x.shape[0], device=torch.device('cuda')))

    def forward(self, x):
        k = self.batch_k
        n, d = x.shape

        debug = False
        # debug = True
        if debug:
            np.set_printoptions(precision=3, suppress=True)
            print(x[:,:5])
            print(x.shape)
        
        distance = self.get_distance(x)
        
        distance = torch.clamp(distance, min=self.cutoff)
        if debug:
            print('distance:')#, end=' ')
            print(distance.detach().cpu().numpy())

        log_weights = ((2.0 - float(d)) * torch.log(distance)
                       - (float(d - 3) / 2) * torch.log(1.0 - 0.25 * (distance ** 2.0)))

        if debug:
            print('log_weights:')#, end=' ')
            print(log_weights.detach().cpu().numpy())
        
        weights = torch.exp(log_weights - torch.max(log_weights))

        if debug:
            print('weights:')#, end=' ')
            print(weights.detach().cpu().numpy())

        # Sample only negative examples by setting weights of
        # the same-class examples to 0.
        mask = torch.ones_like(weights)
        for i in list(range(0,n,k)):
            mask[i:i+k, i:i+k] = 0
            
        if debug:
            print('mask:')#, end=' ')
            print(mask.detach().cpu().numpy())
            print('dist < nonzero:')#, end=' ')
            print((distance < self.nonzero_loss_cutoff).float().detach().cpu().numpy())

        # let's eliminate nans and zeros immediately
        weights[torch.isnan(weights)] = 1.0
        weights[weights < 1e-2] = 1e-2

        weights = weights * mask * (distance < self.nonzero_loss_cutoff).float()
        if debug:
            print('masked weights:')#, end=' ')
            print(weights.detach().cpu().numpy())
        
        weights = weights.detach().cpu().numpy()

        if debug:
            print('np weights:')#, end=' ')
            print(weights)
        
        # weights[np.isnan(weights)] = 1.0
        # weights[weights < 1e-2] = 1e-2

        if debug:
            print('clean weights:')#, end=' ')
            print(weights)

        # careful divison here
        weights = weights / (1e-4 + np.sum(weights, axis=1, keepdims=True))
            
        if debug:
            print('new weights:')#, end=' ')
            # print(weights.detach().cpu().numpy())
            print(weights)
        
        a_indices = []
        p_indices = []
        n_indices = []

        # np_weights = weights.cpu().detach().numpy()
        np_weights = weights
        for i in list(range(n)):
            block_idx = i // k
            try:
                n_indices += np.random.choice(n, k-1, p=np_weights[i]).tolist()
            except:
                n_indices += np.random.choice(n, k-1).tolist()

            for j in list(range(block_idx * k, (block_idx + 1)*k)):
                if j != i:
                    a_indices.append(i)
                    p_indices.append(j)

        return a_indices, x[a_indices], x[p_indices], x[n_indices], x

def shuffle_valid_and_sink_invalid_boxes(boxes, tids, scores):
    # put the good boxes shuffled at the top;
    # sink the bad boxes to the bottom.

    # boxes are B x N x D
    # tids are B x N
    # scores are B x N
    B, N, D = list(boxes.shape)

    boxes_new = torch.zeros_like(boxes)
    tids_new = -1*torch.ones_like(tids)
    scores_new = torch.zeros_like(scores)

    for b in list(range(B)):

        # for the sake of training,
        # we want to mix up the ordering
        inds = list(range(N))
        np.random.shuffle(inds)

        boxes[b] = boxes[b,inds]
        scores[b] = scores[b,inds]
        tids[b] = tids[b,inds]
        
        inds = np.argsort(-1.0*scores[b].cpu().detach().numpy()) # descending
        inds = np.squeeze(inds)

        boxes_new[b] = boxes[b,inds]
        scores_new[b] = scores[b,inds]
        tids_new[b] = tids[b,inds]

        # print('ok, boxes old and new')
        # print(boxes[b])
        # print(boxes_new[b])
        # input()

    return boxes_new, tids_new, scores_new

def get_target_scored_box_single(target, boxes, tids, scores):
    # boxes are N x D
    # tids are N and int32
    # scores are N
    # here we retrieve one target box
    N, D = list(boxes.shape)
    
    box_ = torch.ones(D)
    score_ = torch.zeros(1)
    # print 'target = %d' % (target),

    count = 0
    for i in list(range(N)):
        box = boxes[i]
        tid = tids[i]
        score = scores[i]
        # print 'target = %d; tid = %d; score = %.2f' % (target, tid, score)
        if score > 0.0 and tid==target:
            # print 'got it:',
            # print box,
            # print score
            return box, score
    # did not find it; return empty stuff (with score 0)
    return box_, score_

def get_target_traj(targets, boxlist_s, tidlist_s, scorelist_s):
    # targets are B
    # boxlist_s are B x S x N x D
    # tidlist_s are B x S x N
    # scorelist_s are B x S x N
    
    B, S, N, D = list(boxlist_s.shape)
    # (no asserts on shape; boxlist could instead be lrtlist)

    # return box_traj for the target, sized B x S x D
    # and also the score_traj, sized B x S
    # (note the object may not live across all frames)

    box_traj = torch.zeros(B, S, D)
    score_traj = torch.zeros(B, S)
    for b in list(range(B)):
        for s in list(range(S)):
            box_, score_ = get_target_scored_box_single(targets[b], boxlist_s[b,s], tidlist_s[b,s], scorelist_s[b,s])
            box_traj[b,s] = box_
            score_traj[b,s] = score_
    return box_traj.cuda(), score_traj.cuda()

def collect_object_info(lrtlist_camRs, tidlist_s, scorelist_s, K, boxlist_camRs=None, mod='', summ_writer=None):
    # lrtlist_camRs is B x S x N x 19
    # boxlist_camRs is B x S x N x 9
    # tidlist_s is B x S x N
    # scorelist_s is B x S x N
    
    # K (int): number of objects to collect
    B, S, N, D = list(lrtlist_camRs.shape)
    
    # this returns a bunch of tensors that begin with dim K
    # these tensors are object-centric: along S is all the info for that particular obj
    # this is in contrast to something like boxes, which is frame-centric
    
    obj_lrt_traj = []
    obj_box_traj = []
    obj_tid_traj = []
    obj_score_traj = []
    for target_ind in list(range(K)):
        target_tid = tidlist_s[:,0,target_ind]
        tid_traj = torch.reshape(target_tid, [B, 1]).repeat(1, S)

        # extract its traj from the full tensors
        lrt_traj, score_traj = get_target_traj(
            target_tid,
            lrtlist_camRs,
            tidlist_s,
            scorelist_s)
        # lrt_traj is B x S x 19
        # score_traj is B x S
        if boxlist_camRs is not None:
            box_traj, _ = get_target_traj(
                target_tid,
                boxlist_camRs,
                tidlist_s,
                scorelist_s)
            # box_traj is B x S x 9
        else:
            box_traj = None

        obj_lrt_traj.append(lrt_traj)
        obj_box_traj.append(box_traj)
        obj_tid_traj.append(tid_traj)
        obj_score_traj.append(score_traj)

    ## stack up
    obj_lrt_traj = torch.stack(obj_lrt_traj, axis=0)
    # this is K x B x S x 19
    if boxlist_camRs is not None:
        obj_box_traj = torch.stack(obj_box_traj, axis=0)
        # this is K x B x S x 9
    else:
        obj_box_traj = None
    obj_tid_traj = torch.stack(obj_tid_traj, axis=0)
    # this is K x B x S
    obj_score_traj = torch.stack(obj_score_traj, axis=0)
    # this is K x B x S

    # return obj_lrt_traj, obj_tid_traj, obj_score_traj
    # return obj_lrt_traj, obj_box_traj, obj_score_traj
    return obj_lrt_traj, obj_tid_traj, obj_score_traj

def rescore_boxlist_with_inbound(camX_T_camR, boxlist_camR, tidlist, Z, Y, X, vox_util, only_cars=True, pad=0.0):
    # boxlist_camR is B x N x 9
    B, N, D = list(boxlist_camR.shape)
    assert(D==9)
    xyzlist = boxlist_camR[:,:,:3]
    # this is B x N x 3
    lenlist = boxlist_camR[:,:,3:7]
    # this is B x N x 3

    xyzlist = utils.geom.apply_4x4(camX_T_camR, xyzlist)
    
    validlist = 1.0-(torch.eq(tidlist, -1*torch.ones_like(tidlist))).float()
    # this is B x N
    
    if only_cars:
        biglist = (torch.norm(lenlist, dim=2) > 2.0).float()
        validlist = validlist * biglist

    xlist, ylist, zlist = torch.unbind(xyzlist, dim=2)
    inboundlist_0 = vox_util.get_inbounds(torch.stack([xlist+pad, ylist, zlist], dim=2), Z, Y, X, already_mem=False).float()
    inboundlist_1 = vox_util.get_inbounds(torch.stack([xlist-pad, ylist, zlist], dim=2), Z, Y, X, already_mem=False).float()
    inboundlist_2 = vox_util.get_inbounds(torch.stack([xlist, ylist, zlist+pad], dim=2), Z, Y, X, already_mem=False).float()
    inboundlist_3 = vox_util.get_inbounds(torch.stack([xlist, ylist, zlist-pad], dim=2), Z, Y, X, already_mem=False).float()
    inboundlist = inboundlist_0*inboundlist_1*inboundlist_2*inboundlist_3
    scorelist = validlist * inboundlist
    return scorelist

def rescore_lrtlist_with_inbound(lrtlist_camR, scorelist, Z, Y, X, vox_util, pad=0.0):
    # lrtlist_camR is B x N x 19
    # assume R is the coord where we want to check inbound-ness
    B, N, D = list(lrtlist_camR.shape)
    assert(D==19)
    clist = utils.geom.get_clist_from_lrtlist(lrtlist_camR)
    # this is B x N x 3

    xlist, ylist, zlist = torch.unbind(clist, dim=2)
    inboundlist_0 = vox_util.get_inbounds(torch.stack([xlist+pad, ylist, zlist], dim=2), Z, Y, X, already_mem=False).float()
    inboundlist_1 = vox_util.get_inbounds(torch.stack([xlist-pad, ylist, zlist], dim=2), Z, Y, X, already_mem=False).float()
    inboundlist_2 = vox_util.get_inbounds(torch.stack([xlist, ylist, zlist+pad], dim=2), Z, Y, X, already_mem=False).float()
    inboundlist_3 = vox_util.get_inbounds(torch.stack([xlist, ylist, zlist-pad], dim=2), Z, Y, X, already_mem=False).float()
    inboundlist = inboundlist_0*inboundlist_1*inboundlist_2*inboundlist_3
    scorelist = scorelist * inboundlist
    return scorelist

def rescore_boxlist_with_pointcloud(camX_T_camR, boxlist_camR, xyz_camX, scorelist, tidlist, thresh=1.0):
    # boxlist_camR is B x N x 9
    B, N, D = list(boxlist_camR.shape)
    assert(D==9)
    xyzlist = boxlist_camR[:,:,:3]
    # this is B x N x 3
    lenlist = boxlist_camR[:,:,3:7]
    # this is B x N x 3


    xyzlist = utils.geom.apply_4x4(camX_T_camR, xyzlist)

    # xyz_camX is B x V x 3
    xyz_camX = xyz_camX[:,::10]
    xyz_camX = xyz_camX.unsqueeze(1)
    # xyz_camX is B x 1 x V x 3
    xyzlist = xyzlist.unsqueeze(2)
    # xyzlist is B x N x 1 x 3

    dists = torch.norm(xyz_camX - xyzlist, dim=3)
    # this is B x N x V

    mindists = torch.min(dists, 2)[0]
    ok = (mindists < thresh).float()
    scorelist = scorelist * ok
    return scorelist

def rescore_lrtlist_with_pointcloud(lrtlist_camX, xyz_camX, scorelist, thresh=1.0, min_pts=1):
    # boxlist_camR is B x N x 9
    B, N, D = list(lrtlist_camX.shape)
    assert(D==19)
    xyzlist_camX = utils.geom.get_clist_from_lrtlist(lrtlist_camX)
    # this is B x N x 3

    # xyz_camX is B x V x 3
    xyz_camX = xyz_camX[:,::10] # for speed
    xyz_camX = xyz_camX.unsqueeze(1)
    # xyz_camX is B x 1 x V x 3
    xyzlist_camX = xyzlist_camX.unsqueeze(2)
    # xyzlist_camX is B x N x 1 x 3

    # dists = torch.norm(xyz_camX - xyzlist_camX, dim=3)
    dists = torch.min(xyz_camX - xyzlist_camX, dim=3)[0]
    # this is B x N x V

    # there should be at least 1 pt within 1m of the centroid
    # mindists = torch.min(dists, 2)[0]
    # ok = (mindists < thresh).float()
    num_ok = torch.sum(dists < thresh, dim=2)
    ok = (num_ok >= min_pts).float()
    scorelist = scorelist * ok
    return scorelist

def get_gt_flow(obj_lrtlist_camRs,
                obj_scorelist,
                camRs_T_camXs,
                Z, Y, X, 
                K=2,
                mod='',
                vis=True,
                summ_writer=None):
    # this constructs the flow field according to the given
    # box trajectories (obj_lrtlist_camRs) (collected from a moving camR)
    # and egomotion (encoded in camRs_T_camXs)
    # (so they do not take into account egomotion)
    # so, we first generate the flow for all the objects,
    # then in the background, put the ego flow
    N, B, S, D = list(obj_lrtlist_camRs.shape)
    assert(S==2) # as a flow util, this expects S=2

    flows = []
    masks = []
    for k in list(range(K)):
        obj_masklistR0 = utils.vox.assemble_padded_obj_masklist(
            obj_lrtlist_camRs[k,:,0:1],
            obj_scorelist[k,:,0:1],
            Z, Y, X,
            coeff=1.0)
        # this is B x 1(N) x 1(C) x Z x Y x Z
        # obj_masklistR0 = obj_masklistR0.squeeze(1)
        # this is B x 1 x Z x Y x X
        obj_mask0 = obj_masklistR0.squeeze(1)
        # this is B x 1 x Z x Y x X

        camR_T_cam0 = camRs_T_camXs[:,0]
        camR_T_cam1 = camRs_T_camXs[:,1]
        cam0_T_camR = utils.geom.safe_inverse(camR_T_cam0)
        cam1_T_camR = utils.geom.safe_inverse(camR_T_cam1)
        # camR0_T_camR1 = camR0_T_camRs[:,1]
        # camR1_T_camR0 = utils.geom.safe_inverse(camR0_T_camR1)

        # obj_masklistA1 = utils.vox.apply_4x4_to_vox(camR1_T_camR0, obj_masklistA0)
        # if vis and (summ_writer is not None):
        #     summ_writer.summ_occ('flow/obj%d_maskA0' % k, obj_masklistA0)
        #     summ_writer.summ_occ('flow/obj%d_maskA1' % k, obj_masklistA1)

        if vis and (summ_writer is not None):
            # summ_writer.summ_occ('flow/obj%d_mask0' % k, obj_mask0)
            summ_writer.summ_oned('flow/obj%d_mask0_%s' % (k, mod), torch.mean(obj_mask0, 3))
        
        _, ref_T_objs_list = utils.geom.split_lrtlist(obj_lrtlist_camRs[k])
        # this is B x S x 4 x 4
        ref_T_obj0 = ref_T_objs_list[:,0]
        ref_T_obj1 = ref_T_objs_list[:,1]
        obj0_T_ref = utils.geom.safe_inverse(ref_T_obj0)
        obj1_T_ref = utils.geom.safe_inverse(ref_T_obj1)
        # these are B x 4 x 4
        
        mem_T_ref = utils.vox.get_mem_T_ref(B, Z, Y, X)
        ref_T_mem = utils.vox.get_ref_T_mem(B, Z, Y, X)

        ref1_T_ref0 = utils.basic.matmul2(ref_T_obj1, obj0_T_ref)
        cam1_T_cam0 = utils.basic.matmul3(cam1_T_camR, ref1_T_ref0, camR_T_cam0)
        mem1_T_mem0 = utils.basic.matmul3(mem_T_ref, cam1_T_cam0, ref_T_mem)

        xyz_mem0 = utils.basic.gridcloud3d(B, Z, Y, X)
        xyz_mem1 = utils.geom.apply_4x4(mem1_T_mem0, xyz_mem0)

        xyz_mem0 = xyz_mem0.reshape(B, Z, Y, X, 3)
        xyz_mem1 = xyz_mem1.reshape(B, Z, Y, X, 3)

        # only use these displaced points within the obj mask
        # obj_mask03 = obj_mask0.view(B, Z, Y, X, 1).repeat(1, 1, 1, 1, 3)
        obj_mask0 = obj_mask0.view(B, Z, Y, X, 1)
        # # xyz_mem1[(obj_mask03 < 1.0).bool()] = xyz_mem0
        # cond = (obj_mask03 < 1.0).float()
        cond = (obj_mask0 > 0.0).float()
        xyz_mem1 = cond*xyz_mem1 + (1.0-cond)*xyz_mem0

        flow = xyz_mem1 - xyz_mem0
        flow = flow.permute(0, 4, 1, 2, 3)
        obj_mask0 = obj_mask0.permute(0, 4, 1, 2, 3)

        # if vis and k==0:
        if vis:
            summ_writer.summ_3d_flow('flow/gt_%d_%s' % (k, mod), flow, clip=4.0)

        masks.append(obj_mask0)
        flows.append(flow)

    camR_T_cam0 = camRs_T_camXs[:,0]
    camR_T_cam1 = camRs_T_camXs[:,1]
    cam0_T_camR = utils.geom.safe_inverse(camR_T_cam0)
    cam1_T_camR = utils.geom.safe_inverse(camR_T_cam1)

    mem_T_ref = utils.vox.get_mem_T_ref(B, Z, Y, X)
    ref_T_mem = utils.vox.get_ref_T_mem(B, Z, Y, X)

    cam1_T_cam0 = utils.basic.matmul2(cam1_T_camR, camR_T_cam0)
    mem1_T_mem0 = utils.basic.matmul3(mem_T_ref, cam1_T_cam0, ref_T_mem)

    xyz_mem0 = utils.basic.gridcloud3d(B, Z, Y, X)
    xyz_mem1 = utils.geom.apply_4x4(mem1_T_mem0, xyz_mem0)

    xyz_mem0 = xyz_mem0.reshape(B, Z, Y, X, 3)
    xyz_mem1 = xyz_mem1.reshape(B, Z, Y, X, 3)

    flow = xyz_mem1 - xyz_mem0
    flow = flow.permute(0, 4, 1, 2, 3)

    bkg_flow = flow

    # allow zero motion in the bkg
    any_mask = torch.max(torch.stack(masks, axis=0), axis=0)[0]
    masks.append(1.0-any_mask)
    flows.append(bkg_flow)

    flows = torch.stack(flows, axis=0)
    masks = torch.stack(masks, axis=0)
    masks = masks.repeat(1, 1, 3, 1, 1, 1)
    flow = utils.basic.reduce_masked_mean(flows, masks, dim=0)

    if vis:
        summ_writer.summ_3d_flow('flow/gt_complete', flow, clip=4.0)

    # flow is shaped B x 3 x D x H x W
    return flow

def get_synth_flow(occs,
                   unps,
                   summ_writer,
                   sometimes_zero=False,
                   do_vis=False):
    B,S,C,Z,Y,X = list(occs.shape)
    assert(S==2)
    assert(C==1)

    # we do not sample any rotations here, to keep the distribution purely
    # uniform across all translations
    # (rotation ruins this, since the pivot point is at the camera)
    # cam1_T_cam0 = [utils.geom.get_random_rt(B, r_amount=0.0, t_amount=1.0), # large motion
    #                utils.geom.get_random_rt(B, r_amount=0.0, t_amount=0.1, # small motion
    #                                         sometimes_zero=sometimes_zero)]
    # cam1_T_cam0 = random.sample(cam1_T_cam0, k=1)[0]

    # cam1_T_cam0 = utils.geom.get_random_rt(B, r_amount=0.0, t_amount=0.1)
    cam1_T_cam0 = utils.geom.get_random_rt(B, r_amount=0.0, t_amount=2.0)
    

    occ0 = occs[:,0]
    unp0 = unps[:,0]
    occ1 = utils.vox.apply_4x4_to_vox(cam1_T_cam0, occ0, binary_feat=True)
    unp1 = utils.vox.apply_4x4_to_vox(cam1_T_cam0, unp0)
    occs = [occ0, occ1]
    unps = [unp0, unp1]

    if do_vis:
        summ_writer.summ_occs('synth/occs', occs)
        summ_writer.summ_unps('synth/unps', unps, occs)
        
    mem_T_cam = utils.vox.get_mem_T_ref(B, Z, Y, X)
    cam_T_mem = utils.vox.get_ref_T_mem(B, Z, Y, X)
    mem1_T_mem0 = utils.basic.matmul3(mem_T_cam, cam1_T_cam0, cam_T_mem)
    xyz_mem0 = utils.basic.gridcloud3d(B, Z, Y, X)
    xyz_mem1 = utils.geom.apply_4x4(mem1_T_mem0, xyz_mem0)
    xyz_mem0 = xyz_mem0.reshape(B, Z, Y, X, 3)
    xyz_mem1 = xyz_mem1.reshape(B, Z, Y, X, 3)
    flow = xyz_mem1-xyz_mem0
    # this is B x Z x Y x X x 3
    flow = flow.permute(0, 4, 1, 2, 3)
    # this is B x 3 x Z x Y x X
    if do_vis:
        summ_writer.summ_3d_flow('synth/flow', flow, clip=2.0)

    if do_vis:
        occ0_e = utils.samp.backwarp_using_3d_flow(occ1, flow, binary_feat=True)
        unp0_e = utils.samp.backwarp_using_3d_flow(unp1, flow)
        summ_writer.summ_occs('synth/occs_stab', [occ0, occ0_e])
        summ_writer.summ_unps('synth/unps_stab', [unp0, unp0_e], [occ0, occ0_e])

    occs = torch.stack(occs, dim=1)
    unps = torch.stack(unps, dim=1)

    return occs, unps, flow, cam1_T_cam0

def get_safe_samples(valid, dims, N_to_sample, mode='3d', tol=5.0):
    N, C = list(valid.shape)
    assert(C==1)
    assert(N==np.prod(dims))
    inds, locs, valids = get_safe_samples_py(valid, dims, N_to_sample, mode=mode, tol=tol)
    inds = torch.from_numpy(inds).to('cuda')
    locs = torch.from_numpy(locs).to('cuda')
    valids = torch.from_numpy(valids).to('cuda')
    
    inds = torch.reshape(inds, [N_to_sample, 1])
    inds = inds.long()
    if mode=='3d':
        locs = torch.reshape(locs, [N_to_sample, 3])
    elif mode=='2d':
        locs = torch.reshape(locs, [N_to_sample, 2])
    else:
        assert(False)# choose 3d or 2d please
    locs = locs.float()
    valids = torch.reshape(valids, [N_to_sample])
    valids = valids.float()
    return inds, locs, valids

def get_safe_samples_py(valid, dims, N_to_sample, mode='3d', tol=5.0):
    if mode=='3d':
        Z, Y, X = dims
    elif mode=='2d':
        Y, X = dims
    else:
        assert(False) # please choose 2d or 3d
    valid = valid.detach().cpu()
    valid = np.reshape(valid, [-1])
    N_total = len(valid)
    # assert(N_to_sample < N_total) # otw we need a padding step, and maybe a mask in the loss
    initial_tol = tol

    all_inds = np.arange(N_total)
    # reshape instead of squeeze, in case one or zero come
    valid_inds = all_inds[np.reshape((np.where(valid > 0)), [-1])]
    N_valid = len(valid_inds)
    # print('initial tol = %.2f' % tol)
    # print('N_valid = %d' % N_valid)
    # print('N_to_sample = %d' % N_to_sample)
    if N_to_sample < N_valid:
        # ok we can proceed

        if mode=='3d':
            xyz = utils.basic.gridcloud3d_py(Z, Y, X)
            locs = xyz[np.reshape((np.where(valid > 0)), [-1])]
        elif mode=='2d':
            xy = utils.basic.gridcloud2d_py(Y, X)
            locs = xy[np.reshape((np.where(valid > 0)), [-1])]

        samples_ok = False
        nTries = 0
        while (not samples_ok):
            # print('sample try %d...' % nTries)
            nTries += 1
            sample_inds = np.random.permutation(N_valid).astype(np.int32)[:N_to_sample]
            samples_try = valid_inds[sample_inds]
            locs_try = locs[sample_inds]
            nn_dists = np.zeros([N_to_sample], np.float32)
            samples_ok = True # ok this might work

            for i, loc in enumerate(locs_try):
                # exclude the current samp
                other_locs0 = locs_try[:i]
                other_locs1 = locs_try[i+1:]
                other_locs = np.concatenate([other_locs0, other_locs1], axis=0) 
                dists = np.linalg.norm(
                    np.expand_dims(loc, axis=0).astype(np.float32) - other_locs.astype(np.float32), axis=1)
                mindist = np.min(dists)
                nn_dists[i] = mindist
                if mindist < tol:
                    samples_ok = False
            # ensure we do not get stuck here: every 100 tries, subtract 1px to make it easier
            tol = tol - nTries*0.01
        # print(locs_try)
        if tol < (initial_tol/2.0):
            print('warning: initial_tol = %.2f; final_tol = %.2f' % (initial_tol, tol))
        # utils.basic.print_stats_py('nn_dists_%s' % mode, nn_dists)

        # print('these look ok:')
        # print(samples_try[:10])
        valid = np.ones(N_to_sample, np.float32)
    else:
        print('not enough valid samples! returning a few fakes')
        if mode=='3d':
            perm = np.random.permutation(Z*Y*X)
            samples_try = perm[:N_to_sample].astype(np.int32)
            locs_try = np.zeros((N_to_sample, 3), np.float32)
        elif mode=='2d':
            perm = np.random.permutation(Y*X)
            samples_try = perm[:N_to_sample].astype(np.int32)
            locs_try = np.zeros((N_to_sample, 2), np.float32)
        else:
            assert(False) # 2d or 3d please
        valid = np.zeros(N_to_sample, np.float32)
    return samples_try, locs_try, valid

def get_synth_flow_v2(xyz_cam0,
                      occ0,
                      unp0,
                      vox_util,
                      summ_writer=None,
                      sometimes_zero=False):
    # this version re-voxlizes occ1, rather than warp
    B,C,Z,Y,X = list(unp0.shape)
    assert(C==3)
    
    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    # we do not sample any rotations here, to keep the distribution purely
    # uniform across all translations
    # (rotation ruins this, since the pivot point is at the camera)
    # cam1_T_cam0 = [utils.geom.get_random_rt(B, r_amount=0.0, t_amount=2.0), # large motion
    #                utils.geom.get_random_rt(B, r_amount=0.0, t_amount=0.1, # small motion
    #                                         sometimes_zero=sometimes_zero)]
    # cam1_T_cam0 = random.sample(cam1_T_cam0, k=1)[0]
    # cam1_T_cam0 = utils.geom.get_random_rt(B, r_amount=0.0, t_amount=0.1)
    # cam1_T_cam0 = utils.geom.get_random_rt(B, r_amount=0.0, t_amount=1.0)
    cam1_T_cam0 = utils.geom.get_random_rt(B, r_amount=0.0, t_amount=2.0)

    xyz_cam1 = utils.geom.apply_4x4(cam1_T_cam0, xyz_cam0)
    occ1 = vox_util.voxelize_xyz(xyz_cam1, Z, Y, X)
    unp1 = vox_util.apply_4x4_to_vox(cam1_T_cam0, unp0)
    occs = [occ0, occ1]
    unps = [unp0, unp1]

    if summ_writer is not None:
        summ_writer.summ_occs('synth/occs', occs)
        summ_writer.summ_unps('synth/unps', unps, occs)
        
    mem_T_cam = vox_util.get_mem_T_ref(B, Z, Y, X)
    cam_T_mem = vox_util.get_ref_T_mem(B, Z, Y, X)
    mem1_T_mem0 = utils.basic.matmul3(mem_T_cam, cam1_T_cam0, cam_T_mem)
    xyz_mem0 = utils.basic.gridcloud3d(B, Z, Y, X)
    xyz_mem1 = utils.geom.apply_4x4(mem1_T_mem0, xyz_mem0)
    xyz_mem0 = xyz_mem0.reshape(B, Z, Y, X, 3)
    xyz_mem1 = xyz_mem1.reshape(B, Z, Y, X, 3)
    flow = xyz_mem1-xyz_mem0
    # this is B x Z x Y x X x 3
    flow = flow.permute(0, 4, 1, 2, 3)
    # this is B x 3 x Z x Y x X
    if summ_writer is not None:
        summ_writer.summ_3d_flow('synth/flow', flow, clip=2.0)

    if summ_writer is not None:
        occ0_e = utils.samp.backwarp_using_3d_flow(occ1, flow, binary_feat=True)
        unp0_e = utils.samp.backwarp_using_3d_flow(unp1, flow)
        summ_writer.summ_occs('synth/occs_stab', [occ0, occ0_e])
        summ_writer.summ_unps('synth/unps_stab', [unp0, unp0_e], [occ0, occ0_e])

    occs = torch.stack(occs, dim=1)
    unps = torch.stack(unps, dim=1)

    return occs, unps, flow, cam1_T_cam0

def get_boxes_from_flow_mag(flow_mag, N, min_voxels=3):
    B, Z, Y, X = list(flow_mag.shape)
    # flow_mag is B x Z x Y x X

    ## plan:
    # take a linspace of threhsolds between the min and max
    # for each thresh
    #   create a binary map
    #   turn this into labels with connected_components
    # vis all these

    assert(B==1) # later i will extend

    flow_mag = flow_mag[0]
    # flow_mag is Z x Y x X
    flow_mag = flow_mag.detach().cpu().numpy()

    from cc3d import connected_components

    # adjust for numerical errors
    flow_mag = flow_mag*100.0

    boxlist = np.zeros([N, 9], dtype=np.float32)
    scorelist = np.zeros([N], dtype=np.float32)
    connlist = np.zeros([N, Z, Y, X], dtype=np.float32)
    boxcount = 0

    mag = np.reshape(flow_mag, [Z, Y, X])
    mag_min, mag_max = np.min(mag), np.max(mag)
    # print('mag min, max = %.6f, %.6f' % (mag_min, mag_max))
    
    threshs = np.linspace(mag_min, mag_max, num=20)
    threshs = threshs[1:-1]
    # print('threshs:', threshs)
    
    zg, yg, xg = utils.basic.meshgrid3d_py(Z, Y, X, stack=False, norm=False)
    box3d_list = []

    flow_mag_vis = flow_mag - np.min(flow_mag)
    flow_mag_vis = flow_mag_vis / np.max(flow_mag_vis)
    # utils.py.print_stats('flow_mag_vis', flow_mag_vis)
    image = (np.mean(flow_mag_vis, axis=1)*255.0).astype(np.uint8)
    image = np.stack([image, image, image], axis=2)

    # utils.py.print_stats('image', image)

    # sx = float(X)/np.abs(float(utils.vox.XMAX-utils.vox.XMIN))
    # sy = float(Y)/np.abs(float(utils.vox.YMAX-utils.vox.YMIN))
    # sz = float(Z)/np.abs(float(utils.vox.ZMAX-utils.vox.ZMIN))
    # print('scalars:', sx, sy, sz)
    
    for ti, thresh in enumerate(threshs):
        # print('working on thresh %d: %.2f' % (ti, thresh))
        mask = (mag > thresh).astype(np.int32)
        # if np.sum(mask) > 8: # if we have a few pixels to connect up 
        # if np.sum(mask) > 3: # if we have a few pixels to connect up 
        # if np.sum(mask) > 3 and np.sum(mask) < 10000: # if we have a few pixels to connect up

        vox = torch.from_numpy(mask).float().cuda().reshape(1, 1, Z, Y, X)
        print('vox', vox.shape)
        weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
        vox_fat = (F.conv3d(vox, weights, padding=1)).clamp(0, 1)
        mask = vox_fat.reshape(Z, Y, X).cpu().numpy().astype(np.int32)

        if True:
            # if thresh > 0.5:
            labels = connected_components(mask)
            segids = [ x for x in np.unique(labels) if x != 0 ]
            for si, segid in enumerate(segids):
                extracted_vox = (labels == segid)
                if np.sum(extracted_vox) > min_voxels: # if we have a few voxels to box up
                    # if True:
                    # print('thresh %.2f; segid = %d; size %d' % (thresh, segid, np.sum(extracted_vox)))

                    z = zg[extracted_vox==1]
                    y = yg[extracted_vox==1]
                    x = xg[extracted_vox==1]

                    # find the oriented box in birdview
                    im = np.sum(extracted_vox, axis=1) # reduce on the Y dim
                    im = im.astype(np.uint8)

                    # somehow the versions change 
                    # _, contours, hier = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    contours, hier = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
                    if contours:
                        cnt = contours[0]
                        rect = cv2.minAreaRect(cnt)

                        clip = False
                        if clip:
                            # i want to clip at the index where YMAX dips under the ground
                            # and where YMIN reaches above some reasonable height

                            shift = hyp.YMIN
                            scale = float(Y)/np.abs(float(hyp.YMAX-hyp.YMIN))
                            ymin_ = (hyp.FLOOR-shift)*scale
                            ymax_ = (hyp.CEIL-shift)*scale

                            if ymin_ > ymax_:
                                # this is true if y points downards
                                ymax_, ymin_ = ymin_, ymax_

                            # ymin = np.clip(np.min(y), ymin_, ymax_)
                            # ymax = np.clip(np.max(y), ymin_, ymax_)

                        ymin = np.min(y)
                        ymax = np.max(y)
                            
                        hei = ymax-ymin
                        yc = (ymax+ymin)/2.0

                        (xc,zc),(wid,dep),theta = rect
                        theta = -theta
                        
                        box = cv2.boxPoints(rect)
                        if dep < wid:
                            # dep goes along the long side of an oriented car
                            theta += 90.0
                            wid, dep = dep, wid
                        theta = utils.geom.deg2rad(theta)

                        if boxcount < N:#  and (yc > ymin_) and (yc < ymax_):
                            # bx, by = np.split(box, axis=1)
                            # boxpoints[boxcount,:] = box

                            box3d = [xc, yc, zc, wid, hei, dep, 0, theta, 0]
                            box3d = np.array(box3d).astype(np.float32)

                            already_have = False
                            for box3d_ in box3d_list:
                                if np.all(box3d_==box3d):
                                    already_have = True
                                # else:
                                #     print('already have this box')
                            # print('zc, yc, xc', zc, yc, xc)

                            # # when we run the flow mode, this is our set of conditions (to help flow not fail so hard)
                            # if ((not already_have) and
                            #     (hei >= 1.0) and
                            #     (wid >= 1.0) and
                            #     (dep >= 1.0) and 
                            #     (hei <= 40.0) and
                            #     (wid <= 40.0) and
                            #     (dep <= 40.0) and
                            #     mask[int(zc),int(yc),int(xc)]):

                            # # otw 
                            # if ((not already_have) and
                            #     (hei >= 1.0) and
                            #     (wid >= 1.0) and
                            #     (dep >= 1.0)):

                            # kitti mode
                            if ((not already_have) and
                                (hei >= 1.0) and
                                (wid >= 1.0) and
                                (dep >= 1.0) and 
                                (hei <= 40.0) and
                                (wid <= 40.0) and
                                (dep <= 40.0)):
                                # mask[int(zc),int(yc),int(xc)]):

                            
                                
                            #     # (dep < 20.0) and
                            #     # # be bigger than 1 vox
                            #     # (hei > 2.0) and
                            #     # (wid > 2.0) and
                            #     # (dep > 2.0)
                            # ):
                            # if not already_have:
                            # if ((not already_have) and
                            #     (hei >= 1.0) and
                            #     (wid >= 1.0) and
                            #     (dep >= 1.0)):
                                # print('thresh %.2f; segid = %d; size %d' % (thresh, segid, np.sum(extracted_vox)))
                                # print 'mean(y), min(y) max(y), ymin_, ymax_, ymin, ymax = %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % (
                                #     np.mean(y), np.min(y), np.max(y), ymin_, ymax_, ymin, ymax)
                                # print('got a box: xc, yc, zc = %.2f, %.2f, %.2f; wid, hei, dep = %.2f, %.2f, %.2f' % (
                                #     xc, yc, zc, wid, hei, dep))
                                # if not (
                                
                                # print 'wid, hei, dep = %.2f, %.2f, %.2f' % (wid, hei, dep)
                                # # print 'theta = %.2f' % theta
                                
                                box = np.int0(box)
                                cv2.drawContours(image,[box],-1,(0,191,255),1)

                                boxlist[boxcount,:] = box3d
                                # scorelist[boxcount] = np.random.uniform(0.1, 1.0)
                                scorelist[boxcount] = 1.0

                                conn_ = np.zeros([Z, Y, X], np.float32)
                                conn_[extracted_vox] = 1.0
                                connlist[boxcount] = conn_
                                
                                # imageio.imwrite('boxes_%02d.png' % (boxcount), image)
                                # imageio.imwrite('conn_%02d.png' % (boxcount), np.max(conn_, axis=1))
                                
                                boxcount += 1
                                box3d_list.append(box3d)
                            else:
                                # print('skipping a box that already exists')
                                pass
                    #     else:
                    #         print('dropping at the last second, due to some size issue')
                    # else:
                    #     print('contours did not close')
        #         else:
        #             print('not enough voxels')
        # else:
        #     print('not enough voxels at all')
            


    image = np.transpose(image, [2, 0, 1]) # channels first
    image = torch.from_numpy(image).float().to('cuda').unsqueeze(0)
    boxlist = torch.from_numpy(boxlist).float().to('cuda').unsqueeze(0)
    scorelist = torch.from_numpy(scorelist).float().to('cuda').unsqueeze(0)
    connlist = torch.from_numpy(connlist).float().to('cuda').unsqueeze(0)

    tidlist = torch.linspace(1.0, N, N).long().to('cuda')
    tidlist = tidlist.unsqueeze(0)

    image = utils.improc.preprocess_color(image)
    
    return image, boxlist, scorelist, tidlist, connlist
    # return image, boxlist, scorelist, tidlist, None


def get_boxes_from_binary(binary, N, min_voxels=3, offset_for_occlusion=False):
    B, Z, Y, X = list(binary.shape)

    # if offset_for_occlusion:
    #     print('warning: the occlusion offset code is a bit weird now, and should be cleaned up')
        
    # turn the binary map into labels with connected_components

    assert(B==1) # later i will extend

    binary = binary[0]
    # binary_act = binary_act[0]
    # binary is Z x Y x X
    mask = binary.detach().cpu().numpy().astype(np.int32)
    # mask_act = binary_act.detach().cpu().numpy().astype(np.int32)

    from cc3d import connected_components

    boxlist = np.zeros([N, 9], dtype=np.float32)
    scorelist = np.zeros([N], dtype=np.float32)
    connlist = np.zeros([N, Z, Y, X], dtype=np.float32)
    boxcount = 0

    zg, yg, xg = utils.basic.meshgrid3d_py(Z, Y, X, stack=False, norm=False)
    box3d_list = []

    labels = connected_components(mask)
    segids = [ x for x in np.unique(labels) if x != 0 ]
    for si, segid in enumerate(segids):
        extracted_vox = (labels == segid)
        # extracted_vox = extracted_vox * mask_act # trim to the actual voxels available here
        # if np.sum(extracted_vox) > min_voxels: # if we have a few voxels to box up
        # if True:
        #     # print('thresh %.2f; segid = %d; size %d' % (thresh, segid, np.sum(extracted_vox)))
        #     print('segid = %d; size %d' % (segid, np.sum(extracted_vox)))

        z = zg[extracted_vox==1]
        y = yg[extracted_vox==1]
        x = xg[extracted_vox==1]

        zmin = np.min(z)
        zmax = np.max(z)
        ymin = np.min(y)
        ymax = np.max(y)
        xmin = np.min(x)
        xmax = np.max(x)

        # instead of checking for min total size,
        # let's please check if each dimension is more than 1 voxel
        
        if (zmax-zmin > 2 and
            ymax-ymin > 2 and
            xmax-xmin > 2):
                
            # print('segid = %d; size %d' % (segid, np.sum(extracted_vox)))


            # find the oriented box in birdview
            im = np.sum(extracted_vox, axis=1) # reduce on the Y dim
            im = im.astype(np.uint8)

            # somehow the versions change 
            # _, contours, hier = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours, hier = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
            if contours:
                cnt = contours[0]
                rect = cv2.minAreaRect(cnt)

                ymin = np.min(y)
                ymax = np.max(y)

                hei = ymax-ymin
                yc = (ymax+ymin)/2.0

                (xc,zc),(wid,dep),theta = rect
                theta = -theta

                if offset_for_occlusion:
                    # assume that we just boxed up the front half
                    # print('theta', theta)
                    if theta < 45.0:
                        zc = zc - dep/2.0 + dep
                        dep = dep*2.0
                    else:
                        zc = zc - wid/2.0 + wid
                        wid = wid*2.0

                    # # somehow dep is not the right var right now,
                    # # but zc is!!
                    # zc = zc - wid/2.0 + wid
                    # wid = wid*2.0

                box = cv2.boxPoints(rect)
                if dep < wid:
                    # dep goes along the long side of an oriented car
                    theta += 90.0
                    wid, dep = dep, wid
                theta = utils.geom.deg2rad(theta)
                
                if boxcount < N:#  and (yc > ymin_) and (yc < ymax_):
                    # bx, by = np.split(box, axis=1)
                    # boxpoints[boxcount,:] = box

                    box3d = [xc, yc, zc, wid, hei, dep, 0, theta, 0]
                    box3d = np.array(box3d).astype(np.float32)

                    already_have = False
                    for box3d_ in box3d_list:
                        if np.all(box3d_==box3d):
                            already_have = True
                        # else:
                        #     print('already have this box')
                    # print('zc, yc, xc', zc, yc, xc)

                    # # when we run the flow mode, this is our set of conditions (to help flow not fail so hard)
                    # if ((not already_have) and
                    #     (hei >= 1.0) and
                    #     (wid >= 1.0) and
                    #     (dep >= 1.0) and 
                    #     (hei <= 40.0) and
                    #     (wid <= 40.0) and
                    #     (dep <= 40.0) and
                    #     mask[int(zc),int(yc),int(xc)]):

                    # # otw 
                    # if ((not already_have) and
                    #     (hei >= 1.0) and
                    #     (wid >= 1.0) and
                    #     (dep >= 1.0)):

                    # kitti mode
                    if ((not already_have) and
                        (hei >= 1.0) and
                        (wid >= 1.0) and
                        (dep >= 1.0)):
                        # (hei <= 40.0) and
                        # (wid <= 40.0) and
                        # (dep <= 40.0)):
                        # mask[int(zc),int(yc),int(xc)]):



                    #     # (dep < 20.0) and
                    #     # # be bigger than 1 vox
                    #     # (hei > 2.0) and
                    #     # (wid > 2.0) and
                    #     # (dep > 2.0)
                    # ):
                    # if not already_have:
                    # if ((not already_have) and
                    #     (hei >= 1.0) and
                    #     (wid >= 1.0) and
                    #     (dep >= 1.0)):
                        # print('thresh %.2f; segid = %d; size %d' % (thresh, segid, np.sum(extracted_vox)))
                        # print 'mean(y), min(y) max(y), ymin_, ymax_, ymin, ymax = %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % (
                        #     np.mean(y), np.min(y), np.max(y), ymin_, ymax_, ymin, ymax)
                        # print('got a box: xc, yc, zc = %.2f, %.2f, %.2f; wid, hei, dep = %.2f, %.2f, %.2f' % (
                        #     xc, yc, zc, wid, hei, dep))
                        # if not (

                        # print 'wid, hei, dep = %.2f, %.2f, %.2f' % (wid, hei, dep)
                        # # print 'theta = %.2f' % theta

                        box = np.int0(box)

                        boxlist[boxcount,:] = box3d
                        # scorelist[boxcount] = np.random.uniform(0.1, 1.0)
                        scorelist[boxcount] = 1.0

                        conn_ = np.zeros([Z, Y, X], np.float32)
                        conn_[extracted_vox] = 1.0
                        connlist[boxcount] = conn_

                        boxcount += 1
                        box3d_list.append(box3d)
                    else:
                        # print('skipping a box that already exists')
                        pass
            #     else:
            #         print('dropping at the last second, due to some size issue')
            # else:
            #     print('contours did not close')
            
    boxlist = torch.from_numpy(boxlist).float().to('cuda').unsqueeze(0)
    scorelist = torch.from_numpy(scorelist).float().to('cuda').unsqueeze(0)
    connlist = torch.from_numpy(connlist).float().to('cuda').unsqueeze(0)
    tidlist = torch.linspace(1.0, N, N).long().to('cuda')
    tidlist = tidlist.unsqueeze(0)
    return boxlist, scorelist, tidlist, connlist

def suppress_tiny_segments(occ, min_voxels=3, min_side=1):
    B, C, Z, Y, X = list(occ.shape)
    assert(C==1)

    occ_new = torch.zeros_like(occ)
    
    assert(B==1) # later i will extend

    # occ is Z x Y x X
    occ = occ[0,0].detach().cpu().numpy().astype(np.uint8)
    occ_new = occ_new[0,0].detach().cpu().numpy().astype(np.uint8)

    from cc3d import connected_components

    zg, yg, xg = utils.basic.meshgrid3d_py(Z, Y, X, stack=False, norm=False)

    labels = connected_components(occ)
    segids = [ x for x in np.unique(labels) if x != 0 ]
    for si, segid in enumerate(segids):
        extracted_vox = (labels == segid)
        
        if np.sum(extracted_vox) > min_voxels: # if we have a few voxels to box up
            
            z = zg[extracted_vox==1]
            y = yg[extracted_vox==1]
            x = xg[extracted_vox==1]
            zmin = np.min(z)
            zmax = np.max(z)
            ymin = np.min(y)
            ymax = np.max(y)
            xmin = np.min(x)
            xmax = np.max(x)

            if (zmax-zmin > min_side and
                ymax-ymin > min_side and
                xmax-xmin > min_side):
                occ_new = occ_new + extracted_vox
    return torch.from_numpy(occ_new).reshape(B, C, Z, Y, X).float().cuda()

def fit_box_to_binary(binary, oriented=False, add_pad=0.0):
    B, Z, Y, X = list(binary.shape)
        
    # turn the binary map into a single box

    assert(B==1) # later i will extend
    binary = binary[0]
    # binary is Z x Y x X
    
    mask = binary.detach().cpu().numpy().astype(np.int32)

    zg, yg, xg = utils.basic.meshgrid3d_py(Z, Y, X, stack=False, norm=False)

    z = zg[mask==1]
    y = yg[mask==1]
    x = xg[mask==1]

    zmin = np.min(z) - add_pad
    zmax = np.max(z) + add_pad
    ymin = np.min(y) - add_pad
    ymax = np.max(y) + add_pad
    xmin = np.min(x) - add_pad
    xmax = np.max(x) + add_pad

    if oriented:
        
        # find the oriented box in birdview
        im = np.sum(mask, axis=1) # reduce on the Y dim
        im = im.astype(np.uint8)


        # somehow the versions change 
        # _, contours, hier = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, hier = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

        if contours:
            cnt = contours[0]
            rect = cv2.minAreaRect(cnt)

            hei = ymax-ymin
            yc = (ymax+ymin)/2.0

            (xc,zc),(wid,dep),theta = rect
            theta = -theta

            box = cv2.boxPoints(rect)
            if dep < wid:
                # dep goes along the long side of an oriented car
                theta += 90.0
                wid, dep = dep, wid
            theta = utils.geom.deg2rad(theta)

            box3d = [xc, yc, zc, wid, hei, dep, 0, theta, 0]
            box3d = np.array(box3d).astype(np.float32)

            return torch.from_numpy(box3d).float().to(binary.device).unsqueeze(0)
        else:
            return None
    else:
        dep = zmax-zmin
        zc = (zmax+zmin)/2.0

        hei = ymax-ymin
        yc = (ymax+ymin)/2.0

        wid = xmax-xmin
        xc = (xmax+xmin)/2.0
        
        box3d = [xc, yc, zc, wid, hei, dep, 0, 0, 0]
        box3d = np.array(box3d).astype(np.float32)
        return torch.from_numpy(box3d).float().to(binary.device).unsqueeze(0)
        

def get_any_boxes_from_binary(binary, N, min_voxels=3, min_side=1, count_mask=None):
    B, Z, Y, X = list(binary.shape)
        
    # turn the binary map into labels with connected_components

    assert(B==1) # later i will extend
    binary = binary[0]
    # binary is Z x Y x X

    if count_mask is None:
        count_mask = torch.ones_like(binary)
    else:
        count_mask = count_mask.reshape(Z, Y, X)
    
    mask = binary.detach().cpu().numpy().astype(np.int32)
    count_mask = count_mask.detach().cpu().numpy()

    from cc3d import connected_components

    boxlist = np.zeros([N, 9], dtype=np.float32)
    scorelist = np.zeros([N], dtype=np.float32)
    connlist = np.zeros([N, Z, Y, X], dtype=np.float32)
    boxcount = 0

    zg, yg, xg = utils.basic.meshgrid3d_py(Z, Y, X, stack=False, norm=False)
    box3d_list = []

    labels = connected_components(mask)
    segids = [ x for x in np.unique(labels) if x != 0 ]
    for si, segid in enumerate(segids):
        extracted_vox = (labels == segid)

        z = zg[extracted_vox==1]
        y = yg[extracted_vox==1]
        x = xg[extracted_vox==1]

        zmin = np.min(z)
        zmax = np.max(z)
        ymin = np.min(y)
        ymax = np.max(y)
        xmin = np.min(x)
        xmax = np.max(x)

        if (zmax-zmin > min_side and
            ymax-ymin > min_side and
            xmax-xmin > min_side and
            np.sum(extracted_vox*count_mask) > min_voxels):

            # find the oriented box in birdview
            im = np.sum(extracted_vox, axis=1) # reduce on the Y dim
            im = im.astype(np.uint8)

            # somehow the versions change 
            # _, contours, hier = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours, hier = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
            if contours:
                cnt = contours[0]
                rect = cv2.minAreaRect(cnt)

                ymin = np.min(y)
                ymax = np.max(y)

                hei = ymax-ymin
                yc = (ymax+ymin)/2.0

                (xc,zc),(wid,dep),theta = rect
                theta = -theta

                box = cv2.boxPoints(rect)
                if dep < wid:
                    # dep goes along the long side of an oriented car
                    theta += 90.0
                    wid, dep = dep, wid
                theta = utils.geom.deg2rad(theta)
                
                if boxcount < N:#  and (yc > ymin_) and (yc < ymax_):
                    # bx, by = np.split(box, axis=1)
                    # boxpoints[boxcount,:] = box

                    box3d = [xc, yc, zc, wid, hei, dep, 0, theta, 0]
                    box3d = np.array(box3d).astype(np.float32)

                    already_have = False
                    for box3d_ in box3d_list:
                        if np.all(box3d_==box3d):
                            already_have = True

                    if not already_have:
                        box = np.int0(box)

                        boxlist[boxcount,:] = box3d
                        scorelist[boxcount] = 1.0

                        conn_ = np.zeros([Z, Y, X], np.float32)
                        conn_[extracted_vox] = 1.0
                        connlist[boxcount] = conn_

                        boxcount += 1
                        box3d_list.append(box3d)
                    else:
                        # print('skipping a box that already exists')
                        pass
                # endif boxcount
            # endif contours
        # endif sides
    # endloop over segments
        
    boxlist = torch.from_numpy(boxlist).float().to('cuda').unsqueeze(0)
    scorelist = torch.from_numpy(scorelist).float().to('cuda').unsqueeze(0)
    connlist = torch.from_numpy(connlist).float().to('cuda').unsqueeze(0)
    tidlist = torch.linspace(1.0, N, N).long().to('cuda')
    tidlist = tidlist.unsqueeze(0)
    return boxlist, scorelist, tidlist, connlist


def get_any_segments_from_binary(binary, N, min_voxels=3, min_side=1):
    B, Z, Y, X = list(binary.shape)
        
    # turn the binary map into labels with connected_components

    assert(B==1) # later i will extend
    binary = binary[0]
    # binary is Z x Y x X

    mask = binary.detach().cpu().numpy().astype(np.int32)

    from cc3d import connected_components

    connlist = np.zeros([N, Z, Y, X], dtype=np.float32)
    boxcount = 0

    zg, yg, xg = utils.basic.meshgrid3d_py(Z, Y, X, stack=False, norm=False)
    box3d_list = []

    labels = connected_components(mask)
    segids = [ x for x in np.unique(labels) if x != 0 ]
    for si, segid in enumerate(segids):
        extracted_vox = (labels == segid)

        z = zg[extracted_vox==1]
        y = yg[extracted_vox==1]
        x = xg[extracted_vox==1]

        zmin = np.min(z)
        zmax = np.max(z)
        ymin = np.min(y)
        ymax = np.max(y)
        xmin = np.min(x)
        xmax = np.max(x)

        if (zmax-zmin > min_side and
            ymax-ymin > min_side and
            xmax-xmin > min_side):

            if boxcount < N:
                conn_ = np.zeros([Z, Y, X], np.float32)
                conn_[extracted_vox] = 1.0
                connlist[boxcount] = conn_
            # endif boxcount
        # endif sides
    # endloop over segments
        
    connlist = torch.from_numpy(connlist).float().to('cuda').unsqueeze(0)
    return connlist

def get_boxes_from_masklist(masklist):
    assert(False) # something is wrong in this func
    B, N, Z, Y, X = list(masklist.shape)

    assert(B==1)
    masklist = masklist[0]
    masklist = (masklist > 0).float()
    # masklist is N x Z x Y x X
    masklist = masklist.detach().cpu().numpy().astype(np.int32)

    from cc3d import connected_components

    boxlist = np.zeros([N, 9], dtype=np.float32)
    scorelist = np.zeros([N], dtype=np.float32)
    connlist = np.zeros([N, Z, Y, X], dtype=np.float32)
    boxcount = 0

    zg, yg, xg = utils.basic.meshgrid3d_py(Z, Y, X, stack=False, norm=False)
    box3d_list = []

    for n in range(N):
        extracted_vox = masklist[n]
        print('segid = %d; size %d' % (n, np.sum(extracted_vox)))
        print('extracted_vox', extracted_vox.shape)

        z = zg[extracted_vox==1]
        y = yg[extracted_vox==1]
        x = xg[extracted_vox==1]

        # find the oriented box in birdview
        im = np.sum(extracted_vox, axis=1) # reduce on the Y dim
        im = im.astype(np.uint8)

        # somehow the versions change 
        # _, contours, hier = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, hier = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
        if contours:
            print('got contours; finding box')
            
            cnt = contours[0]
            rect = cv2.minAreaRect(cnt)

            ymin = np.min(y)
            ymax = np.max(y)

            hei = ymax-ymin
            yc = (ymax+ymin)/2.0

            (xc,zc),(wid,dep),theta = rect
            theta = -theta

            box = cv2.boxPoints(rect)
            if dep < wid:
                # dep goes along the long side of an oriented car
                theta += 90.0
                wid, dep = dep, wid
            theta = utils.geom.deg2rad(theta)

            box3d = [xc, yc, zc, wid, hei, dep, 0, theta, 0]
            box3d = np.array(box3d).astype(np.float32)

            already_have = False
            for box3d_ in box3d_list:
                if np.all(box3d_==box3d):
                    already_have = True

                if not already_have:
                    box = np.int0(box)

                    print('ok, setting box')
                    boxlist[boxcount,:] = box3d
                    scorelist[boxcount] = 1.0

                    conn_ = np.zeros([Z, Y, X], np.float32)
                    conn_[extracted_vox] = 1.0
                    connlist[boxcount] = conn_
                    boxcount += 1
            else:
                print('skipping a box that already exists')
                pass
        else:
            print('contouring failed!')
            input()
            
    boxlist = torch.from_numpy(boxlist).float().to('cuda').unsqueeze(0)
    scorelist = torch.from_numpy(scorelist).float().to('cuda').unsqueeze(0)
    connlist = torch.from_numpy(connlist).float().to('cuda').unsqueeze(0)
    tidlist = torch.linspace(1.0, N, N).long().to('cuda')
    tidlist = tidlist.unsqueeze(0)
    return boxlist, scorelist, tidlist, connlist

    

def find_detections_corresponding_to_traj(obj_clist, obj_vlist, xyzlists, vislists):
    # obj_clist is B x S x 3
    # obj_vlist is B x S
    # xyzlists is B x S x N x 3
    # vislists is B x S x N x 3
    
    B, S, N, D = list(xyzlists.shape)
    assert(D==3) # this should be 3 values, for xyz

    # print('obj_clist', obj_clist.shape)
    # print('obj_vlist', obj_vlist.shape)
    # print('xyzlists', xyzlists.shape)
    # print('vislists', vislists.shape)

    # make life easier
    obj_foundlist = torch.zeros_like(obj_vlist)
    for b in range(B):
        obj_foundlist[b] = find_detections_corresponding_to_traj_single(
            obj_clist[b], obj_vlist[b], xyzlists[b], vislists[b])
    return obj_foundlist

def find_detections_corresponding_to_traj_single(obj_clist, obj_vlist, xyzlists, vislists):
    # obj_clist is S x 3
    # obj_vlist is S
    # xyzlists is S x N x 3
    # vislists is S x N x 3
    S, N, D = list(xyzlists.shape)
    assert(D==3) # this should be 3 values, for xyz
    
    obj_foundlist = torch.zeros_like(obj_vlist)

    # step through the trajectory, and
    # see if each location has a detection nearby
    for s in list(range(S)):
        obj_c = obj_clist[s]
        # this is 3

        # print('obj_c:', obj_c.detach().cpu().numpy())

        # look at the list of detections;
        # if there is one within some threshold of dist with this, then ok.

        xyzlist = xyzlists[s]
        # this is N x 3
        vislist = vislists[s]
        # this is N

        # print('xyzlist:', xyzlist.detach().cpu().numpy())
        

        distlist = torch.norm(obj_c.unsqueeze(0)-xyzlist, dim=1)
        # this is N

        # print('distlist:', distlist.detach().cpu().numpy())
        
        dist_thresh = 3.0 
        did_detect_something = 0.0
        for n in list(range(N)):
            if (vislist[n] > 0.5) and (distlist[n] < dist_thresh):
                did_detect_something = 1.0
        obj_foundlist[s] = did_detect_something
        
        # print('got it?', did_detect_something)
        # input()
    return obj_foundlist


def get_traj_loglike(pts, energy_map):
    # energy_map is B x 1 x Z x X; it is not normalized in any way
    # pts is B x T x (2 or 3); it specifies vox coordinates of the traj 

    B, T, D = list(pts.shape)
    _, _, Z, X = list(energy_map.shape)

    if D==3:
        # ignore the Y dim
        x, _, z = torch.unbind(pts, dim=2)
    elif D==2:
        x, z = torch.unbind(pts, dim=2)
    else:
        assert(False) # pts dim should be 2 (xz) or 3 (xyz)

    energy_per_timestep = utils.samp.bilinear_sample2d(energy_map, x, z)
    energy_per_timestep = energy_per_timestep.reshape(B, T) # get rid of the trailing channel dim
    # this is B x T

    # to construct the probability-based loss, i need the log-sum-exp over the spatial dims
    energy_vec = energy_map.reshape(B, Z*X)
    logpartition_function = torch.logsumexp(energy_vec, 1, keepdim=True)
    # this is B x 1

    loglike_per_timestep = energy_per_timestep - logpartition_function
    # this is B x T

    loglike_per_traj = torch.sum(loglike_per_timestep, dim=1)
    # this is B

    return loglike_per_traj

class SimplePool():
    def __init__(self, pool_size, version='pt'):
        self.pool_size = pool_size
        self.version = version
        # random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.items = []
        if not (version=='pt' or version=='np'):
            print('version = %s; please choose pt or np')
            assert(False) # please choose pt or np
            
    def __len__(self):
        return len(self.items)
    
    def mean(self, min_size='none'):
        if min_size=='half':
            pool_size_thresh = self.pool_size/2
        else:
            pool_size_thresh = 1
            
        if self.version=='np':
            if len(self.items) >= pool_size_thresh:
                return np.sum(self.items)/float(len(self.items))
            else:
                return np.nan
        if self.version=='pt':
            if len(self.items) >= pool_size_thresh:
                return torch.sum(self.items)/float(len(self.items))
            else:
                return torch.from_numpy(np.nan)
    
    def sample(self):
        idx = np.random.randint(len(self.items))
        return self.items[idx]
    
    def fetch(self, num=None):
        if self.version=='pt':
            item_array = torch.stack(self.items)
        elif self.version=='np':
            item_array = np.stack(self.items)
        if num is not None:
            # there better be some items
            assert(len(self.items) >= num)
                
            # if there are not that many elements just return however many there are
            if len(self.items) < num:
                return item_array
            else:
                idxs = np.random.randint(len(self.items), size=num)
                return item_array[idxs]
        else:
            return item_array
            
    def is_full(self):
        full = self.num==self.pool_size
        # print 'num = %d; full = %s' % (self.num, full)
        return full
    
    def empty(self):
        self.items = []
        self.num = 0
            
    def update(self, items):
        for item in items:
            if self.num < self.pool_size:
                # the pool is not full, so let's add this in
                self.num = self.num + 1
            else:
                # the pool is full
                # pop from the front
                self.items.pop(0)
            # add to the back
            self.items.append(item)
        return self.items
    
def sample_eight_points(template_mask, max_tries=1000, random_center=True):
    # let's sample corners in python
    B, _, ZZ, ZY, ZX = list(template_mask.shape)
    template_mask_py = template_mask.cpu().detach().numpy()

    failed = False

    sampled_corners = np.zeros([B, 8, 3], np.float32)
    sampled_centers = np.zeros([B, 1, 3], np.float32)
    for b in list(range(B)):

        retry = True
        num_tries = 0

        while retry:
            num_tries += 1
            # make the lengths multiples of two
            lx = np.random.randint(1,ZX/2)*2.0
            ly = np.random.randint(1,ZY/2)*2.0
            lz = np.random.randint(1,ZZ/2)*2.0
            # print('lx, ly, lz', lx, ly, lz)
            xs = np.array([lx/2., lx/2., -lx/2., -lx/2., lx/2., lx/2., -lx/2., -lx/2.])
            ys = np.array([ly/2., ly/2., ly/2., ly/2., -ly/2., -ly/2., -ly/2., -ly/2.])
            zs = np.array([lz/2., -lz/2., -lz/2., lz/2., lz/2., -lz/2., -lz/2., lz/2.])
            corners = np.stack([xs, ys, zs], axis=1)
            # this is 8 x 3

            if random_center:
                # put the centroid within the inner half of the template
                cx = np.random.randint(ZX/4-1,ZX-ZX/4)
                cy = np.random.randint(ZY/4-1,ZY-ZY/4)
                cz = np.random.randint(ZZ/4-1,ZZ-ZZ/4)
                center = np.reshape(np.array([cx, cy, cz]), [1, 3])
            else:
                # center = np.reshape(np.array([ZX/2-1, ZY/2-1, ZZ/2-1]), [1, 3])
                center = np.reshape(np.array([ZX/2, ZY/2, ZZ/2]), [1, 3])

            corners = corners + center
            # now i want to see if those locations are all valid

            # let's start with inbounds
            inb = utils.py.get_inbounds(corners, ZZ, ZY, ZX, already_mem=True)
            if np.sum(inb) == 8:
                # now let's also ensure all valid
                retry = False
                for corner in corners:
                    # print(corner)
                    cin = template_mask_py[b, 0, int(corner[2]), int(corner[1]), int(corner[0])]
                    if cin == 0:
                        retry = True
                        
            # if np.mod(num_tries, 1000)==0:
            if num_tries == int(max_tries/2):
                # print('up to %d tries' % num_tries)
                # # let's dilate the mask by one, since it seems we are stuck
                # print('before, sum was', np.sum(template_mask_py))
                weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                template_mask_ = F.conv3d(template_mask, weights, padding=1)
                template_mask_ = torch.clamp(template_mask_, 0, 1)
                template_mask_py = template_mask_.cpu().detach().numpy()
                # print('now, sum is', np.sum(template_mask_py))
            # if num_tries == 10000:
            #     # give up
            #     retry = False

            if num_tries == max_tries:
                # give up
                retry = False
                failed = True

        # print('that took %d tries' % num_tries)
        sampled_corners[b] = corners
        sampled_centers[b] = center
    sampled_corners = torch.from_numpy(sampled_corners).float().cuda()
    sampled_centers = torch.from_numpy(sampled_centers).float().cuda()
    return sampled_corners, sampled_centers, failed

def parse_boxes(box_camRs, origin_T_camRs):
    B, S, D = box_camRs.shape
    assert(D==9)
    # box_camRs is B x S x 9
    # origin_T_camRs is B x S x 4 x 4
    # in this data, the last three elements are rotation angles, 
    # and these angles are wrt the world origin

    # fix the bikes; their boxes only cover the bottom half
    for b in list(range(B)):
        for s in list(range(S)):
            box = box_camRs[b,s]
            x, y, z, lx, ly, lz, rx, ry, rz = torch.unbind(box, axis=0)
            if lx < 1.0:
                y = y - ly/2.0
                ly = ly * 2.0
                lz = lz * 1.5
            box = torch.stack([x, y, z, lx, ly, lz, rx, ry, rz], dim=0)
            box_camRs[b,s] = box

    obj_lens = box_camRs[:,:,3:6]
    
    rots = utils.geom.deg2rad(box_camRs[:,:,6:])
    roll = rots[:,:,0] 
    pitch = rots[:,:,1] 
    yaw = rots[:,:,2]
    pitch_ = pitch.reshape(-1)
    yaw_ = yaw.reshape(-1)
    roll_ = roll.reshape(-1)
    rots_ = utils.geom.eul2rotm(-pitch_ - np.pi/2.0, -roll_, yaw_ - np.pi/2.0)
    ts_ = torch.zeros_like(rots_[:,0])
    rts_ = utils.geom.merge_rt(rots_, ts_)
    # this B*S x 4 x 4

    origin_T_camRs_ = origin_T_camRs.reshape(B*S, 4, 4)
    camRs_T_origin_ = utils.geom.safe_inverse(origin_T_camRs_)
    rts_ = utils.basic.matmul2(camRs_T_origin_, rts_)

    lrt_camRs = utils.geom.convert_boxlist_to_lrtlist(box_camRs)
    lenlist, rtlist = utils.geom.split_lrtlist(lrt_camRs)
    _, tlist_ = utils.geom.split_rt(rtlist.reshape(-1, 4, 4))
    rlist_, _ = utils.geom.split_rt(rts_)
    rtlist = utils.geom.merge_rt(rlist_, tlist_).reshape(B, S, 4, 4)
    # this is B x S x 4 x 4
    lrt_camRs = utils.geom.merge_lrtlist(lenlist, rtlist)
    return lrt_camRs

def parse_seg_into_mem(seg_camXs, num_seg_labels, occ_memX0s, pix_T_cams, camX0s_T_camXs, vox_util):
    B, S, H, W = list(seg_camXs.shape)
    _, _, _, Z, Y, X = list(occ_memX0s.shape)
    
    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    seg_onehots = torch.zeros(B, S, num_seg_labels, H, W).float().cuda()
    for l in list(range(num_seg_labels)):
        seg_onehots[:,:,l] = (seg_camXs==l).float()

    # now let's unproject each one
    seg_memXs = __u(vox_util.unproject_rgb_to_mem(
        __p(seg_onehots), Z, Y, X, __p(pix_T_cams)))
    # seg_memX0s = vox_util.apply_4x4s_to_voxs(camX0s_T_camXs, seg_memXs).round()
    seg_memX0s = vox_util.apply_4x4s_to_voxs(camX0s_T_camXs, seg_memXs)
    # this is B x S x num_seg_labels x Z x Y x X

    seg_memX0 = utils.basic.reduce_masked_mean(
        seg_memX0s,
        occ_memX0s.repeat(1, 1, num_seg_labels, 1, 1, 1),
        dim=1)
    seg_memX0[seg_memX0 < 0.8] = 0.0
    seg_memX0 = torch.max(seg_memX0, dim=1)[1]
    # this is B x Z x Y x X
    return seg_memX0


def assemble_hypothesis(memY0_T_memX0, lrt_camY0, bkg_memX0, obj_memX0, obj_mask_memX0, vox_util, crop_zyx, norm=True):
    
    def crop_feat(feat_pad):
        Z_pad, Y_pad, X_pad = crop_zyx
        feat = feat_pad[:,:,
                        Z_pad:-Z_pad,
                        Y_pad:-Y_pad,
                        X_pad:-X_pad].clone()
        return feat
    def pad_feat(feat):
        Z_pad, Y_pad, X_pad = crop_zyx
        feat_pad = F.pad(feat, (Z_pad, Z_pad, Y_pad, Y_pad, X_pad, X_pad), 'constant', 0)
        return feat_pad

    if False:
        valid_mask_memX0 = torch.ones_like(obj_mask_memX0)
        valid_mask_memX0_pad = pad_feat(valid_mask_memX0)
        obj_mask_memX0_pad = pad_feat(obj_mask_memX0)
        obj_memX0_pad = pad_feat(obj_memX0)

        valid_mask_memY0_pad = vox_util.apply_4x4_to_vox(memY0_T_memX0, valid_mask_memX0_pad, already_mem=True)
        obj_mask_memY0_pad = vox_util.apply_4x4_to_vox(memY0_T_memX0, obj_mask_memX0_pad, already_mem=True)
        obj_memY0_pad = vox_util.apply_4x4_to_vox(memY0_T_memX0, obj_memX0_pad, already_mem=True)
        obj_memY0 = crop_feat(obj_memY0_pad)
        obj_mask_memY0 = crop_feat(obj_mask_memY0_pad)
        valid_mask_memY0 = crop_feat(valid_mask_memY0_pad)
        obj_mask_memY0 = obj_mask_memY0 * valid_mask_memY0

        scene_memY0 = bkg_memX0 * (1.0 - obj_mask_memY0) + obj_mask_memY0 * obj_memY0

    else:
        bkg_memX0_pad = pad_feat(bkg_memX0)
        B, C, Z, Y, X = list(bkg_memX0_pad.shape)
        obj_mask_memY0 = vox_util.assemble_padded_obj_masklist(
            lrt_camY0.unsqueeze(1), torch.ones_like(lrt_camY0[:,0]).unsqueeze(1), Z, Y, X,
        ).squeeze(1)

        wide_obj_mask_memY0 = vox_util.assemble_padded_obj_masklist(
            lrt_camY0.unsqueeze(1), torch.ones_like(lrt_camY0[:,0]).unsqueeze(1), Z, Y, X,
            coeff=1.1, additive_coeff=2.0).squeeze(1)
        
        obj_mask_memY0 = crop_feat(obj_mask_memY0)
        wide_obj_mask_memY0 = crop_feat(wide_obj_mask_memY0)
        scene_memY0 = bkg_memX0 * (1.0 - obj_mask_memY0) + obj_mask_memY0 * obj_memX0
    if norm:
        scene_memY0 = utils.basic.l2_normalize(scene_memY0, dim=1)
        
    return scene_memY0, wide_obj_mask_memY0

def propose_boxes_by_differencing(
        K, S,
        occ_memXAI_all,
        diff_memXAI_all,
        crop_zyx,
        set_data_name=None,
        data_ind=None,
        super_iter=None,
        use_box_cache=False,
        summ_writer=None):

    have_boxes = False
    if hyp.do_use_cache and use_box_cache:
        box_cache_fn = 'cache/%s_%06d_s%d_box_%d.npz' % (set_data_name, data_ind, S, super_iter)
        # check if the thing exists
        if os.path.isfile(box_cache_fn):
            print('found box cache at %s; we will use this' % box_cache_fn)
            cache = np.load(box_cache_fn, allow_pickle=True)['save_dict'].item()
            # cache = cache['save_dict']

            have_boxes = True
            lrtlist_memXAI_all = torch.from_numpy(cache['lrtlist_memXAI_all']).cuda().unbind(1)
            connlist_memXAI_all = torch.from_numpy(cache['connlist_memXAI_all']).cuda().unbind(1)
            scorelist_all = [s for s in torch.from_numpy(cache['scorelist_all']).cuda().unbind(1)]
            blue_vis = [s for s in torch.from_numpy(cache['blue_vis']).cuda().unbind(1)]
        else:
            print('could not find box cache at %s; we will write this' % box_cache_fn)

    if not have_boxes:
        blue_vis = []
        # conn_vis = []
        # lrtlist_camXIs = []
        lrtlist_memXAI_all = []
        connlist_memXAI_all = []
        scorelist_all = []

        for I in list(range(S)):

            diff_memXAI = diff_memXAI_all[I]
            # border = 1
            # diff_memXAI[:,:,0:border] = 0
            # diff_memXAI[:,:,:,0:border] = 0
            # diff_memXAI[:,:,:,:,0:border] = 0
            # diff_memXAI[:,:,-border:] = 0
            # diff_memXAI[:,:,:,-border:] = 0
            # diff_memXAI[:,:,:,:,-border:] = 0
            if summ_writer is not None:
                summ_writer.summ_oned('proposals/diff_iter_%d' % I, diff_memXAI, bev=True)

            boxes_image, boxlist_memXAI, scorelist_e, tidlist, connlist = get_boxes_from_flow_mag(
                diff_memXAI.squeeze(1), K)
            x, y, z, lx, ly, lz, rx, ry, rz = boxlist_memXAI.unbind(2)
            ly = ly + 1.0
            z = z + crop_zyx[0]
            y = y + crop_zyx[1]
            x = x + crop_zyx[2]
            boxlist_memXAI = torch.stack([x, y, z, lx, ly, lz, rx, ry, rz], dim=2)

            lrtlist_memXAI = utils.geom.convert_boxlist_to_lrtlist(boxlist_memXAI)
            lrtlist_memXAI_all.append(lrtlist_memXAI)
            connlist_memXAI_all.append(connlist)

            scorelist_e[scorelist_e > 0.0] = 1.0
            occ_memXAI = occ_memXAI_all[I]
            diff_memXAI = diff_memXAI_all[I]
            for n in list(range(K)):
                mask_1 = connlist[:,n:n+1]
                weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                mask_3 = (F.conv3d(mask_1, weights, padding=1)).clamp(0, 1)

                center_mask = mask_1.clone()
                surround_mask = (mask_3-mask_1).clamp(0,1)

                center_ = utils.basic.reduce_masked_mean(occ_memXAI, center_mask, dim=[2,3,4])
                surround_ = utils.basic.reduce_masked_mean(occ_memXAI, surround_mask, dim=[2,3,4])
                score_ = center_ - surround_
                score_ = torch.clamp(torch.sigmoid(score_), min=1e-4)
                score_[score_ < 0.55] = 0.0
                scorelist_e[:,n] = score_
            scorelist_all.append(scorelist_e)

            # self.summ_writer.summ_rgb('proposals/anchor_frame', diff_memXAI_vis[self.anchor])
            # self.summ_writer.summ_rgb('proposals/get_boxes', boxes_image)
            blue_vis.append(boxes_image)

            # conn_vis.append(self.summ_writer.summ_occ('', torch.sum(connlist, dim=1, keepdims=True).clamp(0, 1), only_return=True))

        if hyp.do_use_cache and use_box_cache:
            # save this, so that we have it all next time
            save_dict = {}

            save_dict['lrtlist_memXAI_all'] = torch.stack(lrtlist_memXAI_all, dim=1).detach().cpu().numpy()
            save_dict['connlist_memXAI_all'] = torch.stack(connlist_memXAI_all, dim=1).detach().cpu().numpy()
            save_dict['scorelist_all'] = torch.stack(scorelist_all, dim=1).detach().cpu().numpy()
            save_dict['blue_vis'] = torch.stack(blue_vis, dim=1).detach().cpu().numpy()
            np.savez(box_cache_fn, save_dict=save_dict)
            print('saved boxes to %s cache, for next time' % box_cache_fn)

    return lrtlist_memXAI_all, connlist_memXAI_all, scorelist_all, blue_vis

def get_random_valid_lrt(lrtlist, scorelist):
    # lrtlist is B x N x 19
    # scorelist is B x N
    # we want to return lrt, shaped B x 19

    B, N, D = list(lrtlist.shape)
    assert(D==19)

    lrt = []
    score = []
    ind = []
    for b in list(range(B)):
        lrtlist_b = lrtlist[b]
        scorelist_b = scorelist[b]
        if torch.sum(scorelist_b > 0) == 0:
            # nothing valid here
            lrt_ = lrtlist_b[0]
            score_ = scorelist_b[0]
            ind_ = (scorelist_b[0]*0).long()
        else:
            # shuffle
            inds = np.random.permutation(N).astype(np.int32)
            lrtlist_b = lrtlist_b[inds]
            scorelist_b = scorelist_b[inds]
            # of the valids...
            lrtlist_ = lrtlist_b[scorelist_b > 0]
            scorelist_ = scorelist_b[scorelist_b > 0]
            inds_ = torch.from_numpy(np.array(inds)).cuda().long()
            inds_ = inds_[scorelist_b > 0]
            # ...select one
            lrt_ = lrtlist_[0]
            score_ = scorelist_[0]
            ind_ = inds_[0]
        lrt.append(lrt_)
        score.append(score_)
        ind.append(ind_)
    lrt = torch.stack(lrt, dim=0)
    score = torch.stack(score, dim=0)
    ind = torch.stack(ind, dim=0)
    return lrt, score, ind

def shuffle_lrtlist(lrtlist, scorelist):
    B, N, D = list(lrtlist.shape)
    assert(D==19)

    # lrtlist_new = lrtlist.clone()
    # scorelist_new = scorelist.clone()
    
    # for b in list(range(B)):
    #     lrtlist_b = lrtlist[b]
    #     scorelist_b = scorelist[b]
    #     inds = np.random.permutation(N).astype(np.int32)
    #     lrtlist_b = lrtlist_b[inds]
    #     scorelist_b = scorelist_b[inds]
    #     lrtlist_new[b] = lrtlist_b
    #     scorelist_new[b] = scorelist_b
    # return lrtlist_new, scorelist_new

    inds = np.random.permutation(N).astype(np.int32)
    lrtlist = lrtlist[:,inds]
    scorelist = scorelist[:,inds]
    return lrtlist, scorelist

def shuffle_lrtlists(lrtlist_s, scorelist_s, *args):
    B, S, N, D = list(lrtlist_s.shape)
    assert(D==19)

    # lrtlist_s_new = lrtlist_s.clone()
    # scorelist_s_new = scorelist_s.clone()
    
    inds = np.random.permutation(N).astype(np.int32)
    lrtlist_s = lrtlist_s[:,:,inds]
    scorelist_s = scorelist_s[:,:,inds]

    returning_list = [lrtlist_s, scorelist_s]

    for var in args:
        returning_list.append(var[:,:,inds])

    return returning_list

def select_valid_moving_lrtlist(lrtlist_s, scorelist_s, req_early_motion=False, starting_thresh=2.0, min_dist=0.0):
    B, S, N, D = list(lrtlist_s.shape)
    assert(D==19)
    assert(S > 1)
    assert(starting_thresh > min_dist) # otw the loop will never go

    # init with the zeroth guy, so that we are not totally stuck
    lrtlist_new = lrtlist_s[:,:,0].clone()
    scorelist_new = scorelist_s[:,:,0].clone()

    # lrtlist, scorelist = lrtlist_s[:,:,n], scorelist_s[:,:,n]

    # first let's just take a valid one
    for b in list(range(B)):
        for n in list(range(N)):
            lrtlist, scorelist = lrtlist_s[b:b+1,:,n], scorelist_s[b:b+1,:,n]
            # these are 1 x S x 19 and 1 x S
            # make sure it is valid all the way through
            if torch.sum(scorelist) == S:
                lrtlist_new[b] = lrtlist[0]
                scorelist_new[b] = scorelist[0]

    have_valid = torch.sum(scorelist_new)==(B*S)
    # print('have_valid', have_valid)

    if have_valid:
        # now let's try to find a moving one, to potentially replace that valid selection
        for b in list(range(B)):
            have_moving = False
            dist_thresh_ = starting_thresh
            while (not have_moving) and (dist_thresh_ > min_dist):
                for n in list(range(N)):
                    lrtlist, scorelist = lrtlist_s[b:b+1,:,n], scorelist_s[b:b+1,:,n]
                    # these are 1 x S x 19 and 1 x S

                    # make sure it is valid all the way through
                    if torch.sum(scorelist) == S:
                        # make sure it is moving
                        clist = utils.geom.get_clist_from_lrtlist(lrtlist)
                        # this is 1 x S x 3

                        if req_early_motion:
                            dist = torch.norm(clist[:,0] - clist[:,1], dim=1)
                        else:
                            dist = torch.norm(clist[:,0] - clist[:,-1], dim=1)
                        # this is 1
                        # print('dist', dist)
                        if dist.squeeze() > dist_thresh_:
                            # print('found object %d with dist' % n, dist.detach().cpu().numpy())
                            lrtlist_new[b] = lrtlist[0]
                            scorelist_new[b] = scorelist[0]
                            have_moving = True
                            break
                dist_thresh_ -= 0.05
    else:
        have_moving = False
    return lrtlist_new, scorelist_new, have_valid, have_moving

def get_closest_moving_object(lrtlist_object, lrtlist_s, scorelist_s):
    B, S, N, D = list(lrtlist_s.shape)
    clist_object = utils.geom.get_clist_from_lrtlist(lrtlist_object)
    lrt_closest_moving = torch.zeros(B,S,D).cuda()

    for b in range(B):
        min_dist = float('inf')
        for n in range(N):
            lrtlist = lrtlist_s[b:b+1,:,n]
            scorelist = scorelist_s[b:b+1,:,n]
            if torch.sum(scorelist)==S:
                clist = utils.geom.get_clist_from_lrtlist(lrtlist)

def select_valid_lrtlist(lrtlist_s, scorelist_s):
    B, S, N, D = list(lrtlist_s.shape)
    assert(D==19)

    # if sink_carry:
    #     assert end_frame is not None

    # init with the zeroth guy, so that we are not totally stuck
    lrtlist_new = lrtlist_s[:,:,0].clone()
    scorelist_new = scorelist_s[:,:,0].clone()

    all_ok = True

    for b in list(range(B)):
        ok = False
        for n in list(range(N)):
            lrtlist, scorelist = lrtlist_s[b:b+1,:,n], scorelist_s[b:b+1,:,n]

            if torch.sum(scorelist) == S: 
                lrtlist_new[b] = lrtlist[0]
                scorelist_new[b] = scorelist[0]
                ok = True
                break

        all_ok = ok

    return all_ok, lrtlist_new, scorelist_new

def select_valid_lrtlist_with_action(lrtlist_s, scorelist_s, actionlist_s, end_frame=None, sink_carry=False):
    B, S, N, D = list(lrtlist_s.shape)
    assert(D==19)
    assert(S > 1)

    # if sink_carry:
    #     assert end_frame is not None

    # init with the zeroth guy, so that we are not totally stuck
    lrtlist_new = lrtlist_s[:,:,0].clone()
    scorelist_new = scorelist_s[:,:,0].clone()
    actionlist_new = actionlist_s[:,:,0].clone()

    for b in list(range(B)):
        for n in list(range(N)):
            lrtlist, scorelist, actionlist = lrtlist_s[b:b+1,:,n], scorelist_s[b:b+1,:,n], actionlist_s[b:b+1,:,n]
            # these are 1 x S x 19 and 1 x S
            # make sure it is valid all the way through
            if end_frame is None:
                is_carry = torch.any(actionlist_s[b, :, n] > 3)
            else:
                is_carry = actionlist_s[b, end_frame-1, n] > 3

            if torch.sum(scorelist) == S and (not sink_carry or not is_carry): # carrying has label 4 and 5
                lrtlist_new[b] = lrtlist[0]
                scorelist_new[b] = scorelist[0]
                actionlist_new[b] = actionlist[0]

    return lrtlist_new, scorelist_new, actionlist_new

def get_given_frames_from_clist(clist, frame0, frame1, window_size=1):
    B, S, D = list(clist.shape)
    given_s = torch.zeros_like(clist[:,:,0])
    clist_given = torch.ones_like(clist)*100.0 # go away
    for frame in [frame0, frame1]:
        for i in range(-window_size, window_size+1):
            if frame+i >=0 and frame+i<S:
                given_s[:,frame+i] = 1.0
                clist_given[:,frame+i] = clist[:,frame+i]
    return given_s, clist_given

def sample_startframe_endframe_with_mindist(clist, min_dist=1.0, min_accel=0.0, min_framedist=8):
    B, S, D = list(clist.shape)
    assert(S > min_framedist)
    frame0 = S-1
    frame1 = 0
    xyz0 = clist[:,frame0]
    xyz1 = clist[:,frame1]
    delta = xyz1-xyz0
    dist = torch.norm(delta, dim=1)
    accel = 0.0
    tries = 0

    while (frame0 > frame1) or (dist < min_dist) or (accel < min_accel):
        # frame0 = torch.randint(low=0, high=S-1-min_framedist, size=[])
        # frame1 = torch.randint(low=frame0+min_framedist, high=S-1, size=[])
        # print('framedist:', frame1-frame0)

        frame0 = torch.randint(low=0, high=S-min_framedist, size=[])
        frame1 = torch.randint(low=frame0+min_framedist, high=S, size=[])
        
        xyzs = clist[:,frame0:frame1+2]
        # print('xyzs', xyzs.shape)
        vels = xyzs[:,1:] - xyzs[:,:-1]
        # print('vels', vels.shape)

        if min_accel > 0:
            print('this is probably going to fail; when you want accel you need a lower high in randint')
            accels = vels[:,1:] - vels[:,:-1]
            # print('accels', accels.shape)
            accel = torch.max(torch.norm(accels, dim=2), dim=1)[0]
            # print('accel', accel)

        delta = xyz1-xyz0
        dist = torch.norm(delta, dim=1)

        tries += 1
        if tries > 100:
            return None, None, None, None, None, False
    
    return delta, frame0, frame1, xyz0, xyz1, True

def select_valid_rotating_lrtlist(lrtlist_s, scorelist_s):
    B, S, N, D = list(lrtlist_s.shape)
    assert(D==19)
    assert(S > 1)

    # init with the zeroth guy, so that we are not totally stuck
    lrtlist_new = lrtlist_s[:,:,0].clone()
    scorelist_new = scorelist_s[:,:,0].clone()
    select = 0

    eps = 1e-5

    # lrtlist, scorelist = lrtlist_s[:,:,n], scorelist_s[:,:,n]

    # first let's just take a valid one
    for b in list(range(B)):
        for n in list(range(N)):
            lrtlist, scorelist = lrtlist_s[b:b+1,:,n], scorelist_s[b:b+1,:,n]
            # these are 1 x S x 19 and 1 x S
            # make sure it is valid all the way through
            if torch.sum(scorelist) == S:
                lrtlist_new[b] = lrtlist[0]
                scorelist_new[b] = scorelist[0]

    have_valid = torch.sum(scorelist_new)==(B*S)

    def batch_trace(tensor):
        return tensor.diagonal(dim1=-2, dim2=-1).sum(-1)

    if have_valid:
        # now let's try to find a moving one, to potentially replace that valid selection
        have_rotating = torch.zeros(B, 1).float().cuda()
        for b in list(range(B)):
            for n in list(range(N)):
                lrtlist, scorelist = lrtlist_s[b:b + 1, :, n], scorelist_s[b:b + 1, :, n]
                # these are 1 x S x 19 and 1 x S

                # make sure it is valid all the way through
                if torch.sum(scorelist) == S:
                    # make sure it is moving
                    lenlist, rtlist = utils.geom.split_lrtlist(lrtlist)
                    r0, t0 = utils.geom.split_rt(rtlist[:, 0])
                    rt1_inverse = utils.geom.safe_inverse(rtlist[:, 1])
                    r1_inverse, t1_inverse = utils.geom.split_rt(rt1_inverse)
                    obj0_T_obj1 = utils.basic.matmul2(r0, r1_inverse)
                    if torch.abs(batch_trace(obj0_T_obj1) - 3.0) > eps:
                        # print('found object %d with dist' % n, dist.detach().cpu().numpy())
                        lrtlist_new[b] = lrtlist[0]
                        scorelist_new[b] = scorelist[0]
                        have_rotating[b] = 1.
                        select = n
                        break
    return lrtlist_new, scorelist_new, have_valid, have_rotating

class Kalman3d(nn.Module):
    def __init__(self,
                 initial_xyz,
                 measurement_noise=1.0,
                 process_noise=1.0):
        super(Kalman3d, self).__init__()

        N, D = list(initial_xyz.shape)
        assert(D==3)
        self.N = N # we can't call this B because that's a matrix here

        dt = 1 # sampling rate
        self.u = 0.0 # control input (acceleration magnitude); we have no knowledge here so it's 0.0

        self.Q = torch.zeros([N, 6, 1]).float().cuda()
        self.Q[:,:3,0] = initial_xyz

        self.Q_estimate = self.Q # initial state estimate

        self.measurement_noise = measurement_noise
        self.process_noise = process_noise

        # measurement noise matrix
        self.Ez = torch.eye(3).unsqueeze(0).repeat(N, 1, 1).float().cuda()
        self.Ez = self.Ez * measurement_noise**2

        # process noise matrix
        Ex_py = np.array(
            [[dt**2/4, 0, 0, dt/2, 0, 0],
             [0, dt**2/4, 0, 0, dt/2, 0],
             [0, 0, dt**2/4, 0, 0, dt/2],
             [dt/2, 0, 0, 1, 0, 0],
             [0, dt/2, 0, 0, 1, 0],
             [0, 0, dt/2, 0, 0, 1]])*(dt**2)
        self.Ex = torch.from_numpy(Ex_py).float().cuda()
        self.Ex = self.Ex.unsqueeze(0).repeat(N, 1, 1)
        self.Ex = self.Ex * self.process_noise**2

        self.P = self.Ex # estimate of covariance

        # motion model
        A_py = np.array(
            [[1, 0, 0, dt, 0, 0],
             [0, 1, 0, 0, dt, 0],
             [0, 0, 1, 0, 0, dt],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]])
        self.A = torch.from_numpy(A_py).float().cuda()
        self.A = self.A.unsqueeze(0).repeat(N, 1, 1)
        # this is B x 6 x 6

        # second piece of motion model
        B_py = np.array(
            [[dt**2/2],
             [dt**2/2],
             [dt**2/2],
             [dt],
             [dt],
             [dt]])
        self.B = torch.from_numpy(B_py).float().cuda()
        self.B = self.B.unsqueeze(0).repeat(N, 1, 1)
        # this is B x 6 x 1

        # measurement function
        C_py = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0]]) # this is our measurement function C, that we apply to the state estimate Q to get our expect next/new measurement
        self.C = torch.from_numpy(C_py).float().cuda()
        self.C = self.C.unsqueeze(0).repeat(N, 1, 1)
        # this is B x 3 x 6

        self.eye6 = torch.eye(6).unsqueeze(0).repeat(N, 1, 1).float().cuda()

        # print_('A', self.A)
        # print_('B', self.B)
        # print_('C', self.C)

    def forward(self, new_measurement, Ez_factor=1.0, update_q=True):
        N, D = list(new_measurement.shape)
        assert(D==3)
        assert(N==self.N)

        from utils.basic import matmul2, matmul3, matmul4

        # print_('Q_estimate', self.Q_estimate)

        self.Q_estimate = matmul2(self.A, self.Q_estimate) + self.B * self.u

        # predict next covariance
        self.P = matmul3(self.A, self.P, self.A.transpose(1,2)) + self.Ex

        # compute kalman gain
        # matlab: K = P*C'*inv(C*P*C'+Ez)
        CPCt = matmul3(self.C, self.P, self.C.transpose(1,2))
        denom = torch.inverse(CPCt + self.Ez * Ez_factor)
        K = matmul3(self.P, self.C.transpose(1,2), denom)
        # print_('K', K)

        if update_q:
            # Q_estimate = Q_estimate + K * (Q_loc_meas(:,t) - C * Q_estimate);
            CQe = matmul2(self.C, self.Q_estimate)
            residual = new_measurement.reshape(N, 3, 1) - CQe
            self.Q_estimate = self.Q_estimate + matmul2(K, residual)

        # update covariance
        self.P = matmul2(self.eye6 - matmul2(K, self.C), self.P)

        return self.Q_estimate

    def peek_forward(self):
        # tell me the answer but do not update the state
        Q_estimate = utils.basic.matmul2(self.A, self.Q_estimate) + self.B * self.u
        return Q_estimate

def get_radius_in_pix_from_clist(clist, pix_T_cam):
    B,S,_ = list(clist.shape)

    lrt_traj = utils.geom.convert_clist_to_lrtlist(clist, torch.ones((B,3), dtype=torch.float32).cuda())
    corners  = utils.geom.get_xyzlist_from_lrtlist(lrt_traj, include_clist=False)                         
    corners_pix = []          

    for i in range(8):
        corners_pix.append(utils.geom.apply_pix_T_cam(pix_T_cam, corners[:,:,i]))
    
    corners_pix = torch.stack(corners_pix, axis =2)            

    sizes = torch.zeros(B,S,1)

    for b in range(B):
        for s in range(S):
            max_square_distance = 0
            for pair in combinations(corners_pix[b,s],2):                        
                p0 = pair[0]
                p1 = pair[1]
                l2 = torch.dist(p0, p1, 2)
                if l2>max_square_distance:
                    max_square_distance = l2
            sizes[b,s] = max_square_distance
    
    return sizes

def normalize_adj_matrix(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def get_lrtlist_with_initial_speed(lrtlist_s, scorelist_s, speed_thresh=0.02):
    B, S, _  = list(scorelist_s.shape)
    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    inds = scorelist_s[0,0] > 0
    lrtlist_s = lrtlist_s[:,:,inds]
    scorelist_s = scorelist_s[:,:,inds]
    N = torch.sum(inds)

    clist_s = __u(utils.geom.get_clist_from_lrtlist(__p(lrtlist_s)))
    # B x S x N x 3

    initial_speed = torch.norm(clist_s[:,1] - clist_s[:,0], dim=2)
    # B x N
    speed_ok = initial_speed[0] > speed_thresh
    lrtlist_s = lrtlist_s[:,:,speed_ok]
    scorelist_s = scorelist_s[:,:,speed_ok]
    clist_s = __u(utils.geom.get_clist_from_lrtlist(__p(lrtlist_s)))

    return clist_s, scorelist_s, lrtlist_s

def get_possible_trajs_at_endpoint(traj_e, endpoint, clist_original, deviation_thresh=0.05):
    # traj_e is H x S x 3
    # endpoint is B x 3
    # clist_original is B x S x 3

    # make it meet the endpoint
    traj_e = traj_e / (1e-4 + traj_e[:,-1].unsqueeze(1))
    traj_e = traj_e * endpoint.unsqueeze(1)

    # now discard ones that need to turn too much for this to happen
    deviation = torch.norm(traj_e[:,1] - clist_original[:,1], dim=1)

    deviation_thresh = (torch.min(deviation)+0.01).clamp(min=deviation_thresh)
    speed_ok = deviation <= deviation_thresh
    traj_e = traj_e[speed_ok]
    return traj_e

def refine_pairwise_possibilities(all_possibilities, pair_dists, lib_a, lib_b):
    B, H_total, N, S, D = list(all_possibilities.shape)
    assert(D==3)
    print('evaluating %d possibilities' % H_total)
    
    P, Q, E = list(lib_a.shape)
    P2, Q2, E2 = list(lib_b.shape)
    assert(E==3)
    assert(P==P2)
    assert(Q==Q2)

    combos = list(combinations(list(range(N)), 2))
    
    H_new = 0
    refined_possibilities = []
    for h in list(range(H_total)):
        possibility = all_possibilities[:,h]
        # B x N x S x 3
        # print('considering possibility %d' % h)

        looks_ok = True

        for combo in combos:
            n0 = combo[0]
            n1 = combo[1]
            # assert(self.B==1)
            obj0 = possibility[0,n0]
            obj1 = possibility[0,n1]
            # S x 3

            dists = torch.norm(obj0 - obj1, dim=1)
            # S

            min_dist, ind = torch.min(dists, dim=0)

            if min_dist < 3.0:
                # we need to investigate

                mid = int(Q/2)
                # print('ind', ind, 'mid', mid)
                if ind >= mid and ind < S-mid:
                    start = ind-mid
                    end = ind+mid+1
                    # slice_a = all_trajs_a[n,start:end]
                    # slice_b = all_trajs_b[n,start:end]
                    dist_slice = dists[start:end]

                    
                    slice_a = obj0[start:end]
                    slice_b = obj1[start:end]
                    
                    # print_('dist_slice', dist_slice)
                    
                    # lib_a is P x Q x 3
                    
                    # print_('pair_a[0]', pair_a[0])

                    # dist_to_lib = torch.sum(torch.abs(dist_slice.reshape(1, Q) - pair_dists), dim=1)
                    # # P


                    # print('slice_a', slice_a.shape)
                    # print('slice_b', slice_b.shape)
                    # # Q x 3
                    
                    # print('lib_a', lib_a.shape)
                    # print('lib_b', lib_b.shape)
                    # # P x Q x 3

                    # what i want now is:
                    # treat the concat of slices as a mini pointcloud
                    # and for each pair in the lib,
                    # try to find a rigid transform to align things

                    xyz_query = torch.cat([slice_a, slice_b], dim=0)
                    # Q*2 x 3
                    xyz_lib_full = torch.cat([lib_a, lib_b], dim=1)
                    # P x Q*2 x 3

                    initial_lib_dists = torch.zeros_like(lib_a[:,:,0])
                    lib_dists = torch.zeros_like(lib_a[:,:,0])

                    xyz_query = xyz_query.unsqueeze(0).repeat(P, 1, 1)

                    # dist = torch.mean(torch.norm(xyz_lib_full - xyz_query, dim=2), dim=1)
                    # utils.basic.print_stats('initial_lib_dist', dist)
                    
                    lib_T_query = utils.track.batch_rigid_transform_3d(xyz_query, xyz_lib_full)
                    xyz_query = utils.geom.apply_4x4(lib_T_query, xyz_query)

                    dist = torch.mean(torch.norm(xyz_lib_full - xyz_query, dim=2), dim=1)
                    # utils.basic.print_stats('matched lib_dist', dist)

                    min_dist_to_lib = torch.min(dist)
                    
                    # utils.basic.print_stats('initial_lib_dists', initial_lib_dists)
                    # utils.basic.print_stats('lib_dists', lib_dists)

                    # input()

                    # if m==0:
                    #     utils.basic.print_stats('dist_to_lib', dist_to_lib)
                    
                    # min_dist_to_lib = torch.min(dist_to_lib)
                    
                    # if min_dist_to_lib > 1.0:
                    #     looks_ok = False
                    if min_dist_to_lib > 0.5:
                        looks_ok = False
            # endif min_dist
        # end loop over combos
        if looks_ok:
            refined_possibilities.append(possibility)
            
    # end loop over h
    if len(refined_possibilities):
        refined_possibilities = torch.stack(refined_possibilities, dim=1)
    else:
        refined_possibilities = torch.zeros([B, 0, N, S, 3])
    return refined_possibilities

def refine_pairwise_possibilities_in_parallel(all_possibilities, pair_dists, lib_a, lib_b, neighbor_thresh=1.0):
    B, H, N, S, D = list(all_possibilities.shape)
    assert(D==3)

    P, Q, E = list(lib_a.shape)
    P2, Q2, E2 = list(lib_b.shape)
    assert(E==3)
    assert(P==P2)
    assert(Q==Q2)

    print('  refining input list of %d possibilities' % H)
    
    # now i want to check for conflicts within pairs, and make a pruned list

    combos = list(combinations(list(range(N)), 2))

    print('  we have these combos:', combos)

    # xyz_lib_full = torch.cat([lib_a, lib_b], dim=1)
    # # P x Q*2 x 3

    xyz_lib_full = torch.stack([lib_a, lib_b], dim=2)
    # ? x Q x 2 x 3
    # for some partial alignment
    # let's align to the midpoint of item0
    xyz_mid = xyz_lib_full[:,int(Q/2):int(Q/2)+1,0:1]
    xyz_lib_full = xyz_lib_full - xyz_mid
    xyz_lib_full = xyz_lib_full.reshape(P, Q*2, 3)


    # let's also make a lib of the opposite pairings
    xyz_lib_full2 = torch.stack([lib_b, lib_a], dim=2)
    # ? x Q x 2 x 3
    # for some partial alignment
    # let's align to the midpoint of item0
    xyz_mid = xyz_lib_full2[:,int(Q/2):int(Q/2)+1,0:1]
    xyz_lib_full2 = xyz_lib_full2 - xyz_mid
    xyz_lib_full2 = xyz_lib_full2.reshape(P, Q*2, 3)

    # and cat up
    xyz_lib_full = torch.cat([xyz_lib_full, xyz_lib_full2], dim=0)
    P = P*2

    refined_possibilities = torch.zeros((B, 0, N, S, D), dtype=torch.float32).cuda()

    mid = int(Q/2)
        
    for combo_ind, combo in enumerate(combos):
        # we are overwriting all_possibilities as we go,
        # so it is important to tally up H repeatedly
        H = all_possibilities.shape[1]
        if H > 0:
            print('  investigating combo %d' % combo_ind, combo)
            # print('all_possibilities_camR0', all_possibilities_camR0.shape)

            n0 = combo[0]
            n1 = combo[1]

            obj0 = all_possibilities[0,:,n0]
            obj1 = all_possibilities[0,:,n1]
            # H x S x 3
            dists = torch.norm(obj0 - obj1, dim=2)
            # H x S

            dists[:,:mid] += 10.0
            dists[:,-mid:] += 10.0

            min_dist, min_ind = torch.min(dists, dim=1)
            # H

            dist_thresh = 2.0
            min_dist_large = min_dist >= dist_thresh
            safe_possibilities = all_possibilities[0,min_dist_large]
            # print('safe_possibilities', safe_possibilities.shape)

            min_dist_small = min_dist < dist_thresh
            risky_possibilities = all_possibilities[0,min_dist_small]

            # these are ones that we want to check, and possibly reject
            # print('risky_possibilities', risky_possibilities.shape)

            # i don't need to recompute dists, or even recompute the min along S
            min_ind = min_ind[min_dist_small]
            dists = dists[min_dist_small]
            obj0 = obj0[min_dist_small]
            obj1 = obj1[min_dist_small]
            # xyz_query = xyz_query[min_dist_small]
            # print('min_ind', min_ind.shape)

            start = min_ind-mid
            end = min_ind+mid+1

            cannot_check = torch.logical_or(start < 0, end >= S)
            can_check = torch.logical_and(start >= 0, end < S)
            # print_('can_check sum', torch.sum(can_check))
            # print_('  cannot_check sum', torch.sum(cannot_check))
            safe_possibilities = torch.cat([safe_possibilities,
                                            risky_possibilities[cannot_check]], dim=0)
            
            print('  setting aside %d of these, since they do not come close' % (safe_possibilities.shape[0]))
            # print('  this yields %d possibilities to check' % (torch.sum(can_check)).detach().cpu().numpy())

            if torch.sum(can_check) > 0:
                risky_possibilities = risky_possibilities[can_check]
                # print('checking %d possibilities that do intersect' % (risky_possibilities.shape[0]))
                
                min_ind = min_ind[can_check]
                start = min_ind-mid
                end = min_ind+mid+1

                inds = start.reshape(-1, 1) + torch.arange(Q).reshape(1, Q).long().cuda()
                # ? x Q
                dist_slices = torch.gather(dists, 1, inds)
                # ? x Q

                # slightly ugly but ok:
                obj0_slices_x = torch.gather(obj0[:,:,0], 1, inds)
                obj0_slices_y = torch.gather(obj0[:,:,1], 1, inds)
                obj0_slices_z = torch.gather(obj0[:,:,2], 1, inds)

                obj1_slices_x = torch.gather(obj1[:,:,0], 1, inds)
                obj1_slices_y = torch.gather(obj1[:,:,1], 1, inds)
                obj1_slices_z = torch.gather(obj1[:,:,2], 1, inds)

                obj0_slices = torch.stack([obj0_slices_x,
                                           obj0_slices_y,
                                           obj0_slices_z], dim=2)
                obj1_slices = torch.stack([obj1_slices_x,
                                           obj1_slices_y,
                                           obj1_slices_z], dim=2)
                # xyz_query = torch.cat([obj0_slices, obj1_slices], dim=1)
                # # ? x Q*2 x 3

                xyz_query = torch.stack([obj0_slices, obj1_slices], dim=2)
                # ? x Q x 2 x 3
                # for some partial alignment
                # xyz_mid = torch.mean(xyz_query[:,int(Q/2):int(Q/2)+1], dim=2, keepdim=True)
                xyz_mid = xyz_query[:,int(Q/2):int(Q/2)+1,0:1]
                
                xyz_query = xyz_query - xyz_mid

                M = xyz_query.shape[0]
                xyz_query = xyz_query.reshape(M, Q*2, 3)
                
                
                # xyz_lib_full is P x Q*2 x 3
                # xyz_query is M x Q*2 x 3

                if False:

                    full_align = False

                    full_align = True

                    xyz_query_ = xyz_query.reshape(M, 1, Q*2, 3).repeat(1, P, 1, 1).reshape(-1, Q*2, 3)
                    xyz_lib_ = xyz_lib_full.reshape(1, P, Q*2, 3).repeat(M, 1, 1, 1).reshape(-1, Q*2, 3)

                    if full_align:
                        # we want to align the lib to each query
                        lib_T_query_ = utils.track.batch_rigid_transform_3d(xyz_query_, xyz_lib_)
                        xyz_query_ = utils.geom.apply_4x4(lib_T_query_, xyz_query_)
                        # this aligns with lib now

                    dist_to_lib_ = torch.mean(torch.norm(xyz_lib_ - xyz_query_, dim=2), dim=1)
                    # M*P
                    dist_to_lib = dist_to_lib_.reshape(M, P)

                    min_dist_to_lib = torch.min(dist_to_lib, dim=1)[0]
                    # M
                else:
                    # first let's find topk based on translation,
                    # then find the optimal transform

                    xyz_query_ = xyz_query.reshape(M, 1, Q*2, 3).repeat(1, P, 1, 1).reshape(-1, Q*2, 3)
                    xyz_lib_ = xyz_lib_full.reshape(1, P, Q*2, 3).repeat(M, 1, 1, 1).reshape(-1, Q*2, 3)
                    
                    dist_to_lib_ = torch.mean(torch.norm(xyz_lib_ - xyz_query_, dim=2), dim=1)
                    # M*P
                    dist_to_lib = dist_to_lib_.reshape(M, P)
                    # M x P

                    utils.basic.print_stats('  first dist_to_lib', dist_to_lib)
                    
                    min_dist_to_lib = torch.min(dist_to_lib, dim=0)[0]
                    # P
                    # this tells us, across all queries, what the nearest items from the library are

                    P_ = 256
                    best_dists, best_inds = torch.topk(-min_dist_to_lib, P_)

                    xyz_lib_mini = xyz_lib_full[best_inds]

                    # P_ = 4
                    # best_dists, best_inds = torch.topk(-dist_to_lib, P_, dim=1)

                    # print('xyz_lib_full', xyz_lib_full.shape)
                    # print('best_inds', best_inds.shape)
                    # xyz_lib_mini = torch.gather(xyz_lib_full, 0, best_inds)
                    # print('xyz_lib_mini', xyz_lib_mini.shape)

                    xyz_query_ = xyz_query.reshape(M, 1, Q*2, 3).repeat(1, P_, 1, 1).reshape(-1, Q*2, 3)
                    xyz_lib_ = xyz_lib_mini.reshape(1, P_, Q*2, 3).repeat(M, 1, 1, 1).reshape(-1, Q*2, 3)

                    # we want to align the lib to each query
                    lib_T_query_ = utils.track.batch_rigid_transform_3d(xyz_query_, xyz_lib_)
                    xyz_query_ = utils.geom.apply_4x4(lib_T_query_, xyz_query_)
                    # this aligns with lib now

                    dist_to_lib_ = torch.mean(torch.norm(xyz_lib_ - xyz_query_, dim=2), dim=1)
                    # M*P
                    dist_to_lib = dist_to_lib_.reshape(M, P_)

                    min_dist_to_lib = torch.min(dist_to_lib, dim=1)[0]
                

                utils.basic.print_stats('  final dist_to_lib', dist_to_lib)
                
                min_dist_ok = min_dist_to_lib < neighbor_thresh
                # print_('sum(min_dist_ok)', torch.sum(min_dist_ok))

                safe_possibilities = torch.cat([safe_possibilities,
                                                risky_possibilities[min_dist_ok]], dim=0)

                print('  added %d good intersections; we now have %d possibilities' % (
                    risky_possibilities[min_dist_ok].shape[0], safe_possibilities.shape[0]))
                      
                # print('new safe_possibilities', safe_possibilities.shape)
            # end if sum(can_check) > 0
            # refined_possibilities = torch.cat([refined_possibilities, safe_possibilities.unsqueeze(0)], dim=1)
            all_possibilities = safe_possibilities.unsqueeze(0)
        # end if H > 0
    # end loop over combos

    refined_possibilities = all_possibilities
    # input()
    return refined_possibilities

def get_best_value_possibility_per_scenario(possibilities_per_scenario, ways_to_be_true, vox_util, value_halfmemXs_, Z2, Y2, X2):
    M = ways_to_be_true.shape[0]
    _, _, N, S, _ = list(possibilities_per_scenario[0].shape)
    best_value_per_scenario = torch.zeros(M).float().cuda()
    best_possibility_per_scenario = torch.zeros(M, N, S, 3).float().cuda()
       
    for m in list(range(M)):
        possibilities_camX0 = possibilities_per_scenario[m]
        # B x H x N x S x 3
        
        H = ways_to_be_true[m]
        print('scenario %d; we have %d hypotheses to check' % (m, H))
        if H > 0:
            # for h in list(range(H)):
            #     possibility_camR0 = possibilities_camR0[:,h]
            #     # B x S x N x 3

            B, H, N, S, D = possibilities_camX0.shape
            assert(B==1)
            assert(D==3)
            
            traj_camX0_ = possibilities_camX0.permute(0, 3, 1, 2, 4).reshape(S, H*N, 3)
            # S x H*N x 3
            traj_halfmemX0_ = vox_util.Ref2Mem(
                traj_camX0_, Z2, Y2, X2)

            value_samples_, valid_samples_ = utils.samp.trilinear_sample3d(
                value_halfmemXs_, traj_halfmemX0_, return_inbounds=True)
            value_samples_ = value_samples_.squeeze(1) # eliminate channel dim
            valid_samples_[valid_samples_==0] = 0.5
            # B*S x H*N

            value_samples = value_samples_.reshape(S, H, N)
            valid_samples = valid_samples_.reshape(S, H, N)
            value_total = utils.basic.reduce_masked_mean(value_samples, valid_samples, dim=[0,2])
            # H

            H = np.min([H, 8])
            best_values, best_inds = torch.topk(value_total, H)
            print_('%d best_values' % H, best_values)
            # input()

            best_possibility_per_scenario[m] = possibilities_camX0[0,best_inds[0]]
            best_value_per_scenario[m] = best_values[0]

    return best_value_per_scenario, best_possibility_per_scenario

def get_vis_hypothesis_bev_frontal(summ_writer, interpretation, N, xyz_camXs, Z1, Y1, X1, vox_util, max_hypothesis=64):
    vis_all_anyrank_bev = []
    vis_all_anyrank_frontal = []

    S = xyz_camXs.shape[1] 

    # for s in list(range(1, self.S)):
    # for s in list(range(np.max([1, self.S-10]), self.S)):
    for s in list(range(1, S)):
        occ_memX0 = vox_util.voxelize_xyz(xyz_camXs[0:1,s], Z1, Y1, X1)
        vis_clean_anyrank_bev = summ_writer.summ_occ('', occ_memX0[0:1], bev=True, only_return=True)
        vis_clean_anyrank_frontal = summ_writer.summ_occ('', occ_memX0[0:1], frontal=True, only_return=True)
        for n0 in list(range(N)):
            trajlist_camX0 = interpretation['%d' % n0]
            # B x H x S x 3
            H = trajlist_camX0.shape[1]
            # if s==0:
            #     print('interpretation %d; object %d has %d paths' % (m, n0, H))
            
            for h in list(range(np.min([max_hypothesis, H]))):
                print("AAAAA")
                vis = summ_writer.summ_traj_on_occ(
                    '',
                    trajlist_camX0[:,h,:s],
                    occ_memX0[0:1],
                    vox_util,
                    show_bkg=False,
                    already_mem=False,
                    bev=True, 
                    only_return=True,
                    frame_id=s,
                    sigma=1)
                vis_new_any = (torch.max(vis, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
                vis_clean_anyrank_bev[vis_new_any>0] = vis[vis_new_any>0]

                vis = summ_writer.summ_traj_on_occ(
                    '',
                    trajlist_camX0[:,h,:s],
                    occ_memX0[0:1],
                    vox_util,
                    show_bkg=False,
                    already_mem=False,
                    frontal=True, 
                    only_return=True,
                    frame_id=s,
                    sigma=1)
                vis_new_any = (torch.max(vis, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
                vis_clean_anyrank_frontal[vis_new_any>0] = vis[vis_new_any>0]
            # end loop over h
        # end loop over n0
        vis_all_anyrank_bev.append(vis_clean_anyrank_bev)
        vis_all_anyrank_frontal.append(vis_clean_anyrank_frontal)
    
    return vis_all_anyrank_bev, vis_all_anyrank_frontal

def get_vis_bev_frontal_multiobject(summ_writer, clist_camX0s, occ_memX0, vox_util, frame_id):
    _, _, N, _ = list(clist_camX0s.shape)

    vis_clean_anyrank_bev = summ_writer.summ_occ('', occ_memX0[0:1], bev=True, only_return=True)
    vis_clean_anyrank_frontal = summ_writer.summ_occ('', occ_memX0[0:1], frontal=True, only_return=True)

    for n0 in list(range(N)):
        vis = summ_writer.summ_traj_on_occ(
            '',
            clist_camX0s[:,:,n0],
            occ_memX0[0:1],
            vox_util,
            show_bkg=False,
            already_mem=False,
            bev=True, 
            only_return=True,
            frame_id=frame_id,
            sigma=1)
        vis_new_any = (torch.max(vis, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
        vis_clean_anyrank_bev[vis_new_any>0] = vis[vis_new_any>0]

        vis = summ_writer.summ_traj_on_occ(
            '',
            clist_camX0s[:,:,n0],
            occ_memX0[0:1],
            vox_util,
            show_bkg=False,
            already_mem=False,
            frontal=True, 
            only_return=True,
            frame_id=frame_id,
            sigma=1)
        vis_new_any = (torch.max(vis, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
        vis_clean_anyrank_frontal[vis_new_any>0] = vis[vis_new_any>0]
    
    return vis_clean_anyrank_bev, vis_clean_anyrank_frontal

def prune_flow_field(flow_f, flow_b, alpha1=0.1, alpha2=1.0):
    B, C, H, W = flow_f.shape
    # xs_ori, ys_ori = self.meshgrid2d(1, H, W, sample_step=1, margin=0)
    flow_b_at_target_loc = utils.samp.backwarp_using_2d_flow(flow_b, flow_f)
    # utils.basic.print_stats('flow_b_at_target_loc', torch.norm(flow_b_at_target_loc, dim=1))
    diff = torch.norm((flow_f + flow_b_at_target_loc), dim=1) # B x H x W
    off = (diff > alpha1*torch.norm(torch.cat([flow_f, flow_b_at_target_loc], dim=1), dim=1) + alpha2).float()
    # off = (diff**2 > 0.01*(torch.norm(flow_f, dim=1)**2 + torch.norm(flow_b_at_target_loc, dim=1)**2) + 0.5).float()
    on = 1.0 - off
    on = on.unsqueeze(1) # B x 1 x H x W
    return on

def get_filtered_trajs(trajs_XYs, trajs_Ts, min_lifespan, min_dist=0, trajs_XYZs=None):
    trajs_XYs_filtered = []
    trajs_Ts_filtered = []
    if trajs_XYZs is not None:
        trajs_XYZs_filtered = []
    for i in range(len(trajs_XYs)):
        traj = trajs_XYs[i]
        endpoint_dist = torch.norm(traj[-1] - traj[0])
        if len(trajs_XYs[i]) >= min_lifespan:
            if min_dist==0:
                trajs_XYs_filtered.append(trajs_XYs[i])
                trajs_Ts_filtered.append(trajs_Ts[i])
                if trajs_XYZs is not None:
                    trajs_XYZs_filtered.append(trajs_XYZs[i])
            elif endpoint_dist >= min_dist:
                trajs_XYs_filtered.append(trajs_XYs[i])
                trajs_Ts_filtered.append(trajs_Ts[i])
                if trajs_XYZs is not None:
                    trajs_XYZs_filtered.append(trajs_XYZs[i])
    if trajs_XYZs is not None:
        return trajs_XYs_filtered, trajs_XYZs_filtered, trajs_Ts_filtered
    else:
        return trajs_XYs_filtered, trajs_Ts_filtered

def upgrade_2d_trajs_to_3d(trajs_XYs, trajs_Ts, depths, pix_T_cam):

    B, S, C, H, W = list(depths.shape)
    
    trajs_XYZs = []

    fx, fy, x0, y0 = utils.geom.split_intrinsics(pix_T_cam)
    
    for i in range(len(trajs_XYs)):
        xys = trajs_XYs[i]
        # N x 2
        ss = trajs_Ts[i]
        # []
        print_('xy %d' % i, xy)
        print_('s %d' % i, s)

        xyzs = []
        for s in ss:
            z = utils.samp.bilinear_sample2d(depths[:,s], xy[:,0], xy[:,1]) # B x 1 x N

            print_('z %d' % i, z)
            xyz = utils.geom.pixels2camera(x, y, z, fx, fy, x0, y0)
            print_('xyz %d' % i, xyz)
            xyzs.append(xyz)
        
        trajs_XYZs.append(torch.stack(xyz, dim=0))
        
        input()
        # if len(trajs_XYs[i]) >= min_lifespan:
        #     if min_dist==0:
        #         trajs_XYs_filtered.append(trajs_XYs[i])
        #         trajs_Ts_filtered.append(trajs_Ts[i])
        #     elif endpoint_dist >= min_dist:
        #         trajs_XYs_filtered.append(trajs_XYs[i])
        #         trajs_Ts_filtered.append(trajs_Ts[i])
                
    return trajs_XYZs


def get_affinity_matrix(position_matrix, valid_matrix, use_position_affinity=False, smooth=1, norm_velo=False):
    N_traj, S, _ = list(position_matrix.shape)

    position_matrix = position_matrix.half()
    valid_matrix = valid_matrix.half()

    velo_matrix = (position_matrix[:, smooth:] - position_matrix[:, 0:S-smooth])/smooth # the velocity of trajs N x S-1 x 2
    velo_valid_matrix = valid_matrix[:, smooth:] * valid_matrix[:, 0:S-smooth] # we want both positions to be valid

    device_name = 'cuda'
    # create a meshgrid to represent the N_traj * N_traj affinity matrix
    xv, yv = torch.meshgrid(
        [torch.arange(N_traj, device=torch.device(device_name)),
         torch.arange(N_traj, device=torch.device(device_name))])
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)

    velo_common_valid = velo_valid_matrix[xv, :] * velo_valid_matrix[yv, :] # (N*N) x (S-1)
    velo_diff = velo_matrix[xv, :, :] - velo_matrix[yv, :, :] # (N*N) x (S-1) x 2
    velo_diff_norm = torch.norm(velo_diff, dim=2) # (N*N) x (S-1)

    # this would be 0 if no overlap between two traj (because common_valid is all zero)
    # else it would be the max velocity difference
    velo_diff_norm_max = torch.max(velo_diff_norm * velo_common_valid, dim=1)[0] # (N*N)

    del velo_diff, velo_diff_norm
    torch.cuda.empty_cache()

    if norm_velo:
        # normalize by max speed
        velo_norm = torch.norm(velo_matrix, dim=2) # N x S-1
        velo_norm = velo_norm * velo_valid_matrix
        velo_norm = torch.cat([velo_norm[xv, :], velo_norm[yv, :]], dim=-1) # (N*N) x ((S-1) * 2)
        velo_norm_max = torch.max(velo_norm, dim=1)[0] # (N*N)

        velo_diff_norm_max = velo_diff_norm_max / (2*velo_norm_max + 1.0)

        del velo_norm, velo_norm_max
        torch.cuda.empty_cache()
    
    # # free memory
    utils.basic.print_stats('velo_diff_norm_max', velo_diff_norm_max)
    # # let's create affinity matrix 
    # velo_sigma = 0.2 #OG
    velo_sigma = 0.4
    velo_affinity = torch.exp(-velo_diff_norm_max / (2*velo_sigma*velo_sigma)) # (N*N)
    # velo_affinity = torch.exp(-velo_diff_norm_max / (2*velo_sigma*velo_sigma)) # (N*N)

    utils.basic.print_stats('velo_affinity', velo_affinity)

    affinity = velo_affinity

    # del velo_affinity, velo_diff_norm_max
    # torch.cuda.empty_cache()
    
    if use_position_affinity:
        # let's also check the euclidean distance between trajs, in addition to velocity
        position_common_valid = valid_matrix[xv, :] * valid_matrix[yv, :] # (N*N) x (S)
        position_diff = position_matrix[xv, :, :] - position_matrix[yv, :, :] # (N*N) x (S) x 2
        position_diff_norm = torch.norm(position_diff, dim=2) # (N*N) x (S)

        
        # position_diff_norm_max = torch.max(position_diff_norm * position_common_valid, dim=1)[0] # (N*N)
        position_diff_norm_mean = utils.basic.reduce_masked_mean(position_diff_norm, position_common_valid, dim=1) # (N*N)

        utils.basic.print_stats('position_diff_norm_mean', position_diff_norm_mean)
        
        del position_common_valid, position_diff, position_diff_norm
        torch.cuda.empty_cache()
    
        # position_sigma = 0.5 #OG
        position_sigma = 7
        position_affinity = torch.exp(-position_diff_norm_mean / (2*position_sigma*position_sigma)) # (N*N)
        
        utils.basic.print_stats('position_affinity', position_affinity)

        affinity = position_affinity * affinity
        # utils.basic.print_stats('velo_affinity', velo_affinity)

        # affinity = torch.exp(- (position_diff_norm_mean*0.1 * velo_diff_norm_max) / 100.0) # (N*N)

        # del position_affinity 
        # torch.cuda.empty_cache()
    # else:
    #     affinity = torch.exp(- velo_diff_norm_max / 100.0) # (N*N)

    # set the affinity for non-overlapping traj pair to be 0
    has_overlap = torch.sum(velo_common_valid, dim=1) > 0.0
    affinity[~has_overlap] = 0.0

    affinity_matrix = affinity.reshape(N_traj, N_traj)
    return affinity_matrix

def calculate_within_cluster_distance(all_clusters_all_xy_s, num_clusters, S):

    within_cluster_dist = np.zeros((num_clusters))
    for n0 in range(num_clusters):
        c0_xy_s = all_clusters_all_xy_s[n0]
        best_dist = 10000.0
        dist_from_cluster = torch.zeros(S, dtype=torch.float32)
        for s in range(S):
            c0_xy = c0_xy_s[s]
            # this is all points from this cluster on the current timestep
            if len(c0_xy) > 1:
                c0_xy0_ = c0_xy.reshape(1, -1, 2)
                c0_xy1_ = c0_xy.reshape(-1, 1, 2)
                # K x K x 2
                dist_mat = torch.norm(c0_xy0_ - c0_xy1_, dim=2)
                # K x K
                # min_dists = torch.min(dist_mat, dim=1)[0]
                min_dists, _ = torch.topk(dist_mat[0], 2, largest=False)
                # print_('min_dists', min_dists)
                min_dist = min_dists[1]
                # print_('min_dist', min_dist)
        
                dist_from_cluster[s] = min_dist
            # else:
            #     dist_from_cluster[s] = 10000

            max_dist_from_cluster = torch.max(dist_from_cluster)
            # print('max dist (along s) within %d' % (n0), max_dist_from_cluster)
            within_cluster_dist[n0] = max_dist_from_cluster
    
    return within_cluster_dist

def get_cluster_dist_and_affinity(all_clusters_all_xy_s, all_clusters_all_id_s, affinity_matrix, num_clusters, S, separation_thresh=8.0):

    cluster_dist = np.zeros((num_clusters, num_clusters))
    cluster_affinity = np.zeros((num_clusters, num_clusters))
    for n0 in range(num_clusters):
        # for n1 in range(num_clusters):
        for n1 in range(n0+1,num_clusters):
            # print('working on %d,%d' % (n0, n1))

            c0_xy_s = all_clusters_all_xy_s[n0]
            c1_xy_s = all_clusters_all_xy_s[n1]

            c0_id_s = all_clusters_all_id_s[n0]
            c1_id_s = all_clusters_all_id_s[n1]

            # print('c0_xy_s', len(c0_xy_s))
            # print('c1_xy_s', len(c1_xy_s))
            # print('c0_id_s', len(c0_id_s))
            # print('c1_id_s', len(c0_id_s))
            # input()

            min_dist_from_neighbor = torch.zeros(S, dtype=torch.float32)
            affinity_to_neighbor = torch.zeros(S, dtype=torch.float32)
            affinity_to_neighborhood = torch.zeros(S, dtype=torch.float32)

            for s in range(S):
                if torch.max(min_dist_from_neighbor) < separation_thresh: # keep computing
                    c0_xy = c0_xy_s[s]
                    c1_xy = c1_xy_s[s]
                    # this is all points from both clusters on the current timestep

                    c0_id = c0_id_s[s]
                    c1_id = c1_id_s[s]
                    # this is their trajectory ids

                    if len(c0_xy) and len(c1_xy):
                        M, N = len(c0_xy), len(c1_xy)
                            
                        c0_xy_ = c0_xy.reshape(-1, 1, 2)
                        c1_xy_ = c1_xy.reshape(1, -1, 2)
                        # M x N x 2

                        dist_mat = torch.norm(c0_xy_ - c1_xy_, dim=2)
                        # M x N

                        grid_y, grid_x = utils.py.meshgrid2d(M, N)

                        # print('dist_mat', dist_mat.shape)
                        # print('grid_y', grid_y.shape)

                        dist_mat_ = dist_mat.reshape(-1)
                        grid_y_ = grid_y.reshape(-1).astype(np.int32)
                        grid_x_ = grid_x.reshape(-1).astype(np.int32)

                        min_dist = torch.min(dist_mat_)
                        min_dist_ind = torch.argmin(dist_mat_)

                        min_dist_from_neighbor[s] = min_dist

                        neighbor_inds = torch.where(dist_mat_ < separation_thresh)[0]
                        neighbor_inds = neighbor_inds.detach().cpu().numpy()
                        neighbor_affinities = []
                        # if len(neighbor_inds):
                        #     local_affinity_mat = torch.zeros((M, N), dtype=torch.float32)
                        #     print('local_affinity_mat', local_affinity_mat.shape)
                        #     for m in range(M):
                        #         for n in range(N):
                        #             c0i = c0_id[m]
                        #             c1i = c1_id[n]
                        #             local_affinity = affinity_matrix[c0i, c1i]
                        #             local_affinity_mat[m, n] = local_affinity
                        #     local_affinity_mat_ = local_affinity_mat.reshape(-1)
                        #     affinity_to_neighbor[s] = local_affinity_mat_[min_dist_ind]

                        #     neighbor_affinities = local_affinity_mat_[neighbor_inds]
                        #     affinity_to_neighborhood[s] = torch.mean(neighbor_affinities)
                        for neighbor_ind in neighbor_inds:
                            # print('neighbor_ind', neighbor_ind)

                            m = grid_y_[neighbor_ind]
                            n = grid_x_[neighbor_ind]
                            # print('m, n', m, n)
                            
                            c0i = c0_id[m]
                            c1i = c1_id[n]
                            affinity = affinity_matrix[c0i, c1i]
                            neighbor_affinities.append(affinity)
                        if len(neighbor_affinities):
                            # affinity_to_neighborhood[s] = torch.mean(torch.stack(neighbor_affinities))
                            affinity_to_neighborhood[s] = torch.mean(torch.stack(neighbor_affinities))
                            # local_affinity_mat[m, n] = local_affinity
                            # local_affinity_mat_ = local_affinity_mat.reshape(-1)
                            # affinity_to_neighbor[s] = local_affinity_mat_[min_dist_ind]

                            # neighbor_affinities = local_affinity_mat_[neighbor_inds]
                            # affinity_to_neighborhood[s] = torch.mean(neighbor_affinities)
                    # else:
                    #     min_dist_from_neighbor[s] = 10000
                    #     affinity_to_neighbor[s] = 0.0
                    #     affinity_to_neighborhood[s] = 0.0
                    
                # end if max(dist)
            # end loop over S
            
            separation_from_neighbor = torch.max(min_dist_from_neighbor)
            
            # print('comparing %d, %d, min_dist_from_neighbor; sep' % (n0, n1), min_dist_from_neighbor, separation_from_neighbor)

            if separation_from_neighbor > separation_thresh:
                cluster_affinity[n0, n1] = -1
                cluster_affinity[n1, n0] = -1
                cluster_dist[n0, n1] = separation_from_neighbor
                cluster_dist[n1, n0] = separation_from_neighbor
            else:
                # mean_affinity_to_neighbor = torch.mean(affinity_to_neighbor)
                mean_affinity_to_neighborhood = torch.mean(affinity_to_neighborhood)
                # print('max dist (along s) from %d to %d' % (n0, n1), max_dist_from_cluster)

                cluster_dist[n0, n1] = separation_from_neighbor
                cluster_dist[n1, n0] = separation_from_neighbor

                cluster_affinity[n0, n1] = mean_affinity_to_neighborhood
                cluster_affinity[n1, n0] = mean_affinity_to_neighborhood
        # end loop over n1
    # end loop over n0

    return cluster_dist, cluster_affinity

def discard_empty_clusters(labels, num_clusters):
    new_labels = labels*0
    new_num_clusters = 0
    for n0 in range(num_clusters):
        inds = np.where(labels==n0)[0]
        if len(inds):
            new_labels[inds] = new_num_clusters
            new_num_clusters += 1
        else:
            print('discarding cluster %d' % n0)
    return new_labels, new_num_clusters

def get_all_traj_xy_and_id(trajs_XYs, trajs_Ts, labels, num_clusters, S):
    # we want to know, for all clusters,
    # for all timesteps,
    # what are all the xys
    # and what traj ids do those belong to?

    all_clusters_all_xy_s = []
    all_clusters_all_id_s = []
    for n0 in range(num_clusters):
        ind0 = np.where(labels==n0)[0]
        traj0_xy = [trajs_XYs[i] for i in ind0]
        traj0_t = [trajs_Ts[i] for i in ind0]
        all_xy_s = []
        all_id_s = []
        for s in range(S):
            all_xy = []
            all_id = []
            if len(ind0):
                for id, xy, time in zip(ind0, traj0_xy, traj0_t):
                    # print('id', id)
                    # print('xy', xy)p
                    # print('time', time)

                    inds = torch.where(time==s)[0]
                    # print('inds', inds)
                    xy = xy[inds]
                    # print('ind xy', xy)
                    id = np.array([id]*len(inds))
                    all_xy.append(xy)
                    all_id.append(id)
                all_xy = torch.cat(all_xy, dim=0)
                all_id = np.concatenate(all_id, axis=0).astype(np.int32)
            else:
                all_xy = torch.zeros((0, 2), dtype=torch.float32).cuda()
                all_id = np.zeros((0)).astype(np.int32)
                
            # print('all_xy', all_xy.shape)
            # print('all_id', all_id.shape)
            # input()
            all_xy_s.append(all_xy)
            all_id_s.append(all_id)
        all_clusters_all_xy_s.append(all_xy_s)
        all_clusters_all_id_s.append(all_id_s)
        # all_clusters_all_id_s.append(ind0)
    
    return all_clusters_all_xy_s, all_clusters_all_id_s

def discard_disconnected_clusters(all_clusters_all_xy_s, trajs_XYs, trajs_Ts, affinity_matrix, labels, num_clusters, within_cluster_dist, S):

    within_cluster_dist = calculate_within_cluster_distance(
        all_clusters_all_xy_s, num_clusters, S)
    # print('within_cluster_dist\n', within_cluster_dist)
    
    for n0 in range(num_clusters):
        # inds = np.where(labels==n0)[0]
        # print_('inds', inds)

        if within_cluster_dist[n0] > 5.0:
            print('discarding cluster %d' % n0)

            safe_inds = np.where(labels!=n0)[0]
            # print('safe_inds', safe_inds)

            affinity_matrix = affinity_matrix[:, safe_inds]
            affinity_matrix = affinity_matrix[safe_inds, :]
            
            labels = labels[safe_inds]
            trajs_XYs = [trajs_XYs[i] for i in safe_inds]
            trajs_Ts = [trajs_Ts[i] for i in safe_inds]

    labels, num_clusters = discard_empty_clusters(labels, num_clusters)

    return trajs_XYs, trajs_Ts, affinity_matrix, labels, num_clusters

def get_cluster_outliers(all_clusters_all_xy_s, all_clusters_all_id_s, dist_thresh=6.0):
    
    # here, for each cluster,
    # i want to discard the trajectories that are away from the main mass of the cluster,
    # where the "main mass" is determined by dilating the cluster
    # what we had said earlier is: "erode then dilate"
    # but a simpler method here might be:
    # for each point, find its nearest neighbors in the cluster, in each frame.
    # if on any frame, the neighbors are more than some dist away, throw away this guy.

    
    ids_to_discard = []

    num_clusters = len(all_clusters_all_xy_s)
    
    for n0 in range(num_clusters):
        c0_xy_s = all_clusters_all_xy_s[n0]
        c0_id_s = all_clusters_all_id_s[n0]
        S = len(c0_xy_s)
        for s in range(S):
            c0_xy = c0_xy_s[s]
            # this is all points from this cluster on the current timestep
            c0_id = c0_id_s[s]
            # this is all their ids
            K = c0_xy.shape[0]
            if K >= 3:
                c0_xy0_ = c0_xy.reshape(1, -1, 2)
                c0_xy1_ = c0_xy.reshape(-1, 1, 2)
                # K x K x 2
                dist_mat = torch.norm(c0_xy0_ - c0_xy1_, dim=2)
                # K x K
                # c0_id is (K,)
                sorted_dist_mat, _ = torch.sort(dist_mat, dim=1, descending=False)
                min_dist1 = sorted_dist_mat[:, 1] # K
                min_dist2 = sorted_dist_mat[:, 2]

                outliers_ids = torch.where(torch.max(min_dist1, min_dist2) > dist_thresh)[0]

                ids_to_discard.extend(c0_id[outliers_ids.cpu().numpy()])

            else:
                ids_to_discard.extend(c0_id)

                # for k in range(K):
                #     id = c0_id[k]
                #     dists = dist_mat[k]
                #     if len(dists) >= 3:
                #         min_dists, _ = torch.topk(dist_mat[k], 3, largest=False)
                #         min_dist1 = min_dists[1]
                #         min_dist2 = min_dists[2]
                #         # this is the distance to the nearest neighbor
                #         if torch.max(min_dist1, min_dist2) > dist_thresh:
                #             if id not in ids_to_discard:
                #                 ids_to_discard.append(id)
                #     else:
                #         if id not in ids_to_discard:
                #             ids_to_discard.append(id)

    ids_to_discard = np.unique(ids_to_discard)
                
    print('outlier ids_to_discard parallel', ids_to_discard)
    return ids_to_discard

def get_ids_for_empty_clusters(all_clusters_all_xy_s, all_clusters_all_id_s, min_ids=2):
    
    # here, for each cluster,
    # i want to return the ids of the cluster if there are fewer than min_ids
    
    ids_to_discard = []

    num_clusters = len(all_clusters_all_xy_s)

    
    for n0 in range(num_clusters):
        c0_xy_s = all_clusters_all_xy_s[n0]
        c0_id_s = all_clusters_all_id_s[n0]
        
        S = len(c0_xy_s)
        
        unique_ids = []
        for s in range(S):
            c0_xy = c0_xy_s[s]
            # this is all points from this cluster on the current timestep
            c0_id = c0_id_s[s]
            # this is all their ids
            
            K = c0_xy.shape[0]
            for k in range(K):
                id = c0_id[k]
                if id not in unique_ids:
                    unique_ids.append(id)

        if len(unique_ids) < min_ids:
            for id in unique_ids:
                ids_to_discard.append(id)
    print('empty cluster ids_to_discard', ids_to_discard)
    return ids_to_discard
    

def get_all_canvases(trajs_XYs, trajs_Ts, labels, num_clusters, S, H, W, convex_hull=True):

    all_canvases = torch.zeros((num_clusters, S, H, W), dtype=torch.float32, device=torch.device('cuda'))

    for i in range(num_clusters):

        id_to_vis = np.where(labels==i)[0]

        # print('id_to_vis', id_to_vis)

        xy_list = [trajs_XYs[i] for i in id_to_vis]
        t_list = [trajs_Ts[i] for i in id_to_vis]

        # xy_list is a list of positions 
        # t_list is a list of timesteps where those positions are active

        canvas = torch.zeros((S, H, W), dtype=torch.float32, device=torch.device('cuda'))
        for xy, t in zip(xy_list, t_list):
            for xy_, t_ in zip(xy, t):
                x_ = torch.round(xy_[0]).long()
                y_ = torch.round(xy_[1]).long()
                canvas[t_, y_, x_] = 1.0

        if convex_hull:
            for s in list(range(S)):
                # print('working on s=%d' % s)
                canvas_py = (canvas[s]).detach().cpu().numpy()
                indices = np.where(canvas_py!=0)
                # print('indices', indices)
                indices = np.stack(indices, axis=1)
                # print('indices', indices.shape)

                indices = indices.astype(np.float32)
                if indices.shape[0] > 3:

                    y = indices[:,0]
                    x = indices[:,1]
                    xy = np.stack([x,y], axis=1)

                    hull = ConvexHull(xy)

                    x = xy[hull.vertices,0]
                    y = xy[hull.vertices,1]
                    xy = np.stack([x,y], axis=1)

                    img = Image.new('L', (W, H), 0)
                    ImageDraw.Draw(img).polygon(xy, outline=1, fill=1)
                    mask = np.array(img)

                    canvas[s] = torch.from_numpy(mask)
                else:
                    canvas[s] = 0
        else:
            # dilate

            # # fancy kernel
            # weights = torch.ones(1, 1, 5, 5, device=torch.device('cuda'))
            # weights[0,0,0,0] = 0
            # weights[0,0,0,-1] = 0
            # weights[0,0,-1,0] = 0
            # weights[0,0,-1,-1] = 0

            weights = torch.ones(1, 1, 3, 3, device=torch.device('cuda'))
            
            canvas = canvas.unsqueeze(1)
            canvas = (F.conv2d(canvas, weights, padding=1)).clamp(0, 1)
            canvas = (F.conv2d(canvas, weights, padding=1)).clamp(0, 1)
            canvas = (F.conv2d(canvas, weights, padding=1)).clamp(0, 1)
            canvas = (F.conv2d(canvas, weights, padding=1)).clamp(0, 1)
            canvas = canvas.squeeze(1)
                
        # all_canvases.append(canvas)
        all_canvases[i] = canvas
    
    return all_canvases



def get_color_distortion(s=1.0, p_jitter=0.8, p_grayscale=0.2):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=p_jitter)
    rnd_gray = transforms.RandomGrayscale(p_grayscale)
    color_distort = transforms.Compose([
    rnd_color_jitter,
    rnd_gray])
    return color_distort

def do_random_erase_rgb_depth_valid(rgb, depth, valid, scaling):
    i, j, h, w, v = transforms.RandomErasing.get_params(rgb, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=[0])
    if random.random() > 0.5:
        rgb = TF.erase(rgb, i, j, h, w, v)
        depth = TF.erase(depth, int(i*scaling), int(j*scaling), int(h*scaling), int(w*scaling), v)
        valid = TF.erase(valid, int(i*scaling), int(j*scaling), int(h*scaling), int(w*scaling), v)
    
    return rgb, depth, valid

def apply_random_erase(rgb, scale_max=0.33):
    i, j, h, w, v = transforms.RandomErasing.get_params(rgb, scale=(0.02, scale_max), ratio=(0.3, 3.3), value=[0])

    B, C, H, W = list(rgb.shape)

    for b in list(range(B)):
        rgb_b = rgb[b]
        # rgb_b = rgb_b+0.5
        if random.random() > 0.5:
            rgb_b = TF.erase(rgb_b, i, j, h, w, v)
            # noise_mask = (torch.randn_like(rgb_b)*0.05).clamp(0, 1)
            # rgb_b[rgb_b==0] = noise_mask[rgb_b==0]
        # rgb_b = rgb_b - 0.5
        rgb[b] = rgb_b
    return rgb

def get_composed_transforms(rgb, depth, valid, scaling):
    color_distort = get_color_distortion()
    gaussian_blur = transforms.GaussianBlur(kernel_size=int(0.1 * rgb.shape[-2]+1))

    rgb = rgb+0.5

    rgb = color_distort(rgb)

    if random.random() > 0.5:
        rgb = gaussian_blur(rgb)
        # print("applying gaussian blur")

    if random.random() > 0.5:
        rgb = TF.hflip(rgb)
        depth = TF.hflip(depth)
        valid = TF.hflip(valid)
        # print("doing hflip")

    if random.random() > 0.5:
        rgb = TF.vflip(rgb)
        depth = TF.vflip(depth)
        valid = TF.vflip(valid)
        # print("doing vflip")

    angle = random.choice([-30, -15, 0, 15, 30])
    # print('rotating by : ', angle)
    rgb = TF.rotate(rgb, angle)
    depth = TF.rotate(depth, angle)
    valid = TF.rotate(valid, angle)

    rgb, depth, valid = do_random_erase_rgb_depth_valid(rgb, depth, valid, scaling)

    rgb = rgb - 0.5

    return rgb, depth, valid

def get_color_augs(rgb):
    # print('deprecated; please use "apply_color_augs" instead; will call that func now....')
    return apply_color_augs(rgb)

def apply_random_blur_and_noise(rgb):
    B, C, H, W = list(rgb.shape)

    for b in list(range(B)):
        rgb_b = rgb[b]

        rgb_b = rgb_b+0.5
    
        gaussian_blur = transforms.GaussianBlur(kernel_size=9)

        if random.random() > 0.5:
            rgb_b = gaussian_blur(rgb_b)

        if random.random() > 0.5:
            rgb_b = (rgb_b + torch.randn_like(rgb_b)*0.05).clamp(0, 1)

        rgb_b = rgb_b - 0.5

        rgb[b] = rgb_b
    
    return rgb
    
def apply_color_augs(rgb, amount=0.5):
    B, C, H, W = list(rgb.shape)

    to_pil = transforms.ToPILImage()
    from_pil = transforms.ToTensor()
    
    for b in list(range(B)):
        rgb_b = rgb[b]

        # rgb_b = rgb_b.detach().cpu()
        rgb_b = rgb_b+0.5
        # rgb_b = to_pil(rgb_b)
    
        color_distort = get_color_distortion(s=amount)
        gaussian_blur = transforms.GaussianBlur(kernel_size=9)

        rgb_b = color_distort(rgb_b)

        if random.random() > 0.5:
            rgb_b = gaussian_blur(rgb_b)
            # print("applying gaussian blur")

        if False:
            # i want to avoid these for a minute
            if random.random() > 0.5:
                rgb_b = TF.hflip(rgb_b)
                # print("doing hflip")

            if random.random() > 0.5:
                rgb_b = TF.vflip(rgb_b)
                # print("doing vflip")
            
        # rgb_b = from_pil(rgb_b)
        # rgb_b = rgb_b.cuda()
        
        rgb_b = rgb_b - 0.5

        rgb[b] = rgb_b
    
    return rgb


def get_class_masks_from_trajs(all_clusters_all_id_s, all_clusters_all_xy_s, H, W,
                               num_clusters, anchor_frame, pix_count_threshold=50,
                               dilate=0, erode=0):

    S = len(all_clusters_all_xy_s[0])
    middle_frame_cluster_ids = []
    pix_count_threshold = pix_count_threshold
    for n in range(num_clusters):
        traj_count = len(all_clusters_all_id_s[n][int(S/2)])
        if traj_count >= pix_count_threshold:
            middle_frame_cluster_ids.append(n)
    
    n_middle = len(middle_frame_cluster_ids)

    masks = torch.zeros(n_middle, S, H, W).cuda()
    weights = torch.ones(1, 1, 3, 3).float().cuda()

    for s in range(S):
        for n in range(n_middle):
            cluster_id = middle_frame_cluster_ids[n]
            all_xy = all_clusters_all_xy_s[cluster_id][s].long()
            x = all_xy[:,0]
            y = all_xy[:,1]
            mask = torch.zeros(H,W).float().cuda()
            mask[y,x] = 1.0

            for dil in range(dilate):
                mask = mask.reshape(1, 1, H, W)
                mask = (F.conv2d(mask, weights, padding=1)).clamp(0, 1)
                mask = mask.reshape(H, W)
            
            masks[n,s] = mask

    if dilate:
        # it is important that the masks do not overlap with one another
        # let's just throw away all pixels that have this overlap
        mask_overlap = (torch.sum(masks, dim=0, keepdim=True) > 1).float()
        masks = masks * (1.0 - mask_overlap)

    if erode:
        assert(dilate>0) # otw probably we will eliminate everything
        
        # to be safe, let's erode each mask once,
        # to step away from the boundaries
        for s in range(S):
            for n in range(n_middle):
                mask = masks[n,s]

                for er in range(erode):
                    mask = mask.reshape(1, 1, H, W)
                    mask = 1.0 - (F.conv2d(1.0 - mask, weights, padding=1)).clamp(0, 1)
                    mask = mask.reshape(H, W)
                
                masks[n,s] = mask
        
    
    return masks

def do_random_translation(input, h_frac, v_frac):
    random_translate = transforms.RandomAffine(0, (h_frac, v_frac))
    return random_translate(input)
        

def apply_random_flips(image, vertical=False):
    B, C, H, W = list(image.shape)

    if random.random() > 0.5:
        image = torch.flip(image, [3])
    if vertical and random.random() > 0.5:
        image = torch.flip(image, [2])
    
    return image

def apply_random_rot90(image):
    B, C, H, W = list(image.shape)

    if random.random() > 0.5:
        H_ = min(H, W)
        image = image[:,:,:H_,:H_]
        image = torch.rot90(image,1,[2,3])
    
    return image

def apply_random_pointcloud_scaling(xyz, scale_low=0.8, scale_high=1.2):
    B, N, C = list(xyz.shape)
    scales = np.random.uniform(scale_low, scale_high, B)
    for b in range(B):
        xyz[b] = xyz[b] * scales[b]
    return xyz

def get_rgb_and_boxlists_after_center_crop(rgb, boxlist2d, scorelist, H_crop, W_crop, random_crop=False):
    B, C, H, W = list(rgb.shape)
    
    min_y = boxlist2d[:,:,0] * H
    min_x = boxlist2d[:,:,1] * W

    max_y = boxlist2d[:,:,2] * H
    max_x = boxlist2d[:,:,3] * W

    if random_crop:
        i, j, h, w = transforms.RandomCrop.get_params(rgb, (H_crop, W_crop))
    else:
        i = (H - H_crop) // 2
        h = H_crop
        j = (W - W_crop) // 2
        w = W_crop

    rgb_cropped = rgb[:, :, i:i+h, j:j+w]

    min_y = torch.clamp((min_y - i)/H_crop, min=0.0, max=1.0)
    max_y = torch.clamp((max_y - i)/W_crop, min=0.0, max=1.0)
    min_x = torch.clamp((min_x - j)/H_crop, min=0.0, max=1.0)
    max_x = torch.clamp((max_x - j)/W_crop, min=0.0, max=1.0)

    boxlist2d[:,:,0] = min_y
    boxlist2d[:,:,1] = min_x
    boxlist2d[:,:,2] = max_y
    boxlist2d[:,:,3] = max_x

    range_x = max_x - min_x
    range_y = max_y - min_y

    x_zero = (range_x==0.0)
    y_zero = (range_y==0.0)

    scorelist[x_zero] = 0.0
    scorelist[y_zero] = 0.0


    return rgb_cropped, boxlist2d, scorelist

def get_boxlist_from_connected_components(component_label):
    H, W = list(component_label.shape)
    H = float(H)
    W = float(W)
    unique_labels, counts = torch.unique(component_label, return_counts=True)
    counts[0] = 0
    max_label = torch.argmax(counts)
    temp = torch.zeros_like(component_label)
    temp[component_label==max_label] = 1
    component_label = temp
    y_ind = component_label.nonzero()[:,0]
    x_ind = component_label.nonzero()[:,1]
    min_x = torch.min(x_ind)/W
    min_y = torch.min(y_ind)/H
    max_x = torch.max(x_ind)/W
    max_y = torch.max(y_ind)/H

    return component_label, torch.FloatTensor([min_y, min_x, max_y, max_x])

def sort_lrtlist_by_size(lrtlist, tids, scores):
    # put large objects at the top

    # put the good boxes shuffled at the top;
    # sink the bad boxes to the bottom.

    # boxes are B x N x D
    # tids are B x N
    # scores are B x N
    B, N, D = list(lrtlist.shape)
    assert (D==19)

    lrtlist_new = torch.zeros_like(lrtlist)
    tids_new = -1*torch.ones_like(tids)
    scores_new = torch.zeros_like(scores)

    lenlist, _ = utils.geom.split_lrtlist(lrtlist) # lenlist is B x N x 3
    sizelist = torch.sum(lenlist, dim=2) # B x N

    for b in list(range(B)):
        sizes = sizelist[b, :]
        inds = np.argsort(-1.0*sizes.cpu().detach().numpy()) # descending
        inds = np.squeeze(inds)

        lrtlist_new[b] = lrtlist[b,inds]
        scores_new[b] = scores[b,inds]
        tids_new[b] = tids[b,inds]
        
        inds = np.argsort(-1.0*scores_new[b].cpu().detach().numpy(), kind='mergesort') # descending, we want to keep the order, so use a stable sorting
        inds = np.squeeze(inds)

        lrtlist_new[b] = lrtlist_new[b,inds]
        scores_new[b] = scores_new[b,inds]
        tids_new[b] = tids_new[b,inds]

        # print('ok, boxes old and new')
        # print(boxes[b])
        # print(boxes_new[b])
        # input()

    return lrtlist_new, tids_new, scores_new

def update_obj_dict_with_match(obj_dict, s0, lrt0, max_dot, feat0):
    # lrtlist = obj_dict['lrtlist'].cuda() # S x 19
    # scorelist = obj_dict['scorelist'].cuda() # S
    # plauslist = obj_dict['plauslist'].cuda() # S
    # featlist = obj_dict['featlist'].cuda() # S x 128
    # forecastlist = obj_dict['forecastlist'].cuda() # S x 128
    # libcostlist = obj_dict['libcostlist'].cuda() # S x 128
    # vislist_bev = obj_dict['vislist_bev'].cuda() # 1 x S x 3 x Z x X
    # vislist_fro = obj_dict['vislist_fro'].cuda() # 1 x S x 3 x Y x X

    # lrtlist[s0] = lrt0
    # scorelist[s0] = max_dot
    # plauslist[s0] = 1.0
    # featlist[s0] = feat0
    

    obj_dict['lrtlist'][s0] = lrt0    
    obj_dict['scorelist'][s0] = max_dot    
    obj_dict['plauslist'][s0] = 1.0
    obj_dict['featlist'][s0] = feat0
    return obj_dict

def calibrate_library(traj_lib, S):
    K = traj_lib.shape[0]
    T = traj_lib.shape[1]

    # i want index S in the lib to meet the endpoint
    
    # maybe i never need to trim the lib actually
    
    
    # i want to calibrate wrt S, but using that as the mid of the lib
    #

    # if S==T:
    #     print('trimming not necessary')
    #     return traj_lib
    # elif T > S:
    #     print('trimming the library, since T > S')

    local_traj_lib = traj_lib.clone()

    # we need to re-calibrate

    end = S-1

    xyz0 = local_traj_lib[:,0]
    xyz1 = local_traj_lib[:,end]

    print('calibration setting index %d as the local endpoint' % (end))

    delta = xyz1 - xyz0

    dx = delta[:,0]
    dy = delta[:,1]
    dz = delta[:,2]

    dx_ok = torch.abs(dx) > 0.001
    dy_ok = torch.abs(dy) > 0.001
    dz_ok = torch.abs(dz) > 0.001
    all_ok = torch.logical_and(torch.logical_and(dx_ok, dy_ok), dz_ok)
    local_traj_lib = local_traj_lib[all_ok]

    xyz0 = local_traj_lib[:,0]
    xyz1 = local_traj_lib[:,end]
    delta = xyz1 - xyz0
    dx = delta[:,0]
    dy = delta[:,1]
    dz = delta[:,2]

    bot_hyp = torch.sqrt(dz**2 + dx**2)

    pitch = -torch.atan2(dy, bot_hyp)
    yaw = torch.atan2(dz, dx)

    rot = utils.geom.eul2rotm(yaw*0.0, yaw, pitch)
    rt = utils.geom.merge_rt(rot, 0.0*rot[:,0])
    local_traj_lib = utils.geom.apply_4x4(rt, local_traj_lib)
    return local_traj_lib

def update_obj_dict_with_forecasts(obj_dict, s0, traj_lib, H):
    S = obj_dict['lrtlist'].shape[0]
    K, T, _ = list(traj_lib.shape)
    
    print_('sum(scorelist>0)', torch.sum(scorelist>0))

    if torch.sum(scorelist[max(s0-int(T/2)+1,0):s0+1]>0) > 1:

        clist = utils.geom.get_clist_from_lrtlist(obj_dict['lrtlist'].reshape(1, S, 19)).squeeze(0) # S x 3

        # let's see the path this match implies, according to the library

        steplist = torch.arange(S)

        # use T/2 steps, so that we can always forecast T/2 into the future
        scorelist_mini = obj_dict['scorelist'][max(s0-int(T/2)+1,0):s0+1]
        steplist_mini = steplist[max(s0-int(T/2)+1,0):s0+1]
        clist_mini = clist[max(s0-int(T/2)+1,0):s0+1]

        print_('scorelist_mini', scorelist_mini)
        print_('steplist_mini', steplist_mini)
        print_('clist_mini', clist_mini)

        mini_inds = scorelist_mini > 0
        # lrtlist_mini = lrtlist_mini[mini_inds]
        scorelist_mini = scorelist_mini[mini_inds]
        steplist_mini = steplist_mini[mini_inds]
        clist_mini = clist_mini[mini_inds]

        print_('mini_inds', mini_inds)

        print_('valid scorelist_mini', scorelist_mini)
        print_('valid steplist_mini', steplist_mini)
        print_('valid clist_mini', clist_mini)
        # input()

        first_ind = steplist_mini[0]
        last_ind = steplist_mini[-1]

        print('first_ind, last_ind', first_ind, last_ind)

        offset_steplist = steplist_mini - first_ind
        print_('offset_steplist', offset_steplist)

        local_T = last_ind - first_ind + 1

        print('local_T', local_T)

        local_traj_lib = calibrate_library(traj_lib, local_T)

        lib_K = local_traj_lib.shape[0]
        lib_T = local_traj_lib.shape[1]
        # lib_mid = int(lib_T/2)

        print('lib_K', lib_K)
        print('lib_T', lib_T)
        # print('lib_mid', lib_mid)

        start_lrt = lrtlist[first_ind]
        end_lrt = lrtlist[last_ind]

        # clist = utils.geom.get_clist_from_lrtlist(lrtlist[first_ind:last_ind+1].unsqueeze(0)).squeeze(0) # local_T x 3

        startpoint = utils.geom.get_clist_from_lrtlist(start_lrt.reshape(1,1,19)).reshape(1, 3)
        endpoint = utils.geom.get_clist_from_lrtlist(end_lrt.reshape(1,1,19)).reshape(1, 3)
        print('startpoint, endpoint', startpoint, endpoint)
        # print('clist', clist)

        # we need to scale the lib according to the travel distance
        query_dist = torch.norm(endpoint - startpoint, dim=1) # 1

        lib_dist = torch.norm(local_traj_lib[:,0] - local_traj_lib[:,local_T-1], dim=1) # K

        # add noise to the query dist, so that we don't need to follow the traj precisely
        query_dist = (query_dist + torch.randn(lib_dist.shape).float().cuda() * 0.1).clamp(min=0.0) # K
        traj_camNY = local_traj_lib * query_dist.reshape(lib_K, 1, 1) / lib_dist.reshape(lib_K, 1, 1)
        # # set the last one to zero, so that we always have that option
        # traj_camNY[-1] = 0.0

        print('scaling using %d' % (local_T-1))
        # input()

        # we also need to orient the lib according to the travel direction
        camNY_T_camX0 = utils.geom.get_NY_transforms_from_endpoints(startpoint, endpoint)#, eps=0.1)
        camX0_T_camNY = utils.geom.safe_inverse(camNY_T_camX0)
        # 1 x 4 x 4

        traj_camNY_ = traj_camNY.reshape(1, lib_K*lib_T, 3)
        traj_camX0_ = utils.geom.apply_4x4(camX0_T_camNY, traj_camNY_) # 1 x KT x 3
        traj_camX0_ = traj_camX0_.reshape(lib_K, lib_T, 3).permute(1, 0, 2) # T x K x 3

        # clist_lib_mini = traj_camX0_[steplist_mini] # local_T x K x 3
        # clist_lib_mini = traj_camX0_[:local_T] # local_T x K x 3
        clist_lib_mini = traj_camX0_[offset_steplist] # local_T x K x 3
        print('clist_lib_mini', clist_lib_mini.shape)
        print('clist_lib_mini[:,0]', clist_lib_mini[:,0])
        print_('scorelist_mini', scorelist_mini)
        # clist_dist = torch.sum(torch.norm(clist_mini.unsqueeze(1) - clist_lib_mini, dim=2) * scorelist_mini.reshape(local_T, 1), dim=0)/torch.sum(scorelist_mini) # K
        clist_dist = utils.basic.reduce_masked_mean(
            torch.norm(clist_mini.unsqueeze(1) - clist_lib_mini, dim=2),
            scorelist_mini.reshape(-1, 1),
            dim=0)

        H_early = np.min([H, lib_K])
        early_dists, early_inds = torch.topk(clist_dist, H_early, largest=False)
        print_('early_dists', early_dists)
        libcostlist = torch.exp(-3*early_dists)
        # i want to be quite sensitive to this, since my library is already large
        best_lib_value = torch.max(libcostlist)
        traj_camX0_ = traj_camX0_[:,early_inds] # T x H x 3
        lib_K = traj_camX0_.shape[1]
        traj_camX0_[offset_steplist] = clist_mini.unsqueeze(1).repeat(1, lib_K, 1)

        # set one forecast to zero, so that we always have that option
        traj_camX0_[:,-1] = traj_camX0_[0:1,0]

        traj_halfmemX0_ = self.vox_util.Ref2Mem(traj_camX0_, Z2, Y2, X2) # T x K x 3

        print('traj_halfmemX0_', traj_halfmemX0_.shape)
        traj_halfmemX0_trim_ = traj_halfmemX0_[:local_T]
        print('traj_halfmemX0_trim_', traj_halfmemX0_trim_.shape)

        value_halfmemXs_trim = value_halfmemXs[:,first_ind:last_ind+1].cuda()

        print('value_halfmemXs_trim', value_halfmemXs_trim.shape)

        value_halfmemXs_trim_ = value_halfmemXs_trim.reshape(-1, 1, Z2, Y2, X2)
        print('value_halfmemXs_trim_', value_halfmemXs_trim_.shape)

        value_samples_, valid_samples_ = utils.samp.trilinear_sample3d(
            value_halfmemXs_trim_, traj_halfmemX0_trim_, return_inbounds=True)
        value_samples_ = value_samples_.squeeze(1) # eliminate channel dim
        # S x K

        # values = utils.basic.reduce_masked_mean(value_samples_, valid_samples_, dim=0)

        # # let's prefer trajectories that cross through the valid area of the scene
        # valid_samples_[valid_samples_==0] = 0.5
        # values = torch.mean(value_samples_*valid_samples_, dim=0)

        # values = torch.mean(value_samples_*valid_samples_, dim=0)

        if torch.sum(valid_samples_) > 0:
            values = utils.basic.reduce_masked_mean(value_samples_, valid_samples_, dim=0)
            # K
            # utils.basic.print_stats('values', values)
        else:
            # we cannot really choose based on value
            values = torch.ones_like(torch.mean(value_samples_, dim=0)) # we cannot really choose
            # use the ranking provided by distance

        H_per_object = np.min([self.H_to_keep, lib_K])
        best_values, best_inds = torch.topk(values*libcostlist, H_per_object)
        traj_camX0 = traj_camX0_[:,best_inds] # T x H x 3

        print('forecastlist', forecastlist.shape)
        print('traj_camX0', traj_camX0.shape)
        print_('best_values', best_values)
        print_('best_lib_value', best_lib_value)

        forecastlist[first_ind:min(first_ind+lib_T,self.S),:H_per_object] = traj_camX0[:self.S-first_ind]
        forecastlist[min(first_ind+lib_T,self.S-1):,:H_per_object] = traj_camX0[-1].unsqueeze(0)

        libcostlist = early_dists[best_inds]

        plauslist[s0] = torch.max(best_values)

        # visualizing this, we should see trajs going from startpoints to endpoints
        # occ_halfmemX = self.vox_util.voxelize_xyz(xyz_camX, Z2, Y2, X2).detach().cpu()
        # occ_memR0 = self.vox_util.voxelize_xyz(self.xyz_camRs[0:1,self.S-1], Z1, Y1, X1)

    #     vis_clean_anyrank_bev = self.summ_writer.summ_lrtlist_bev(
    #         '',
    #         occ_memX,
    #         lrt0.reshape(1, 1, 19),
    #         score0.reshape(1, 1),
    #         torch.ones_like(score0).reshape(1, 1).long()*max_ind,
    #         self.vox_util,
    #         include_zeros=False,
    #         frame_id=s0,
    #         only_return=True)

    #     # vis_clean_anyrank_bev = self.summ_writer.summ_occ('', occ_memX, bev=True, only_return=True)
    #     vis_clean_anyrank_fro = self.summ_writer.summ_occ('', occ_memX, frontal=True, only_return=True)
    #     # vis_clean_anyrank_per = self.summ_writer.summ_rgb('', self.rgb_camXs[:,s0], only_return=True)
    #     # H = traj_camR0.shape[1]
    #     for h in list(range(np.min([3, H_per_object]))):
    #         # traj_camX0_h = traj_camX0[:,h].unsqueeze(0) # 1 x S x 3
    #         traj_camX0_h = forecastlist[:,h].unsqueeze(0) # 1 x S x 3
    #         vis = self.summ_writer.summ_traj_on_occ(
    #             '',
    #             traj_camX0_h,
    #             occ_memX,
    #             self.vox_util,
    #             show_bkg=False,
    #             already_mem=False,
    #             bev=True,
    #             only_return=True,
    #             frame_id=s0,
    #             sigma=1)
    #         vis_new_any = (torch.max(vis, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
    #         vis_clean_anyrank_bev[vis_new_any>0] = vis[vis_new_any>0]

    #         vis = self.summ_writer.summ_traj_on_occ(
    #             '',
    #             traj_camX0_h,
    #             occ_memX,
    #             self.vox_util,
    #             show_bkg=False,
    #             already_mem=False,
    #             frontal=True,
    #             only_return=True,
    #             frame_id=s0,
    #             sigma=1)
    #         vis_new_any = (torch.max(vis, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
    #         vis_clean_anyrank_fro[vis_new_any>0] = vis[vis_new_any>0]

    #     # end loop over h
    #     # self.summ_writer.summ_rgb('explain/%d_%d_interpretation_bev' % (n0, n1), vis_clean_anyrank_bev)
    #     # self.summ_writer.summ_rgb('explain/%d_%d_interpretation_frontal' % (n0, n1), vis_clean_anyrank_frontal)

    #     vislist_bev[:,s0] = vis_clean_anyrank_bev
    #     vislist_fro[:,s0] = vis_clean_anyrank_fro
    #     # input()
    # else:
    #     # vislist_bev[:,s0] = self.summ_writer.summ_lrtlist_bev(
    #     #     '',
    #     #     occ_memX,
    #     #     lrt0.reshape(1, 1, 19),
    #     #     score0.reshape(1, 1),
    #     #     score0.reshape(1, 1).long(),
    #     #     self.vox_util,
    #     #     include_zeros=False,
    #     #     frame_id=s0,
    #     #     only_return=True)

    #     vis_clean_anyrank_bev = self.summ_writer.summ_lrtlist_bev(
    #         '',
    #         occ_memX,
    #         lrt0.reshape(1, 1, 19),
    #         score0.reshape(1, 1),
    #         torch.ones_like(score0).reshape(1, 1).long()*max_ind,
    #         self.vox_util,
    #         include_zeros=False,
    #         frame_id=s0,
    #         only_return=True)
    #     traj_ = utils.geom.get_clist_from_lrtlist(lrtlist.reshape(1, -1, 19))
    #     vis = self.summ_writer.summ_traj_on_occ(
    #         '',
    #         traj_,
    #         occ_memX,
    #         self.vox_util,
    #         frame_id=s0,
    #         bev=True,
    #         only_return=True)
    #     vis_new_any = (torch.max(vis, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
    #     vis_clean_anyrank_bev[vis_new_any>0] = vis[vis_new_any>0]
    #     vislist_bev[:,s0] = vis_clean_anyrank_bev

    #     vislist_fro[:,s0] = self.summ_writer.summ_traj_on_occ(
    #         '',
    #         traj_,
    #         occ_memX,
    #         self.vox_util,
    #         frame_id=s0,
    #         frontal=True,
    #         only_return=True)

    #     # # we don't have summ_lrtlist_frontal yet, so let's just summ occ

def normalize_keypoints(kp_xy, H, W):
    B, N, D = kp_xy.shape
    assert(D==2)
    kp_xy = torch.stack([kp_xy[:,:,0]/W, kp_xy[:,:,1]/H], dim=2)
    return kp_xy

def unnormalize_keypoints(kp_xy, H, W):
    B, N, D = kp_xy.shape
    assert(D==2)
    kp_xy = torch.stack([kp_xy[:,:,0]*W, kp_xy[:,:,1]*H], dim=2)
    return kp_xy
    
def get_pts_inside_outside_occ(xyz_camR, occ_memR, vox_util):
    _, _, Z, Y, X = occ_memR.shape
    xyz_memR = vox_util.Ref2Mem(xyz_camR, Z, Y, X)
    x, y, z = xyz_memR.unbind(2)
    x = x.round().clamp(0,X-1).long()
    y = y.round().clamp(0,Y-1).long()
    z = z.round().clamp(0,Z-1).long()
    have = occ_memR[0,0,z,y,x]==1
    return xyz_camR[:,have.reshape(-1)], xyz_camR[:,~have.reshape(-1)]


class Vgg16(nn.Module):
    
    def __init__(self):
        from torchvision import models
        
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, cheap=False):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        if cheap:
            return (h_relu_1_2, h_relu_2_2)
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out
    
def convert_xywh_to_yxyx(xywh):
    xc, yc, w, h = xywh.unbind(dim=-1)
    x0, y0, x1, y1 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
    return torch.stack([y0, x0, y1, x1], dim=-1)
    
def convert_yxyx_to_xywh(yxyx):
    y0, x0, y1, x1 = yxyx.unbind(dim=-1)
    xc = (x0+x1)/2.0
    yc = (y0+y1)/2.0
    h = y1-y0
    w = x1-x0
    return torch.stack([xc, yc, w, h], dim=-1)

def perceptual_dist(vgg, rgb_e, rgb_g, vgg_g=None, cheap=False):
    B, C, H, W = rgb_e.shape
    B2, C2, H2, W2 = rgb_g.shape
    # don't assert same batch, in case we have multiple recons of one target
    # assert(B==B2)
    assert(C==C2)
    assert(H==H2)
    assert(W==W2)
    vgg_e = vgg(rgb_e+0.5, cheap=cheap)
    if vgg_g is None:
        vgg_g = vgg(rgb_g+0.5, cheap=cheap)
    perceptual_loss_sc0 = torch.mean(F.mse_loss(vgg_e[0], vgg_g[0], reduction='none'), dim=[1,2,3]) # B
    perceptual_loss_sc1 = torch.mean(F.mse_loss(vgg_e[1], vgg_g[1], reduction='none'), dim=[1,2,3]) # B
    if cheap:
        vgg_loss = torch.mean(torch.stack([perceptual_loss_sc0,
                                           perceptual_loss_sc1], dim=1), dim=1) # B
    else:
        perceptual_loss_sc2 = torch.mean(F.mse_loss(vgg_e[2], vgg_g[2], reduction='none'), dim=[1,2,3]) # B
        perceptual_loss_sc3 = torch.mean(F.mse_loss(vgg_e[3], vgg_g[3], reduction='none'), dim=[1,2,3]) # B
        vgg_loss = torch.mean(torch.stack([perceptual_loss_sc0,
                                           perceptual_loss_sc1,
                                           perceptual_loss_sc2,
                                           perceptual_loss_sc3], dim=1), dim=1) # B
    return vgg_loss

def perceptual_dists(vgg, rgbs_e, rgbs_g, vggs_g=None):
    B, S, C, H, W = rgbs_e.shape
    B2, S, C2, H2, W2 = rgbs_g.shape
    # don't assert same batch, in case we have multiple recons of one target
    # assert(B==B2)
    assert(C==C2)
    assert(H==H2)
    assert(W==W2)
    vggs_e_ = vgg(rgbs_e.reshape(B*S, C, H, W)+0.5)

    vggs_e = []
    for feat in vggs_e_:
        BS, D, vH, vW = feat.shape
        vggs_e.append(feat.reshape(B, S, D, vH, vW))
            
    if vggs_g is None:
        vggs_g_ = vgg(rgbs_g.reshape(B2*S, C, H, W)+0.5)
        vggs_g = []
        for feat in vggs_g_:
            BS, D, vH, vW = feat.shape
            vggs_g.append(feat.reshape(B2, S, D, vH, vW))
        
    perceptual_loss_sc0 = torch.mean(F.mse_loss(vggs_e[0], vggs_g[0], reduction='none'), dim=[1,2,3,4]) # B
    perceptual_loss_sc1 = torch.mean(F.mse_loss(vggs_e[1], vggs_g[1], reduction='none'), dim=[1,2,3,4]) # B
    perceptual_loss_sc2 = torch.mean(F.mse_loss(vggs_e[2], vggs_g[2], reduction='none'), dim=[1,2,3,4]) # B
    perceptual_loss_sc3 = torch.mean(F.mse_loss(vggs_e[3], vggs_g[3], reduction='none'), dim=[1,2,3,4]) # B
    vgg_loss = torch.mean(torch.stack([perceptual_loss_sc0,
                                       perceptual_loss_sc1,
                                       perceptual_loss_sc2,
                                       perceptual_loss_sc3], dim=1), dim=1) # B
    return vgg_loss


def perceptual_dist_image(vgg, rgb_e, rgb_g, return_smaller=True):
    B, C, H, W = rgb_e.shape
    B2, C2, H2, W2 = rgb_g.shape
    # don't assert same batch, in case we have multiple recons of one target
    # assert(B==B2)
    assert(C==C2)
    assert(H==H2)
    assert(W==W2)
    vgg_e = vgg(rgb_e+0.5)
    vgg_g = vgg(rgb_g+0.5)
    
    perceptual_loss_sc0 = torch.mean(F.mse_loss(vgg_e[0], vgg_g[0], reduction='none'), dim=1, keepdim=True)
    perceptual_loss_sc1 = torch.mean(F.mse_loss(vgg_e[1], vgg_g[1], reduction='none'), dim=1, keepdim=True)
    perceptual_loss_sc2 = torch.mean(F.mse_loss(vgg_e[2], vgg_g[2], reduction='none'), dim=1, keepdim=True)
    perceptual_loss_sc3 = torch.mean(F.mse_loss(vgg_e[3], vgg_g[3], reduction='none'), dim=1, keepdim=True)

    if return_smaller:
        _, _, H, W = perceptual_loss_sc3.shape

    perceptual_loss_sc0 = F.interpolate(perceptual_loss_sc0, (H, W))
    perceptual_loss_sc1 = F.interpolate(perceptual_loss_sc1, (H, W))
    perceptual_loss_sc2 = F.interpolate(perceptual_loss_sc2, (H, W))
    perceptual_loss_sc3 = F.interpolate(perceptual_loss_sc3, (H, W))

    perceptual_loss = (perceptual_loss_sc0 + perceptual_loss_sc1 + perceptual_loss_sc2 + perceptual_loss_sc3)/4.0

    return perceptual_loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def get_1d_embedding(x, C, cat_coords=True):
    B, N, D = x.shape
    assert(D==1)
    div_term = (torch.arange(0, C, 2, device='cuda', dtype=torch.float32) * (1000.0 / C)).reshape(1, 1, int(C/2))
    pe_x = torch.zeros(B, N, C, device='cuda', dtype=torch.float32)
    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)
    if cat_coords:
        pe_x = torch.cat([pe_x, x], dim=2)
    return pe_x
    
def get_2d_embedding(xy, C, cat_coords=True):
    B, N, D = xy.shape
    assert(D==2)

    x = xy[:,:,0:1]
    y = xy[:,:,1:2]
    div_term = (torch.arange(0, C, 2, device='cuda', dtype=torch.float32) * (1000.0 / C)).reshape(1, 1, int(C/2))
    
    pe_x = torch.zeros(B, N, C, device='cuda', dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device='cuda', dtype=torch.float32)
    
    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)
    
    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)
    
    pe = torch.cat([pe_x, pe_y], dim=2) # B, C, Z, Y, X
    if cat_coords:
        pe = torch.cat([pe, xy], dim=2)
    return pe
    
def get_3d_embedding(xyz, C, cat_coords=True):
    B, N, D = xyz.shape
    assert(D==3)

    x = xyz[:,:,0:1]
    y = xyz[:,:,1:2]
    z = xyz[:,:,2:3]
    div_term = (torch.arange(0, C, 2, device='cuda', dtype=torch.float32) * (1000.0 / C)).reshape(1, 1, int(C/2))
    
    pe_x = torch.zeros(B, N, C, device='cuda', dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device='cuda', dtype=torch.float32)
    pe_z = torch.zeros(B, N, C, device='cuda', dtype=torch.float32)
    
    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)
    
    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)
    
    pe_z[:, :, 0::2] = torch.sin(z * div_term)
    pe_z[:, :, 1::2] = torch.cos(z * div_term)
    
    pe = torch.cat([pe_x, pe_y, pe_z], dim=2) # B, N, C*3
    if cat_coords:
        pe = torch.cat([pe, xyz], dim=2) # B, N, C*3+3
    return pe
    
def get_pc_embedding(xyz, C, cat_coords=True):
    B, N, D = xyz.shape
    assert(D==3)

    x = xyz[:,:,0:1]
    y = xyz[:,:,1:2]
    z = xyz[:,:,2:3]
    div_term = (torch.arange(0, C, 2, device='cuda', dtype=torch.float32) * (1000.0 / C)).reshape(1, 1, int(C/2))
    # print('z', z.shape)
    
    pe_z = torch.zeros(B, N, C, device='cuda', dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device='cuda', dtype=torch.float32)
    pe_x = torch.zeros(B, N, C, device='cuda', dtype=torch.float32)
    
    # print('pe_z', pe_z.shape)
    
    pe_z[:, :, 0::2] = torch.sin(z * div_term)
    pe_z[:, :, 1::2] = torch.cos(z * div_term)

    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)
    
    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)

    # print('pe_x', pe_x.shape)
    # print('pe_y', pe_y.shape)
    # print('pe_z', pe_z.shape)
    
    pe = torch.cat([pe_x, pe_y, pe_z], dim=2) # B, N, C*3
    if cat_coords:
        pe = torch.cat([pe, xyz], dim=2) # B, N, C*3+3
    return pe

def get_relative_2d_embedding_map(xy, H, W, C, cat_coords=True):
    B, D = xy.shape
    assert(D==2)

    grid_xy = utils.basic.gridcloud2d(B, H, W, device='cuda') # B, H*W, 2
    N = grid_xy.shape[1]
    x = grid_xy[:,:,0:1] - xy[:,0].reshape(B, 1, 1)
    y = grid_xy[:,:,1:2] - xy[:,1].reshape(B, 1, 1)
    div_term = (torch.arange(0, C, 2, device='cuda', dtype=torch.float32) * (1000.0 / C)).reshape(1, 1, int(C/2))
    
    pe_x = torch.zeros(B, N, C, device='cuda', dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device='cuda', dtype=torch.float32)
    
    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)
    
    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)
    
    pe = torch.cat([pe_x, pe_y], dim=2) # B, C, Z, Y, X
    if cat_coords:
        pe = torch.cat([pe, grid_xy], dim=2)
    return pe
