import torch
import numpy as np
import utils.geom
import utils.vox
import torch.nn.functional as F
import utils.basic
import utils.track
from utils.basic import print_stats, print_
import skimage
import scipy.spatial

device_name = 'cuda'

def vis_affinities(name, rgbs, traj_dict, affinity, frames_to_vis, summ_writer):

    trajs_XYZs_cam = traj_dict['XYZs_cam']
    trajs_XYZs_pix = traj_dict['XYZs_pix']
    trajs_Ts = traj_dict['Ts']

    S = rgbs.shape[1]
    position_matrix_cam, position_matrix_pix, valid_matrix = prep_reference_mats(trajs_XYZs_cam, trajs_XYZs_pix, trajs_Ts, S)
    
    return summ_writer.summ_anchor_affinity_on_images(
        'affinity/%s' % name,
        rgbs[:,:frames_to_vis].unbind(1),
        position_matrix_pix[:,:frames_to_vis].cpu().numpy(), 
        valid_matrix[:,:frames_to_vis].cpu().numpy(),
        utils.basic.normalize_single(affinity).cpu().numpy(),
        frame_ids=range(frames_to_vis),
    )
    
def prep_anchor_mats(anchor_id, trajs_XYZs_cam, trajs_XYZs_pix, trajs_Ts, S):
    # prep tensors for the anchor
    anchor_position_cam = torch.zeros((1, S, 3), device=torch.device(device_name))
    anchor_position_pix = torch.zeros((1, S, 3), device=torch.device(device_name))
    anchor_valid = torch.zeros((1, S), device=torch.device(device_name)) # element i,j is 1 if traj_i is alive at time_j
    Ts = trajs_Ts[anchor_id] # K (the lifespan of this traj)
    XYZs_cam = trajs_XYZs_cam[anchor_id] # K x 3
    XYZs_pix = trajs_XYZs_pix[anchor_id] # K x 3
    anchor_position_cam[0, Ts] = XYZs_cam
    anchor_position_pix[0, Ts] = XYZs_pix
    anchor_valid[0, Ts] = 1.0
    return anchor_position_cam, anchor_position_pix, anchor_valid

def prep_reference_mats(trajs_XYZs_cam, trajs_XYZs_pix, trajs_Ts, S):
    # prep tensors for the rest
    N_traj = len(trajs_XYZs_pix)
    position_matrix_cam = torch.zeros((N_traj, S, 3), device=torch.device(device_name))
    position_matrix_pix = torch.zeros((N_traj, S, 3), device=torch.device(device_name))
    valid_matrix = torch.zeros((N_traj, S), device=torch.device(device_name)) # element i,j is 1 if traj_i is alive at time_j
    for i in range(N_traj):
        Ts = trajs_Ts[i] # K (the lifespan of this traj)
        XYZs_cam = trajs_XYZs_cam[i] # K x 3
        XYZs_pix = trajs_XYZs_pix[i] # K x 3
        position_matrix_cam[i, Ts] = XYZs_cam
        position_matrix_pix[i, Ts] = XYZs_pix
        valid_matrix[i, Ts] = 1.0
    return position_matrix_cam, position_matrix_pix, valid_matrix
            
def compute_dist_affinity(anchor_id, traj_dict, S, coeff=1.0, s0=None, H=None, W=None, summ_writer=None, part_id=0):
    trajs_XYZs_cam = traj_dict['XYZs_cam']
    trajs_XYZs_pix = traj_dict['XYZs_pix']
    trajs_Ts = traj_dict['Ts']
    
    if summ_writer is not None:
        # draw a map of all the pixels that are activated in s0
        N_traj = len(trajs_XYZs_pix)
        inlier_mask0 = torch.zeros((H*W), device=torch.device(device_name))
        for i in range(N_traj):
            xys = trajs_XYZs_pix[i][:,:2]
            ts = trajs_Ts[i]
            if (s0 in ts):
                ind0 = ts==s0
                xy0 = xys[ind0].long()
                inds = utils.basic.sub2ind(H, W, xy0[:,1], xy0[:,0])
                inlier_mask0[inds] = 1.0
        inlier_mask0 = inlier_mask0.reshape(1, 1, H, W)
        summ_writer.summ_oned('affinity/dist_inlier_mask_%d_part%d' % (s0, part_id), inlier_mask0, norm=False)
    
    anchor_position_cam, anchor_position_pix, anchor_valid = prep_anchor_mats(anchor_id, trajs_XYZs_cam, trajs_XYZs_pix, trajs_Ts, S)
    position_matrix_cam, position_matrix_pix, valid_matrix = prep_reference_mats(trajs_XYZs_cam, trajs_XYZs_pix, trajs_Ts, S)

    anchor_dist = torch.norm(position_matrix_cam - anchor_position_cam, dim=2) # N x S
    print_stats('anchor_dist', anchor_dist)
    anchor_dist_valid = anchor_valid * valid_matrix # N x S
    N_traj = len(trajs_XYZs_cam)
    dist_affinity = torch.zeros((N_traj), device=torch.device(device_name))
    for i in range(N_traj):
        dist = anchor_dist[i]
        val = anchor_dist_valid[i]
        if torch.sum(val) > 3:
            dist = dist[val > 0]
            # dist_var = torch.var(dist)
            # dist_affinity[i] = torch.exp(-100.0*dist_var)
            dist_change = torch.abs(dist[0] - dist[-1])/(1+torch.mean(dist)) # proportional to dist
            dist_affinity[i] = torch.exp(-coeff*dist_change)
    print_stats('dist_affinity', dist_affinity)
    return dist_affinity

def compute_rigid_affinity(cam1_T_cam0, traj_dict, s0, s1, S, H=None, W=None, summ_writer=None, part_id=0):
    trajs_XYZs_cam = traj_dict['XYZs_cam']
    trajs_XYZs_pix = traj_dict['XYZs_pix']
    trajs_Ts = traj_dict['Ts']

    if summ_writer is not None:
        # draw a map of all the pixels that are activated in s0
        N_traj = len(trajs_XYZs_pix)
        inlier_mask0 = torch.zeros((H*W), device=torch.device(device_name))
        for i in range(N_traj):
            xys = trajs_XYZs_pix[i][:,:2]
            ts = trajs_Ts[i]
            if (s0 in ts):
                ind0 = ts==s0
                xy0 = xys[ind0].long()
                inds = utils.basic.sub2ind(H, W, xy0[:,1], xy0[:,0])
                inlier_mask0[inds] = 1.0
        inlier_mask0 = inlier_mask0.reshape(1, 1, H, W)
        summ_writer.summ_oned('affinity/rigid_inlier_mask_%d_part%d' % (s0, part_id), inlier_mask0, norm=False)
    
    position_matrix_cam, position_matrix_pix, valid_matrix = prep_reference_mats(trajs_XYZs_cam, trajs_XYZs_pix, trajs_Ts, S)
    
    N_traj = position_matrix_cam.shape[0]
    rigid_affinity = torch.zeros((N_traj), device=torch.device(device_name))
    for i in range(N_traj):
        xyz_cam0 = position_matrix_cam[i, s0]
        xyz_cam1 = position_matrix_cam[i, s1]
        valid_0 = valid_matrix[i, s0]
        valid_1 = valid_matrix[i, s1]
        both_valid = valid_0*valid_1
        if both_valid==1:
            xyz_cam1_prime = utils.geom.apply_4x4_single(cam1_T_cam0, xyz_cam0.unsqueeze(0)).squeeze(0)
            err = torch.norm(xyz_cam1_prime-xyz_cam1, dim=0)
            rigid_affinity[i] = torch.exp(-2.0*err) # 10 is too large
    return rigid_affinity

def compute_component_affinity(anchor_xy, traj_dict, H, W, s0, summ_writer=None, part_id=0):
    trajs_XYZs_cam = traj_dict['XYZs_cam']
    trajs_XYZs_pix = traj_dict['XYZs_pix']
    trajs_Ts = traj_dict['Ts']

    weights = torch.ones(1, 1, 3, 3, device=torch.device(device_name))
    
    # draw a map of all the pixels that are activated in s0
    N_traj = len(trajs_XYZs_pix)
    inlier_mask0 = torch.zeros((H*W), device=torch.device(device_name))
    for i in range(N_traj):
        xys = trajs_XYZs_pix[i][:,:2]
        ts = trajs_Ts[i]
        if (s0 in ts):
            ind0 = ts==s0
            xy0 = xys[ind0].long()
            inds = utils.basic.sub2ind(H, W, xy0[:,1], xy0[:,0])
            inlier_mask0[inds] = 1.0
    inlier_mask0 = inlier_mask0.reshape(1, 1, H, W)
    # since the trajectories are at stride 4, we need to dilate things to connect them
    inlier_mask0 = F.conv2d(inlier_mask0, weights, padding=1).clamp(0,1)
    inlier_mask0 = F.conv2d(inlier_mask0, weights, padding=1).clamp(0,1)

    component_label0 = skimage.measure.label(inlier_mask0.reshape(H, W).cpu().numpy())

    if summ_writer is not None:
        summ_writer.summ_oned('affinity/component_inlier_mask_%d_part%d' % (s0, part_id), inlier_mask0, norm=False)
        summ_writer.summ_seg('affinity/component_mask_%d_part%d' % (s0, part_id), torch.from_numpy(component_label0).cuda().reshape(1, H, W))

    anchor_xy_np = anchor_xy.detach().cpu().numpy()
    component_num = component_label0[round(anchor_xy_np[1]), round(anchor_xy_np[0])]
    # print('component_num', component_num)
    component_on = torch.from_numpy(component_label0==component_num).cuda().reshape(1, 1, H, W).float()
    component_safe = component_on.clone()
    component_safe = F.conv2d(component_safe, weights, padding=1).clamp(0,1)
    if summ_writer is not None:
        summ_writer.summ_oned('affinity/component_on_%d_part%d' % (s0, part_id), component_on, norm=False)
        summ_writer.summ_oned('affinity/component_safe_%d_part%d' % (s0, part_id), component_safe, norm=False)

    component_label = torch.from_numpy(component_label0).cuda().reshape(H*W)
    component_affinity = torch.zeros((N_traj), device=torch.device(device_name))
    component_safe = component_safe.reshape(H*W)
    for i in range(len(trajs_Ts)):
        xys = trajs_XYZs_pix[i][:,:2]
        ts = trajs_Ts[i]
        if (s0 in ts):
            ind0 = ts==s0
            xy0 = xys[ind0].long()
            inds = utils.basic.sub2ind(H, W, xy0[:,1], xy0[:,0])
            assert(len(inds==1))
            if component_safe[inds]==1:
                component_affinity[i] = 1.0
    return component_affinity, component_safe

def mark_outliers_on_component(traj_dict, component_safe, s0, H, W, summ_writer=None):
    trajs_XYZs_cam = traj_dict['XYZs_cam']
    trajs_XYZs_pix = traj_dict['XYZs_pix']
    trajs_Ts = traj_dict['Ts']

    should_reject = torch.zeros((len(trajs_Ts)), device=torch.device(device_name))
    for i in range(len(trajs_Ts)):
        xys = trajs_XYZs_pix[i][:,:2]
        ts = trajs_Ts[i]
        if (s0 in ts):
            ind0 = ts==s0
            xy0 = xys[ind0].long()
            inds = utils.basic.sub2ind(H, W, xy0[:,1], xy0[:,0])
            assert(len(inds==1))
            if component_safe[inds]==1:
                should_reject[i] = 1.0
    return should_reject


def shift_inliers_or_outliers(traj_dict_inlier, traj_dict_outlier, new_inlier_ids=None, new_outlier_ids=None):
    # transfer inlier_ids from traj_dict_outlier to traj_dict_inlier
    
    trajs_XYZs_cam_inlier = traj_dict_inlier['XYZs_cam']
    trajs_XYZs_pix_inlier = traj_dict_inlier['XYZs_pix']
    trajs_Ts_inlier = traj_dict_inlier['Ts']
    
    trajs_XYZs_cam_outlier = traj_dict_outlier['XYZs_cam']
    trajs_XYZs_pix_outlier = traj_dict_outlier['XYZs_pix']
    trajs_Ts_outlier = traj_dict_outlier['Ts']
    
    print('trajs_Ts_inlier before', len(trajs_Ts_inlier))
    print('trajs_Ts_outlier before', len(trajs_Ts_outlier))

    assert((new_inlier_ids is None) or (new_outlier_ids is None)) # only shift one way please
    assert((new_inlier_ids is not None) or (new_outlier_ids is not None)) # shift one please
    
    if new_inlier_ids is not None:
        trajs_XYZs_cam_inlier_ = [trajs_XYZs_cam_outlier[i] for i in new_inlier_ids]
        trajs_XYZs_pix_inlier_ = [trajs_XYZs_pix_outlier[i] for i in new_inlier_ids]
        trajs_Ts_inlier_ = [trajs_Ts_outlier[i] for i in new_inlier_ids]
        trajs_XYZs_cam_inlier_new = trajs_XYZs_cam_inlier + trajs_XYZs_cam_inlier_
        trajs_XYZs_pix_inlier_new = trajs_XYZs_pix_inlier + trajs_XYZs_pix_inlier_
        trajs_Ts_inlier_new = trajs_Ts_inlier + trajs_Ts_inlier_

        all_original_ids = list(range(len(trajs_Ts_outlier)))
        remaining_ids = [x for x in all_original_ids if x not in new_inlier_ids]
        trajs_XYZs_cam_outlier_new = [trajs_XYZs_cam_outlier[rem] for rem in remaining_ids]
        trajs_XYZs_pix_outlier_new = [trajs_XYZs_pix_outlier[rem] for rem in remaining_ids]
        trajs_Ts_outlier_new = [trajs_Ts_outlier[rem] for rem in remaining_ids]
        
    if new_outlier_ids is not None:
        trajs_XYZs_cam_outlier_ = [trajs_XYZs_cam_inlier[i] for i in new_outlier_ids]
        trajs_XYZs_pix_outlier_ = [trajs_XYZs_pix_inlier[i] for i in new_outlier_ids]
        trajs_Ts_outlier_ = [trajs_Ts_inlier[i] for i in new_outlier_ids]
        trajs_XYZs_cam_outlier_new = trajs_XYZs_cam_outlier + trajs_XYZs_cam_outlier_
        trajs_XYZs_pix_outlier_new = trajs_XYZs_pix_outlier + trajs_XYZs_pix_outlier_
        trajs_Ts_outlier_new = trajs_Ts_outlier + trajs_Ts_outlier_

        all_original_ids = list(range(len(trajs_Ts_inlier)))
        remaining_ids = [x for x in all_original_ids if x not in new_outlier_ids]
        trajs_XYZs_cam_inlier_new = [trajs_XYZs_cam_inlier[rem] for rem in remaining_ids]
        trajs_XYZs_pix_inlier_new = [trajs_XYZs_pix_inlier[rem] for rem in remaining_ids]
        trajs_Ts_inlier_new = [trajs_Ts_inlier[rem] for rem in remaining_ids]

    print('trajs_Ts_inlier after', len(trajs_Ts_inlier_new))
    print('trajs_Ts_outlier after', len(trajs_Ts_outlier_new))
        
    traj_dict_inlier_new = {
        'XYZs_cam': trajs_XYZs_cam_inlier_new,
        'XYZs_pix': trajs_XYZs_pix_inlier_new,
        'Ts': trajs_Ts_inlier_new,
    }
    traj_dict_outlier_new = {
        'XYZs_cam': trajs_XYZs_cam_outlier_new,
        'XYZs_pix': trajs_XYZs_pix_outlier_new,
        'Ts': trajs_Ts_outlier_new,
    }
    return traj_dict_inlier_new, traj_dict_outlier_new

def get_rigid_motion_anchor(cam1_T_cam0, traj_dict, s0, s1, H, W):
    trajs_XYZs_cam = traj_dict['XYZs_cam']
    trajs_XYZs_pix = traj_dict['XYZs_pix']
    trajs_Ts = traj_dict['Ts']
    
    xy_pix0, xyz_cam0, xyz_cam1, id_0 = get_pointclouds_for_two_frames(traj_dict, s0, s1)
    xyz_cam1_prime = utils.geom.apply_4x4_single(cam1_T_cam0, xyz_cam0)
    # N x 3
    err = torch.norm(xyz_cam1_prime-xyz_cam1, dim=1)
    inlier_dists, inlier_inds = torch.topk(err, min(10, len(err)), largest=False)

    # simply taking the argmax here is unreliable, since it may be a floater pixel
    # so, instead, we create an image, blur, and sample from there

    # rigid_affinity = torch.exp(-err*100.0) # N
    rigid_affinity = torch.exp(-err*2.0) # N
    rigid_affinity_normalized = utils.basic.normalize_single(rigid_affinity)
    rigid_affinity_map, _ = utils.geom.create_depth_image_single(xy_pix0, rigid_affinity_normalized, H, W, force_positive=False)
    rigid_affinity_map = rigid_affinity_map.reshape(1, 1, H, W)

    weights = torch.ones(1, 1, 3, 3, device=torch.device(device_name))
    rigid_affinity_blurred = rigid_affinity_map.clone()
    rigid_affinity_blurred = F.conv2d(rigid_affinity_blurred, weights, padding=1)
    rigid_affinity_blurred = F.conv2d(rigid_affinity_blurred, weights, padding=1)
    rigid_affinity_blurred = F.conv2d(rigid_affinity_blurred, weights, padding=1)

    argmax_y, argmax_x = utils.basic.hard_argmax2d(rigid_affinity_blurred)
    anchor_xy = torch.stack([argmax_x, argmax_y], dim=1).squeeze(0)
    anchor_id = get_traj_id_at_xy_t(trajs_XYZs_pix, trajs_Ts, anchor_xy, s0)
    # assert(anchor_id is not None) # this should never happen
    return anchor_id, anchor_xy, rigid_affinity_blurred
    
def get_pointclouds_for_two_frames(traj_dict, s0, s1):
    trajs_XYZs_cam = traj_dict['XYZs_cam']
    trajs_XYZs_pix = traj_dict['XYZs_pix']
    trajs_Ts = traj_dict['Ts']
    
    xy_pix0 = []
    xyz_cam0 = []
    xyz_cam1 = []
    id_0 = []
    for i in range(len(trajs_XYZs_cam)):
        xys = trajs_XYZs_pix[i][:,:2]
        xyzs = trajs_XYZs_cam[i]
        ts = trajs_Ts[i]
        if (s0 in ts) and (s1 in ts):
            # print('xyzs', xyzs)
            # print('ts', ts)
            ind0 = ts==s0
            ind1 = ts==s1
            # print('ts[ind0]', ts[ind0])
            # print('ts[ind1]', ts[ind1])

            xy0 = xys[ind0]
            xyz0 = xyzs[ind0]
            xyz1 = xyzs[ind1]

            id_0.append(i)
            xy_pix0.append(xy0)
            xyz_cam0.append(xyz0)
            xyz_cam1.append(xyz1)
            # input()
    xy_pix0 = torch.stack(xy_pix0, dim=0).reshape(-1, 2)
    xyz_cam0 = torch.stack(xyz_cam0, dim=0).reshape(-1, 3)
    xyz_cam1 = torch.stack(xyz_cam1, dim=0).reshape(-1, 3)
    # print('xy_pix0', xy_pix0.shape)
    # print('xyz_cam0', xyz_cam0.shape)
    # print('xyz_cam1', xyz_cam1.shape)
    return xy_pix0, xyz_cam0, xyz_cam1, id_0

def get_stats(trajs_XYs):
    N_traj = len(trajs_XYs)
    mean_lifespan = np.mean([len(x) for x in trajs_XYs])
    print('N_traj', N_traj)
    if N_traj > 0:
        print('mean_lifespan', mean_lifespan)
        mean_dist = torch.mean(torch.stack([torch.norm(x[-1]-x[0]) for x in trajs_XYs]))
        print_('mean_dist', mean_dist)
    return N_traj

def get_rigid_transform(xyz0, xyz1, inlier_thresh=0.04, ransac_steps=256, recompute_with_inliers=False):
    xyz0 = xyz0.detach().cpu().numpy()
    xyz1 = xyz1.detach().cpu().numpy()
    # xyz0 and xyz1 are each N x 3
    assert len(xyz0) == len(xyz1)

    # utils.py.print_stats('xyz0', xyz0)
    # utils.py.print_stats('xyz1', xyz1)
    
    N = xyz0.shape[0] # total points
    nPts = 8
    # assert(N > nPts)
    if N < nPts:
        print('grt: too few points; returning translation')
        R = np.eye(3, dtype=np.float32)
        t = np.mean(xyz1-xyz0, axis=0)
        print('t', t)
        rt = utils.py.merge_rt(R, t)
        return torch.from_numpy(rt).cuda()

    # print('N = %d' % N)
    # print('doing ransac')
    rts = []
    errs = []
    inliers = []
    for step in list(range(ransac_steps)):
        # assert(N > nPts) 
        perm = np.random.permutation(N)
        cam1_T_cam0, _ = utils.track.rigid_transform_3d_py_helper(xyz0[perm[:nPts]], xyz1[perm[:nPts]])
        # i got some errors in matmul when the arrays were too big,
        # so let's just use 1k points for the error 
        perm = np.random.permutation(N)
        xyz1_prime = utils.track.apply_4x4_py(cam1_T_cam0, xyz0[perm[:min([1000,N])]])
        xyz1_actual = xyz1[perm[:min([1000,N])]]
        # N x 3
        
        # print('xyz1_prime', xyz1_prime.shape)
        # print('xyz1_actual', xyz1_prime.shape)
        # err = np.mean(np.sum(np.abs(xyz1_prime-xyz1_actual), axis=1))
        err = np.linalg.norm(xyz1_prime-xyz1_actual, axis=1)
        # utils.py.print_stats('err', err)
        # print('err', err)
        inlier = (err < inlier_thresh).astype(np.float32)
        # print('inlier', inlier)
        inlier_count = np.sum(err < inlier_thresh)
        # print('inlier_count', inlier_count)
        # input()
        rts.append(cam1_T_cam0)
        errs.append(np.mean(err))
        inliers.append(inlier_count)
    # print('errs', errs)
    # print('inliers', inliers)
    ind0 = np.argmin(errs)
    ind1 = np.argmax(inliers)
    # print('ind0=%d, err=%.3f, inliers=%d' % (ind0, errs[ind0], inliers[ind0]))
    # print('ind1=%d, err=%.3f, inliers=%d' % (ind1, errs[ind1], inliers[ind1]))
    rt0 = rts[ind0]
    rt1 = rts[ind1]
    # print('rt0', rt0)
    # print('rt1', rt1)

    cam1_T_cam0 = rt1

    if recompute_with_inliers:
        xyz1_prime = utils.track.apply_4x4_py(cam1_T_cam0, xyz0)
        # N x 3
        err = np.linalg.norm(xyz1_prime-xyz1, axis=1)
        inlier = (err < inlier_thresh).astype(np.float32)
        xyz0_inlier = xyz0[inlier > 0]
        xyz1_inlier = xyz1[inlier > 0]
        cam1_T_cam0, _ = utils.track.rigid_transform_3d_py_helper(xyz0_inlier, xyz1_inlier)
        
    cam1_T_cam0 = torch.from_numpy(cam1_T_cam0).cuda().float()
    
    return cam1_T_cam0

def get_traj_id_at_xy_t(trajs_XYZs_pix, trajs_Ts, xy, t):
    all_ids = []
    all_dists = []
    for i in range(len(trajs_XYZs_pix)):
        xys = trajs_XYZs_pix[i][:,:2]
        ts = trajs_Ts[i]

        if (t in ts) and ((t+1) in ts):
            ind0 = ts==t
            xy0 = xys[ind0]
            all_ids.append(i)
            all_dists.append(torch.norm(xy0-xy))
            # if torch.norm(xy0-xy) < 1.0:
            #     print('returning an id')
            #     return i
    # print('giving up')
    ind = torch.argmin(torch.stack(all_dists))
    id = all_ids[ind]
    return id

def get_part_traj_xy_and_id(trajs_XYs, trajs_Ts, ind0, S):
    # we want to know, for this cluster
    # for all timesteps,
    # what are all the xys
    # and what traj ids do those belong to?

    # ind0 specifies the indices of the part

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

        all_xy_s.append(all_xy)
        all_id_s.append(all_id)
    return all_xy_s, all_id_s

def get_part_outliers(c0_xy_s, c0_id_s, dist_thresh=0.1):
    
    # i want to discard the trajectories that are away from the main mass of the cluster,
    # where the "main mass" is determined by dilating the cluster
    # what we had said earlier is: "erode then dilate"
    # but a simpler method here might be:
    # for each point, find its nearest neighbors in the cluster, in each frame.
    # if on any frame, the neighbors are more than some dist away, throw away this guy.

    ids_to_discard = []

    S = len(c0_xy_s)
    for s in range(S):
        c0_xy = c0_xy_s[s]
        # this is all points from this cluster on the current timestep
        c0_id = c0_id_s[s]
        # this is all their ids

        # print('c0_xy', c0_xy.shape)
        # print('c0_id', c0_id.shape)
        
        K = c0_xy.shape[0]
        if K >= 3:
            c0_xy0_ = c0_xy.reshape(1, -1, 3)
            c0_xy1_ = c0_xy.reshape(-1, 1, 3)
            # K x K x 2
            dist_mat = torch.norm(c0_xy0_ - c0_xy1_, dim=2)
            # K x K
            # c0_id is (K,)
            sorted_dist_mat, _ = torch.sort(dist_mat, dim=1, descending=False)
            min_dist1 = sorted_dist_mat[:, 1] # K
            min_dist2 = sorted_dist_mat[:, 2]

            outlier_ids = torch.where(torch.max(min_dist1, min_dist2) > dist_thresh)[0]
            # print_('outlier_ids', outlier_ids)
            
            ids_to_discard.extend(c0_id[outlier_ids.cpu().numpy()])
        else:
            ids_to_discard.extend(c0_id)
        
        # if id not in ids_to_discard
        #     if K >= 3:
        #         for k in range(K):
        #             id = c0_id[k]
        #             dists = dist_mat[k]

        #             min_dists, _ = torch.topk(dist_mat[k], 3, largest=False)
        #             min_dist1 = min_dists[1]
        #             min_dist2 = min_dists[2]
        #             # this is the distance to the nearest neighbor
        #             if torch.max(min_dist1, min_dist2) > dist_thresh:
        #                 ids_to_discard.append(id)
        #             else:
        #                 ids_to_discard.append(id)
        #     else:
        #         ids_to_discard.extend(c0_id)

    ids_to_discard = np.unique(ids_to_discard)
                
    print('outlier ids_to_discard', ids_to_discard)
    return ids_to_discard

def get_cycle_consistent_transform_helper(
        xyz_cam0, xyz_cam1,
        flow_01,
        pix_T_cam, H, W,
        flow_valid=None,
        inlier_thresh=0.25):
    # this just does one direction

    xyz_cam1_i, valid_i = utils.geom.get_point_correspondence_from_flow(
        xyz_cam0, xyz_cam1, flow_01, pix_T_cam, H, W, flow_valid=flow_valid)
    xyz_cam0_i = xyz_cam0[valid_i>0]
    xyz_cam1_i = xyz_cam1_i[valid_i>0]
    
    cam1_T_cam0_i = get_rigid_transform(
        xyz_cam0_i, xyz_cam1_i,
        inlier_thresh=inlier_thresh,
        ransac_steps=512,
        recompute_with_inliers=True)
    # xyz_cam0_o = xyz_cam0[valid_i==0]
    # xyz_cam1_o = xyz_cam1[valid_i==0]

    corresp_tuple = (xyz_cam0_i.unsqueeze(0), xyz_cam1_i.unsqueeze(0))
    return cam1_T_cam0_i.unsqueeze(0), corresp_tuple
# (xyz_cam0_o.unsqueeze(0), xyz_cam1_o.unsqueeze(0)))

def get_cycle_consistent_transform(
        xyz_cam0, xyz_cam1,
        flow_01, flow_10,
        pix_T_cam, H, W,
        flow_01_valid=None,
        flow_10_valid=None,
        inlier_thresh=0.25):

    # forward direction
    cam1_T_cam0_fw, corresp_tuple = get_cycle_consistent_transform_helper(
        xyz_cam0, xyz_cam1,
        flow_01,
        pix_T_cam, H, W,
        flow_valid=flow_01_valid,
        inlier_thresh=inlier_thresh)
    cam0_T_cam1_bw, _ = get_cycle_consistent_transform_helper(
        xyz_cam1, xyz_cam0,
        flow_10,
        pix_T_cam, H, W,
        flow_valid=flow_10_valid,
        inlier_thresh=inlier_thresh)

    # now we want to see if these are inverses of each other
    
    # first gather the valids
    xyz_cam0 = xyz_cam0.reshape(-1, 3)
    # valid_cam0 = torch.norm(xyz_cam0, dim=1) > 1e-4
    # xyz_cam0 = xyz_cam0[xyz_cam0[:,2] > 0]
    # xyz_cam0 = xyz_cam0.unsqueeze(0)
    # # print('xyz_cam0', xyz_cam0.shape)
    # xyz_cam1 = xyz_cam1.reshape(-1, 3)
    # xyz_cam1 = xyz_cam1[xyz_cam1[:,2] > 0]
    # xyz_cam1 = xyz_cam1.unsqueeze(0)
    # print('xyz_cam1', xyz_cam1.shape)
    xyz_cam0 = xyz_cam0[torch.norm(xyz_cam0, dim=1) > 1e-4]
    xyz_cam0 = xyz_cam0.unsqueeze(0)

    cam0_T_cam0 = utils.basic.matmul2(cam0_T_cam1_bw, cam1_T_cam0_fw)
    xyz_cam0_prime = utils.geom.apply_4x4(cam0_T_cam0, xyz_cam0)

    dist = torch.norm(xyz_cam0-xyz_cam0_prime, dim=2)

    return cam1_T_cam0_fw, dist, corresp_tuple#, noncorresp_tuple
    

def get_2d_inliers(pix_T_cam,
                   xyz_cam0_i,
                   xyz_cam0_o,
                   inlier_thresh=2.0):
    # project things, and find more inliers in pixel coords
    xy_pix0_i = utils.geom.apply_pix_T_cam(pix_T_cam, xyz_cam0_i)
    xy_pix0_o = utils.geom.apply_pix_T_cam(pix_T_cam, xyz_cam0_o)
    _, N_i, _ = xy_pix0_i.shape
    _, N_o, _ = xy_pix0_o.shape
    xy_pix0_i_ = xy_pix0_i.reshape(N_i, 1, 2)
    xy_pix0_o_ = xy_pix0_o.reshape(1, N_o, 2)
    dist_mat = torch.norm(xy_pix0_i_ - xy_pix0_o_, dim=2) # N_i x N_o
    also_inlier = torch.min(dist_mat, dim=0)[0] < inlier_thresh
    xyz_cam0_i_also = xyz_cam0_o[0][also_inlier].unsqueeze(0)
    xyz_cam0_o = xyz_cam0_o[0][~also_inlier].unsqueeze(0)
    xyz_cam0_i = torch.cat([xyz_cam0_i, xyz_cam0_i_also], dim=1)
    return xyz_cam0_i, xyz_cam0_o


def get_dominant_cloud(xyz_cam0, xyz_cam1, 
                       flow_01, flow_10,
                       pix_T_cam, H, W,
                       flow_01_valid=None,
                       flow_10_valid=None,
                       inlier_thresh_3d=0.25,
                       inlier_thresh_2d=2.0,
                       corresp_thresh_3d=0.1):

    cam1_T_cam0, align_error, corresp_tuple = get_cycle_consistent_transform(
        xyz_cam0, xyz_cam1,
        flow_01, flow_10,
        pix_T_cam, H, W,
        flow_01_valid=flow_01_valid,
        flow_10_valid=flow_10_valid,
        inlier_thresh=inlier_thresh_3d)
    cam0_T_cam1 = utils.geom.safe_inverse(cam1_T_cam0)
    xyz_cam0_c, xyz_cam1_c = corresp_tuple
    # xyz_cam0_n, xyz_cam1_n = noncorresp_tuple
    # c = corresp
    # n = noncorresp

    # separate inliers and outliers
    
    xyz_cam1_stab = utils.geom.apply_4x4(cam0_T_cam1, xyz_cam1)
    xyz_cam1_c_stab = utils.geom.apply_4x4(cam0_T_cam1, xyz_cam1_c)
    # measure error in the corresponded pointclouds
    align_error_c = torch.norm(xyz_cam0_c-xyz_cam1_c_stab, dim=2)

    xyz_cam0_c_ = xyz_cam0_c.squeeze(0)
    xyz_cam1_c_ = xyz_cam1_c.squeeze(0)
    align_error_ = align_error_c.squeeze(0)

    # ok but now,
    # i don't care so much about those flow corresps
    # i want to know based on my rigid transform

    # i = inlier; o = outlier
    xyz_cam0_i = xyz_cam0_c_[align_error_ <= inlier_thresh_3d].unsqueeze(0)
    xyz_cam1_i = xyz_cam1_c_[align_error_ <= inlier_thresh_3d].unsqueeze(0)
    xyz_cam0_o = xyz_cam0_c_[align_error_ > inlier_thresh_3d].unsqueeze(0)
    xyz_cam1_o = xyz_cam1_c_[align_error_ > inlier_thresh_3d].unsqueeze(0)
    
    # # add the 2d inliers
    # if len(xyz_cam0_o[0]) > 0:
    #     xyz_cam0_i, xyz_cam0_o = get_2d_inliers(pix_T_cam, xyz_cam0_i, xyz_cam0_o,
    #                                             inlier_thresh=inlier_thresh_2d)
    # if len(xyz_cam1_o[0]) > 0:
    #     xyz_cam1_i, xyz_cam1_o = get_2d_inliers(pix_T_cam, xyz_cam1_i, xyz_cam1_o,
    #                                             inlier_thresh=inlier_thresh_2d)

    # add additional 3d inliers
    if len(xyz_cam0_o[0]) > 0 and len(xyz_cam0_i[0]) > 0:
        (xyz_cam0_i,
         xyz_cam1_i,
         xyz_cam0_o,
         xyz_cam1_o) = get_additional_3d_inliers(
             cam0_T_cam1,
             xyz_cam0_i,
             xyz_cam1_i,
             xyz_cam0_o,
             xyz_cam1_o,
             corresp_thresh=corresp_thresh_3d)
        # note that the inliers no longer correspond  
        
    return (xyz_cam0_i,
            xyz_cam1_i,
            xyz_cam0_o,
            xyz_cam1_o,
            cam0_T_cam1,
            cam1_T_cam0,
            align_error) 

def get_additional_3d_inliers(cam0_T_cam1,
                              xyz_cam0_i,
                              xyz_cam1_i,
                              xyz_cam0_o,
                              xyz_cam1_o,
                              corresp_thresh=0.1):
    # what i'm picturing here is:
    # there are points in the outlier set that are perfectly statisfied by the rigid warp,
    # even though the flow field does not quite correspond them

    xyz_cam0 = xyz_cam0_o.clone()
    xyz_cam1 = utils.geom.apply_4x4(cam0_T_cam1, xyz_cam1_o)
    
    _, N_0, _ = xyz_cam0.shape
    _, N_1, _ = xyz_cam1.shape
    
    xyz_cam0_ = xyz_cam0.reshape(N_0, 1, 3)
    xyz_cam1_ = xyz_cam1.reshape(1, N_1, 3)
    
    dist_mat = torch.norm(xyz_cam0_ - xyz_cam1_, dim=2) # N_0 x N_1
    
    corresp0 = torch.min(dist_mat, dim=0)[0] < corresp_thresh
    # these are the cam0 points that appear to have correspondences in cam1 under this transform
    corresp1 = torch.min(dist_mat, dim=1)[0] < corresp_thresh
    # these are the cam1 points that appear to have correspondences in cam0 under this transform

    # print_('additional rigid inliers0', torch.sum(corresp0))

    xyz_cam0_i_also = xyz_cam0_o[0][corresp0].unsqueeze(0)
    xyz_cam1_i_also = xyz_cam1_o[0][corresp1].unsqueeze(0)
    xyz_cam0_o = xyz_cam0_o[0][~corresp0].unsqueeze(0)
    xyz_cam1_o = xyz_cam1_o[0][~corresp1].unsqueeze(0)
    xyz_cam0_i = torch.cat([xyz_cam0_i, xyz_cam0_i_also], dim=1)
    xyz_cam1_i = torch.cat([xyz_cam1_i, xyz_cam1_i_also], dim=1)

    # let's also add points that are close spatial neighbors of the inliers

    N0_i = xyz_cam0_i.shape[1]
    N0_o = xyz_cam0_o.shape[1]
    N1_i = xyz_cam1_i.shape[1]
    N1_o = xyz_cam1_o.shape[1]
    xyz_cam0_i_ = xyz_cam0_i.reshape(N0_i, 1, 3)
    xyz_cam0_o_ = xyz_cam0_o.reshape(1, N0_o, 3)
    xyz_cam1_i_ = xyz_cam1_i.reshape(N1_i, 1, 3)
    xyz_cam1_o_ = xyz_cam1_o.reshape(1, N1_o, 3)

    dist_mat0 = torch.norm(xyz_cam0_i_ - xyz_cam0_o_, dim=2) # N0_i x N0_o
    dist_mat1 = torch.norm(xyz_cam1_i_ - xyz_cam1_o_, dim=2) # N1_i x N1_o
    corresp0 = torch.min(dist_mat0, dim=0)[0] < corresp_thresh
    corresp1 = torch.min(dist_mat1, dim=0)[0] < corresp_thresh
    # these are the outlier points that are pretty close to an inlier

    # print_('additional close inliers0', torch.sum(corresp0))
    
    xyz_cam0_i_also = xyz_cam0_o[0][corresp0].unsqueeze(0)
    xyz_cam1_i_also = xyz_cam1_o[0][corresp1].unsqueeze(0)
    xyz_cam0_o = xyz_cam0_o[0][~corresp0].unsqueeze(0)
    xyz_cam1_o = xyz_cam1_o[0][~corresp1].unsqueeze(0)
    xyz_cam0_i = torch.cat([xyz_cam0_i, xyz_cam0_i_also], dim=1)
    xyz_cam1_i = torch.cat([xyz_cam1_i, xyz_cam1_i_also], dim=1)
    
    return (xyz_cam0_i,
            xyz_cam1_i,
            xyz_cam0_o,
            xyz_cam1_o)

def get_3d_inliers(pix_T_cam,
                   cam0_T_cam1,
                   xyz_cam0,
                   xyz_cam1,
                   corresp_thresh=0.1):
    # what i'm picturing here is:
    # there are points in the outlier set that are perfectly statisfied by the rigid warp,
    # even though the flow field does not quite correspond them
    
    B, N_0, _ = xyz_cam0.shape
    _, N_1, _ = xyz_cam1.shape

    xyz_cam0_ = xyz_cam0.clone()
    xyz_cam1_ = utils.geom.apply_4x4(cam0_T_cam1, xyz_cam1)
    
    have_infinite_memory = False
    if have_infinite_memory:
        xyz_cam0_ = xyz_cam0_.reshape(N_0, 1, 3)
        xyz_cam1_ = xyz_cam1_.reshape(1, N_1, 3)
        
        dist_mat = torch.norm(xyz_cam0_ - xyz_cam1_, dim=2) # N_0 x N_1
        corresp0 = torch.min(dist_mat, dim=1)[0] < corresp_thresh
        # these are the cam0 points that appear to have a correspondence in cam1 under this transform
        corresp1 = torch.min(dist_mat, dim=0)[0] < corresp_thresh
        # these are the cam1 points that appear to have a correspondence in cam0 under this transform
    else:
        # let's solve this with a kd tree
        assert(B==1)
        xyz_cam0_py = xyz_cam0_[0].detach().cpu().numpy()
        xyz_cam1_py = xyz_cam1_[0].detach().cpu().numpy()
        
        kdt0 = scipy.spatial.KDTree(xyz_cam0_py)
        print('got kdt0')
        kdt1 = scipy.spatial.KDTree(xyz_cam1_py)
        print('got kdt1')
        # d, i = kdt.query(dat, k=2)

        # these distances and indices are anchored to the data
        d0, i0 = kdt0.query(xyz_cam1_py, k=1, distance_upper_bound=corresp_thresh)
        d1, i1 = kdt1.query(xyz_cam0_py, k=1, distance_upper_bound=corresp_thresh)
        # print('got queries0')
        print('got queries0', d0.shape, i1.shape)
        print('got queries1', d1.shape, i0.shape)
        i0[i0==N_0] = 0
        i1[i1==N_1] = 0
        
        # corresp0 = get_vox_corresp(xyz_cam0_, xyz_cam1_, corresp_thresh)
        # corresp1 = get_vox_corresp(xyz_cam1_, xyz_cam0_, corresp_thresh)

        # corresp1 = i1[d1 < corresp_thresh]
        # corresp0 = i1[d1 < corresp_thresh]
        
        # corresp0 = torch.from_numpy(d0 < corresp_thresh)
        # corresp1 = torch.from_numpy(d1 < corresp_thresh)
        
        # corresp1 = get_vox_corresp(xyz_cam1_, xyz_cam0_, corresp_thresh)

        # but actually, i only want the ones that are nearest neighbors of each other
        # this way, we can't have back of car and front of car both matching to middle
        # and also, maybe i can use a wider distance this way

        # i1 is the index in xyz1 of the nearest neighbor of xyz0
        # i0 is the index in xyz0 of the nearest neighbor of xyz1
        # i want to make sure the guys are neighbors of each other


        # in other words, if i1[m] == n, i would like i0[n] == m
        # so i want i1 == i0[i1]

        # in other words, if
        # if i1[m] == n, i would like i0[n] == m
        # if i1[m] == n, i would like i0[n] == m


        # 

        # if i1 == i0[i1]
        # corresp0 = torch.from_numpy((d0 < corresp_thresh) & (i1[i0] == i0))#[i1]))
        # corresp1 = torch.from_numpy((d1 < corresp_thresh) & (i0[i1] == i1))#[i0]))
        # corresp0 = torch.from_numpy((d0 < corresp_thresh) & (i1[i0] == i0[i1[i0]])
        # corresp1 = torch.from_numpy((d1 < corresp_thresh) & (i0[i1] == i1))#[i0]))
        # corresp0 = torch.from_numpy((d0 < corresp_thresh) & (i1 == i0[i1]))
        # corresp1 = torch.from_numpy((d1 < corresp_thresh) & (i0 == i1[i0]))
        # corresp0 = torch.from_numpy((d0 < corresp_thresh) & (i0 == i1[i0]))
        # corresp1 = torch.from_numpy((d1 < corresp_thresh) & (i1 == i0[i1]))
        # corresp0 = torch.from_numpy(d0 < corresp_thresh)
        # corresp1 = torch.from_numpy(d1 < corresp_thresh)
        
        corresp0 = torch.from_numpy(d1 < corresp_thresh)
        corresp1 = torch.from_numpy(d0 < corresp_thresh)

        
    # else:
    #     # let's solve this by voxelizing
    #     assert(B==1)
    #     corresp0 = get_vox_corresp(xyz_cam0_, xyz_cam1_, corresp_thresh)
    #     corresp1 = get_vox_corresp(xyz_cam1_, xyz_cam0_, corresp_thresh)

    corresp0 = corresp0.reshape(-1)
    corresp1 = corresp1.reshape(-1)

    # index into the unstabilized 
    xyz_cam0_i = xyz_cam0[0][corresp0].unsqueeze(0)
    xyz_cam0_o = xyz_cam0[0][~corresp0].unsqueeze(0)
    xyz_cam1_i = xyz_cam1[0][corresp1].unsqueeze(0)
    xyz_cam1_o = xyz_cam1[0][~corresp1].unsqueeze(0)
    
    return (xyz_cam0_i,
            xyz_cam1_i,
            xyz_cam0_o,
            xyz_cam1_o)

def get_vox_corresp(xyz_cam0, xyz_cam1, corresp_thresh):
    # voxelize xyz_cam1, and then for each xyz_cam0, see if it lands in an occupied voxel
    B, N_0, _ = xyz_cam0.shape
    xyz_all = torch.cat([xyz_cam0, xyz_cam1], dim=1)
    centroid = torch.mean(xyz_all, dim=1)
    xmin = torch.min(xyz_all[0,:,0]).item()
    ymin = torch.min(xyz_all[0,:,1]).item()
    zmin = torch.min(xyz_all[0,:,2]).item()
    xmax = torch.max(xyz_all[0,:,0]).item()
    ymax = torch.max(xyz_all[0,:,1]).item()
    zmax = torch.max(xyz_all[0,:,2]).item()
    bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
    Z = int((zmax-zmin)/corresp_thresh)
    Y = int((ymax-ymin)/corresp_thresh)
    X = int((xmax-xmin)/corresp_thresh)
    print('creating vox util with resolution %d, %d, %d' % (Z, Y, X))
    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=centroid,
        bounds=bounds,
        assert_cube=False)
    print('vox sizes:', vox_util.default_vox_size_Z, vox_util.default_vox_size_Y, vox_util.default_vox_size_X)
    occ_mem1 = vox_util.voxelize_xyz(xyz_cam1, Z, Y, X, assert_cube=False)
    xyz_mem0 = vox_util.Ref2Mem(xyz_cam0, Z, Y, X, assert_cube=False)

    corresp0 = utils.samp.trilinear_sample3d(occ_mem1, xyz_mem0)
    print('corresp0', corresp0.shape)
    corresp0 = corresp0 > 0
    print_('corresp0 sum', torch.sum(corresp0))
    
    # print('xyz_cam0', xyz_cam0.shape)
    # print('xyz_mem0', xyz_mem0.shape)
    # x = xyz_mem0[:,:,0].clamp(0, X-1).long().reshape(-1)
    # y = xyz_mem0[:,:,1].clamp(0, Y-1).long().reshape(-1)
    # z = xyz_mem0[:,:,2].clamp(0, Z-1).long().reshape(-1)
    # corresp0 = occ_mem1[0,0,z,y,x] > 0
    # print('corresp0', corresp0.shape)
    # corresp0 = corresp0.reshape(-1)
    # print_('corresp0 sum', torch.sum(corresp0))
    return corresp0

