import utils.geom
import torch
import numpy as np
from cmaes import CMA

def initialize_lrtlist_with_lrt(initial_lrt, S):
    l, cam_T_obj = utils.geom.split_lrt(initial_lrt)
    r, t = utils.geom.split_rt(cam_T_obj)
    llist = l.unsqueeze(1).repeat(1, S, 1)
    rlist = r.unsqueeze(1).repeat(1, S, 1, 1)
    tlist = t.unsqueeze(1).repeat(1, S, 1)
    rtlist = utils.geom.merge_rtlist(rlist, tlist)
    lrtlist = utils.geom.merge_lrtlist(llist, rtlist)
    return lrtlist

def update_lrtlist_with_tlist(lrtlist, tlist_new):
    lenlist, rtlist = utils.geom.split_lrtlist(lrtlist)
    rlist, tlist_old = utils.geom.split_rtlist(rtlist)
    rtlist = utils.geom.merge_rtlist(rlist, tlist_new)
    lrtlist = utils.geom.merge_lrtlist(lenlist, rtlist)
    return lrtlist

def update_lrtlist_with_trlist(lrtlist, trlist_new):
    B, S, D = lrtlist.shape
    for s in range(S):
        lrtlist[:,s] = update_lrt_with_tr(lrtlist[:,s], trlist_new[:,s])
    return lrtlist

def update_lrt_with_tr(lrt, tr_new):
    l, rt = utils.geom.split_lrt(lrt)
    r_old, t_old = utils.geom.split_rt(rt)

    t_new = tr_new[:,:3]
    rx = tr_new[:,3]
    ry = tr_new[:,4]
    rz = tr_new[:,5]
    r_new = utils.geom.eul2rotm(rx, ry, rz)
    rt_new = utils.geom.merge_rt(r_new, t_new)
    lrt_new = utils.geom.merge_lrt(l, rt_new)
    return lrt_new

def render_one_object_3d(xyz_obj, lrt_camRs, vox_util, Z1, Y1, X1, Z2, Y2, X2):
    B, N, C = xyz_obj.shape
    assert(C==3)
    _, S, D = lrt_camRs.shape
    assert(D==19)
    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)
    
    # now that i have some boxes, i want to use them to render some new scenes
    camRs_T_objs = __u(vox_util.get_ref_T_zoom(__p(lrt_camRs), Z2, Y2, X2))
    xyz_objs = xyz_obj.unsqueeze(1).repeat(1, S, 1, 1) # B, S, N, 3
    xyz_camRs = __u(utils.geom.apply_4x4(__p(camRs_T_objs), __p(xyz_objs)))
    occ_memRs = __u(vox_util.voxelize_xyz(__p(xyz_camRs), Z1, Y1, X1))
    return occ_memRs

# def render_one_object_2d(xyz_obj, lrt_camXs, pix_T_cam, H, W):
#     B, N, C = xyz_obj.shape
#     assert(C==3)
#     _, S, D = lrt_camXs.shape
#     assert(D==19)
#     __p = lambda x: utils.basic.pack_seqdim(x, B)
#     __u = lambda x: utils.basic.unpack_seqdim(x, B)

#     # now that i have some boxes, i want to use them to render some new scenes
#     camXs_T_objs = __u(vox_util.get_ref_T_zoom(__p(lrt_camXs), Z2, Y2, X2))
#     xyz_objs = xyz_obj.unsqueeze(1).repeat(1, S, 1, 1) # B, S, N, 3
#     xyz_camXs = __u(utils.geom.apply_4x4(__p(camXs_T_objs), __p(xyz_objs)))
#     depth_camXs_, _ = utils.geom.create_depth_image(pix_T_cam, __p(xyz_camXs), H, W)
#     depth_camXs = __u(depth_camXs_)
#     return depth_camXs

def render_one_object_2d(xyz_obj, lrt_camX, pix_T_cam, H, W, max_dist):
    B, N, C = xyz_obj.shape
    assert(C==3)
    _, D = lrt_camX.shape
    assert(D==19)

    # now that i have some boxes, i want to use them to render some new scenes
    _, camX_T_obj = utils.geom.split_lrt(lrt_camX)
    xyz_camX = utils.geom.apply_4x4(camX_T_obj, xyz_obj)
    depth_camX, valid_camX = utils.geom.create_depth_image(pix_T_cam, xyz_camX, H, W, serial=True, slices=30, max_val=max_dist)
    return depth_camX, valid_camX

def render_one_object_with_feat_2d(xyz_obj, feat, lrt_camX, pix_T_cam, H, W, max_dist):
    B, N, C = xyz_obj.shape
    assert(C==3)
    _, D = lrt_camX.shape
    assert(D==19)

    # now that i have some boxes, i want to use them to render some new scenes
    _, camX_T_obj = utils.geom.split_lrt(lrt_camX)
    xyz_camX = utils.geom.apply_4x4(camX_T_obj, xyz_obj)
    depth_camX, valid_camX = utils.geom.create_depth_image(pix_T_cam, xyz_camX, H, W, serial=True, slices=30, max_val=max_dist)
    feat_camX, _ = utils.geom.create_feat_image(pix_T_cam, xyz_camX, feat, H, W, serial=True, slices=30, max_val=max_dist)
    return depth_camX, feat_camX, valid_camX

def create_featimg_from_depth(obj_depth_cam, arm_depth_cam, bkg_depth_cam, obj_featimg_cam, arm_featimg_cam, bkg_featimg_cam):
    full_depth_cam = torch.min(torch.cat([obj_depth_cam, arm_depth_cam, bkg_depth_cam], dim=1), dim=1, keepdim=True)[0]
    obj_mask = (obj_depth_cam == full_depth_cam).float()
    arm_mask = (obj_mask == 0) * (arm_depth_cam == full_depth_cam).float()
    bkg_mask = 1 - (obj_mask + arm_mask)
    full_featimg_cam = obj_mask * obj_featimg_cam + arm_mask * arm_featimg_cam + bkg_mask * bkg_featimg_cam
    return full_featimg_cam

def go_optim(
        xyz_obj,
        lrt_camX,
        camRs_T_camXs,
        rgb_camXs,
        xyz_camX_list,
        pix_T_cam,
        vox_util,
        sw,
        Z2, Y2, X2,
        max_dist=16,
        sigma=0.1,
        t_only=True,
        population_size=5,
        max_stale=10,
        max_generations=80):

    B, S, C, H, W = rgb_camXs.shape
    assert(B==1)
    assert(D==3)

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)
    
    # assume cam is not moving, and we only need the one mat
    camR_T_camX = camRs_T_camXs[:,0] 
    camX_T_camR = utils.geom.safe_inverse(camR_T_camX)
    
    lrt_camXs = initialize_lrtlist_with_lrt(lrt_camX, S)
    lrt_camRs = utils.geom.apply_4x4_to_lrtlist(camR_T_camX, lrt_camXs)

    _, rt_camRs = utils.geom.split_lrtlist(lrt_camRs)
    r_camRs, t_camRs = utils.geom.split_rtlist(rt_camRs)
    rx_, ry_, rz_ = utils.geom.rotm2eul(__p(r_camRs))
    tr_camRs = torch.cat([t_camRs.reshape(1, S, 3),
                          rx_.reshape(1, S, 1),
                          ry_.reshape(1, S, 1),
                          rz_.reshape(1, S, 1)], dim=2)

    if False:
        vis = []
        for s in range(S):
            vis.append(sw.summ_lrtlist(
                '',
                rgb_camXs[:,s],
                lrt_camXs[:,s:s+1],
                torch.ones_like(lrt_camXs[:,s:s+1,0]),
                torch.ones_like(lrt_camXs[:,s:s+1,0]).long(),
                pix_T_cam,
                frame_id=s,
                only_return=True)
            )
        sw.summ_rgbs('1_proposals/initial_traj', vis)

    # set up occ/freespace supervision
    xyz_camRs = __u(utils.geom.apply_4x4(__p(camRs_T_camXs), __p(xyz_camXs)))
    occ_memRs = __u(vox_util.voxelize_xyz(__p(xyz_camRs), Z2, Y2, X2))
    
    vis_memRs = __u(vox_util.convert_xyz_to_visibility(__p(xyz_camXs), Z2, Y2, X2, target_T_given=__p(camRs_T_camXs)))
    free_memRs = vis_memRs * (1.0 - occ_memRs)
    
    sw.summ_occs('4_optim/occ_memRs', occ_memRs.unbind(1), bev=True)
    # sw.summ_occs('4_optim/free_memRs', free_memRs.unbind(1), bev=True)
    
    print('optimizing traj')
    best_t_gens = []
    min_values = []
    min_values.append(0)
    for s in range(1,S):
        print('working on s=%d' % s)

        prev_tr_camR = tr_camRs[:,s-1]
        prev_tr_py = prev_tr_camR.cpu().reshape(-1).numpy() # 6

        if t_only:
            delta_py = np.zeros_like(prev_tr_py[:3])
            bounds = np.array([
                -1, 1,
                -1, 1,
                -1, 1,
            ]).reshape(-1, 2)
        else:
            delta_py = np.zeros_like(prev_tr_py)
            bounds = np.array([
                -1, 1,
                -1, 1,
                -1, 1,
                -np.pi/4, np.pi/4,
                -np.pi/4, np.pi/4,
                -np.pi/4, np.pi/4,
            ]).reshape(-1, 2)
        optimizer = CMA(mean=delta_py, sigma=sigma, population_size=population_size, bounds=bounds)
        
        min_value = np.inf
        best_delta_py = delta_py.copy()

        stale_count = 0
        generation_count = 0
        
        while stale_count < max_stale and generation_count < max_generations:
            generation_count += 1
            improved_things = False
            solutions = []

            for pi in range(optimizer.population_size):

                if pi==optimizer.population_size-1:
                    # explicitly consider a static sol
                    delta_py = delta_py * 0
                else:
                    delta_py = optimizer.ask()
                    
                if t_only:
                    delta_py_ = np.concatenate([delta_py, delta_py*0])
                    tr_py = prev_tr_py + delta_py_
                else:
                    tr_py = prev_tr_py + delta_py
                    
                tr_camR = torch.from_numpy(tr_py).reshape(1, 6).float().to('cuda')
                lrt_camR = update_lrt_with_tr(lrt_camRs[:,s], tr_camR)
                obj_occ_memR_ = render_one_object_3d(xyz_obj, lrt_camR.unsqueeze(1), vox_util, Z2, Y2, X2, Z2, Y2, X2)

                occ_loss = torch.abs(obj_occ_memR_ - occ_memRs[:,s:s+1])
                occ_value = torch.sum(occ_loss).detach().cpu().numpy()

                value = occ_value

                if value < min_value:
                    min_value = value
                    best_delta_py = delta_py.copy()
                    improved_things = True
                solutions.append((delta_py, value))

            if improved_things:
                stale_count = 0
            else:
                stale_count += 1
            print(f"{generation_count} {stale_count} {value} {min_value}")
            optimizer.tell(solutions)
        min_values.append(min_value)
        if t_only:
            best_delta_py_ = np.concatenate([best_delta_py, best_delta_py*0])
            tr_camRs[:,s] = prev_tr_camR + torch.from_numpy(best_delta_py_).reshape(1, 6).float().to('cuda')
        else:
            tr_camRs[:,s] = prev_tr_camR + torch.from_numpy(best_delta_py).reshape(1, 6).float().to('cuda')
        print('best_delta_py', best_delta_py)
    
    lrt_camRs = update_lrtlist_with_trlist(lrt_camRs, tr_camRs)
    obj_occ_memRs = render_one_object_3d(xyz_obj, lrt_camRs, vox_util, Z2, Y2, X2, Z2, Y2, X2)
    sw.summ_occs('4_optim/obj_occ_memRs', obj_occ_memRs.unbind(1), bev=True)
    
    lrt_camXs = utils.geom.apply_4x4_to_lrtlist(camX_T_camR, lrt_camRs)
    vis = []
    for s in range(S):
        vis.append(sw.summ_lrtlist(
            '',
            rgb_camXs[:,s],
            lrt_camXs[:,s:s+1],
            torch.ones_like(lrt_camXs[:,s:s+1,0]),
            torch.ones_like(lrt_camXs[:,s:s+1,0]).long(),
            pix_T_cam,
            # # frame_id=s,
            # frame_id=np.mean(min_values),
            frame_id=min_values[s],
            only_return=True)
        )
    sw.summ_rgbs('4_optim/optim_traj', vis)

    return True


def depth_optim_free(
        xyz_camXs,
        rgb_camXs,
        arm_camXs, 
        lrt_camX,
        obj_xyz_camX,
        bkg_xyz_camX, 
        pix_T_cam,
        H, W,
        sw,
        max_dist=16,
        sigma=0.1,
        t_only=True,
        population_size=5,
        max_stale=10,
        max_generations=80):

    B, S, N, D = xyz_camXs.shape
    assert(B==1)
    assert(D==3)

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    # let's first see what it looks like if i render the given pointcloud

    # keep_camXs = 1.0 - ignore_camXs

    full_xyz_camX = torch.cat([obj_xyz_camX,
                               bkg_xyz_camX], dim=1)
    depth_camX, _ = utils.geom.create_depth_image(pix_T_cam, full_xyz_camX, H, W, serial=True, slices=30) # B x 1 x H x W
    bkg_depth_camX, bkg_valid_camX = utils.geom.create_depth_image(pix_T_cam, bkg_xyz_camX, H, W, serial=True, slices=30) # B x 1 x H x W

    sw.summ_oned('8_optim/initial_depth_camX', depth_camX, max_val=max_dist, frame_id=0)

    sw.summ_oned('8_optim/bkg_depth_camX', bkg_depth_camX, max_val=max_dist)
    # return False
    
    lrt_camXs = initialize_lrtlist_with_lrt(lrt_camX, S)
    
    _, rt_camXs = utils.geom.split_lrtlist(lrt_camXs)
    r_camXs, t_camXs = utils.geom.split_rtlist(rt_camXs)
    rx_, ry_, rz_ = utils.geom.rotm2eul(__p(r_camXs))
    tr_camXs = torch.cat([t_camXs.reshape(1, S, 3),
                          rx_.reshape(1, S, 1),
                          ry_.reshape(1, S, 1),
                          rz_.reshape(1, S, 1)], dim=2)

    lens, camX_T_obj = utils.geom.split_lrt(lrt_camX)
    obj_T_camX = camX_T_obj.inverse()
    obj_xyz_obj = utils.geom.apply_4x4(obj_T_camX, obj_xyz_camX)

    full_depth_camXs_g = []
    arm_depth_camXs_g = []
    for s in range(S):
        full_depth_camX, _ = utils.geom.create_depth_image(pix_T_cam, xyz_camXs[:,s], H, W) # this is gt singleview depth; does not need serial
        arm_depth_camX = full_depth_camX.clone()
        arm_depth_camX[arm_camXs[:,s] < 1] = max_dist
        
        full_depth_camXs_g.append(full_depth_camX)
        arm_depth_camXs_g.append(arm_depth_camX)
    full_depth_camXs_g = torch.stack(full_depth_camXs_g, dim=1)
    arm_depth_camXs_g = torch.stack(arm_depth_camXs_g, dim=1)
    
    sw.summ_oneds('8_optim/arm_depth_camXs', arm_depth_camXs_g.unbind(1), max_val=max_dist)
    sw.summ_oneds('8_optim/full_depth_camXs_g', full_depth_camXs_g.unbind(1), max_val=max_dist)
    
    print('optimizing traj')
    best_t_gens = []
    min_values = []

    min_values.append(0)
    for s in range(1,S):
        print('working on s=%d' % s)

        # full_depth_camX_g, _ = utils.geom.create_depth_image(pix_T_cam, xyz_camXs[:,s], H, W)
        # arm_depth_camX = full_depth_camX_g * arm_camXs[:,s]

        prev_tr_camX = tr_camXs[:,s-1]
        prev_tr_py = prev_tr_camX.cpu().reshape(-1).numpy() # 6

        if t_only:
            delta_py = np.zeros_like(prev_tr_py[:3])
            bounds = np.array([
                -1, 1,
                -1, 1,
                -1, 1,
            ]).reshape(-1, 2)
        else:
            delta_py = np.zeros_like(prev_tr_py)
            bounds = np.array([
                -1, 1,
                -1, 1,
                -1, 1,
                -np.pi/4, np.pi/4,
                -np.pi/4, np.pi/4,
                -np.pi/4, np.pi/4,
            ]).reshape(-1, 2)
        optimizer = CMA(mean=delta_py, sigma=sigma, population_size=population_size, bounds=bounds)
        
        min_value = np.inf
        best_delta_py = delta_py.copy()

        stale_count = 0
        generation_count = 0
        
        while stale_count < max_stale and generation_count < max_generations:
            generation_count += 1
            improved_things = False
            solutions = []

            for pi in range(optimizer.population_size):

                if pi==optimizer.population_size-1:
                    # explicitly consider a static sol
                    delta_py = delta_py * 0
                else:
                    delta_py = optimizer.ask()
                    
                if t_only:
                    delta_py_ = np.concatenate([delta_py, delta_py*0])
                    tr_py = prev_tr_py + delta_py_
                else:
                    tr_py = prev_tr_py + delta_py
                    
                tr_camX = torch.from_numpy(tr_py).reshape(1, 6).float().to('cuda')
                lrt_camX = update_lrt_with_tr(lrt_camXs[:,s], tr_camX)
                obj_depth_camX, obj_valid_camX = render_one_object_2d(obj_xyz_obj, lrt_camX, pix_T_cam, H, W)


                full_valid_camX = (bkg_valid_camX + obj_valid_camX).clamp(0,1)# * keep_camXs[:,s]
                full_depth_camX_e = torch.min(torch.cat([obj_depth_camX,
                                                         arm_depth_camXs_g[:,s],
                                                         bkg_depth_camX], dim=1), dim=1, keepdim=True)[0]
                depth_loss = torch.abs(full_depth_camX_e - full_depth_camXs_g[:,s])# * keep_camXs[:,s] #* full_valid_camX
                value = torch.sum(depth_loss).detach().cpu().numpy()


                if value < min_value:
                    min_value = value
                    best_delta_py = delta_py.copy()
                    improved_things = True


                solutions.append((delta_py, value))

            if improved_things:
                stale_count = 0
            else:
                stale_count += 1
            print(f"{generation_count} {stale_count} {value} {min_value}")
            optimizer.tell(solutions)
        min_values.append(min_value)
        if t_only:
            best_delta_py_ = np.concatenate([best_delta_py, best_delta_py*0])
            tr_camXs[:,s] = prev_tr_camX + torch.from_numpy(best_delta_py_).reshape(1, 6).float().to('cuda')
        else:
            tr_camXs[:,s] = prev_tr_camX + torch.from_numpy(best_delta_py).reshape(1, 6).float().to('cuda')
        print('best_delta_py', best_delta_py)
    
    lrt_camXs = update_lrtlist_with_trlist(lrt_camXs, tr_camXs)

    full_depth_camXs_e = []
    for s in range(S):
        obj_depth_camX, _ = render_one_object_2d(obj_xyz_obj, lrt_camXs[:,s], pix_T_cam, H, W)
        full_depth_camX_e = torch.min(torch.cat([obj_depth_camX,
                                                 arm_depth_camXs_g[:,s],
                                                 bkg_depth_camX], dim=1), dim=1, keepdim=True)[0]
        full_depth_camXs_e.append(full_depth_camX_e)
    full_depth_camXs_e = torch.stack(full_depth_camXs_e, dim=1)
    # print('full_depth_camXs', full_depth_camXs.shape)
    sw.summ_oneds('8_optim/full_depth_camXs_e', full_depth_camXs_e.unbind(1), max_val=max_dist)
    sw.summ_oneds('8_optim/diff_depth_camXs_e', (full_depth_camXs_e - full_depth_camXs_g).unbind(1))
    
    vis = []
    for s in range(S):
        vis.append(sw.summ_lrtlist(
            '',
            rgb_camXs[:,s],
            lrt_camXs[:,s:s+1],
            torch.ones_like(lrt_camXs[:,s:s+1,0]),
            torch.ones_like(lrt_camXs[:,s:s+1,0]).long(),
            pix_T_cam,
            # # frame_id=s,
            # frame_id=np.mean(min_values),
            frame_id=min_values[s],
            only_return=True)
        )
    sw.summ_rgbs('8_optim/optim_traj', vis)

    return lrt_camXs


def depth_optim_prismatic(
        direction_camX,
        xyz_camXs,
        rgb_camXs,
        arm_camXs, 
        lrt_camX,
        obj_xyz_camX,
        bkg_xyz_camX, 
        pix_T_cam,
        H, W,
        sw,
        max_dist=16,
        sigma=0.1,
        t_only=True,
        population_size=5,
        max_stale=10,
        max_generations=80):

    B, S, N, D = xyz_camXs.shape
    assert(B==1)
    assert(D==3)

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    # let's first see what it looks like if i render the given pointcloud

    # keep_camXs = 1.0 - ignore_camXs

    full_xyz_camX = torch.cat([obj_xyz_camX,
                               bkg_xyz_camX], dim=1)
    depth_camX, _ = utils.geom.create_depth_image(pix_T_cam, full_xyz_camX, H, W, serial=True, slices=30) # B x 1 x H x W
    bkg_depth_camX, bkg_valid_camX = utils.geom.create_depth_image(pix_T_cam, bkg_xyz_camX, H, W, serial=True, slices=30) # B x 1 x H x W

    sw.summ_oned('9_joint/initial_depth_camX', depth_camX, max_val=max_dist, frame_id=0)

    sw.summ_oned('9_joint/bkg_depth_camX', bkg_depth_camX, max_val=max_dist)
    # return False
    
    lrt_camXs = initialize_lrtlist_with_lrt(lrt_camX, S)
    
    _, rt_camXs = utils.geom.split_lrtlist(lrt_camXs)
    r_camXs, t_camXs = utils.geom.split_rtlist(rt_camXs)
    rx_, ry_, rz_ = utils.geom.rotm2eul(__p(r_camXs))
    tr_camXs = torch.cat([t_camXs.reshape(1, S, 3),
                          rx_.reshape(1, S, 1),
                          ry_.reshape(1, S, 1),
                          rz_.reshape(1, S, 1)], dim=2)

    lens, camX_T_obj = utils.geom.split_lrt(lrt_camX)
    obj_T_camX = camX_T_obj.inverse()
    obj_xyz_obj = utils.geom.apply_4x4(obj_T_camX, obj_xyz_camX)

    full_depth_camXs_g = []
    arm_depth_camXs_g = []
    for s in range(S):
        full_depth_camX, _ = utils.geom.create_depth_image(pix_T_cam, xyz_camXs[:,s], H, W) # gt singleview depth; does not need serial
        arm_depth_camX = full_depth_camX.clone()
        arm_depth_camX[arm_camXs[:,s] < 1] = max_dist
        
        full_depth_camXs_g.append(full_depth_camX)
        arm_depth_camXs_g.append(arm_depth_camX)
    full_depth_camXs_g = torch.stack(full_depth_camXs_g, dim=1)
    arm_depth_camXs_g = torch.stack(arm_depth_camXs_g, dim=1)
    
    sw.summ_oneds('9_joint/arm_depth_camXs', arm_depth_camXs_g.unbind(1), max_val=max_dist)
    sw.summ_oneds('9_joint/full_depth_camXs_g', full_depth_camXs_g.unbind(1), max_val=max_dist)

    direction_py = direction_camX[0].cpu().numpy()
    
    print('optimizing prismatic traj with direction', direction_py)
    best_t_gens = []
    min_values = []

    min_values.append(0)
    for s in range(1,S):
        print('working on s=%d' % s)

        prev_tr_camX = tr_camXs[:,s-1]
        prev_tr_py = prev_tr_camX.cpu().reshape(-1).numpy() # 6

        amount_py = np.zeros_like(prev_tr_py[0:2])
        bounds = np.array([
            -1, 1,
            -1, 1,
        ]).reshape(-1, 2)
        optimizer = CMA(mean=amount_py, sigma=sigma, population_size=population_size, bounds=bounds)
        
        min_value = np.inf
        best_amount_py = amount_py.copy()

        stale_count = 0
        generation_count = 0
        
        while stale_count < max_stale and generation_count < max_generations:
            generation_count += 1
            improved_things = False
            solutions = []

            for pi in range(optimizer.population_size):

                if pi==optimizer.population_size-1:
                    # explicitly consider a static sol
                    amount_py = amount_py * 0
                else:
                    amount_py = optimizer.ask()
                    
                delta_py_ = np.concatenate([amount_py[0:1]*direction_py, direction_py*0])
                tr_py = prev_tr_py + delta_py_
                    
                tr_camX = torch.from_numpy(tr_py).reshape(1, 6).float().to('cuda')
                lrt_camX = update_lrt_with_tr(lrt_camXs[:,s], tr_camX)
                obj_depth_camX, obj_valid_camX = render_one_object_2d(obj_xyz_obj, lrt_camX, pix_T_cam, H, W)

                full_valid_camX = (bkg_valid_camX + obj_valid_camX).clamp(0,1)# * keep_camXs[:,s]
                full_depth_camX_e = torch.min(torch.cat([obj_depth_camX,
                                                         arm_depth_camXs_g[:,s],
                                                         bkg_depth_camX], dim=1), dim=1, keepdim=True)[0]
                depth_loss = torch.abs(full_depth_camX_e - full_depth_camXs_g[:,s])# * keep_camXs[:,s] #* full_valid_camX
                value = torch.sum(depth_loss).detach().cpu().numpy()

                if value < min_value:
                    min_value = value
                    best_amount_py = amount_py.copy()
                    improved_things = True


                solutions.append((amount_py, value))

            if improved_things:
                stale_count = 0
            else:
                stale_count += 1
            print(f"{generation_count} {stale_count} {value} {min_value}")
            optimizer.tell(solutions)
        min_values.append(min_value)
        
        best_delta_py_ = np.concatenate([best_amount_py[0:1]*direction_py, direction_py*0])
        tr_camXs[:,s] = prev_tr_camX + torch.from_numpy(best_delta_py_).reshape(1, 6).float().to('cuda')
        print('best_amount_py', best_amount_py)
    
    lrt_camXs = update_lrtlist_with_trlist(lrt_camXs, tr_camXs)

    full_depth_camXs_e = []
    for s in range(S):
        obj_depth_camX, _ = render_one_object_2d(obj_xyz_obj, lrt_camXs[:,s], pix_T_cam, H, W)
        full_depth_camX_e = torch.min(torch.cat([obj_depth_camX,
                                                 arm_depth_camXs_g[:,s],
                                                 bkg_depth_camX], dim=1), dim=1, keepdim=True)[0]
        full_depth_camXs_e.append(full_depth_camX_e)
    full_depth_camXs_e = torch.stack(full_depth_camXs_e, dim=1)
    # print('full_depth_camXs', full_depth_camXs.shape)
    sw.summ_oneds('9_joint/full_depth_camXs_e', full_depth_camXs_e.unbind(1), max_val=max_dist)
    sw.summ_oneds('9_joint/diff_depth_camXs_e', (full_depth_camXs_e - full_depth_camXs_g).unbind(1))
    
    vis = []
    for s in range(S):
        vis.append(sw.summ_lrtlist(
            '',
            rgb_camXs[:,s],
            lrt_camXs[:,s:s+1],
            torch.ones_like(lrt_camXs[:,s:s+1,0]),
            torch.ones_like(lrt_camXs[:,s:s+1,0]).long(),
            pix_T_cam,
            # # frame_id=s,
            # frame_id=np.mean(min_values),
            frame_id=min_values[s],
            only_return=True)
        )
    sw.summ_rgbs('9_joint/optim_traj', vis)

    return True


def featimg_optim_prismatic(
        direction_camX,
        xyz_camXs,
        rgb_camXs,
        arm_camXs,
        lrt_camX,
        obj_xyz_camX,
        bkg_xyz_camX,
        obj_feat_camX,
        bkg_feat_camX,
        pix_T_cam,
        H, W,
        sw,
        max_dist=16,
        sigma=0.1,
        t_only=True,
        population_size=5,
        max_stale=10,
        max_generations=80):
    B, S, N, D = xyz_camXs.shape
    assert (B == 1)
    assert (D == 3)

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    # let's first see what it looks like if i render the given pointcloud

    # keep_camXs = 1.0 - ignore_camXs

    full_xyz_camX = torch.cat([obj_xyz_camX,
                               bkg_xyz_camX], dim=1)
    full_feat_camX = torch.cat([obj_feat_camX, bkg_feat_camX], dim=2)
    depth_camX, _ = utils.geom.create_depth_image(pix_T_cam, full_xyz_camX, H, W)  # B x 1 x H x W
    featimg_camX, _ = utils.geom.create_feat_image(pix_T_cam, full_xyz_camX, full_feat_camX, H, W)
    bkg_depth_camX, bkg_valid_camX = utils.geom.create_depth_image(pix_T_cam, bkg_xyz_camX, H, W)  # B x 1 x H x W
    bkg_featimg_camX, _ = utils.geom.create_feat_image(pix_T_cam, bkg_xyz_camX, bkg_feat_camX, H, W)

    sw.summ_oned('9_joint/initial_depth_camX', depth_camX, max_val=max_dist, frame_id=0)

    sw.summ_oned('9_joint/bkg_depth_camX', bkg_depth_camX, max_val=max_dist)
    sw.summ_rgb('9_joint/initial_feat_camX', featimg_camX, frame_id=0)
    sw.summ_rgb('9_joint/bkg_feat_camX', bkg_featimg_camX)
    # return False

    lrt_camXs = initialize_lrtlist_with_lrt(lrt_camX, S)

    _, rt_camXs = utils.geom.split_lrtlist(lrt_camXs)
    r_camXs, t_camXs = utils.geom.split_rtlist(rt_camXs)
    rx_, ry_, rz_ = utils.geom.rotm2eul(__p(r_camXs))
    tr_camXs = torch.cat([t_camXs.reshape(1, S, 3),
                          rx_.reshape(1, S, 1),
                          ry_.reshape(1, S, 1),
                          rz_.reshape(1, S, 1)], dim=2)

    lens, camX_T_obj = utils.geom.split_lrt(lrt_camX)
    obj_T_camX = camX_T_obj.inverse()
    obj_xyz_obj = utils.geom.apply_4x4(obj_T_camX, obj_xyz_camX)

    full_depth_camXs_g = []
    arm_depth_camXs_g = []
    for s in range(S):
        full_depth_camX, _ = utils.geom.create_depth_image(pix_T_cam, xyz_camXs[:, s], H, W)
        arm_depth_camX = full_depth_camX.clone()
        arm_depth_camX[arm_camXs[:, s] < 1] = max_dist

        full_depth_camXs_g.append(full_depth_camX)
        arm_depth_camXs_g.append(arm_depth_camX)
    full_depth_camXs_g = torch.stack(full_depth_camXs_g, dim=1)
    arm_depth_camXs_g = torch.stack(arm_depth_camXs_g, dim=1)

    sw.summ_oneds('9_joint/arm_depth_camXs', arm_depth_camXs_g.unbind(1), max_val=max_dist)
    sw.summ_oneds('9_joint/full_depth_camXs_g', full_depth_camXs_g.unbind(1), max_val=max_dist)
    sw.summ_rgbs('9_joint/full_featimg_camXs_g', rgb_camXs.unbind(1))

    direction_py = direction_camX[0].cpu().numpy()

    print('optimizing prismatic traj with direction', direction_py)
    best_t_gens = []
    min_values = []

    min_values.append(0)
    for s in range(1, S):
        print('working on s=%d' % s)

        prev_tr_camX = tr_camXs[:, s - 1]
        prev_tr_py = prev_tr_camX.cpu().reshape(-1).numpy()  # 6

        amount_py = np.zeros_like(prev_tr_py[0:2])
        bounds = np.array([
            -1, 1,
            -1, 1,
        ]).reshape(-1, 2)
        optimizer = CMA(mean=amount_py, sigma=sigma, population_size=population_size, bounds=bounds)

        min_value = np.inf
        best_amount_py = amount_py.copy()

        stale_count = 0
        generation_count = 0

        while stale_count < max_stale and generation_count < max_generations:
            generation_count += 1
            improved_things = False
            solutions = []

            for pi in range(optimizer.population_size):

                if pi == optimizer.population_size - 1:
                    # explicitly consider a static sol
                    amount_py = amount_py * 0
                else:
                    amount_py = optimizer.ask()

                delta_py_ = np.concatenate([amount_py[0:1] * direction_py, direction_py * 0])
                tr_py = prev_tr_py + delta_py_

                tr_camX = torch.from_numpy(tr_py).reshape(1, 6).float().to('cuda')
                lrt_camX = update_lrt_with_tr(lrt_camXs[:, s], tr_camX)
                obj_depth_camX, obj_featimg_camX, obj_valid_camX = render_one_object_with_feat_2d(obj_xyz_obj,
                                                                                                  obj_feat_camX,
                                                                                                  lrt_camX, pix_T_cam,
                                                                                                  H, W)

                full_valid_camX = (bkg_valid_camX + obj_valid_camX).clamp(0, 1)  # * keep_camXs[:,s]
                full_depth_camX_e = torch.min(torch.cat([obj_depth_camX,
                                                         arm_depth_camXs_g[:, s],
                                                         bkg_depth_camX], dim=1), dim=1, keepdim=True)[0]
                full_featimg_camX_e = create_featimg_from_depth(obj_depth_camX, arm_depth_camXs_g[:, s], bkg_depth_camX,
                                                                obj_featimg_camX, rgb_camXs[:, s], bkg_featimg_camX)
                depth_loss = torch.abs(
                    full_depth_camX_e - full_depth_camXs_g[:, s])  # * keep_camXs[:,s] #* full_valid_camX
                featimg_loss = torch.abs(full_featimg_camX_e - rgb_camXs[:, s])
                value = torch.sum(depth_loss).detach().cpu().numpy()

                if value < min_value:
                    min_value = value
                    best_amount_py = amount_py.copy()
                    improved_things = True

                solutions.append((amount_py, value))

            if improved_things:
                stale_count = 0
            else:
                stale_count += 1
            print(f"{generation_count} {stale_count} {value} {min_value}")
            optimizer.tell(solutions)
        min_values.append(min_value)

        best_delta_py_ = np.concatenate([best_amount_py[0:1] * direction_py, direction_py * 0])
        tr_camXs[:, s] = prev_tr_camX + torch.from_numpy(best_delta_py_).reshape(1, 6).float().to('cuda')
        print('best_amount_py', best_amount_py)

    lrt_camXs = update_lrtlist_with_trlist(lrt_camXs, tr_camXs)

    full_depth_camXs_e = []
    full_featimg_camXs_e = []
    for s in range(S):
        obj_depth_camX, obj_featimg_camX, _ = render_one_object_with_feat_2d(obj_xyz_obj, obj_feat_camX,
                                                                             lrt_camXs[:, s], pix_T_cam, H, W)
        full_depth_camX_e = torch.min(torch.cat([obj_depth_camX,
                                                 arm_depth_camXs_g[:, s],
                                                 bkg_depth_camX], dim=1), dim=1, keepdim=True)[0]
        full_depth_camXs_e.append(full_depth_camX_e)
        full_featimg_camX_e = create_featimg_from_depth(obj_depth_camX, arm_depth_camXs_g[:, s], bkg_depth_camX,
                                                        obj_featimg_camX, rgb_camXs[:, s], bkg_featimg_camX)
        full_featimg_camXs_e.append(full_featimg_camX_e)
    full_depth_camXs_e = torch.stack(full_depth_camXs_e, dim=1)
    full_featimg_camXs_e = torch.stack(full_featimg_camXs_e, dim=1)
    # print('full_depth_camXs', full_depth_camXs.shape)
    sw.summ_oneds('9_joint/full_depth_camXs_e', full_depth_camXs_e.unbind(1), max_val=max_dist)
    sw.summ_oneds('9_joint/diff_depth_camXs_e', (full_depth_camXs_e - full_depth_camXs_g).unbind(1))
    sw.summ_rgbs('9_joint/full_featimg_camXs_e', full_featimg_camXs_e.unbind(1))
    sw.summ_rgbs('9_joint/diff_featimg_camXs_e', (full_featimg_camXs_e - rgb_camXs).unbind(1))

    vis = []
    for s in range(S):
        vis.append(sw.summ_lrtlist(
            '',
            rgb_camXs[:, s],
            lrt_camXs[:, s:s + 1],
            torch.ones_like(lrt_camXs[:, s:s + 1, 0]),
            torch.ones_like(lrt_camXs[:, s:s + 1, 0]).long(),
            pix_T_cam,
            # # frame_id=s,
            # frame_id=np.mean(min_values),
            frame_id=min_values[s],
            only_return=True)
        )
    sw.summ_rgbs('9_joint/optim_traj', vis)

    return True


def featimg_optim_free_traj(
        xyz_camX_list_full,
        rgb_camXs,
        arm_camXs,
        lrt_camX,
        obj_xyz_camX,
        bkg_xyz_camX,
        obj_feat_camX,
        bkg_feat_camX,
        pix_T_cam,
        seed_s,
        H, W,
        sw,
        max_dist=16,
        sigma=0.1,
        population_size=5,
        max_stale=10,
        max_generations=80,
):
    # obj_xyz_camX is 1 x N0 x 3, obj_feat_camX is 1 x C x N0
    # bkg_xyz_camX is 1 x N1 x 3, bkg_feat_camX is 1 x C x N1
    B, S, C, H, W = rgb_camXs.shape
    assert (B == 1)
    assert (C == 3)

    assert (obj_xyz_camX.shape[1] == obj_feat_camX.shape[2])
    assert (bkg_xyz_camX.shape[1] == bkg_feat_camX.shape[2])

    # a small update: now it can take a short traj, S1 is the length of the short traj.
    # if S1=1, same as before.
    S1, _ = lrt_camX.shape
    assert (S1 < S)

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    full_xyz_camX = torch.cat([obj_xyz_camX, bkg_xyz_camX], dim=1)
    full_feat_camX = torch.cat([obj_feat_camX, bkg_feat_camX], dim=2)
    full_depth_camX, full_valid_camX = utils.geom.create_depth_image(pix_T_cam, full_xyz_camX, H, W,
                                                                     serial=True, slices=30,
                                                                     max_val=max_dist)  # B x 1 x H x W
    obj_depth_camX, obj_valid_camX = utils.geom.create_depth_image(pix_T_cam, obj_xyz_camX, H, W, serial=True,
                                                                   slices=30, max_val=max_dist)  # B x 1 x H x W
    bkg_depth_camX, bkg_valid_camX = utils.geom.create_depth_image(pix_T_cam, bkg_xyz_camX, H, W, serial=True,
                                                                   slices=30, max_val=max_dist)  # B x 1 x H x W

    full_featimg_camX, _ = utils.geom.create_feat_image(pix_T_cam, full_xyz_camX, full_feat_camX, H, W,
                                                        serial=True, slices=30, max_val=max_dist)
    obj_featimg_camX, _ = utils.geom.create_feat_image(pix_T_cam, obj_xyz_camX, obj_feat_camX, H, W, serial=True,
                                                       slices=30, max_val=max_dist)
    bkg_featimg_camX, _ = utils.geom.create_feat_image(pix_T_cam, bkg_xyz_camX, bkg_feat_camX, H, W, serial=True,
                                                       slices=30, max_val=max_dist)  # B x 1 x H x W

    sw.summ_oned('8_optim/0_full_depth_camX', full_depth_camX, max_val=max_dist, frame_id=0)
    sw.summ_oned('8_optim/0_obj_depth_camX', obj_depth_camX, max_val=max_dist)
    sw.summ_oned('8_optim/0_bkg_depth_camX', bkg_depth_camX, max_val=max_dist)
    sw.summ_rgb('8_optim/0_full_featimg_camX', full_featimg_camX, frame_id=0)
    sw.summ_rgb('8_optim/0_obj_featimg_camX', obj_featimg_camX)
    sw.summ_rgb('8_optim/0_bkg_featimg_camX', bkg_featimg_camX)
    # return False

    # use the last lrt to init all else
    lrt_camXs = initialize_lrtlist_with_lrt(lrt_camX[-1:], S - S1 + 1)
    lrt_camXs = torch.cat([lrt_camX[:-1].unsqueeze(0), lrt_camXs], dim=1)
    lrt_camX_bak = lrt_camX.clone()

    _, rt_camXs = utils.geom.split_lrtlist(lrt_camXs)
    r_camXs, t_camXs = utils.geom.split_rtlist(rt_camXs)
    rx_, ry_, rz_ = utils.geom.rotm2eul(__p(r_camXs))
    tr_camXs = torch.cat([t_camXs.reshape(1, S, 3),
                          rx_.reshape(1, S, 1),
                          ry_.reshape(1, S, 1),
                          rz_.reshape(1, S, 1)], dim=2)

    lens, camX_T_obj = utils.geom.split_lrt(lrt_camX[0:1])
    obj_T_camX = camX_T_obj.inverse()
    obj_xyz_obj = utils.geom.apply_4x4(obj_T_camX, obj_xyz_camX)

    full_depth_camXs_g = []
    full_valid_camXs_g = []
    arm_depth_camXs_g = []

    keep_camXs = []  # = 1.0 - ignore_camXs

    for s in range(S):
        full_depth_camX, full_valid_camX = utils.geom.create_depth_image(
            pix_T_cam, xyz_camX_list_full[s], H, W,
            max_val=max_dist)  # this is gt singleview depth; does not need serial
        arm_depth_camX = full_depth_camX.clone()
        arm_depth_camX[arm_camXs[:, s] < 1] = max_dist
        full_depth_camXs_g.append(full_depth_camX)
        full_valid_camXs_g.append(full_valid_camX)
        arm_depth_camXs_g.append(arm_depth_camX)
        keep_camXs.append(1.0 - utils.improc.dilate2d(arm_camXs[:, s]))

    full_depth_camXs_g = torch.stack(full_depth_camXs_g, dim=1)
    full_valid_camXs_g = torch.stack(full_valid_camXs_g, dim=1)
    arm_depth_camXs_g = torch.stack(arm_depth_camXs_g, dim=1)
    keep_camXs = torch.stack(keep_camXs, dim=1)

    sw.summ_oneds('8_optim/1_arm_depth_camXs', arm_depth_camXs_g.unbind(1), max_val=max_dist)
    sw.summ_oneds('8_optim/1_full_depth_camXs_g', full_depth_camXs_g.unbind(1), max_val=max_dist)
    sw.summ_oneds('8_optim/1_full_valid_camXs_g', full_valid_camXs_g.unbind(1), norm=False)
    sw.summ_rgbs('8_optim/1_full_featimg_camXs_g', rgb_camXs.unbind(1))

    print('optimizing traj')
    best_t_gens = []
    min_values = []

    t_camXs_py = t_camXs.cpu().reshape(-1).numpy()  # S*3
    xyz_py = xyz_camX_list_full[0].cpu().numpy()[0]  # N, 3
    optimizer = CMA(mean=t_camXs_py, sigma=sigma, population_size=population_size)  # no bounds for now

    min_value = np.inf
    best_t_camXs = t_camXs_py.copy()

    stale_count = 0
    generation_count = 0

    while stale_count < max_stale and generation_count < max_generations:
        generation_count += 1
        improved_things = False
        solutions = []

        for pi in range(optimizer.population_size):

            t_camXs_py = optimizer.ask()

            t_camXs = torch.from_numpy(t_camXs_py).reshape(1, S, 3).float().to('cuda')
            lrt_camXs = update_lrtlist_with_tlist(lrt_camXs, t_camXs)
            lrt_camXs[0, seed_s:seed_s + S1] = lrt_camX_bak

            total_featimg_error = 0

            for s in range(S):
                obj_depth_camX, obj_featimg_camX, obj_valid_camX = render_one_object_with_feat_2d(obj_xyz_obj,
                                                                                                  obj_feat_camX,
                                                                                                  lrt_camXs[:, s],
                                                                                                  pix_T_cam, H, W,
                                                                                                  max_dist)

                full_valid_camX = (bkg_valid_camX + obj_valid_camX).clamp(0, 1)
                full_depth_camX_e = torch.min(torch.cat([obj_depth_camX,
                                                         arm_depth_camXs_g[:, s],
                                                         bkg_depth_camX], dim=1), dim=1, keepdim=True)[0]
                full_featimg_camX_e = create_featimg_from_depth(obj_depth_camX, arm_depth_camXs_g[:, s], bkg_depth_camX,
                                                                obj_featimg_camX, rgb_camXs[:, s], bkg_featimg_camX)

                full_depth_camX_e[~full_valid_camX.bool()] = full_depth_camXs_g[:, s][~full_valid_camX.bool()]
                featimg_loss = torch.abs(full_featimg_camX_e - rgb_camXs[:, s]) * keep_camXs[:, s]
                # depth_error = torch.abs(full_depth_camX_e - full_depth_camXs_g[:, s]).clamp(0, 10.0) * keep_camXs[:, s]
                # utils.basic.print_stats('depth_error', depth_error)
                total_featimg_error += torch.mean(featimg_loss)

            use_accel = False
            if use_accel:
                velo = t_camXs[:, 1:] - t_camXs[:, :-1]
                accel = velo[:, 1:] - velo[:, :-1]
                accel_error = torch.sum(torch.abs(accel))  # higher -> worse
                value = total_featimg_error.item() + accel_error.item()
            else:
                value = total_featimg_error.item()

            if value < min_value:
                min_value = value
                best_t_camXs_py = t_camXs_py.copy()
                improved_things = True

            solutions.append((t_camXs_py, value))

        if improved_things:
            stale_count = 0
        else:
            stale_count += 1
        print(f"{generation_count} {stale_count} {value} {min_value}")
        optimizer.tell(solutions)

    print('best_t_camXs_py', best_t_camXs_py.reshape(S, 3))

    t_camXs = torch.from_numpy(best_t_camXs_py).reshape(1, S, 3).float().to('cuda')
    lrt_camXs = update_lrtlist_with_tlist(lrt_camXs, t_camXs)
    lrt_camXs[0, seed_s:seed_s + S1] = lrt_camX_bak

    _, rt_camXs = utils.geom.split_lrtlist(lrt_camXs)
    r_camXs, t_camXs = utils.geom.split_rtlist(rt_camXs)
    print('final t_camXs', t_camXs)

    full_depth_camXs_e = []
    full_featimg_camXs_e = []
    for s in range(S):
        obj_depth_camX, obj_featimg_camX, _ = render_one_object_with_feat_2d(obj_xyz_obj, obj_feat_camX,
                                                                             lrt_camXs[:, s], pix_T_cam, H, W,
                                                                             max_dist=max_dist)
        full_depth_camX_e = torch.min(torch.cat([obj_depth_camX,
                                                 arm_depth_camXs_g[:, s],
                                                 bkg_depth_camX], dim=1), dim=1, keepdim=True)[0]
        full_depth_camXs_e.append(full_depth_camX_e)
        full_featimg_camX_e = create_featimg_from_depth(obj_depth_camX, arm_depth_camXs_g[:, s], bkg_depth_camX,
                                                        obj_featimg_camX, rgb_camXs[:, s], bkg_featimg_camX)
        full_featimg_camXs_e.append(full_featimg_camX_e)
    full_depth_camXs_e = torch.stack(full_depth_camXs_e, dim=1)
    full_featimg_camXs_e = torch.stack(full_featimg_camXs_e, dim=1)
    # print('full_depth_camXs', full_depth_camXs.shape)
    sw.summ_oneds('8_optim/full_depth_camXs_e', full_depth_camXs_e.unbind(1), max_val=max_dist)
    sw.summ_oneds('8_optim/diff_depth_camXs_e', (torch.abs(full_depth_camXs_e - full_depth_camXs_g) * keep_camXs).clamp(0, 1).unbind(1), norm=False)
    sw.summ_rgbs('8_optim/full_featimg_camXs_e', full_featimg_camXs_e.unbind(1))
    sw.summ_rgbs('8_optim/diff_featimg_camXs_e', (full_featimg_camXs_e - rgb_camXs).unbind(1))

    vis = []
    for s in range(S):
        vis.append(sw.summ_lrtlist(
            '',
            rgb_camXs[:, s],
            lrt_camXs[:, s:s + 1],
            torch.ones_like(lrt_camXs[:, s:s + 1, 0]),
            torch.ones_like(lrt_camXs[:, s:s + 1, 0]).long(),
            pix_T_cam,
            # # frame_id=s,
            # frame_id=np.mean(min_values),
            frame_id=min_value,
            only_return=True)
        )
    sw.summ_rgbs('8_optim/optim_traj', vis)

    return lrt_camXs


def depth_optim_free_traj(
        xyz_camX_list_full,
        rgb_camXs,
        arm_camXs, 
        lrt_camX,
        obj_xyz_camX,
        bkg_xyz_camX, 
        pix_T_cam,
        seed_s,
        H, W,
        sw,
        max_dist=16,
        sigma=0.1,
        population_size=5,
        max_stale=10,
        max_generations=80,
):

    B, S, C, H, W = rgb_camXs.shape
    assert(B==1)
    assert(C==3)

    # a small update: now it can take a short traj, S1 is the length of the short traj.
    # if S1=1, same as before.
    S1, _ = lrt_camX.shape
    assert (S1 < S)

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    full_depth_camX, full_valid_camX = utils.geom.create_depth_image(pix_T_cam, xyz_camX_list_full[0], H, W, serial=True, slices=30, max_val=max_dist) # B x 1 x H x W
    obj_depth_camX, obj_valid_camX = utils.geom.create_depth_image(pix_T_cam, obj_xyz_camX, H, W, serial=True, slices=30, max_val=max_dist) # B x 1 x H x W
    bkg_depth_camX, bkg_valid_camX = utils.geom.create_depth_image(pix_T_cam, bkg_xyz_camX, H, W, serial=True, slices=30, max_val=max_dist) # B x 1 x H x W

    sw.summ_oned('8_optim/0_full_depth_camX', full_depth_camX, max_val=max_dist, frame_id=0)
    sw.summ_oned('8_optim/0_obj_depth_camX', obj_depth_camX, max_val=max_dist)
    sw.summ_oned('8_optim/0_bkg_depth_camX', bkg_depth_camX, max_val=max_dist)
    # return False

    # use the last lrt to init all else
    lrt_camXs = initialize_lrtlist_with_lrt(lrt_camX[-2:-1], S - S1 + 1)
    lrt_camXs = torch.cat([lrt_camX[:-1].unsqueeze(0), lrt_camXs], dim=1)
    lrt_camX_bak = lrt_camX.clone()
    
    _, rt_camXs = utils.geom.split_lrtlist(lrt_camXs)
    r_camXs, t_camXs = utils.geom.split_rtlist(rt_camXs)
    rx_, ry_, rz_ = utils.geom.rotm2eul(__p(r_camXs))
    tr_camXs = torch.cat([t_camXs.reshape(1, S, 3),
                          rx_.reshape(1, S, 1),
                          ry_.reshape(1, S, 1),
                          rz_.reshape(1, S, 1)], dim=2)

    lens, camX_T_obj = utils.geom.split_lrt(lrt_camX[0:1])
    obj_T_camX = camX_T_obj.inverse()
    obj_xyz_obj = utils.geom.apply_4x4(obj_T_camX, obj_xyz_camX)

    full_depth_camXs_g = []
    full_valid_camXs_g = []
    arm_depth_camXs_g = []

    keep_camXs = [] # = 1.0 - ignore_camXs
    
    for s in range(S):
        full_depth_camX, full_valid_camX = utils.geom.create_depth_image(
            pix_T_cam, xyz_camX_list_full[s], H, W, max_val=max_dist) # this is gt singleview depth; does not need serial
        arm_depth_camX = full_depth_camX.clone()
        arm_depth_camX[arm_camXs[:,s] < 1] = max_dist
        full_depth_camXs_g.append(full_depth_camX)
        full_valid_camXs_g.append(full_valid_camX)
        arm_depth_camXs_g.append(arm_depth_camX)
        keep_camXs.append(1.0 - utils.improc.dilate2d(arm_camXs[:,s]))

    full_depth_camXs_g = torch.stack(full_depth_camXs_g, dim=1)
    full_valid_camXs_g = torch.stack(full_valid_camXs_g, dim=1)
    arm_depth_camXs_g = torch.stack(arm_depth_camXs_g, dim=1)
    keep_camXs = torch.stack(keep_camXs, dim=1)

    sw.summ_oneds('8_optim/1_arm_depth_camXs', arm_depth_camXs_g.unbind(1), max_val=max_dist)
    sw.summ_oneds('8_optim/1_full_depth_camXs_g', full_depth_camXs_g.unbind(1), max_val=max_dist)
    sw.summ_oneds('8_optim/1_full_valid_camXs_g', full_valid_camXs_g.unbind(1), norm=False)

    print('optimizing traj')
    best_t_gens = []
    min_values = []

    t_camXs_py = t_camXs.cpu().reshape(-1).numpy() # S*3
    xyz_py = xyz_camX_list_full[0].cpu().numpy()[0] # N, 3
    optimizer = CMA(mean=t_camXs_py, sigma=sigma, population_size=population_size) # no bounds for now
        
    min_value = np.inf
    best_t_camXs = t_camXs_py.copy()

    stale_count = 0
    generation_count = 0

    while stale_count < max_stale and generation_count < max_generations:
        generation_count += 1
        improved_things = False
        solutions = []

        for pi in range(optimizer.population_size):

            t_camXs_py = optimizer.ask()

            t_camXs = torch.from_numpy(t_camXs_py).reshape(1, S, 3).float().to('cuda')
            lrt_camXs = update_lrtlist_with_tlist(lrt_camXs, t_camXs)
            lrt_camXs[0,seed_s:seed_s+S1] = lrt_camX_bak

            total_depth_error = 0
            
            for s in range(S):
                obj_depth_camX, obj_valid_camX = render_one_object_2d(obj_xyz_obj, lrt_camXs[:,s], pix_T_cam, H, W, max_dist)
                
                full_valid_camX = (bkg_valid_camX + obj_valid_camX).clamp(0,1)
                full_depth_camX_e = torch.min(torch.cat([obj_depth_camX,
                                                         arm_depth_camXs_g[:,s],
                                                         bkg_depth_camX], dim=1), dim=1, keepdim=True)[0]

                full_depth_camX_e[~full_valid_camX.bool()] = full_depth_camXs_g[:, s][~full_valid_camX.bool()]
                depth_error = torch.abs(full_depth_camX_e - full_depth_camXs_g[:,s]).clamp(0,10.0) * keep_camXs[:,s]
                # utils.basic.print_stats('depth_error', depth_error)
                total_depth_error += torch.mean(depth_error)

            use_accel = False
            if use_accel:
                velo = t_camXs[:,1:] - t_camXs[:,:-1]
                accel = velo[:,1:] - velo[:,:-1]
                accel_error = torch.sum(torch.abs(accel)) # higher -> worse
                value = total_depth_error.item() + accel_error.item()
            else:
                value = total_depth_error.item()

            if value < min_value:
                min_value = value
                best_t_camXs_py = t_camXs_py.copy()
                improved_things = True

            solutions.append((t_camXs_py, value))

        if improved_things:
            stale_count = 0
        else:
            stale_count += 1
        print(f"{generation_count} {stale_count} {value} {min_value}")
        optimizer.tell(solutions)

    print('best_t_camXs_py', best_t_camXs_py.reshape(S, 3))
    
    t_camXs = torch.from_numpy(best_t_camXs_py).reshape(1, S, 3).float().to('cuda')
    lrt_camXs = update_lrtlist_with_tlist(lrt_camXs, t_camXs)
    lrt_camXs[0,seed_s:seed_s+S1] = lrt_camX_bak

    _, rt_camXs = utils.geom.split_lrtlist(lrt_camXs)
    r_camXs, t_camXs = utils.geom.split_rtlist(rt_camXs)
    print('final t_camXs', t_camXs)

    full_depth_camXs_e = []
    for s in range(S):
        obj_depth_camX, _ = render_one_object_2d(obj_xyz_obj, lrt_camXs[:,s], pix_T_cam, H, W, max_dist)
        full_depth_camX_e = torch.min(torch.cat([obj_depth_camX,
                                                 arm_depth_camXs_g[:,s],
                                                 bkg_depth_camX], dim=1), dim=1, keepdim=True)[0]
        full_depth_camXs_e.append(full_depth_camX_e)
    full_depth_camXs_e = torch.stack(full_depth_camXs_e, dim=1)
    # print('full_depth_camXs', full_depth_camXs.shape)
    sw.summ_oneds('8_optim/full_depth_camXs_e', full_depth_camXs_e.unbind(1), max_val=max_dist)
    sw.summ_oneds('8_optim/diff_depth_camXs_e', (torch.abs(full_depth_camXs_e - full_depth_camXs_g) * keep_camXs).clamp(0, 1).unbind(1), norm=False)
    
    vis = []
    for s in range(S):
        vis.append(sw.summ_lrtlist(
            '',
            rgb_camXs[:,s],
            lrt_camXs[:,s:s+1],
            torch.ones_like(lrt_camXs[:,s:s+1,0]),
            torch.ones_like(lrt_camXs[:,s:s+1,0]).long(),
            pix_T_cam,
            # # frame_id=s,
            # frame_id=np.mean(min_values),
            frame_id=min_value,
            only_return=True)
        )
    sw.summ_rgbs('8_optim/optim_traj', vis)

    return lrt_camXs


def depth_optim_prismatic_traj(
        xyz_camX_list_full,
        rgb_camXs,
        arm_camXs, 
        lrt_camX,
        obj_xyz_camX,
        bkg_xyz_camX, 
        pix_T_cam,
        seed_s,
        H, W,
        sw,
        max_dist=16,
        sigma=0.1,
        cma_trials=10,
        population_size=5,
        max_stale=10,
        max_generations=80,
):

    B, S, C, H, W = rgb_camXs.shape
    assert(B==1)
    assert(C==3)

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    full_depth_camX, full_valid_camX = utils.geom.create_depth_image(pix_T_cam, xyz_camX_list_full[0], H, W, serial=True, slices=30, max_val=max_dist) # B x 1 x H x W
    obj_depth_camX, obj_valid_camX = utils.geom.create_depth_image(pix_T_cam, obj_xyz_camX, H, W, serial=True, slices=30, max_val=max_dist) # B x 1 x H x W
    bkg_depth_camX, bkg_valid_camX = utils.geom.create_depth_image(pix_T_cam, bkg_xyz_camX, H, W, serial=True, slices=30, max_val=max_dist) # B x 1 x H x W

    interest_mask = utils.improc.dilate2d(obj_valid_camX, times=8)

    sw.summ_oned('8_optim/0_full_depth_camX', full_depth_camX, max_val=max_dist, frame_id=0)
    sw.summ_oned('8_optim/0_obj_depth_camX', obj_depth_camX, max_val=max_dist)
    sw.summ_oned('8_optim/0_bkg_depth_camX', bkg_depth_camX, max_val=max_dist)
    sw.summ_oned('8_optim/0_interest_mask', interest_mask, norm=False)
    # return False
    
    lrt_camXs = initialize_lrtlist_with_lrt(lrt_camX, S)
    lrt_camX_bak = lrt_camX.clone()
    
    _, rt_camXs = utils.geom.split_lrtlist(lrt_camXs)
    r_camXs, t_camXs = utils.geom.split_rtlist(rt_camXs)
    rx_, ry_, rz_ = utils.geom.rotm2eul(__p(r_camXs))
    tr_camXs = torch.cat([t_camXs.reshape(1, S, 3),
                          rx_.reshape(1, S, 1),
                          ry_.reshape(1, S, 1),
                          rz_.reshape(1, S, 1)], dim=2)
    t_camXs_bak = t_camXs.clone()

    lens, camX_T_obj = utils.geom.split_lrt(lrt_camX)
    obj_T_camX = camX_T_obj.inverse()
    obj_xyz_obj = utils.geom.apply_4x4(obj_T_camX, obj_xyz_camX)

    full_depth_camXs_g = []
    full_valid_camXs_g = []
    arm_depth_camXs_g = []

    keep_camXs = [] # = 1.0 - ignore_camXs
    
    for s in range(S):
        full_depth_camX, full_valid_camX = utils.geom.create_depth_image(
            pix_T_cam, xyz_camX_list_full[s], H, W, max_val=max_dist) # this is gt singleview depth; does not need serial
        arm_depth_camX = full_depth_camX.clone()
        arm_depth_camX[arm_camXs[:,s] < 1] = max_dist
        full_depth_camXs_g.append(full_depth_camX)
        full_valid_camXs_g.append(full_valid_camX)
        arm_depth_camXs_g.append(arm_depth_camX)

        keep_camX = 1.0 - utils.improc.dilate2d(arm_camXs[:,s])
        keep_camX = keep_camX * utils.improc.erode2d(full_valid_camX) 
        keep_camXs.append(keep_camX)

    full_depth_camXs_g = torch.stack(full_depth_camXs_g, dim=1)
    full_valid_camXs_g = torch.stack(full_valid_camXs_g, dim=1)
    arm_depth_camXs_g = torch.stack(arm_depth_camXs_g, dim=1)
    keep_camXs = torch.stack(keep_camXs, dim=1)
    
    sw.summ_oneds('8_optim/1_arm_depth_camXs', arm_depth_camXs_g.unbind(1), max_val=max_dist)
    sw.summ_oneds('8_optim/1_full_depth_camXs_g', full_depth_camXs_g.unbind(1), max_val=max_dist)
    sw.summ_oneds('8_optim/1_full_valid_camXs_g', full_valid_camXs_g.unbind(1), norm=False)
    sw.summ_oneds('8_optim/1_keep_camXs', keep_camXs.unbind(1), norm=False)

    print('optimizing traj')

    min_value = np.inf
    best_var_py = None

    for trial in range(cma_trials):
    
        # randomly initialize direction
        direction_py = np.random.randn(3).astype(np.float32)
        direction_py = direction_py / np.linalg.norm(direction_py)
        amount_py = np.zeros((S)).astype(np.float32)
        var_py = np.concatenate([direction_py, amount_py])
        optimizer = CMA(mean=var_py, sigma=sigma, population_size=population_size) # no bounds for now

        print('trial %d' % trial, 'starting with direction', direction_py) 
        
        # optimizer1 = CMA(mean=direction_py, sigma=sigma, population_size=population_size) # no bounds for now
        # optimizer2 = CMA(mean=amount_py, sigma=sigma, population_size=population_size) # no bounds for now

        stale_count = 0
        generation_count = 0

        while stale_count < max_stale and generation_count < max_generations:
            generation_count += 1
            improved_things = False
            solutions = []

            for pi in range(optimizer.population_size):

                var_py = optimizer.ask()
                direction_py = var_py[:3]
                amount_py = var_py[3:]

                # direction_py = optimizer1.ask()
                # amount_py = optimizer2.ask()
                # var_py = np.concatenate([direction_py, amount_py])

                direction_py = direction_py / np.linalg.norm(direction_py)
                direction_pt = torch.from_numpy(direction_py).to('cuda').reshape(1, 1, 3)
                amount_pt = torch.from_numpy(amount_py).to('cuda').reshape(1, S, 1)
                delta_pt = direction_pt * amount_pt
                t_camXs = t_camXs_bak + delta_pt
                lrt_camXs = update_lrtlist_with_tlist(lrt_camXs, t_camXs)
                lrt_camXs[:,seed_s] = lrt_camX_bak # ensure we keep the seed obj where we found it 

                total_depth_error = 0

                for s in range(S):
                    obj_depth_camX, obj_valid_camX = render_one_object_2d(obj_xyz_obj, lrt_camXs[:,s], pix_T_cam, H, W, max_dist)

                    full_valid_camX = (bkg_valid_camX + obj_valid_camX).clamp(0,1)
                    full_depth_camX_e = torch.min(torch.cat([obj_depth_camX,
                                                             arm_depth_camXs_g[:,s],
                                                             bkg_depth_camX], dim=1), dim=1, keepdim=True)[0]

                    depth_error = torch.abs(full_depth_camX_e - full_depth_camXs_g[:,s]).clamp(0,0.1) # * keep_camXs[:,s]# * interest_mask
                    # utils.basic.print_stats('depth_error', depth_error)
                    total_depth_error += torch.mean(depth_error)

                use_accel = False
                if use_accel:
                    velo = t_camXs[:,1:] - t_camXs[:,:-1]
                    accel = velo[:,1:] - velo[:,:-1]
                    accel_error = torch.sum(torch.abs(accel)) # higher -> worse
                    value = total_depth_error.item() + accel_error.item()
                else:
                    value = total_depth_error.item()

                if value < min_value:
                    min_value = value
                    best_var_py = var_py.copy()
                    improved_things = True

                solutions.append((var_py, value))

            if improved_things:
                stale_count = 0
            else:
                stale_count += 1
            print(f"{generation_count} {stale_count} {value} {min_value}")
            optimizer.tell(solutions)

    print('best_var_py', best_var_py)
    
    direction_py = best_var_py[:3]
    direction_py = direction_py / np.linalg.norm(direction_py)
    amount_py = best_var_py[3:]
    direction_pt = torch.from_numpy(direction_py).to('cuda').reshape(1, 1, 3)
    amount_pt = torch.from_numpy(amount_py).to('cuda').reshape(1, S, 1)
    delta_pt = direction_pt * amount_pt
    t_camXs = t_camXs_bak + delta_pt
    lrt_camXs = update_lrtlist_with_tlist(lrt_camXs, t_camXs)
    lrt_camXs[:,seed_s] = lrt_camX_bak # ensure we keep the seed obj where we found it 

    full_depth_camXs_e = []
    for s in range(S):
        obj_depth_camX, _ = render_one_object_2d(obj_xyz_obj, lrt_camXs[:,s], pix_T_cam, H, W, max_dist)
        full_depth_camX_e = torch.min(torch.cat([obj_depth_camX,
                                                 arm_depth_camXs_g[:,s],
                                                 bkg_depth_camX], dim=1), dim=1, keepdim=True)[0]
        full_depth_camXs_e.append(full_depth_camX_e)
    full_depth_camXs_e = torch.stack(full_depth_camXs_e, dim=1)
    # print('full_depth_camXs', full_depth_camXs.shape)
    sw.summ_oneds('8_optim/full_depth_camXs_e', full_depth_camXs_e.unbind(1), max_val=max_dist)
    sw.summ_oneds('8_optim/diff_depth_camXs_e', (torch.abs(full_depth_camXs_e - full_depth_camXs_g)).unbind(1))
    
    vis = []
    for s in range(S):
        vis.append(sw.summ_lrtlist(
            '',
            rgb_camXs[:,s],
            lrt_camXs[:,s:s+1],
            torch.ones_like(lrt_camXs[:,s:s+1,0]),
            torch.ones_like(lrt_camXs[:,s:s+1,0]).long(),
            pix_T_cam,
            # # frame_id=s,
            # frame_id=np.mean(min_values),
            frame_id=min_value,
            only_return=True)
        )
    sw.summ_rgbs('8_optim/optim_traj', vis)

    return lrt_camXs


def depth_optim_mcs(
        xyz_camX_list_full,
        rgb_camXs,
        lrt_camX,
        obj_xyz_camX,
        bkg_xyz_camX, 
        pix_T_cam,
        seed_s,
        H, W,
        sw,
        max_dist=16,
        sigma=0.1,
        population_size=5,
        max_stale=10,
        max_generations=80,
):

    B, S, C, H, W = rgb_camXs.shape
    assert(B==1)
    assert(C==3)

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    full_depth_camX, full_valid_camX = utils.geom.create_depth_image(pix_T_cam, xyz_camX_list_full[0], H, W, serial=True, slices=30, max_val=max_dist) # B x 1 x H x W
    obj_depth_camX, obj_valid_camX = utils.geom.create_depth_image(pix_T_cam, obj_xyz_camX, H, W, serial=True, slices=30, max_val=max_dist) # B x 1 x H x W
    bkg_depth_camX, bkg_valid_camX = utils.geom.create_depth_image(pix_T_cam, bkg_xyz_camX, H, W, serial=True, slices=30, max_val=max_dist) # B x 1 x H x W

    sw.summ_oned('8_optim/0_full_depth_camX', full_depth_camX, max_val=max_dist, frame_id=0)
    sw.summ_oned('8_optim/0_obj_depth_camX', obj_depth_camX, max_val=max_dist)
    sw.summ_oned('8_optim/0_bkg_depth_camX', bkg_depth_camX, max_val=max_dist)
    # return False
    
    lrt_camXs = initialize_lrtlist_with_lrt(lrt_camX, S)
    lrt_camX_bak = lrt_camX.clone()
    
    _, rt_camXs = utils.geom.split_lrtlist(lrt_camXs)
    r_camXs, t_camXs = utils.geom.split_rtlist(rt_camXs)
    rx_, ry_, rz_ = utils.geom.rotm2eul(__p(r_camXs))
    tr_camXs = torch.cat([t_camXs.reshape(1, S, 3),
                          rx_.reshape(1, S, 1),
                          ry_.reshape(1, S, 1),
                          rz_.reshape(1, S, 1)], dim=2)

    lens, camX_T_obj = utils.geom.split_lrt(lrt_camX)
    obj_T_camX = camX_T_obj.inverse()
    obj_xyz_obj = utils.geom.apply_4x4(obj_T_camX, obj_xyz_camX)

    full_depth_camXs_g = []
    full_valid_camXs_g = []

    for s in range(S):
        full_depth_camX, full_valid_camX = utils.geom.create_depth_image(
            pix_T_cam, xyz_camX_list_full[s], H, W, max_val=max_dist) # this is gt singleview depth; does not need serial
        full_depth_camXs_g.append(full_depth_camX)
        full_valid_camXs_g.append(full_valid_camX)

    full_depth_camXs_g = torch.stack(full_depth_camXs_g, dim=1)
    full_valid_camXs_g = torch.stack(full_valid_camXs_g, dim=1)
    
    sw.summ_oneds('8_optim/1_full_depth_camXs_g', full_depth_camXs_g.unbind(1), max_val=max_dist)
    sw.summ_oneds('8_optim/1_full_valid_camXs_g', full_valid_camXs_g.unbind(1), norm=False)

    print('optimizing traj')
    best_t_gens = []
    min_values = []

    t_camXs_py = t_camXs.cpu().reshape(-1).numpy() # S*3
    xyz_py = xyz_camX_list_full[0].cpu().numpy()[0] # N, 3
    optimizer = CMA(mean=t_camXs_py, sigma=sigma, population_size=population_size) # no bounds for now
        
    min_value = np.inf
    best_t_camXs = t_camXs_py.copy()

    stale_count = 0
    generation_count = 0

    while stale_count < max_stale and generation_count < max_generations:
        generation_count += 1
        improved_things = False
        solutions = []

        for pi in range(optimizer.population_size):

            t_camXs_py = optimizer.ask()

            t_camXs = torch.from_numpy(t_camXs_py).reshape(1, S, 3).float().to('cuda')
            lrt_camXs = update_lrtlist_with_tlist(lrt_camXs, t_camXs)
            lrt_camXs[:,seed_s] = lrt_camX_bak

            total_depth_error = 0
            
            for s in range(S):
                obj_depth_camX, obj_valid_camX = render_one_object_2d(obj_xyz_obj, lrt_camXs[:,s], pix_T_cam, H, W, max_dist)
                
                full_valid_camX = (bkg_valid_camX + obj_valid_camX).clamp(0,1)
                full_depth_camX_e = torch.min(torch.cat([obj_depth_camX,
                                                         bkg_depth_camX], dim=1), dim=1, keepdim=True)[0]

                depth_error = torch.abs(full_depth_camX_e - full_depth_camXs_g[:,s]).clamp(0,0.1)
                # utils.basic.print_stats('depth_error', depth_error)
                total_depth_error += torch.mean(depth_error)

            use_accel = False
            if use_accel:
                velo = t_camXs[:,1:] - t_camXs[:,:-1]
                accel = velo[:,1:] - velo[:,:-1]
                accel_error = torch.sum(torch.abs(accel)) # higher -> worse
                value = total_depth_error.item() + accel_error.item()
            else:
                value = total_depth_error.item()

            if value < min_value:
                min_value = value
                best_t_camXs_py = t_camXs_py.copy()
                improved_things = True

            solutions.append((t_camXs_py, value))

        if improved_things:
            stale_count = 0
        else:
            stale_count += 1
        print(f"{generation_count} {stale_count} {value} {min_value}")
        optimizer.tell(solutions)

    print('best_t_camXs_py', best_t_camXs_py.reshape(S, 3))
    
    t_camXs = torch.from_numpy(best_t_camXs_py).reshape(1, S, 3).float().to('cuda')
    lrt_camXs = update_lrtlist_with_tlist(lrt_camXs, t_camXs)
    lrt_camXs[:,seed_s] = lrt_camX_bak

    full_depth_camXs_e = []
    for s in range(S):
        obj_depth_camX, _ = render_one_object_2d(obj_xyz_obj, lrt_camXs[:,s], pix_T_cam, H, W, max_dist)
        full_depth_camX_e = torch.min(torch.cat([obj_depth_camX,
                                                 bkg_depth_camX], dim=1), dim=1, keepdim=True)[0]
        full_depth_camXs_e.append(full_depth_camX_e)
    full_depth_camXs_e = torch.stack(full_depth_camXs_e, dim=1)
    # print('full_depth_camXs', full_depth_camXs.shape)
    sw.summ_oneds('8_optim/full_depth_camXs_e', full_depth_camXs_e.unbind(1), max_val=max_dist)
    sw.summ_oneds('8_optim/diff_depth_camXs_e', (torch.abs(full_depth_camXs_e - full_depth_camXs_g)).unbind(1))
    
    vis = []
    for s in range(S):
        vis.append(sw.summ_lrtlist(
            '',
            rgb_camXs[:,s],
            lrt_camXs[:,s:s+1],
            torch.ones_like(lrt_camXs[:,s:s+1,0]),
            torch.ones_like(lrt_camXs[:,s:s+1,0]).long(),
            pix_T_cam,
            # # frame_id=s,
            # frame_id=np.mean(min_values),
            frame_id=min_value,
            only_return=True)
        )
    sw.summ_rgbs('8_optim/optim_traj', vis)

    return lrt_camXs

def featimg_optim_free_traj_2d(
        rgb_camXs,
        box2d_camX,
        obj_mask_camX,
        obj_box_camX, # the box of obj_mask_camX, mainly for translation purpose
        obj_feat_camX,
        bkg_feat_camX,
        seed_s,
        H, W,
        sw,
        max_dist=16,
        sigma=0.1,
        population_size=5,
        max_stale=10,
        max_generations=80,
):
    # obj_xyz_camX is 1 x N0 x 3, obj_feat_camX is 1 x C x N0
    # bkg_xyz_camX is 1 x N1 x 3, bkg_feat_camX is 1 x C x N1
    B, S, C, H, W = rgb_camXs.shape
    assert (B == 1)
    assert (C == 3)

    # a small update: now it can take a short traj, S1 is the length of the short traj.
    # if S1=1, same as before.
    S1, _ = lrt_camX.shape
    assert (S1 < S)

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    full_feat_camX = obj_mask_camX * obj_feat_camX + (1 - obj_mask_camX) * bkg_feat_camX
    sw.summ_rgb('8_optim/0_full_featimg_camX', full_feat_camX, frame_id=0)
    sw.summ_rgb('8_optim/0_obj_featimg_camX', obj_feat_camX)
    sw.summ_rgb('8_optim/0_bkg_featimg_camX', bkg_feat_camX)

    # use the last lrt to init all else
    box2d_camXs = box2d_camX.unsqueeze(1).repeat(1, S, 1)
    box2d_camX_bak = box2d_camX.clone()

    t_camXs = box2d_camXs[:, :, :2]

    sw.summ_rgbs('8_optim/1_full_featimg_camXs_g', rgb_camXs.unbind(1))

    print('optimizing traj')
    best_t_gens = []
    min_values = []

    t_camXs_py = t_camXs.cpu().reshape(-1).numpy()  # S*2
    optimizer = CMA(mean=t_camXs_py, sigma=sigma, population_size=population_size)  # no bounds for now

    min_value = np.inf
    best_t_camXs = t_camXs_py.copy()

    stale_count = 0
    generation_count = 0

    while stale_count < max_stale and generation_count < max_generations:
        generation_count += 1
        improved_things = False
        solutions = []

        for pi in range(optimizer.population_size):

            t_camXs_py = optimizer.ask()

            t_camXs = torch.from_numpy(t_camXs_py).reshape(1, S, 3).float().to('cuda')
            box2d_camXs[:, :, :2] = t_camXs
            box2d_camXs[0, seed_s:seed_s+S1] = box2d_camX_bak

            total_featimg_error = 0

            for s in range(S):
                full_featimg_camX_e = render_full_scene_2d(obj_feat_camX, obj_mask_camX, obj_box_camX, box2d_camXs, bkg_feat_camX)
                featimg_loss = torch.abs(full_featimg_camX_e - rgb_camXs[:, s])
                # utils.basic.print_stats('depth_error', depth_error)
                total_featimg_error += torch.mean(featimg_loss)

            use_accel = False
            if use_accel:
                velo = t_camXs[:, 1:] - t_camXs[:, :-1]
                accel = velo[:, 1:] - velo[:, :-1]
                accel_error = torch.sum(torch.abs(accel))  # higher -> worse
                value = total_featimg_error.item() + accel_error.item()
            else:
                value = total_featimg_error.item()

            if value < min_value:
                min_value = value
                best_t_camXs_py = t_camXs_py.copy()
                improved_things = True

            solutions.append((t_camXs_py, value))

        if improved_things:
            stale_count = 0
        else:
            stale_count += 1
        print(f"{generation_count} {stale_count} {value} {min_value}")
        optimizer.tell(solutions)

    print('best_t_camXs_py', best_t_camXs_py.reshape(S, 2))

    t_camXs = torch.from_numpy(best_t_camXs_py).reshape(1, S, 2).float().to('cuda')
    box2d_camXs[:, :, :2] = t_camXs
    box2d_camXs[0, seed_s:seed_s + S1] = box2d_camX

    t_camXs = box2d_camXs[:, :, :2]
    print('final t_camXs', t_camXs)

    full_featimg_camXs_e = []
    for s in range(S):
        full_featimg_camX_e = render_full_scene_2d(obj_feat_camX, obj_mask_camX, obj_box_camX, box2d_camXs, bkg_feat_camX)
        full_featimg_camXs_e.append(full_featimg_camX_e)
    full_featimg_camXs_e = torch.stack(full_featimg_camXs_e, dim=1)
    # print('full_depth_camXs', full_depth_camXs.shape)
    sw.summ_rgbs('8_optim/full_featimg_camXs_e', full_featimg_camXs_e.unbind(1))
    sw.summ_rgbs('8_optim/diff_featimg_camXs_e', (full_featimg_camXs_e - rgb_camXs).unbind(1))

    vis = []
    for s in range(S):
        vis.append(sw.summ_lrtlist(
            '',
            rgb_camXs[:, s],
            lrt_camXs[:, s:s + 1],
            torch.ones_like(lrt_camXs[:, s:s + 1, 0]),
            torch.ones_like(lrt_camXs[:, s:s + 1, 0]).long(),
            pix_T_cam,
            # # frame_id=s,
            # frame_id=np.mean(min_values),
            frame_id=min_value,
            only_return=True)
        )
    sw.summ_rgbs('8_optim/optim_traj', vis)

    return lrt_camXs
