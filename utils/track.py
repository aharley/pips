import torch
import numpy as np
import utils.geom
import utils.basic
# import utils.vox
import utils.samp
import utils.misc
import torch.nn.functional as F
import sklearn

np.set_printoptions(suppress=True, precision=6, threshold=2000)

def merge_rt_py(r, t):
    # r is 3 x 3
    # t is 3 or maybe 3 x 1
    t = np.reshape(t, [3, 1])
    rt = np.concatenate((r,t), axis=1)
    # rt is 3 x 4
    br = np.reshape(np.array([0,0,0,1], np.float32), [1, 4])
    # br is 1 x 4
    rt = np.concatenate((rt, br), axis=0)
    # rt is 4 x 4
    return rt

def split_rt_py(rt):
    r = rt[:3,:3]
    t = rt[:3,3]
    r = np.reshape(r, [3, 3])
    t = np.reshape(t, [3, 1])
    return r, t

def apply_4x4_py(rt, xyz):
    # rt is 4 x 4
    # xyz is N x 3
    r, t = split_rt_py(rt)
    xyz = np.transpose(xyz, [1, 0])
    # xyz is xyz1 x 3 x N
    xyz = np.dot(r, xyz)
    # xyz is xyz1 x 3 x N
    xyz = np.transpose(xyz, [1, 0])
    # xyz is xyz1 x N x 3
    t = np.reshape(t, [1, 3])
    xyz = xyz + t
    return xyz

def rigid_transform_3d(xyz_cam0, xyz_cam1, do_ransac=True, ransac_steps=256, device='cuda', do_scaling=False):
    # inputs are N x 3
    xyz_cam0 = xyz_cam0.detach().cpu().numpy()
    xyz_cam1 = xyz_cam1.detach().cpu().numpy()
    cam1_T_cam0, scaling = rigid_transform_3d_py(xyz_cam0, xyz_cam1, do_ransac=do_ransac, ransac_steps=ransac_steps, do_scaling=do_scaling)
    cam1_T_cam0 = torch.from_numpy(cam1_T_cam0).float().to(device)
    scaling = torch.from_numpy(scaling).float().to(device)
    if do_scaling:
        return cam1_T_cam0, scaling
    else:
        return cam1_T_cam0

def batch_rigid_transform_3d(xyz0, xyz1):
    # xyz0 and xyz1 are each B x N x 3
    B, N, D = list(xyz0.shape)
    B2, N2, D2 = list(xyz1.shape)
    assert(B==B2)
    assert(N==N2)
    assert(D==3 and D2==3)
    assert(N >= 3) 

    centroid_0 = torch.mean(xyz0, dim=1, keepdim=True)
    centroid_1 = torch.mean(xyz1, dim=1, keepdim=True)
    # B x 1 x 3

    # center the points
    xyz0 = xyz0 - centroid_0
    xyz1 = xyz1 - centroid_1
    # B x N x 3

    H = torch.matmul(xyz0.transpose(1,2), xyz1)
    # B x 3 x 3

    U, S, Vt = torch.svd(H)
    # B x 3 x 3

    R = torch.matmul(Vt.transpose(1,2), U.transpose(1,2))
    # B x 3 x 3

    # # special reflection case
    # if np.linalg.det(R) < 0:
    #    Vt[2,:] *= -1
    #    R = np.dot(Vt.transpose(0,1), U.transpose(0,1))

    t = torch.matmul(-R, centroid_0.transpose(1,2)) + centroid_1.transpose(1,2)
    t = t.reshape([B, 3])

    rt = utils.geom.merge_rt(R, t)
    # this is cam1_T_cam0
    return rt

def rigid_transform_3d_py_helper(xyz0, xyz1, do_scaling=False):
    assert len(xyz0) == len(xyz1)
    N = xyz0.shape[0] # total points
    if N >= 3:
        centroid_xyz0 = np.mean(xyz0, axis=0)
        centroid_xyz1 = np.mean(xyz1, axis=0)
        # print('centroid_xyz0', centroid_xyz0)
        # print('centroid_xyz1', centroid_xyz1)

        # center the points
        xyz0 = xyz0 - np.tile(centroid_xyz0, (N, 1))
        xyz1 = xyz1 - np.tile(centroid_xyz1, (N, 1))

        H = np.dot(xyz0.T, xyz1) / N

        U, S, Vt = np.linalg.svd(H)

        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
           Vt[2,:] *= -1
           S[-1] *= -1
           R = np.dot(Vt.T, U.T) 

        # varP = np.var(xyz0, axis=0).sum()
        # c = 1/varP * np.sum(S) # scale factor
        varP = np.var(xyz0, axis=0)
        # varQ_aligned = np.var(np.dot(xyz1, R.T), axis=0)
        varQ_aligned = np.var(np.dot(xyz1, R), axis=0)

        c = np.sqrt(varQ_aligned / varP) # anisotropic

        if not do_scaling:
            # c = 1.0 # keep it 1.0
            c = np.ones(3) # keep it 1.0

        # t = c * np.dot(-R, centroid_xyz0.T) + centroid_xyz1.T
        t = -np.dot(np.dot(R, np.diag(c)), centroid_xyz0.T) + centroid_xyz1.T

        t = np.reshape(t, [3])
    else:
        # print('too few points; returning identity')
        # R = np.eye(3, dtype=np.float32)
        # t = np.zeros(3, dtype=np.float32)

        print('too few points; returning translation')
        R = np.eye(3, dtype=np.float32)
        t = np.mean(xyz1-xyz0, axis=0)
        # c = 1.0
        c = np.ones(3) # keep it 1.0
        
    rt = merge_rt_py(R, t)
    return rt, c

def rigid_transform_3d_pt_helper(xyz0, xyz1):
    assert len(xyz0) == len(xyz1)
    N = xyz0.shape[0] # total points
    assert(N >= 3)
    centroid_xyz0 = torch.mean(xyz0, axis=0, keepdim=True)
    centroid_xyz1 = torch.mean(xyz1, axis=0, keepdim=True)

    # center the points
    xyz0 = xyz0 - centroid_xyz0.repeat(N, 1)
    xyz1 = xyz1 - centroid_xyz1.repeat(N, 1)

    H = torch.matmul(xyz0.transpose(0,1), xyz1)

    U, S, Vt = torch.svd(H)

    R = torch.matmul(Vt.transpose(0,1), U.transpose(0,1))

    # # special reflection case
    # if np.linalg.det(R) < 0:
    #    Vt[2,:] *= -1
    #    R = np.dot(Vt.transpose(0,1), U.transpose(0,1))

    t = torch.matmul(-R, centroid_xyz0.transpose(0,1)) + centroid_xyz1.transpose(0,1)
    t = t.reshape([3])

    rt = utils.geom.merge_rt_single(R, t)
    return rt

def rigid_transform_3d_py(xyz0, xyz1, do_ransac=True, ransac_steps=256, do_scaling=False):
    # xyz0 and xyz1 are each N x 3
    assert len(xyz0) == len(xyz1)

    N = xyz0.shape[0] # total points

    nPts = 3 # anything >3 is ok really
    if N < nPts:
        print('N = %d; returning an translation only matrix using avg flow' % N)
        R = np.eye(3, dtype=np.float32)
        if N > 0:
            t = np.average(xyz1 - xyz0, axis=0)
        else:
            t = np.zeros(3, dtype=np.float32)
        
        rt = merge_rt_py(R, t)
        # c = 1.0
        c = np.ones(3)

    elif not do_ransac:
        rt, c = rigid_transform_3d_py_helper(xyz0, xyz1, do_scaling=do_scaling)

    else:
        # print('N = %d' % N)
        # print('doing ransac')
        rts = []
        errs = []
        cs = []
        for step in list(range(ransac_steps)):
            assert(N > nPts) 
            perm = np.random.permutation(N)
            
            cam1_T_cam0, c = rigid_transform_3d_py_helper(xyz0[perm[:nPts]], xyz1[perm[:nPts]], do_scaling=do_scaling)


            # final_cam1_T_cam0 = np.dot(cam1_T_cam0, np.diag([c,c,c,1.0])) # consider scaling
            final_cam1_T_cam0 = np.dot(cam1_T_cam0, np.diag([c[0],c[1],c[2],1.0])) # consider scaling, anisotropic
            # i got some errors in matmul when the arrays were too big, 
            # so let's just use 1k points for the error 
            perm = np.random.permutation(N)
            xyz1_prime = apply_4x4_py(final_cam1_T_cam0, xyz0[perm[:min([1000,N])]])
            xyz1_actual = xyz1[perm[:min([1000,N])]]
            err = np.mean(np.sum(np.abs(xyz1_prime-xyz1_actual), axis=1))
            rts.append(cam1_T_cam0)
            errs.append(err)
            cs.append(c)
        ind = np.argmin(errs)
        rt = rts[ind]
        c = cs[ind]
    return rt, c

def compute_mem1_T_mem0_from_object_flow(flow_mem, mask_mem, occ_mem):
    B, C, Z, Y, X = list(flow_mem.shape)
    assert(C==3)
    mem1_T_mem0 = utils.geom.eye_4x4(B)

    xyz_mem0 = utils.basic.gridcloud3d(B, Z, Y, X, norm=False)
    
    for b in list(range(B)):
        # i think there is a way to parallelize the where/gather but it is beyond me right now
        occ = occ_mem[b]
        mask = mask_mem[b]
        flow = flow_mem[b]
        xyz0 = xyz_mem0[b]
        # cam_T_obj = camR_T_obj[b]
        # mem_T_cam = mem_T_ref[b]

        flow = flow.reshape(3, -1).permute(1, 0)
        # flow is -1 x 3
        inds = torch.where((occ*mask).reshape(-1) > 0.5)
        # inds is ?
        flow = flow[inds]

        xyz0 = xyz0[inds]
        xyz1 = xyz0 + flow

        mem1_T_mem0_ = rigid_transform_3d(xyz0, xyz1)
        # this is 4 x 4 
        mem1_T_mem0[b] = mem1_T_mem0_

    return mem1_T_mem0

def compute_mem1_T_mem0_from_object_flow(flow_mem, mask_mem, occ_mem):
    B, C, Z, Y, X = list(flow_mem.shape)
    assert(C==3)
    mem1_T_mem0 = utils.geom.eye_4x4(B)

    xyz_mem0 = utils.basic.gridcloud3d(B, Z, Y, X, norm=False)
    
    for b in list(range(B)):
        # i think there is a way to parallelize the where/gather but it is beyond me right now
        occ = occ_mem[b]
        mask = mask_mem[b]
        flow = flow_mem[b]
        xyz0 = xyz_mem0[b]
        # cam_T_obj = camR_T_obj[b]
        # mem_T_cam = mem_T_ref[b]

        flow = flow.reshape(3, -1).permute(1, 0)
        # flow is -1 x 3
        inds = torch.where((occ*mask).reshape(-1) > 0.5)
        # inds is ?
        flow = flow[inds]

        xyz0 = xyz0[inds]
        xyz1 = xyz0 + flow

        mem1_T_mem0_ = rigid_transform_3d(xyz0, xyz1)
        # this is 4 x 4 
        mem1_T_mem0[b] = mem1_T_mem0_

    return mem1_T_mem0


def track_via_chained_flows(
        lrt_camIs_g,
        mask_mem0,
        model,
        occ_mems,
        occ_mems_half,
        unp_mems,
        summ_writer,
        include_image_summs=False,
        use_live_nets=False,
):
    B, S, _, Z, Y, X = list(occ_mems.shape)
    B, S, _, Z2, Y2, X2 = list(occ_mems_half.shape)
    
    flow_mem0 = torch.zeros(B, 3, Z2, Y2, X2, dtype=torch.float32, device=torch.device('cuda'))
    cam0_T_camI = utils.geom.eye_4x4(B)

    obj_lengths, cams_T_obj0 = utils.geom.split_lrtlist(lrt_camIs_g)
    # this is B x S x 4 x 4

    cam0_T_obj = cams_T_obj0[:,0]
    obj_length = obj_lengths[:,0]

    occ_mem0 = occ_mems_half[:,0]

    input_mems = torch.cat([occ_mems, occ_mems*unp_mems], dim=2)

    mem_T_cam = utils.vox.get_mem_T_ref(B, Z2, Y2, X2)
    cam_T_mem = utils.vox.get_ref_T_mem(B, Z2, Y2, X2)

    lrt_camIs_e = torch.zeros_like(lrt_camIs_g)
    lrt_camIs_e[:,0] = lrt_camIs_g[:,0] # init with gt box on frame0

    all_ious = []
    for s in list(range(1, S)):
        input_mem0 = input_mems[:,0]
        input_memI = input_mems[:,s]

        if use_live_nets:

            use_rigid_warp = True
            if use_rigid_warp:
                xyz_camI = model.xyz_camX0s[:,s]
                xyz_camI = utils.geom.apply_4x4(cam0_T_camI, xyz_camI)
                occ_memI = utils.vox.voxelize_xyz(xyz_camI, Z, Y, X)
                unp_memI = unp_mems[:,s]
                unp_memI = utils.vox.apply_4x4_to_vox(cam0_T_camI, unp_memI, already_mem=False, binary_feat=False)
                input_memI = torch.cat([occ_memI, occ_memI*unp_memI], dim=1)
                # input_memI = utils.vox.apply_4x4_to_vox(cam0_T_camI, input_memI, already_mem=False, binary_feat=False)
            else:
                input_memI = utils.samp.backwarp_using_3d_flow(input_memI, F.interpolate(flow_mem0, scale_factor=2, mode='trilinear'))
                
            featnet_output_mem0, _, _ = model.featnet(input_mem0, None, None)
            featnet_output_memI, _, _ = model.featnet(input_memI, None, None)
            _, residual_flow_mem0 = model.flownet(
                    featnet_output_mem0,
                    featnet_output_memI,
                    torch.zeros([B, 3, Z2, Y2, X2]).float().cuda(),
                    occ_mem0, 
                    False,
                    None)
        else:
            featnet_output_mem0 = model.feat_net.infer_pt(input_mem0)
            featnet_output_memI = model.feat_net.infer_pt(input_memI)
            featnet_output_memI = utils.samp.backwarp_using_3d_flow(featnet_output_memI, flow_mem0)
            residual_flow_mem0 = model.flow_net.infer_pt([featnet_output_mem0,
                                                          featnet_output_memI])

        # if use_live_nets:
        #     _, residual_flow_mem0 = model.flownet(
        #             featnet_output_mem0,
        #             featnet_output_memI,
        #             torch.zeros([B, 3, Z2, Y2, X2]).float().cuda(),
        #             occ_mem0, 
        #             False,
        #             summ_writer)
        # else:
        #     residual_flow_mem0 = model.flow_net.infer_pt([featnet_output_mem0,
        #                                                  featnet_output_memI])

        flow_mem0 = flow_mem0 + residual_flow_mem0

        if include_image_summs:
            summ_writer.summ_feats('3d_feats/featnet_inputs_%02d' % s, [input_mem0, input_memI], pca=True)
            summ_writer.summ_feats('3d_feats/featnet_outputs_warped_%02d' % s, [featnet_output_mem0, featnet_output_memI], pca=True)
            summ_writer.summ_3d_flow('flow/residual_flow_mem0_%02d' % s, residual_flow_mem0, clip=0.0)
            summ_writer.summ_3d_flow('flow/residual_masked_flow_mem0_%02d' % s, residual_flow_mem0*mask_mem0, clip=0.0)
            summ_writer.summ_3d_flow('flow/flow_mem0_%02d' % s, flow_mem0, clip=0.0)

        # compute the rigid motion of the object; we will use this for eval
        memI_T_mem0 = compute_mem1_T_mem0_from_object_flow(
            flow_mem0, mask_mem0, occ_mem0)
        mem0_T_memI = utils.geom.safe_inverse(memI_T_mem0)
        cam0_T_camI = utils.basic.matmul3(cam_T_mem, mem0_T_memI, mem_T_cam)

        # eval
        camI_T_obj = utils.basic.matmul4(cam_T_mem, memI_T_mem0, mem_T_cam, cam0_T_obj)
        # this is B x 4 x 4

        lrt_camIs_e[:,s] = utils.geom.merge_lrt(obj_length, camI_T_obj)
        ious = utils.geom.get_iou_from_corresponded_lrtlists(lrt_camIs_e[:,s:s+1],
                                                             lrt_camIs_g[:,s:s+1])
        all_ious.append(ious)
        summ_writer.summ_scalar('box/mean_iou_%02d' % s, torch.mean(ious).cpu().item())

    # lrt_camIs_e is B x S x 19
    # this is B x S x 1 x 19
    return lrt_camIs_e, all_ious

def cross_corr_with_template(search_region, template):
    B, C, ZZ, ZY, ZX = list(template.shape)
    B2, C2, Z, Y, X = list(search_region.shape)
    assert(B==B2)
    assert(C==C2)
    corr = []

    Z_new = Z-ZZ+1
    Y_new = Y-ZY+1
    X_new = X-ZX+1
    corr = torch.zeros([B, 1, Z_new, Y_new, X_new]).float().cuda()

    # this loop over batch is ~2x faster than the grouped version
    for b in list(range(B)):
        search_region_b = search_region[b:b+1]
        template_b = template[b:b+1]
        corr[b] = F.conv3d(search_region_b, template_b).squeeze(0)

    # grouped version, for reference:
    # corr = F.conv3d(search_region.view(1, B*C, Z, Y, X), template, groups=B) # fast valid conv
    
    # adjust the scale of responses, for stability early on
    corr = 0.001 * corr

    # since we did valid conv (which is smart), the corr map is offset from the search region
    # so we need to offset the xyz of the answer
    # _, _, Z_new, Y_new, X_new = list(corr.shape)
    Z_clipped = (Z - Z_new)/2.0
    Y_clipped = (Y - Y_new)/2.0
    X_clipped = (X - X_new)/2.0
    xyz_offset = np.array([X_clipped, Y_clipped, Z_clipped], np.float32).reshape([1, 3])
    xyz_offset = torch.from_numpy(xyz_offset).float().to('cuda')
    return corr, xyz_offset

def cross_corr_with_templates(search_region, templates):
    B, C, Z, Y, X = list(search_region.shape)
    B2, N, C2, ZZ, ZY, ZX = list(templates.shape)
    assert(B==B2)
    assert(C==C2)

    Z_new = Z-ZZ+1
    Y_new = Y-ZY+1
    X_new = X-ZX+1
    corr = torch.zeros([B, N, Z_new, Y_new, X_new]).float().cuda()

    # this loop over batch is ~2x faster than the grouped version
    for b in list(range(B)):
        search_region_b = search_region[b:b+1]
        for n in list(range(N)):
            template_b = templates[b:b+1,n]
            corr[b,n] = F.conv3d(search_region_b, template_b).squeeze(0)

    # grouped version, for reference:
    # corr = F.conv3d(search_region.view(1, B*C, Z, Y, X), template, groups=B) # fast valid conv
    
    # adjust the scale of responses, for stability early on
    corr = 0.01 * corr

    # since we did valid conv (which is smart), the corr map is offset from the search region
    # so we need to offset the xyz of the answer
    # _, _, Z_new, Y_new, X_new = list(corr.shape)
    Z_clipped = (Z - Z_new)/2.0
    Y_clipped = (Y - Y_new)/2.0
    X_clipped = (X - X_new)/2.0
    xyz_offset = np.array([X_clipped, Y_clipped, Z_clipped], np.float32).reshape([1, 3])
    xyz_offset = torch.from_numpy(xyz_offset).float().to('cuda')
    return corr, xyz_offset

def track_via_inner_products(lrt_camIs_g, mask_mems, feat_mems, vox_util, mask_boxes=False, summ_writer=None):
    B, S, feat3d_dim, Z, Y, X = list(feat_mems.shape)

    feat_vecs = feat_mems.view(B, S, feat3d_dim, -1)
    # this is B x S x C x huge

    feat0_vec = feat_vecs[:,0]
    # this is B x C x huge
    feat0_vec = feat0_vec.permute(0, 2, 1)
    # this is B x huge x C

    obj_mask0_vec = mask_mems[:,0].reshape(B, -1).round()
    # this is B x huge

    orig_xyz = utils.basic.gridcloud3d(B, Z, Y, X)
    # this is B x huge x 3

    obj_lengths, cams_T_obj0 = utils.geom.split_lrtlist(lrt_camIs_g)
    obj_length = obj_lengths[:,0]
    # this is B x S x 4 x 4

    # this is B x S x 4 x 4
    cam0_T_obj = cams_T_obj0[:,0]

    lrt_camIs_e = torch.zeros_like(lrt_camIs_g)
    # we will fill this up

    mask_e_mems = torch.zeros_like(mask_mems)
    mask_e_mems_masked = torch.zeros_like(mask_mems)

    mem_T_cam = vox_util.get_mem_T_ref(B, Z, Y, X)
    cam_T_mem = vox_util.get_ref_T_mem(B, Z, Y, X)

    ious = torch.zeros([B, S]).float().cuda()
    point_counts = np.zeros([B, S])
    for s in list(range(S)):
        feat_vec = feat_vecs[:,s]
        feat_vec = feat_vec.permute(0, 2, 1)
        # B x huge x C

        memI_T_mem0 = utils.geom.eye_4x4(B)
        # we will fill this up

        # Use ground truth box to mask
        if s == 0:
            lrt = lrt_camIs_g[:,0].unsqueeze(1)
        # Use predicted box to mask
        else:
            lrt = lrt_camIs_e[:,s-1].unsqueeze(1)

        # Equal box length
        lrt[:,:,:3] = torch.ones_like(lrt[:,:,:3])*10

        # Remove rotation
        transform = lrt[:,:,3:].reshape(B, 1, 4, 4)
        transform[:,:,:3,:3] = torch.eye(3).unsqueeze(0).unsqueeze(0)
        transform = transform.reshape(B,-1)

        lrt[:,:,3:] = transform

        box_mask = vox_util.assemble_padded_obj_masklist(lrt, torch.ones(1,1).cuda(), Z, Y, X)

        # to simplify the impl, we will iterate over the batchmin
        for b in list(range(B)):
            feat_vec_b = feat_vec[b]
            feat0_vec_b = feat0_vec[b]
            obj_mask0_vec_b = obj_mask0_vec[b]
            orig_xyz_b = orig_xyz[b]
            # these are huge x C

            obj_inds_b = torch.where(obj_mask0_vec_b > 0)
            obj_vec_b = feat0_vec_b[obj_inds_b]
            xyz0 = orig_xyz_b[obj_inds_b]
            # these are med x C

            obj_vec_b = obj_vec_b.permute(1, 0)
            # this is is C x med

            corr_b = torch.matmul(feat_vec_b, obj_vec_b)
            # this is huge x med

            heat_b = corr_b.permute(1, 0).reshape(-1, 1, Z, Y, X)
            # this is med x 1 x Z4 x Y4 x X4

            
            # Mask by box to restrict area
            if mask_boxes:
                
                # Vanilla heatmap
                if summ_writer != None:
                    heat_map = heat_b.max(0)[0]
                    summ_writer.summ_feat("heatmap/vanilla", heat_map.unsqueeze(0), pca=False)
                    mask_e_mems[b,s] = heat_map
                
                box_mask = box_mask.squeeze(0).repeat(heat_b.shape[0],1,1,1,1)
                heat_b = heat_b*box_mask
               
                # Masked heatmap
                if summ_writer != None:
                    heat_map = heat_b.max(0)[0]
                    mask_e_mems_masked[b,s] = heat_map

                    summ_writer.summ_feat("heatmap/masked", heat_map.unsqueeze(0), pca=False)
            
            
            # for numerical stability, we sub the max, and mult by the resolution
            heat_b_ = heat_b.reshape(-1, Z*Y*X)
            heat_b_max = (torch.max(heat_b_, dim=1).values).reshape(-1, 1, 1, 1, 1)
            heat_b = heat_b - heat_b_max
            heat_b = heat_b * float(len(heat_b[0].reshape(-1)))

            xyzI = utils.basic.argmax3d(heat_b*float(Z*10), hard=False, stack=True)
            # this is med x 3
            memI_T_mem0[b] = rigid_transform_3d(xyz0, xyzI)

            # record #points, since ransac depends on this
            point_counts[b, s] = len(xyz0)
        # done stepping through batch

        mem0_T_memI = utils.geom.safe_inverse(memI_T_mem0)
        cam0_T_camI = utils.basic.matmul3(cam_T_mem, mem0_T_memI, mem_T_cam)

        # eval
        camI_T_obj = utils.basic.matmul4(cam_T_mem, memI_T_mem0, mem_T_cam, cam0_T_obj)
        # this is B x 4 x 4
        lrt_camIs_e[:,s] = utils.geom.merge_lrt(obj_length, camI_T_obj)
        ious[:,s] = utils.geom.get_iou_from_corresponded_lrtlists(lrt_camIs_e[:,s:s+1], lrt_camIs_g[:,s:s+1]).squeeze(1)

    if summ_writer != None:
        summ_writer.summ_feats('heatmap/mask_e_memX0s', torch.unbind(mask_e_mems, dim=1), pca=False)
        summ_writer.summ_feats('heatmap/mask_e_memX0s_masked', torch.unbind(mask_e_mems_masked, dim=1), pca=False)

    return lrt_camIs_e, point_counts, ious

                             
def convert_corr_to_xyz(corr, xyz_offset, hard=True):
    # corr is B x 1 x Z x Y x X
    # xyz_offset is 1 x 3
    peak_z, peak_y, peak_x = utils.basic.argmax3d(corr, hard=hard)
    # these are B
    peak_xyz_corr = torch.stack([peak_x, peak_y, peak_z], dim=1)
    # this is B x 3, and in corr coords
    peak_xyz_search = xyz_offset + peak_xyz_corr
    # this is B x 3, and in search coords
    return peak_xyz_search

def convert_corrlist_to_xyzr(corrlist, radlist, xyz_offset, hard=True):
    # corrlist is a list of N different B x Z x Y x X tensors
    # radlist is N angles in radians
    # xyz_offset is 1 x 3
    corrcat = torch.stack(corrlist, dim=1)
    # this is B x N x Z x Y x X
    radcat = torch.from_numpy(np.array(radlist).astype(np.float32)).cuda()
    radcat = radcat.reshape(-1)
    # this is N
    peak_r, peak_z, peak_y, peak_x = utils.basic.argmax3dr(corrcat, radcat, hard=hard)
    # these are B
    peak_xyz_corr = torch.stack([peak_x, peak_y, peak_z], dim=1)
    # this is B x 3, and in corr coords
    peak_xyz_search = xyz_offset + peak_xyz_corr
    # this is B x 3, and in search coords
    return peak_r, peak_xyz_search

def convert_corrs_to_xyzr(corrcat, radcat, xyz_offset, hard=True, grid=None):
    # corrcat is B x N x Z x Y x X tensors
    # radcat is N
    # xyz_offset is 1 x 3
    # if grid is None we'll compute it during the argmax
    peak_r, peak_z, peak_y, peak_x = utils.basic.argmax3dr(corrcat, radcat, hard=hard, grid=grid)
    # these are B
    peak_xyz_corr = torch.stack([peak_x, peak_y, peak_z], dim=1)
    # this is B x 3, and in corr coords
    peak_xyz_search = xyz_offset + peak_xyz_corr
    # this is B x 3, and in search coords
    return peak_r, peak_xyz_search

def convert_corrs_to_xyzr3(corrcat, radcat, xyz_offset, hard=True, grid=None):
    # this function is to deal with rotation in three dimensions
    # corrcat is B x N x Z x Y x X tensors
    # radcat is N x 3
    # xyz_offset is 1 x 3
    # if grid is None we'll compute it during the argmax
    peak_rx, peak_ry, peak_rz, peak_z, peak_y, peak_x = utils.basic.argmax3dr3(corrcat, radcat, hard=hard, grid=grid)
    # these are B
    peak_xyz_corr = torch.stack([peak_x, peak_y, peak_z], dim=1)
    # this is B x 3, and in corr coords
    peak_xyz_search = xyz_offset + peak_xyz_corr
    # this is B x 3, and in search coords
    peak_r = torch.stack([peak_rx, peak_ry, peak_rz], dim=1)
    # this is B x 3
    return peak_r, peak_xyz_search

def remask_via_inner_products(lrt_camIs_g, mask_mems, feat_mems, vox_util, mask_distance=False, summ_writer=None):
    B, S, feat3d_dim, Z, Y, X = list(feat_mems.shape)

    mask = mask_mems[:,0]
    distance_masks = torch.zeros_like(mask_mems)
    
    feat_vecs = feat_mems.view(B, S, feat3d_dim, -1)
    # this is B x S x C x huge

    feat0_vec = feat_vecs[:,0]
    # this is B x C x huge
    feat0_vec = feat0_vec.permute(0, 2, 1)
    # this is B x huge x C
    
    obj_mask0_vec = mask_mems[:,0].reshape(B, -1).round()
    # this is B x huge
    
    orig_xyz = utils.basic.gridcloud3d(B, Z, Y, X)
    # this is B x huge x 3
    
    obj_lengths, cams_T_obj0 = utils.geom.split_lrtlist(lrt_camIs_g)
    obj_length = obj_lengths[:,0]
    # this is B x S x 4 x 4
    
    # this is B x S x 4 x 4
    cam0_T_obj = cams_T_obj0[:,0]
    
    lrt_camIs_e = torch.zeros_like(lrt_camIs_g)
    # we will fill this up

    mem_T_cam = vox_util.get_mem_T_ref(B, Z, Y, X)
    cam_T_mem = vox_util.get_ref_T_mem(B, Z, Y, X)

    mask_e_mems = torch.zeros_like(mask_mems)
    mask_e_mems_thres = torch.zeros_like(mask_mems)
    mask_e_mems_hard  = torch.zeros_like(mask_mems)
    mask_e_mems_spatial = torch.zeros_like(mask_mems)

    ious = torch.zeros([B, S]).float().cuda()
    ious_hard = torch.zeros([B, S]).float().cuda()
    ious_spatial = torch.zeros([B, S]).float().cuda()

    point_counts = np.zeros([B, S])
    rough_centroids_mem = torch.zeros(B, S, 3).float().cuda()
    for s in list(range(S)):
        feat_vec = feat_vecs[:,s]
        feat_vec = feat_vec.permute(0, 2, 1)
        # B x huge x C

        memI_T_mem0 = utils.geom.eye_4x4(B)
        # we will fill this up
        # to simplify the impl, we will iterate over the batchmin
        
        for b in list(range(B)):
            # Expand mask
            # Code for growing a distance constraint mask
            if s == 0:
                 distance_mask = mask_mems[b,s]
                 distance_masks[b,s] = distance_mask
            else:
                 distance_mask = torch.nn.functional.conv3d(distance_masks[b,s-1].unsqueeze(0), torch.ones(1,1,3,3,3).cuda(), padding=1)
                 distance_mask = (distance_mask > 0).float()
                 distance_masks[b,s] = distance_mask

            distance_mask = distance_mask.reshape(X*Y*Z)

            feat_vec_b  = feat_vec[b]
            feat0_vec_b = feat0_vec[b]
            obj_mask0_vec_b = obj_mask0_vec[b]
            orig_xyz_b = orig_xyz[b]
            # these are huge x C
            
            obj_inds_b = torch.where(obj_mask0_vec_b > 0)
            obj_vec_b = feat0_vec_b[obj_inds_b]
            xyz0 = orig_xyz_b[obj_inds_b]
            # these are med x C
            
            obj_vec_b = obj_vec_b.permute(1, 0)
            # this is is C x med
            
            # Calculate similarities
            similarity_b = torch.exp(torch.matmul(feat_vec_b, obj_vec_b))
            
            # Remove entries which could never happen
            if mask_distance == True:
                similarity_b = torch.mul(distance_mask.repeat(similarity_b.shape[1],1).permute(1,0),similarity_b)
            
            # calculate attention
            similarity_b = similarity_b/torch.sum(similarity_b, dim=0, keepdim=True)    
            
            num_mask_channels = similarity_b.shape[1]
            
            # Calculate hard attention
            similarity_argmax = similarity_b.max(0)[1]
            hard_attention = torch.zeros_like(similarity_b)
            
            for i in range(num_mask_channels):
                hard_attention[similarity_argmax[i], i] = 1
    
            # Calculate positional average attention
            spatial_attention = hard_attention.permute(1,0)

            grid = utils.basic.gridcloud3d(1,Z,Y,X)
            
            pos_average = torch.zeros(3)
            
            spatial_attention_mask = torch.zeros(Z,Y,X)
            for i in range(num_mask_channels):
                weighted_grid = torch.mul(grid.squeeze(0), spatial_attention[i].unsqueeze(1))
                grid_average  = torch.sum(weighted_grid, dim=0)
                grid_average  = torch.round(grid_average)
                spatial_attention_mask[list(grid_average.long())] = 1
                #pos_average = pos_average + grid_average
                #import ipdb; ipdb.set_trace()

            #pos_average = pos_average/num_mask_channels
            #import ipdb; ipdb.set_trace()

            # this is huge x med normalized
            
            obj_mask0_vec_b = obj_mask0_vec[b]
            values = obj_mask0_vec_b[obj_inds_b].unsqueeze(1)
            #this is med x C

            # Propagated values are 1
            mask_e_mem = torch.matmul(similarity_b,values)
            mask_e_mem_hard = torch.matmul(hard_attention,values)
            # this is huge x 1
            
            # Threshold probabilities to be 1 close to the max
            mask_e_mem_t = (mask_e_mem > (mask_e_mem.mean()*0.3 + mask_e_mem.max()*0.7)).float()
            # this is huge x 1

            # Constrain to search region
            #mask_e_mem = torch.mul(distance_mask,mask_e_mem.reshape(-1))

            mask_e_mems[b,s] = mask_e_mem.reshape(1,Z,Y,X)
            mask_e_mems_thres[b,s] = mask_e_mem_t.reshape(1,Z,Y,X)
            mask_e_mems_hard[b,s]  = mask_e_mem_hard.reshape(1,Z,Y,X)
            mask_e_mems_spatial[b,s] = spatial_attention_mask.reshape(1,Z,Y,X)

            set_A = mask_mems[b,s].reshape(Z*Y*X).bool()
            set_B = mask_e_mem_t.reshape(Z*Y*X).bool()

            iou = sklearn.metrics.jaccard_score(set_A.bool().cpu().data.numpy(), set_B.bool().cpu().data.numpy(), average='binary')
            ious[b,s] = iou

            iou_hard = sklearn.metrics.jaccard_score(mask_mems[b,s].reshape(Z*Y*X).bool().bool().cpu().data.numpy(), mask_e_mem_hard.reshape(Z*Y*X).bool().cpu().data.numpy(), average='binary') 
            ious_hard[b,s] = iou_hard

            iou_spatial = sklearn.metrics.jaccard_score(mask_mems[b,s].reshape(Z*Y*X).bool().bool().cpu().data.numpy(), spatial_attention_mask.reshape(Z*Y*X).bool().cpu().data.numpy(), average='binary') 
            ious_spatial[b,s] = iou_spatial


    # Visualization Logs
    if summ_writer != None:
        summ_writer.summ_feats('track/mask_e_memX0s', torch.unbind(mask_e_mems, dim=1), pca=False)
        summ_writer.summ_feats('track/mask_e_memX0s_t', torch.unbind(mask_e_mems_thres, dim=1), pca=False)
        summ_writer.summ_feats('track/mask_e_memX0s_h', torch.unbind(mask_e_mems_hard, dim=1), pca=False)
        summ_writer.summ_feats('track/mask_e_memX0s_s', torch.unbind(mask_e_mems_spatial, dim=1), pca=False)

        for s in range(S):
            summ_writer.summ_scalar('track/mean_iou_hard_%02d' % s, torch.mean(ious_hard[:,s]).cpu().item())
            summ_writer.summ_scalar('track/mean_iou_spatial_%02d' % s, torch.mean(ious_spatial[:,s]).cpu().item())

    return ious
    
