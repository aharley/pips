import torch
# import hyperparams as hyp
import numpy as np
import utils.geom
import utils.samp
import utils.improc
import torch.nn.functional as F
import utils.basic
from utils.basic import print_, print_stats

class Vox_util(object):
    def __init__(self, Z, Y, X, scene_centroid, bounds, pad=None, assert_cube=False):
        # on every step, we create this object
        
        self.XMIN, self.XMAX, self.YMIN, self.YMAX, self.ZMIN, self.ZMAX = bounds
        # self.XMIN = self.XMIN.cpu().item()
        # self.XMAX = self.XMAX.cpu().item()
        # self.YMIN = self.YMIN.cpu().item()
        # self.YMAX = self.YMAX.cpu().item()
        # self.ZMIN = self.ZMIN.cpu().item()
        # self.ZMAX = self.ZMAX.cpu().item()
            
        # print('bounds for this iter:',
        #       'X = %.2f to %.2f' % (self.XMIN, self.XMAX), 
        #       'Y = %.2f to %.2f' % (self.YMIN, self.YMAX), 
        #       'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX), 
        # )
        # scene_centroid is B x 3
        B, D = list(scene_centroid.shape)
        # this specifies the location of the world box

        self.Z, self.Y, self.X = Z, Y, X

        scene_centroid = scene_centroid.detach().cpu().numpy()
        x_centroid, y_centroid, z_centroid = scene_centroid[0]
        self.XMIN += x_centroid
        self.XMAX += x_centroid
        self.YMIN += y_centroid
        self.YMAX += y_centroid
        self.ZMIN += z_centroid
        self.ZMAX += z_centroid
        # print('bounds for this iter:',
        #       'X = %.2f to %.2f' % (self.XMIN, self.XMAX), 
        #       'Y = %.2f to %.2f' % (self.YMIN, self.YMAX), 
        #       'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX), 
        # )

        self.default_vox_size_X = (self.XMAX-self.XMIN)/float(X)
        self.default_vox_size_Y = (self.YMAX-self.YMIN)/float(Y)
        self.default_vox_size_Z = (self.ZMAX-self.ZMIN)/float(Z)
        # print('self.default_vox_size_X', self.default_vox_size_X)
        # print('self.default_vox_size_Y', self.default_vox_size_Y)
        # print('self.default_vox_size_Z', self.default_vox_size_Z)

        if pad:
            Z_pad, Y_pad, X_pad = pad
            self.ZMIN -= self.default_vox_size_Z * Z_pad
            self.ZMAX += self.default_vox_size_Z * Z_pad
            self.YMIN -= self.default_vox_size_Y * Y_pad
            self.YMAX += self.default_vox_size_Y * Y_pad
            self.XMIN -= self.default_vox_size_X * X_pad
            self.XMAX += self.default_vox_size_X * X_pad

        if assert_cube:
            # we assume cube voxels
            if (not np.isclose(self.default_vox_size_X, self.default_vox_size_Y)) or (not np.isclose(self.default_vox_size_X, self.default_vox_size_Z)):
                print('Z, Y, X', Z, Y, X)
                print('bounds for this iter:',
                      'X = %.2f to %.2f' % (self.XMIN, self.XMAX), 
                      'Y = %.2f to %.2f' % (self.YMIN, self.YMAX), 
                      'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX), 
                )
                print('self.default_vox_size_X', self.default_vox_size_X)
                print('self.default_vox_size_Y', self.default_vox_size_Y)
                print('self.default_vox_size_Z', self.default_vox_size_Z)
            assert(np.isclose(self.default_vox_size_X, self.default_vox_size_Y))
            assert(np.isclose(self.default_vox_size_X, self.default_vox_size_Z))

    def Ref2Mem(self, xyz, Z, Y, X, assert_cube=False):
        # xyz is B x N x 3, in ref coordinates
        # transforms ref coordinates into mem coordinates
        B, N, C = list(xyz.shape)
        device = xyz.device
        assert(C==3)
        mem_T_ref = self.get_mem_T_ref(B, Z, Y, X, assert_cube=assert_cube, device=device)
        xyz = utils.geom.apply_4x4(mem_T_ref, xyz)
        return xyz
    
    def apply_mem_T_ref_to_lrtlist(self, lrtlist_cam, Z, Y, X, assert_cube=False):
        # lrtlist is B x N x 19, in cam coordinates
        # transforms them into mem coordinates, including a scale change for the lengths
        B, N, C = list(lrtlist_cam.shape)
        assert(C==19)
        mem_T_cam = self.get_mem_T_ref(B, Z, Y, X, assert_cube=assert_cube)

        # apply_4x4 will work for the t part
        lenlist_cam, rtlist_cam = utils.geom.split_lrtlist(lrtlist_cam)
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rlist_cam_, tlist_cam_ = utils.geom.split_rt(__p(rtlist_cam))
        # rlist_cam_ is B*N x 3 x 3
        # tlist_cam_ is B*N x 3
        # tlist_cam = __u(tlist_cam_)
        tlist_mem_ = __p(utils.geom.apply_4x4(mem_T_cam, __u(tlist_cam_)))
        # rlist does not need to change, since cam is aligned with mem
        rlist_mem_ = rlist_cam_.clone()
        rtlist_mem = __u(utils.geom.merge_rt(rlist_mem_, tlist_mem_))
        # this is B x N x 4 x 4

        # next we need to scale the lengths
        lenlist_cam, _ = utils.geom.split_lrtlist(lrtlist_cam)
        # this is B x N x 3
        xlist, ylist, zlist = lenlist_cam.chunk(3, dim=2)
        
        vox_size_X = (self.XMAX-self.XMIN)/float(X)
        vox_size_Y = (self.YMAX-self.YMIN)/float(Y)
        vox_size_Z = (self.ZMAX-self.ZMIN)/float(Z)
        lenlist_mem = torch.cat([xlist / vox_size_X,
                                 ylist / vox_size_Y,
                                 zlist / vox_size_Z], dim=2)
        # merge up
        lrtlist_mem = utils.geom.merge_lrtlist(lenlist_mem, rtlist_mem)
        return lrtlist_mem
    
    def apply_ref_T_mem_to_lrtlist(self, lrtlist_mem, Z, Y, X, assert_cube=False):
        # lrtlist is B x N x 19, in mem coordinates
        # transforms them into cam coordinates, including a scale change for the lengths
        B, N, C = list(lrtlist_mem.shape)
        assert(C==19)
        cam_T_mem = self.get_ref_T_mem(B, Z, Y, X, assert_cube=assert_cube)

        # apply_4x4 will work for the t part
        lenlist_mem, rtlist_mem = utils.geom.split_lrtlist(lrtlist_mem)
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rlist_mem_, tlist_mem_ = utils.geom.split_rt(__p(rtlist_mem))
        # rlist_cam_ is B*N x 3 x 3
        # tlist_cam_ is B*N x 3
        tlist_cam_ = __p(utils.geom.apply_4x4(cam_T_mem, __u(tlist_mem_)))
        # rlist does not need to change, since cam is aligned with mem
        rlist_cam_ = rlist_mem_.clone()
        rtlist_cam = __u(utils.geom.merge_rt(rlist_cam_, tlist_cam_))
        # this is B x N x 4 x 4

        # next we need to scale the lengths
        lenlist_mem, _ = utils.geom.split_lrtlist(lrtlist_mem)
        # this is B x N x 3
        vox_size_X = (self.XMAX-self.XMIN)/float(X)
        vox_size_Y = (self.YMAX-self.YMIN)/float(Y)
        vox_size_Z = (self.ZMAX-self.ZMIN)/float(Z)
        xlist, ylist, zlist = lenlist_mem.chunk(3, dim=2)
        lenlist_cam = torch.cat([xlist * vox_size_X,
                                 ylist * vox_size_Y,
                                 zlist * vox_size_Z], dim=2)
        # merge up
        lrtlist_cam = utils.geom.merge_lrtlist(lenlist_cam, rtlist_cam)
        return lrtlist_cam

    def Mem2Ref(self, xyz_mem, Z, Y, X, assert_cube=False):
        # xyz is B x N x 3, in mem coordinates
        # transforms mem coordinates into ref coordinates
        B, N, C = list(xyz_mem.shape)
        ref_T_mem = self.get_ref_T_mem(B, Z, Y, X, assert_cube=assert_cube)
        xyz_ref = utils.geom.apply_4x4(ref_T_mem, xyz_mem)
        return xyz_ref

    def get_ref_T_mem(self, B, Z, Y, X, assert_cube=False, device='cuda'):
        mem_T_ref = self.get_mem_T_ref(B, Z, Y, X, assert_cube=assert_cube, device=device)
        # note safe_inverse is inapplicable here,
        # since the transform is nonrigid
        ref_T_mem = mem_T_ref.inverse()
        return ref_T_mem

    def get_mem_T_ref(self, B, Z, Y, X, assert_cube=False, device='cuda'):
        # sometimes we want the mat itself
        # note this is not a rigid transform

        # note we need to (re-)compute the vox sizes, for this new resolution
        vox_size_X = (self.XMAX-self.XMIN)/float(X)
        vox_size_Y = (self.YMAX-self.YMIN)/float(Y)
        vox_size_Z = (self.ZMAX-self.ZMIN)/float(Z)

        # for safety, let's check that this is cube
        if assert_cube:
            if (not np.isclose(vox_size_X, vox_size_Y)) or (not np.isclose(vox_size_X, vox_size_Z)):
                print('Z, Y, X', Z, Y, X)
                print('bounds for this iter:',
                      'X = %.2f to %.2f' % (self.XMIN, self.XMAX), 
                      'Y = %.2f to %.2f' % (self.YMIN, self.YMAX), 
                      'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX), 
                )
                print('vox_size_X', vox_size_X)
                print('vox_size_Y', vox_size_Y)
                print('vox_size_Z', vox_size_Z)
            assert(np.isclose(vox_size_X, vox_size_Y))
            assert(np.isclose(vox_size_X, vox_size_Z))

        # translation
        # (this makes the left edge of the leftmost voxel correspond to XMIN)
        center_T_ref = utils.geom.eye_4x4(B, device=device)
        center_T_ref[:,0,3] = -self.XMIN-vox_size_X/2.0
        center_T_ref[:,1,3] = -self.YMIN-vox_size_Y/2.0
        center_T_ref[:,2,3] = -self.ZMIN-vox_size_Z/2.0

        # scaling
        # (this makes the right edge of the rightmost voxel correspond to XMAX)
        mem_T_center = utils.geom.eye_4x4(B, device=device)
        mem_T_center[:,0,0] = 1./vox_size_X
        mem_T_center[:,1,1] = 1./vox_size_Y
        mem_T_center[:,2,2] = 1./vox_size_Z
        mem_T_ref = utils.basic.matmul2(mem_T_center, center_T_ref)

        return mem_T_ref

    def get_inbounds(self, xyz, Z, Y, X, already_mem=False, padding=0.0, assert_cube=False):
        # xyz is B x N x 3
        # padding should be 0 unless you are trying to account for some later cropping
        if not already_mem:
            xyz = self.Ref2Mem(xyz, Z, Y, X, assert_cube=assert_cube)

        x = xyz[:,:,0]
        y = xyz[:,:,1]
        z = xyz[:,:,2]

        x_valid = ((x-padding)>-0.5).byte() & ((x+padding)<float(X-0.5)).byte()
        y_valid = ((y-padding)>-0.5).byte() & ((y+padding)<float(Y-0.5)).byte()
        z_valid = ((z-padding)>-0.5).byte() & ((z+padding)<float(Z-0.5)).byte()
        nonzero = (~(z==0.0)).byte()

        inbounds = x_valid & y_valid & z_valid & nonzero
        return inbounds.bool()

    def get_inbounds_single(self, xyz, Z, Y, X, already_mem=False):
        # xyz is N x 3
        xyz = xyz.unsqueeze(0)
        inbounds = self.get_inbounds(xyz, Z, Y, X, already_mem=already_mem)
        inbounds = inbounds.squeeze(0)
        return inbounds

    def voxelize_xyz(self, xyz_ref, Z, Y, X, already_mem=False, assert_cube=False, clean_eps=0):
        B, N, D = list(xyz_ref.shape)
        assert(D==3)
        if already_mem:
            xyz_mem = xyz_ref
        else:
            xyz_mem = self.Ref2Mem(xyz_ref, Z, Y, X, assert_cube=assert_cube)
        vox = self.get_occupancy(xyz_mem, Z, Y, X, clean_eps=clean_eps)
        return vox

    def get_3d_flow(self, xyz_ori_ref, xyz_tar_ref, Z, Y, X, already_mem=False, assert_cube=False):
        B, N, D = list(xyz_ori_ref.shape)
        assert(D==3)
        if already_mem:
            xyz_ori_mem = xyz_ori_ref
            xyz_tar_mem = xyz_tar_ref
        else:
            xyz_ori_mem = self.Ref2Mem(xyz_ori_ref, Z, Y, X, assert_cube=assert_cube)
            xyz_tar_mem = self.Ref2Mem(xyz_tar_ref, Z, Y, X, assert_cube=assert_cube)

        flow_vec = xyz_tar_mem - xyz_ori_mem # this is B x N x 3
        flow_vox = self.voxelize_flow_vec(xyz_ori_mem, flow_vec, Z, Y, X)

        return flow_vox

    def voxelize_flow_vec(self, xyz, flow_vec, Z, Y, X):
        # xyz is B x N x 3 and in mem coords
        # we want to fill a voxel tensor with 1's at these inds
        B, N, C = list(xyz.shape)
        B1, N1, C1 = list(flow_vec.shape)
        assert(C==3)
        assert(B==B1)
        assert(N==N1)
        assert(C1==3)

        # these papers say simple 1/0 occupancy is ok:
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3d_CVPR_2018_paper.pdf
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf
        # cont fusion says they do 8-neighbor interp
        # voxelnet does occupancy but with a bit of randomness in terms of the reflectance value i think

        inbounds = self.get_inbounds(xyz, Z, Y, X, already_mem=True)
        x, y, z = xyz[:,:,0], xyz[:,:,1], xyz[:,:,2]
        flow_x, flow_y, flow_z = flow_vec[:,:,0], flow_vec[:,:,0], flow_vec[:,:,0] # B x N
        mask = torch.zeros_like(x)
        mask[inbounds] = 1.0

        # set the invalid guys to zero
        # we then need to zero out 0,0,0
        # (this method seems a bit clumsy)
        x = x*mask
        y = y*mask
        z = z*mask

        x = torch.round(x)
        y = torch.round(y)
        z = torch.round(z)
        x = torch.clamp(x, 0, X-1).int()
        y = torch.clamp(y, 0, Y-1).int()
        z = torch.clamp(z, 0, Z-1).int()

        x = x.view(B*N)
        y = y.view(B*N)
        z = z.view(B*N)

        dim3 = X
        dim2 = X * Y
        dim1 = X * Y * Z

        base = torch.arange(0, B, dtype=torch.int32, device=torch.device('cuda'))*dim1
        base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B*N)

        vox_inds = base + z * dim2 + y * dim3 + x
        # voxels = torch.zeros(B*Z*Y*X, device=torch.device('cuda')).float()
        # voxels[vox_inds.long()] = 1.0
        flow_vox = torch.zeros(B*Z*Y*X, 3, device=torch.device('cuda')).float()
        flow_vox[vox_inds.long(), 0] = flow_x
        flow_vox[vox_inds.long(), 1] = flow_y
        flow_vox[vox_inds.long(), 2] = flow_y

        # zero out the singularity
        flow_vox[base.long(), :] = 0.0
        flow_vox = flow_vox.reshape(B, Z, Y, X, 3)

        flow_vox = flow_vox.permute(0, 4, 1, 2, 3) # B x 3 x Z x Y x X
        # B x 1 x Z x Y x X
        return flow_vox

    def get_occupancy_single(self, xyz, Z, Y, X, clean_eps=0):
        # xyz is N x 3 and in mem coords
        # we want to fill a voxel tensor with 1's at these inds

        # (we have a full parallelized version, but fill_ray_single needs this)

        inbounds = self.get_inbounds_single(xyz, Z, Y, X, already_mem=True)
        xyz = xyz[inbounds]
        # xyz is N x 3

        if clean_eps > 0:
            # only take points that are already near centers
            xyz_round = torch.round(xyz)
            absdiff = torch.norm(xyz_round - xyz, dim=1)
            xyz = xyz_round[absdiff < 0.1]
        else:
            # this is more accurate than a cast/floor, but runs into issues when a dim==0
            xyz = torch.round(xyz).int()
            
        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]

        vox_inds = utils.basic.sub2ind3d(Z, Y, X, z, y, x)
        vox_inds = vox_inds.flatten().long()
        voxels = torch.zeros(Z*Y*X, dtype=torch.float32, device=xyz.device)
        voxels[vox_inds] = 1.0
        voxels = voxels.reshape(1, Z, Y, X)
        # 1 x Z x Y x X
        return voxels

    def get_occupancy(self, xyz, Z, Y, X, clean_eps=0):
        # xyz is B x N x 3 and in mem coords
        # we want to fill a voxel tensor with 1's at these inds
        B, N, C = list(xyz.shape)
        assert(C==3)

        # these papers say simple 1/0 occupancy is ok:
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3d_CVPR_2018_paper.pdf
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf
        # cont fusion says they do 8-neighbor interp
        # voxelnet does occupancy but with a bit of randomness in terms of the reflectance value i think

        inbounds = self.get_inbounds(xyz, Z, Y, X, already_mem=True)
        x, y, z = xyz[:,:,0], xyz[:,:,1], xyz[:,:,2]
        mask = torch.zeros_like(x)
        mask[inbounds] = 1.0

        if clean_eps > 0:
            # only take points that are already near centers
            xyz_round = torch.round(xyz) # B, N, 3
            dist = torch.norm(xyz_round - xyz, dim=2)
            mask[dist > clean_eps] = 0
        
        # set the invalid guys to zero
        # we then need to zero out 0,0,0
        # (this method seems a bit clumsy)
        x = x*mask
        y = y*mask
        z = z*mask

        x = torch.round(x)
        y = torch.round(y)
        z = torch.round(z)
        x = torch.clamp(x, 0, X-1).int()
        y = torch.clamp(y, 0, Y-1).int()
        z = torch.clamp(z, 0, Z-1).int()

        x = x.view(B*N)
        y = y.view(B*N)
        z = z.view(B*N)

        dim3 = X
        dim2 = X * Y
        dim1 = X * Y * Z

        base = torch.arange(0, B, dtype=torch.int32, device=xyz.device)*dim1
        base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B*N)

        vox_inds = base + z * dim2 + y * dim3 + x
        voxels = torch.zeros(B*Z*Y*X, device=xyz.device).float()
        voxels[vox_inds.long()] = 1.0
        # zero out the singularity
        voxels[base.long()] = 0.0
        voxels = voxels.reshape(B, 1, Z, Y, X)
        # B x 1 x Z x Y x X
        return voxels

    

    def unproject_image_to_mem(self, rgb_camB, Z, Y, X, pixB_T_camA, assert_cube=False):
        # rgb_camB is B x C x H x W
        # pixB_T_camA is B x 4 x 4

        # rgb lives in B pixel coords
        # we want everything in A memory coords

        # this puts each C-dim pixel in the rgb_camB
        # along a ray in the voxelgrid
        B, C, H, W = list(rgb_camB.shape)

        xyz_memA = utils.basic.gridcloud3d(B, Z, Y, X, norm=False)
        # grid_z, grid_y, grid_x = utils.basic.meshgrid3d(B, Z, Y, X)
        # # these are B x Z x Y x X
        # # these represent the mem grid coordinates

        # # we need to convert these to pixel coordinates
        # x = torch.reshape(grid_x, [B, -1])
        # y = torch.reshape(grid_y, [B, -1])
        # z = torch.reshape(grid_z, [B, -1])
        # # these are B x N
        # xyz_mem = torch.stack([x, y, z], dim=2)

        xyz_camA = self.Mem2Ref(xyz_memA, Z, Y, X, assert_cube=assert_cube)

        xyz_pixB = utils.geom.apply_4x4(pixB_T_camA, xyz_camA)
        normalizer = torch.unsqueeze(xyz_pixB[:,:,2], 2)
        EPS=1e-6
        xy_pixB = xyz_pixB[:,:,:2]/torch.clamp(normalizer, min=EPS)
        # this is B x N x 2
        # this is the (floating point) pixel coordinate of each voxel
        x_pixB, y_pixB = xy_pixB[:,:,0], xy_pixB[:,:,1]
        # these are B x N

        if (0):
            # handwritten version
            values = torch.zeros([B, C, Z*Y*X], dtype=torch.float32)
            for b in list(range(B)):
                values[b] = utils.samp.bilinear_sample_single(rgb_camB[b], x_pixB[b], y_pixB[b])
        else:
            # native pytorch version
            y_pixB, x_pixB = utils.basic.normalize_grid2d(y_pixB, x_pixB, H, W)
            # since we want a 3d output, we need 5d tensors
            z_pixB = torch.zeros_like(x_pixB)
            xyz_pixB = torch.stack([x_pixB, y_pixB, z_pixB], axis=2)
            rgb_camB = rgb_camB.unsqueeze(2)
            xyz_pixB = torch.reshape(xyz_pixB, [B, Z, Y, X, 3])
            values = F.grid_sample(rgb_camB, xyz_pixB)

        values = torch.reshape(values, (B, C, Z, Y, X))
        return values

    def unproject_rgb_to_zoom(self, rgb_camB, lrt, Z, Y, X, pixB_T_camA):
        # rgb_camB is B x C x H x W
        # pixB_T_camA is B x 4 x 4

        # rgb lives in B pixel coords
        # we want everything in A memory coords

        # this puts each C-dim pixel in the rgb_camB
        # along a ray in the voxelgrid
        B, C, H, W = list(rgb_camB.shape)

        xyz_zoom = utils.basic.gridcloud3d(B, Z, Y, X, norm=False)
        # grid_z, grid_y, grid_x = utils.basic.meshgrid3d(B, Z, Y, X)
        # # these are B x Z x Y x X
        # # these represent the mem grid coordinates

        # # we need to convert these to pixel coordinates
        # x = torch.reshape(grid_x, [B, -1])
        # y = torch.reshape(grid_y, [B, -1])
        # z = torch.reshape(grid_z, [B, -1])
        # # these are B x N
        # xyz_mem = torch.stack([x, y, z], dim=2)

        xyz_camA = self.Zoom2Ref(xyz_zoom, lrt, Z, Y, X)

        xyz_pixB = utils.geom.apply_4x4(pixB_T_camA, xyz_camA)
        normalizer = torch.unsqueeze(xyz_pixB[:,:,2], 2)
        EPS=1e-6
        xy_pixB = xyz_pixB[:,:,:2]/torch.clamp(normalizer, min=EPS)
        # this is B x N x 2
        # this is the (floating point) pixel coordinate of each voxel
        x_pixB, y_pixB = xy_pixB[:,:,0], xy_pixB[:,:,1]
        # these are B x N

        if (0):
            # handwritten version
            values = torch.zeros([B, C, Z*Y*X], dtype=torch.float32)
            for b in list(range(B)):
                values[b] = utils.samp.bilinear_sample_single(rgb_camB[b], x_pixB[b], y_pixB[b])
        else:
            # native pytorch version
            y_pixB, x_pixB = utils.basic.normalize_grid2d(y_pixB, x_pixB, H, W)
            # since we want a 3d output, we need 5d tensors
            z_pixB = torch.zeros_like(x_pixB)
            xyz_pixB = torch.stack([x_pixB, y_pixB, z_pixB], axis=2)
            rgb_camB = rgb_camB.unsqueeze(2)
            xyz_pixB = torch.reshape(xyz_pixB, [B, Z, Y, X, 3])
            values = F.grid_sample(rgb_camB, xyz_pixB)

        values = torch.reshape(values, (B, C, Z, Y, X))
        return values
    
    def apply_pixX_T_memR_to_voxR(self, pix_T_camX, camX_T_camR, voxR, D, H, W, noise_amount=0.0, grid_z_vec=None, logspace_slices=False, lrt=None, full_vol=True):
        # mats are B x 4 x 4
        # voxR is B x C x Z x Y x X
        # H, W, D indicates how big to make the output 
        # returns B x C x D x H x W

        B, C, Z, Y, X = list(voxR.shape)
        # z_near = np.maximum(self.ZMIN, 0.1)
        # z_far = self.ZMAX

        # for b in range(B)
        if lrt is not None:
            corners_camR = utils.geom.get_xyzlist_from_lrtlist(lrt.unsqueeze(1)).squeeze(1)
            corners_camX = utils.geom.apply_4x4(camX_T_camR, corners_camR)
            # this is B x 8 x 3
            z_near = torch.min(corners_camX[:,:,2], dim=1)[0] - 0.1
            z_near = z_near.clamp(min=0.1)
            z_far = torch.max(corners_camX[:,:,2], dim=1)[0] + 0.1

            assert(B==1)
            z_near = z_near[0]
            z_far = z_far[0]
            # print('z_near', z_near.detach().cpu().numpy())
            # print('z_far', z_far.detach().cpu().numpy())
        else:
            if not full_vol:
                z_near = 0.1
                z_far = self.ZMAX
            else:
                z_near = 0.1
                z_far = np.sqrt(self.ZMAX**2 + self.YMAX**2 + self.XMAX**2)
                # all_xyz_mem = utils.basic.gridcloud3d(B, Z, Y, X) # B, -1, 3
                # # utils.basic.print_stats('all_xyz_mem', all_xyz_mem)
                # cam_T_mem = self.get_ref_T_mem(B, Z, Y, X)
                # camX_T_mem = utils.basic.matmul2(camX_T_camR, cam_T_mem)
                # xyz = utils.geom.apply_4x4(camX_T_mem, all_xyz_mem)
                # xyz = xyz.reshape(-1, 3)
                # # utils.basic.print_stats('xyz', xyz)
                # z = xyz[:,2]
                # xyz = xyz[z>z_near]
                # # utils.basic.print_stats('xyz', xyz)
                # dist = torch.norm(xyz, dim=1)
                # z_far = torch.max(dist).item()
                # # z_far = torch.max(z).item()
                # # print('choosing z_far', z_far)
                
        if grid_z_vec is None:
            if logspace_slices:
                grid_z_vec = torch.exp(torch.linspace(np.log(z_near), np.log(z_far), steps=D, dtype=torch.float32, device=torch.device('cuda')))
                if noise_amount > 0.:
                    print('cannot add noise to logspace sampling yet')
            else:
                grid_z_vec = torch.linspace(z_near, z_far, steps=D, dtype=torch.float32, device=torch.device('cuda'))
                if noise_amount > 0.:
                    diff = grid_z_vec[1] - grid_z_vec[0]
                    noise = torch.rand(grid_z_vec.shape).float().cuda() * diff * 0.5 * noise_amount
                    # noise = torch.randn(grid_z_vec.shape).float().cuda() * noise_std
                    # noise = torch.randn(grid_z_vec.shape).float().cuda() * noise_std
                    grid_z_vec = grid_z_vec + noise
                    grid_z_vec = grid_z_vec.clamp(min=z_near)
            
        grid_z = torch.reshape(grid_z_vec, [1, 1, D, 1, 1])
        grid_z = grid_z.repeat([B, 1, 1, H, W])
        grid_z = torch.reshape(grid_z, [B*D, 1, H, W])

        pix_T_camX__ = torch.unsqueeze(pix_T_camX, axis=1).repeat([1, D, 1, 1])
        pix_T_camX = torch.reshape(pix_T_camX__, [B*D, 4, 4])
        xyz_camX = utils.geom.depth2pointcloud(grid_z, pix_T_camX)

        camR_T_camX = utils.geom.safe_inverse(camX_T_camR)
        camR_T_camX_ = torch.unsqueeze(camR_T_camX, dim=1).repeat([1, D, 1, 1])
        camR_T_camX = torch.reshape(camR_T_camX_, [B*D, 4, 4])

        if lrt is None:
            mem_T_cam = self.get_mem_T_ref(B*D, Z, Y, X)
            memR_T_camX = utils.basic.matmul2(mem_T_cam, camR_T_camX)
            xyz_memR = utils.geom.apply_4x4(memR_T_camX, xyz_camX)
            xyz_memR = torch.reshape(xyz_memR, [B, D*H*W, 3])
            samp = utils.samp.sample3d(voxR, xyz_memR, D, H, W)
        else:
            xyz_camR = utils.geom.apply_4x4(camR_T_camX, xyz_camX)
            xyz_memR = self.Ref2Zoom(xyz_camR, lrt, Z, Y, X, additive_pad=0.0)
            xyz_memR = torch.reshape(xyz_memR, [B, D*H*W, 3])
            samp = utils.samp.sample3d(voxR, xyz_memR, D, H, W)

        # samp is B x H x W x D x C
        return samp, grid_z_vec

    def voxelize_zoom(self, xyz_ref, lrt, Z, Y, X):
        B, N, D = list(xyz_ref.shape)
        assert(D==3)
        xyz_zoom = self.Ref2Zoom(xyz_ref, lrt, Z, Y, X, additive_pad=0.0)    
        vox = self.get_occupancy(xyz_zoom, Z, Y, X)
        return vox

    def voxelize_near_xyz(self, xyz_ref, xyz, Z, Y, X, sz=16.0, sy=16.0, sx=16.0):
        # xyz_ref is B x N x 3; it is a pointcloud in ref coords
        # xyz is B x 3; it is a point in ref coords
        # sz, sy, sz are the size to grab in 3d, in ref units (usually meters)
        B, N, D = list(xyz_ref.shape)
        assert(D==3)

        lrt = self.get_lrt_of_region_around_xyz(xyz, Z, Y, X, sz=sz, sy=sy, sx=sx)
        xyz_zoom = self.Ref2Zoom(xyz_ref, lrt, Z, Y, X, additive_pad=0.0)
        vox = self.get_occupancy(xyz_zoom, Z, Y, X)
        # vox = torch.zeros([B, 1, Z, Y, X]).float().cpu()
        # for b in list(range(B)):
        #     vox[b] = self.get_occupancy_single(xyz_zoom[b].cpu(), Z, Y, X)
        # vox = vox.cuda()
        # return the lrt also, so that we can convert from here back to ref coords (with Zoom2Ref)
        return vox, lrt

    def get_lrt_of_region_around_xyz(self, xyz, Z, Y, X, sz=16.0, sy=16.0, sx=16.0):
        # xyz is B x 3; it is a point in ref coords
        # sz, sy, sz are the size to grab in 3d, in ref units (usually meters)
        B, D = list(xyz.shape)
        assert(D==3)

        xyzlist = xyz.unsqueeze(1) # B x 1 x 3
        lxlist = torch.ones_like(xyzlist[:,:,0])*sx
        lylist = torch.ones_like(xyzlist[:,:,0])*sy
        lzlist = torch.ones_like(xyzlist[:,:,0])*sz
        lenlist = torch.stack([lxlist, lylist, lzlist], dim=2) # cube this size
        rotlist = torch.zeros_like(xyzlist) # no rot
        boxlist = torch.cat([xyzlist, lenlist, rotlist], dim=2)
        # boxlist is B x 1 x 9
        lrtlist = utils.geom.convert_boxlist_to_lrtlist(boxlist)
        lrt = lrtlist.squeeze(1)
        # lrt is B x 19
        
        return lrt

    def resample_to_target_views(occRs, camRs_T_camPs):
        # resample to the target view

        # occRs is B x S x Y x X x Z x 1
        # camRs_T_camPs is B x S x 4 x 4

        B, S, _, Z, Y, X = list(occRs.shape)

        # we want to construct a mat memR_T_memP

        cam_T_mem = self.get_ref_T_mem(B, Z, Y, X)
        mem_T_cam = self.get_mem_T_ref(B, Z, Y, X)
        cams_T_mems = cam_T_mem.unsqueeze(1).repeat(1, S, 1, 1)
        mems_T_cams = mem_T_cam.unsqueeze(1).repeat(1, S, 1, 1)

        cams_T_mems = torch.reshape(cams_T_mems, (B*S, 4, 4))
        mems_T_cams = torch.reshape(mems_T_cams, (B*S, 4, 4))
        camRs_T_camPs = torch.reshape(camRs_T_camPs, (B*S, 4, 4))

        memRs_T_memPs = torch.matmul(torch.matmul(mems_T_cams, camRs_T_camPs), cams_T_mems)
        memRs_T_memPs = torch.reshape(memRs_T_memPs, (B, S, 4, 4))

        occRs, valid = resample_to_view(occRs, memRs_T_memPs, multi=True)
        return occRs, valid

    def resample_to_target_view(occRs, camR_T_camP):
        B, S, Z, Y, X, _ = list(occRs.shape)
        cam_T_mem = self.get_ref_T_mem(B, Z, Y, X)
        mem_T_cam = self.get_mem_T_ref(B, Z, Y, X)
        memR_T_memP = torch.matmul(torch.matmul(mem_T_cam, camR_T_camP), cam_T_mem)
        occRs, valid = resample_to_view(occRs, memR_T_memP, multi=False)
        return occRs, valid

    def resample_to_view(feats, new_T_old, multi=False):
        # feats is B x S x c x Y x X x Z 
        # it represents some scene features in reference/canonical coordinates
        # we want to go from these coords to some target coords

        # new_T_old is B x 4 x 4
        # it represents a transformation between two "mem" systems
        # or if multi=True, it's B x S x 4 x 4

        B, S, C, Z, Y, X = list(feats.shape)

        # we want to sample for each location in the bird grid
        # xyz_mem = gridcloud3d(B, Z, Y, X)
        grid_y, grid_x, grid_z = utils.basic.meshgrid3d(B, Z, Y, X)
        # these are B x BY x BX x BZ
        # these represent the mem grid coordinates

        # we need to convert these to pixel coordinates
        x = torch.reshape(grid_x, [B, -1])
        y = torch.reshape(grid_y, [B, -1])
        z = torch.reshape(grid_z, [B, -1])
        # these are B x N

        xyz_mem = torch.stack([x, y, z], dim=2)
        # this is B x N x 3

        xyz_mems = xyz_mem.unsqueeze(1).repeat(1, S, 1, 1)
        # this is B x S x N x 3

        xyz_mems_ = xyz_mems.view(B*S, Y*X*Z, 3)

        feats_ = feats.view(B*S, C, Z, Y, X)

        if multi:
            new_T_olds = new_T_old.clone()
        else:
            new_T_olds = new_T_old.unsqueeze(1).repeat(1, S, 1, 1)
        new_T_olds_ = new_T_olds.view(B*S, 4, 4)

        xyz_new_ = utils.geom.apply_4x4(new_T_olds_, xyz_mems_)
        # we want each voxel to replace its value
        # with whatever is at these new coordinates

        # i.e., we are back-warping from the "new" coords

        feats_, valid_ = utils.samp.resample3d(feats_, xyz_new_)
        feats = feats_.view(B, S, C, Z, Y, X)
        valid = valid_.view(B, S, 1, Z, Y, X)
        return feats, valid

    def get_top_occ(self, occ_mem):
        B, C, Z, Y, X = occ_mem.shape
        
        # note that y increases downward in the tensor
        y_values = torch.linspace(float(Y), 1.0, steps=Y, dtype=torch.float32, device=occ_mem.device)
        y_values = y_values.view(1, 1, 1, Y, 1)
        y_values = torch.max(occ_mem*y_values, dim=3)[0] # B, 1, Z, X
        top_mem = torch.zeros_like(occ_mem)
        for b in range(B):
            xyz_mem = utils.basic.gridcloud3d(1, Z, 1, X) # B, -1, 3
            xyz_mem_ = xyz_mem.reshape(-1, 3)
            xyz_mem_[:,1] = y_values[b].reshape(-1)
            xyz_mem_ = xyz_mem_[xyz_mem_[:,1] > 0]
            xyz_mem_[:,1] = Y - xyz_mem_[:,1]
            # xyz_mem_ = torch.stack([xyz_mem_[:,0], values[b].reshape(-1), xyz_mem_[:,2]], dim=1)
            occ_mem_ = self.voxelize_xyz(xyz_mem_.reshape(1, -1, 3), Z, Y, X, already_mem=True)
            # print('occ_mem_b', occ_mem_.shape)
            top_mem[b] = occ_mem_[0]
            # print('occ_mem_', occ_mem_.shape)
        return top_mem
        
    def fill_above_occ(self, occ_mem):
        # the idea here is to mark voxels that are above occupied ones,
        # as a noisy signal for freespace
        B, C, Z, Y, X = occ_mem.shape

        # values = torch.linspace(float(Y), 1.0, steps=Y, dtype=torch.float32, device=occ_mem.device)
        # values = values.view(1, 1, 1, Y, 1)
        # values = torch.max(occ_mem*values, dim=3)[0]
        # print_stats('values', values)
        
        # input()

        # maybe
        # maybe a simple solution re-using my existing tools is:
        # get the coords, then add all y values, 
        # then revox,
        # then subtract occ_mem
        # but note we need to start with just the top surface

        free_mem = torch.zeros_like(occ_mem)
        for b in range(B):
            # xyz_mem = utils.basic.gridcloud3d(1, Z, 1, X) # B, -1, 3
            # xyz_mem_ = xyz_mem.reshape(-1, 3)
            # xyz_mem_[:,2] = values[b].reshape(-1)
            # # xyz_mem_ = torch.stack([xyz_mem_[:,0], values[b].reshape(-1), xyz_mem_[:,2]], dim=1)
            # occ_mem_ = voxelize_xyz(xyz_mem_.reshape(1, -1, 3), Z, Y, X, already_mem=True)
            
            xyz_mem = utils.basic.gridcloud3d(1, Z, Y, X) # B, -1, 3
            xyz_mem_ = xyz_mem.reshape(-1, 3)
            occ_mem_ = occ_mem[b].reshape(-1)
            xyz_mem_ = xyz_mem_[occ_mem_ > 0] # -1, 3
            # print_stats('xyz_mem_', xyz_mem_)
            xyz_mem_ = xyz_mem_.reshape(-1, 1, 3)
            deltas = -torch.linspace(1.0, float(Y), steps=Y, dtype=torch.float32, device=occ_mem.device).reshape(1, Y, 1)
            deltas = torch.cat([0*deltas, deltas, 0*deltas], dim=2)
            # print('xyz_mem_', xyz_mem_.shape)
            # print('deltas', deltas.shape)
            xyz_mem_ = (xyz_mem_ + deltas).reshape(1, -1, 3).long()
            # print_stats('delta xyz_mem_', xyz_mem_)
            inb = self.get_inbounds(xyz_mem_, Z, Y, X, already_mem=True).reshape(-1)
            # x, y, z = xyz_mem_[:,0], xyz_mem_[:,1], xyz_mem_[:,2]
            x, y, z = xyz_mem_[0,inb,0], xyz_mem_[0,inb,1], xyz_mem_[0,inb,2]
            free_mem[b,:,z,y,x] = 1
        free_mem = (free_mem - occ_mem).clamp(0,1)
        return free_mem
        

    def convert_xyz_to_visibility(self, xyz, Z, Y, X, target_T_given=None, ray_add=0.0):
        # xyz is in camera coordinates
        B, N, C = list(xyz.shape)
        assert(C==3)
        voxels = torch.zeros(B, 1, Z, Y, X, dtype=torch.float32, device=xyz.device)
        for b in list(range(B)):
            if target_T_given is not None:
                voxels[b,0] = self.fill_ray_single(xyz[b], Z, Y, X, target_T_given=target_T_given[b], ray_add=ray_add)
            else:
                voxels[b,0] = self.fill_ray_single(xyz[b], Z, Y, X, ray_add=ray_add)
        return voxels

    def convert_xyz_to_visibility_samples(self, xyz, target_T_given=None, ray_add=0.0, samps=100, dist_eps=0.01, rand=True):
        # xyz is in camera coordinates
        B, N, C = list(xyz.shape)
        assert(C==3)
        free_xyz = torch.zeros(B, samps*N, 3, dtype=torch.float32, device=xyz.device)
        for b in list(range(B)):
            if target_T_given is not None:
                free_xyz[b] = self.continuous_fill_ray_single(xyz[b], samps=samps, target_T_given=target_T_given[b], ray_add=ray_add, dist_eps=dist_eps, rand=rand)
            else:
                free_xyz[b] = self.continuous_fill_ray_single(xyz[b], samps=samps, ray_add=ray_add, dist_eps=dist_eps, rand=rand)
        return free_xyz

    def fill_ray_single(self, xyz, Z, Y, X, target_T_given=None, ray_add=0.0):
        # xyz is N x 3, and in cam coords
        # we want to fill a voxel tensor with 1's at these inds,
        # and also at any ind along the ray before it

        # target_T_given, if it exists, takes us to the coords we want to be in;
        # it is 4 x 4

        xyz = torch.reshape(xyz, (-1, 3))
        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
        # these are N

        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        z = z.unsqueeze(1)

        # get the hypotenuses
        u = torch.sqrt(x**2+z**2) # flat to ground
        v = torch.sqrt(x**2+y**2+z**2)
        w = torch.sqrt(x**2+y**2)

        # the ray is along the v line
        # we want to find xyz locations along this line

        # get the angles
        EPS = 1e-6
        u = torch.clamp(u, min=EPS) # note >=0 already
        v = torch.clamp(v, min=EPS) # note >=0 already
        sin_theta = y/v # soh 
        cos_theta = u/v # cah
        sin_alpha = z/u # soh
        cos_alpha = x/u # cah

        samps = int(np.sqrt(Y**2 + Z**2))*2
        # for each proportional distance in [0.0, 1.0], generate a new hypotenuse
        dists = torch.linspace(0.0, 1.0, samps, device=xyz.device)
        dists = torch.reshape(dists, (1, samps))
        v_ = dists * v.repeat(1, samps)
        v_ = v_ + ray_add

        # now, for each of these v_, we want to generate the xyz
        y_ = sin_theta*v_
        u_ = torch.abs(cos_theta*v_)
        z_ = sin_alpha*u_
        x_ = cos_alpha*u_
        # these are the ref coordinates we want to fill
        x = x_.flatten()
        y = y_.flatten()
        z = z_.flatten()

        xyz = torch.stack([x,y,z], dim=1).unsqueeze(0)
        if target_T_given is not None:
            target_T_given = target_T_given.unsqueeze(0)
            xyz = utils.geom.apply_4x4(target_T_given, xyz)
        xyz = self.Ref2Mem(xyz, Z, Y, X)
        xyz = torch.squeeze(xyz, dim=0)
        # these are the mem coordinates we want to fill

        return self.get_occupancy_single(xyz, Z, Y, X)

    def continuous_fill_ray_single(self, xyz, samps=100, target_T_given=None, ray_add=0.0, dist_eps=0.01, rand=True):
        # xyz is N x 3, and in cam coords
        # we want to fill a voxel tensor with 1's at these inds,
        # and also at any ind along the ray before it

        # target_T_given, if it exists, takes us to the coords we want to be in;
        # it is 4 x 4

        xyz = torch.reshape(xyz, (-1, 3))
        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
        # these are N
        N = x.shape[0]

        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        z = z.unsqueeze(1)

        # get the hypotenuses
        u = torch.sqrt(x**2+z**2) # flat to ground
        v = torch.sqrt(x**2+y**2+z**2)
        w = torch.sqrt(x**2+y**2)

        # the ray is along the v line
        # we want to find xyz locations along this line

        # get the angles
        EPS = 1e-6
        u = torch.clamp(u, min=EPS)
        v = torch.clamp(v, min=EPS)
        sin_theta = y/v # soh 
        cos_theta = u/v # cah
        sin_alpha = z/u # soh
        cos_alpha = x/u # cah

        # for each proportional distance in [0.0, 1.0-eps], generate a new hypotenuse
        if rand:
            dists = torch.rand(N*samps, device=xyz.device) * (1.0 - dist_eps)
            dists = torch.reshape(dists, (N, samps))
            v_ = dists * v.repeat(1, samps)
        else:
            dists = torch.linspace(0.0, 1.0-dist_eps, samps, device=xyz.device)
            dists = torch.reshape(dists, (1, samps))
            v_ = dists * v.repeat(1, samps)
        v_ = v_ + ray_add

        # now, for each of these v_, we want to generate the xyz
        y_ = sin_theta*v_
        u_ = torch.abs(cos_theta*v_)
        z_ = sin_alpha*u_
        x_ = cos_alpha*u_
        # these are the ref coordinates we want to fill
        x = x_.flatten()
        y = y_.flatten()
        z = z_.flatten()

        xyz = torch.stack([x,y,z], dim=1).unsqueeze(0)
        if target_T_given is not None:
            target_T_given = target_T_given.unsqueeze(0)
            xyz = utils.geom.apply_4x4(target_T_given, xyz)
        xyz = torch.squeeze(xyz, dim=0) # N, 3
        # these are the cam coordinates we want to fill
        return xyz
    

    def get_freespace(self, xyz, occ, ray_add=0.0):
        # xyz is B x N x 3
        # occ is B x H x W x D x 1
        B, C, Z, Y, X = list(occ.shape)
        assert(C==1)
        vis = self.convert_xyz_to_visibility(xyz, Z, Y, X, ray_add=ray_add)
        # visible space is all free unless it's occupied
        free = (1.0-(occ>0.0).float())*vis
        return free

    def apply_4x4_to_vox(self, B_T_A, feat_A, already_mem=False, binary_feat=False, rigid=True):
        # B_T_A is B x 4 x 4
        # if already_mem=False, it is a transformation between cam systems
        # if already_mem=True, it is a transformation between mem systems

        # feat_A is B x C x Z x Y x X
        # it represents some scene features in reference/canonical coordinates
        # we want to go from these coords to some target coords

        # since this is a backwarp,
        # the question to ask is:
        # "WHERE in the tensor do you want to sample,
        # to replace each voxel's current value?"

        # the inverse of B_T_A represents this "where";
        # it transforms each coordinate in B
        # to the location we want to sample in A

        B, C, Z, Y, X = list(feat_A.shape)

        # we have B_T_A in input, since this follows the other utils.geom.apply_4x4
        # for an apply_4x4 func, but really we need A_T_B
        if rigid:
            A_T_B = utils.geom.safe_inverse(B_T_A)
        else:
            # this op is slower but more powerful
            A_T_B = B_T_A.inverse()

        device = feat_A.device
        if not already_mem:
            cam_T_mem = self.get_ref_T_mem(B, Z, Y, X, device=device)
            mem_T_cam = self.get_mem_T_ref(B, Z, Y, X, device=device)
            A_T_B = utils.basic.matmul3(mem_T_cam, A_T_B, cam_T_mem)

        # we want to sample for each location in the bird grid
        xyz_B = utils.basic.gridcloud3d(B, Z, Y, X, device=device)
        # this is B x N x 3

        # transform
        xyz_A = utils.geom.apply_4x4(A_T_B, xyz_B)
        # we want each voxel to take its value
        # from whatever is at these A coordinates
        # i.e., we are back-warping from the "A" coords

        # feat_B = F.grid_sample(feat_A, normalize_grid(xyz_A, Z, Y, X))
        feat_B = utils.samp.resample3d(feat_A, xyz_A, binary_feat=binary_feat)

        # feat_B, valid = utils.samp.resample3d(feat_A, xyz_A, binary_feat=binary_feat)
        # return feat_B, valid
        return feat_B

    def apply_4x4s_to_voxs(self, Bs_T_As, feat_As, already_mem=False, binary_feat=False):
        # plural wrapper for apply_4x4_to_vox

        B, S, C, Z, Y, X = list(feat_As.shape)

        # utils for packing/unpacking along seq dim
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)

        Bs_T_As_ = __p(Bs_T_As)
        feat_As_ = __p(feat_As)
        feat_Bs_ = self.apply_4x4_to_vox(Bs_T_As_, feat_As_, already_mem=already_mem, binary_feat=binary_feat)
        feat_Bs = __u(feat_Bs_)
        return feat_Bs

    def prep_occs_supervision(self,
                              camRs_T_camXs,
                              xyz_camXs,
                              Z, Y, X,
                              agg=False):
        B, S, N, D = list(xyz_camXs.size())
        assert(D==3)
        # occRs_half = __u(utils.vox.voxelize_xyz(__p(xyz_camRs), Z2, Y2, X2))

        # utils for packing/unpacking along seq dim
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)

        camRs_T_camXs_ = __p(camRs_T_camXs)
        xyz_camXs_ = __p(xyz_camXs)
        xyz_camRs_ = utils.geom.apply_4x4(camRs_T_camXs_, xyz_camXs_)
        occXs_ = self.voxelize_xyz(xyz_camXs_, Z, Y, X)
        occRs_ = self.voxelize_xyz(xyz_camRs_, Z, Y, X)

        # note we must compute freespace in the given view,
        # then warp to the target view
        freeXs_ = self.get_freespace(xyz_camXs_, occXs_)
        freeRs_ = self.apply_4x4_to_vox(camRs_T_camXs_, freeXs_)

        occXs = __u(occXs_)
        occRs = __u(occRs_)
        freeXs = __u(freeXs_)
        freeRs = __u(freeRs_)
        # these are B x S x 1 x Z x Y x X

        if agg:
            # note we should only agg if we are in STATIC mode (time frozen)
            freeR = torch.max(freeRs, dim=1)[0]
            occR = torch.max(occRs, dim=1)[0]
            # these are B x 1 x Z x Y x X
            occR = (occR>0.5).float()
            freeR = (freeR>0.5).float()
            return occR, freeR, occXs, freeXs
        else:
            occRs = (occRs>0.5).float()
            freeRs = (freeRs>0.5).float()
            return occRs, freeRs, occXs, freeXs
        
    def assemble_padded_obj_masklist(self, lrtlist, scorelist, Z, Y, X, coeff=1.0, additive_coeff=0.0):
        # compute a binary mask in 3d for each object
        # we use this when computing the center-surround objectness score
        # lrtlist is B x N x 19
        # scorelist is B x N

        # returns masklist shaped B x N x 1 x Z x Y x X

        B, N, D = list(lrtlist.shape)
        assert(D==19)
        masks = torch.zeros(B, N, Z, Y, X)

        lenlist, ref_T_objlist = utils.geom.split_lrtlist(lrtlist)
        # lenlist is B x N x 3
        # ref_T_objlist is B x N x 4 x 4

        lenlist_ = lenlist.reshape(B*N, 3)
        ref_T_objlist_ = ref_T_objlist.reshape(B*N, 4, 4)
        obj_T_reflist_ = utils.geom.safe_inverse(ref_T_objlist_)

        # we want a value for each location in the mem grid
        xyz_mem_ = utils.basic.gridcloud3d(B*N, Z, Y, X)
        # this is B*N x V x 3, where V = Z*Y*X
        xyz_ref_ = self.Mem2Ref(xyz_mem_, Z, Y, X)
        # this is B*N x V x 3

        lx, ly, lz = torch.unbind(lenlist_, dim=1)
        # these are B*N

        # ref_T_obj = convert_box_to_ref_T_obj(boxes3d)
        # obj_T_ref = ref_T_obj.inverse()

        xyz_obj_ = utils.geom.apply_4x4(obj_T_reflist_, xyz_ref_)
        x, y, z = torch.unbind(xyz_obj_, dim=2)
        # these are B*N x V

        lx = lx.unsqueeze(1)*coeff + additive_coeff
        ly = ly.unsqueeze(1)*coeff + additive_coeff
        lz = lz.unsqueeze(1)*coeff + additive_coeff
        # these are B*N x 1

        x_valid = (x > -lx/2.0).byte() & (x < lx/2.0).byte()
        y_valid = (y > -ly/2.0).byte() & (y < ly/2.0).byte()
        z_valid = (z > -lz/2.0).byte() & (z < lz/2.0).byte()
        inbounds = x_valid.byte() & y_valid.byte() & z_valid.byte()
        masklist = inbounds.float()
        # print(masklist.shape)
        masklist = masklist.reshape(B, N, 1, Z, Y, X)
        # print(masklist.shape)
        # print(scorelist.shape)
        masklist = masklist*scorelist.view(B, N, 1, 1, 1, 1)
        return masklist
        
    def assemble_padded_obj_masklist_within_region(self, lrtlist, scorelist, lrt_region, Z, Y, X, coeff=1.0):
        # compute a binary mask in 3d for each object
        # we use this when computing the center-surround objectness score
        # lrtlist is B x N x 19
        # scorelist is B x N
        # lrt_region is the lrt that defines the search region

        # returns masklist shaped B x N x 1 x Z x Y x X

        B, N, D = list(lrtlist.shape)
        assert(D==19)
        masks = torch.zeros(B, N, Z, Y, X)

        lenlist, ref_T_objlist = utils.geom.split_lrtlist(lrtlist)
        # lenlist is B x N x 3
        # ref_T_objlist is B x N x 4 x 4

        lenlist_ = lenlist.reshape(B*N, 3)
        ref_T_objlist_ = ref_T_objlist.reshape(B*N, 4, 4)
        obj_T_reflist_ = utils.geom.safe_inverse(ref_T_objlist_)

        # we want a value for each location in the mem grid
        xyz_search_ = utils.basic.gridcloud3d(B*N, Z, Y, X)
        # this is B*N x V x 3, where V = Z*Y*X
        lrt_region_ = lrt_region.unsqueeze(1).repeat(1, N, 1).reshape(B*N, 19)
        xyz_ref_ = self.Zoom2Ref(xyz_search_, lrt_region_, Z, Y, X)
        # this is B*N x V x 3

        lx, ly, lz = torch.unbind(lenlist_, dim=1)
        # these are B*N

        # ref_T_obj = convert_box_to_ref_T_obj(boxes3d)
        # obj_T_ref = ref_T_obj.inverse()

        xyz_obj_ = utils.geom.apply_4x4(obj_T_reflist_, xyz_ref_)
        x, y, z = torch.unbind(xyz_obj_, dim=2)
        # these are B*N x V

        lx = lx.unsqueeze(1)*coeff
        ly = ly.unsqueeze(1)*coeff
        lz = lz.unsqueeze(1)*coeff
        # these are B*N x 1

        x_valid = (x > -lx/2.0).byte() & (x < lx/2.0).byte()
        y_valid = (y > -ly/2.0).byte() & (y < ly/2.0).byte()
        z_valid = (z > -lz/2.0).byte() & (z < lz/2.0).byte()
        inbounds = x_valid.byte() & y_valid.byte() & z_valid.byte()
        masklist = inbounds.float()
        # print(masklist.shape)
        masklist = masklist.reshape(B, N, 1, Z, Y, X)
        # print(masklist.shape)
        # print(scorelist.shape)
        masklist = masklist*scorelist.view(B, N, 1, 1, 1, 1)
        return masklist

    def get_zoom_T_ref(self, lrt, Z, Y, X, additive_pad=0.0):
        # lrt is B x 19
        B, E = list(lrt.shape)
        assert(E==19)
        lens, ref_T_obj = utils.geom.split_lrt(lrt)
        lx, ly, lz = lens.unbind(1)

        debug = False

        if debug:
            print('lx, ly, lz')
            print(lx)
            print(ly)
            print(lz)

        obj_T_ref = utils.geom.safe_inverse(ref_T_obj)
        # this is B x 4 x 4

        if debug:
            print('ok, got obj_T_ref:')
            print(obj_T_ref)

        # we want a tiny bit of padding
        # additive helps avoid nans with invalid objects
        # mult helps expand big objects
        lx = lx + additive_pad
        ly = ly + additive_pad
        lz = lz + additive_pad
        lx = lx.clamp(min=0.01)
        ly = ly.clamp(min=0.01)
        lz = lz.clamp(min=0.01)

        # scaling
        Z_VOX_SIZE_X = (lx)/float(X)
        Z_VOX_SIZE_Y = (ly)/float(Y)
        Z_VOX_SIZE_Z = (lz)/float(Z)

        # translation
        center_T_obj_r = utils.geom.eye_3x3(B)
        center_T_obj_t = torch.stack([
            lx/2. - Z_VOX_SIZE_X/2.0,
            ly/2. - Z_VOX_SIZE_Y/2.0,
            lz/2. - Z_VOX_SIZE_Z/2.0,
        ], dim=1)
        if debug:
            print('merging these:')
            print(center_T_obj_r.shape)
            print(center_T_obj_t.shape)
        center_T_obj = utils.geom.merge_rt(center_T_obj_r, center_T_obj_t)

        if debug:
            print('ok, got center_T_obj:')
            print(center_T_obj)

        diag = torch.stack([1./Z_VOX_SIZE_X,
                            1./Z_VOX_SIZE_Y,
                            1./Z_VOX_SIZE_Z,
                            torch.ones([B], device=torch.device('cuda'))],
                           axis=1).view(B, 4)
        if debug:
            print('diag:')
            print(diag)
            print(diag.shape)
        zoom_T_center = torch.diag_embed(diag)
        if debug:
            print('ok, got zoom_T_center:')
            print(zoom_T_center)
            print(zoom_T_center.shape)

        # compose these
        zoom_T_obj = utils.basic.matmul2(zoom_T_center, center_T_obj)

        if debug:
            print('ok, got zoom_T_obj:')
            print(zoom_T_obj)
            print(zoom_T_obj.shape)

        zoom_T_ref = utils.basic.matmul2(zoom_T_obj, obj_T_ref)

        if debug:
            print('ok, got zoom_T_ref:')
            print(zoom_T_ref)

        return zoom_T_ref

    def get_ref_T_zoom(self, lrt, Z, Y, X, additive_pad=0.0):
        # lrt is B x 19
        zoom_T_ref = self.get_zoom_T_ref(lrt, Z, Y, X, additive_pad=additive_pad)
        # note safe_inverse is inapplicable here,
        # since the transform is nonrigid
        ref_T_zoom = zoom_T_ref.inverse()
        return ref_T_zoom

    def Ref2Zoom(self, xyz_ref, lrt_ref, Z, Y, X, additive_pad=0.0):
        # xyz_ref is B x N x 3, in ref coordinates
        # lrt_ref is B x 19, specifying the box in ref coordinates
        # this transforms ref coordinates into zoom coordinates
        B, N, _ = list(xyz_ref.shape)
        zoom_T_ref = self.get_zoom_T_ref(lrt_ref, Z, Y, X, additive_pad=additive_pad)
        xyz_zoom = utils.geom.apply_4x4(zoom_T_ref, xyz_ref)
        return xyz_zoom

    def Zoom2Ref(self, xyz_zoom, lrt_ref, Z, Y, X, additive_pad=0.0):
        # xyz_zoom is B x N x 3, in zoom coordinates
        # lrt_ref is B x 9, specifying the box in ref coordinates
        B, N, _ = list(xyz_zoom.shape)
        ref_T_zoom = self.get_ref_T_zoom(lrt_ref, Z, Y, X, additive_pad=additive_pad)
        xyz_ref = utils.geom.apply_4x4(ref_T_zoom, xyz_zoom)
        return xyz_ref

    def crop_zoom_from_mem(self, mem, lrt, Z2, Y2, X2, additive_pad=0.0, mode='bilinear'):
        # mem is B x C x Z x Y x X
        # lrt is B x 19

        B, C, Z, Y, X = list(mem.shape)
        B2, E = list(lrt.shape)

        assert(E==19)
        assert(B==B2)

        # for each voxel in the zoom grid, i want to
        # sample a voxel from the mem

        xyz_zoom = utils.basic.gridcloud3d(B, Z2, Y2, X2, norm=False)
        # these represent the zoom grid coordinates
        # we need to convert these to mem coordinates
        xyz_ref = self.Zoom2Ref(xyz_zoom, lrt, Z2, Y2, X2, additive_pad=additive_pad)
        xyz_mem = self.Ref2Mem(xyz_ref, Z, Y, X)

        zoom = utils.samp.sample3d(mem, xyz_mem, Z2, Y2, X2, mode=mode)
        zoom = torch.reshape(zoom, [B, C, Z2, Y2, X2])
        return zoom

    def xyz2circles(self, xyz, radius, Z, Y, X, soft=False, already_mem=True):
        # xyz is B x N x 3
        # radius is B x N
        # output is B x N x Z x Y x X
        B, N, D = list(xyz.shape)
        assert(D==3)
        if not already_mem:
            xyz = self.Ref2Mem(xyz, Z, Y, X)
        grid_z, grid_y, grid_x = utils.basic.meshgrid3d(B, Z, Y, X, stack=False, norm=False)
        # note the default stack is on -1
        grid = torch.stack([grid_x, grid_y, grid_z], dim=1)
        # this is B x 3 x Z x Y x X
        xyz = xyz.reshape(B, N, 3, 1, 1, 1)
        grid = grid.reshape(B, 1, 3, Z, Y, X)
        # this is B x N x Z x Y x X

        # round the xyzs, so that at least one value matches the grid perfectly,
        # and we get a value of 1 there (since exp(0)==1)
        xyz = xyz.round()

        radius = radius.clamp(min=0.01)
        
        if soft:
            # interpret radius as sigma
            dist_grid = torch.sum((grid - xyz)**2, dim=2, keepdim=False)
            # this is B x N x Z x Y x X
            radius = radius.reshape(B, N, 1, 1, 1)
            mask = torch.exp(-dist_grid/(2*radius*radius))
            # zero out near zero 
            mask[mask < 0.001] = 0.0
            # h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
            # h[h < np.finfo(h.dtype).eps * h.max()] = 0
            # return h
            return mask
        else:
            assert(False) # something is wrong with this. come back later to debug
            
            dist_grid = torch.norm(grid - xyz, dim=2, keepdim=False)
            # this is 0 at/near the xyz, and increases by 1 for each voxel away
            
            radius = radius.reshape(B, N, 1, 1, 1)
            
            within_radius_mask = (dist_grid < radius).float()
            within_radius_mask = torch.sum(within_radius_mask, dim=1, keepdim=True).clamp(0, 1)
            return within_radius_mask

    def xyz2circle(xyz, Z, Y, X, radius=10.0, already_mem=True):
        # xyz is B x 3
        B, D = list(xyz.shape)
        assert(D==3)
        if not already_mem:
            xyz = self.Ref2Mem(xyz, Z, Y, X)
        grid_z, grid_y, grid_x = utils.basic.meshgrid3d(B, Z, Y, X, stack=False, norm=False)
        # note the default stack is on -1
        grid = torch.stack([grid_x, grid_y, grid_z], dim=1)
        # this is B x 3 x Z x Y x X
        xyz = xyz.reshape(B, 3, 1, 1, 1)
        dist_grid = torch.norm(grid - xyz, dim=1, keepdim=True)
        # this is B x 1 x Z x Y x X
        # this is 0 at/near the xyz, and increases by 1 for each voxel away
        within_radius_mask = (dist_grid < radius).float()
        return within_radius_mask

    

# def center_mem_on_xyz(mem, xyz, Z2, Y2, X2):
#     # mem is B x C x Z x Y x X
#     # xyz is B x 3

#     B, C, Z, Y, X = list(mem.shape)
#     B2, D = list(xyz.shape)

#     assert(D==3)
#     assert(B==B2)

#     # from the xyz i'll make a fat lrt
#     # then call crop_zoom_from_mem

#     xyzlist = xyz.unsqueeze(1) # B x 1 x 3
#     lenlist = torch.ones_like(xyzlist)*10.0 # 10m cube
#     rotlist = torch.zeros_like(xyzlist) # no rot
    
#     boxlist = torch.cat([xyzlist, lenlist, rotlist], dim=2)
#     # boxlist is B x 1 x 9
    
#     lrtlist = utils.geom.convert_boxlist_to_lrtlist(boxlist)
#     lrt = lrtlist.squeeze(1)
#     # lrt is B x 19

#     return crop_zoom_from_mem(mem, lrt, Z2, Y2, X2, additive_pad=0.0)

# def assemble(bkg_feat0, obj_feat0, origin_T_camRs, camRs_T_zoom):
#     # let's first assemble the seq of background tensors
#     # this should effectively CREATE egomotion
#     # i fully expect we can do this all in one shot

#     # note it makes sense to create egomotion here, because
#     # we want to predict each view

#     B, C, Z, Y, X = list(bkg_feat0.shape)
#     B2, C2, Z2, Y2, X2 = list(obj_feat0.shape)
#     assert(B==B2)
#     assert(C==C2)
    
#     B, S, _, _ = list(origin_T_camRs.shape)
#     # ok, we have everything we need
#     # for each timestep, we want to warp the bkg to this timestep
    
#     # utils for packing/unpacking along seq dim
#     __p = lambda x: utils.basic.pack_seqdim(x, B)
#     __u = lambda x: utils.basic.unpack_seqdim(x, B)

#     # we in fact have utils for this already
#     cam0s_T_camRs = utils.geom.get_camM_T_camXs(origin_T_camRs, ind=0)
#     camRs_T_cam0s = __u(utils.geom.safe_inverse(__p(cam0s_T_camRs)))

#     bkg_feat0s = bkg_feat0.unsqueeze(1).repeat(1, S, 1, 1, 1, 1)
#     bkg_featRs = apply_4x4s_to_voxs(camRs_T_cam0s, bkg_feat0s)

#     # now for the objects
    
#     # we want to sample for each location in the bird grid
#     xyz_mems_ = utils.basic.gridcloud3d(B*S, Z, Y, X, norm=False)
#     # this is B*S x Z*Y*X x 3
#     xyz_camRs_ = self.Mem2Ref(xyz_mems_, Z, Y, X)
#     camRs_T_zoom_ = __p(camRs_T_zoom)
#     zoom_T_camRs_ = camRs_T_zoom_.inverse() # note this is not a rigid transform
#     xyz_zooms_ = utils.geom.apply_4x4(zoom_T_camRs_, xyz_camRs_)

#     # we will do the whole traj at once (per obj)
#     # note we just have one feat for the whole traj, so we tile up
#     obj_feats = obj_feat0.unsqueeze(1).repeat(1, S, 1, 1, 1, 1)
#     obj_feats_ = __p(obj_feats)
#     # this is B*S x Z x Y x X x C
    
#     # to sample, we need feats_ in ZYX order
#     obj_featRs_ = utils.samp.sample3d(obj_feats_, xyz_zooms_, Z, Y, X)
#     obj_featRs = __u(obj_featRs_)

#     # overweigh objects, so that we essentially overwrite
#     # featRs = 0.05*bkg_featRs + 0.95*obj_featRs

#     # overwrite the bkg at the object
#     obj_mask = (bkg_featRs > 0).float()
#     featRs = obj_featRs + (1.0-obj_mask)*bkg_featRs
    
#     # note the normalization (next) will restore magnitudes for the bkg

#     # # featRs = bkg_featRs
#     # featRs = obj_featRs
                        
#     # l2 normalize on chans
#     featRs = l2_normalize(featRs, dim=2)

#     validRs = 1.0 - (featRs==0).all(dim=2, keepdim=True).float().cuda()
                        
#     return featRs, validRs, bkg_featRs, obj_featRs

# def convert_boxlist_memR_to_camR(boxlist_memR, Z, Y, X):
#     B, N, D = list(boxlist_memR.shape)
#     assert(D==9)
#     cornerlist_memR_legacy = utils.geom.transform_boxes_to_corners(boxlist_memR, legacy_format=True)
#     ref_T_mem = self.get_ref_T_mem(B, Z, Y, X)
#     cornerlist_camR_legacy = utils.geom.apply_4x4_to_corners(ref_T_mem, cornerlist_memR_legacy)
#     boxlist_camR = utils.geom.corners_to_boxes(cornerlist_camR_legacy, legacy_format=True)
#     return boxlist_camR

# def convert_boxlist_camR_to_memR(boxlist_camR, Z, Y, X):
#     B, N, D = list(boxlist_camR.shape)
#     assert(D==9)
#     cornerlist_camR_legacy = utils.geom.transform_boxes_to_corners(boxlist_camR, legacy_format=True)
#     mem_T_ref = self.get_mem_T_ref(B, Z, Y, X)
#     cornerlist_memR_legacy = utils.geom.apply_4x4_to_corners(mem_T_ref, cornerlist_camR_legacy)
#     boxlist_memR = utils.geom.corners_to_boxes(cornerlist_memR_legacy, legacy_format=True)
#     return boxlist_memR
