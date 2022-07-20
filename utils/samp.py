import torch
from utils.basic import *
import utils.basic
import utils.geom
# import resampler_lib.grid_interpolate as interpolate_cuda
import torch.nn.functional as F

def trilinear_sample3d(vox, xyz, return_inbounds=False):
    # xyz is B x N x 3
    B, C, D, H, W = list(vox.shape)
    B2, N, E = list(xyz.shape)
    assert(B==B2)
    assert(E==3)
    x, y, z = torch.unbind(xyz, dim=2)

    x = x.float()
    y = y.float()
    z = z.float()
    D_f = torch.tensor(D, dtype=torch.float32)
    H_f = torch.tensor(H, dtype=torch.float32)
    W_f = torch.tensor(W, dtype=torch.float32)
    max_z = (D_f - 1).int()
    max_y = (H_f - 1).int()
    max_x = (W_f - 1).int()

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1
    z0 = torch.floor(z).int()
    z1 = z0 + 1

    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    z0_clip = torch.clamp(z0, 0, max_z)
    z1_clip = torch.clamp(z1, 0, max_z)
    dim3 = W
    dim2 = W * H
    dim1 = W * H * D

    base = torch.arange(0, B, dtype=torch.int32).cuda()*dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N])

    base_z0_y0 = base + z0_clip * dim2 + y0_clip * dim3
    base_z0_y1 = base + z0_clip * dim2 + y1_clip * dim3
    base_z1_y0 = base + z1_clip * dim2 + y0_clip * dim3
    base_z1_y1 = base + z1_clip * dim2 + y1_clip * dim3

    idx_z0_y0_x0 = base_z0_y0 + x0_clip
    idx_z0_y0_x1 = base_z0_y0 + x1_clip
    idx_z0_y1_x0 = base_z0_y1 + x0_clip
    idx_z0_y1_x1 = base_z0_y1 + x1_clip
    idx_z1_y0_x0 = base_z1_y0 + x0_clip
    idx_z1_y0_x1 = base_z1_y0 + x1_clip
    idx_z1_y1_x0 = base_z1_y1 + x0_clip
    idx_z1_y1_x1 = base_z1_y1 + x1_clip

    # use the indices to lookup pixels in the flat image
    # vox is B x C x H x W x D
    # move C out to last dim
    vox_flat = (vox.permute(0, 2, 3, 4, 1)).reshape(B*D*H*W, C)
    i_z0_y0_x0 = vox_flat[idx_z0_y0_x0.long()]
    i_z0_y0_x1 = vox_flat[idx_z0_y0_x1.long()]
    i_z0_y1_x0 = vox_flat[idx_z0_y1_x0.long()]
    i_z0_y1_x1 = vox_flat[idx_z0_y1_x1.long()]
    i_z1_y0_x0 = vox_flat[idx_z1_y0_x0.long()]
    i_z1_y0_x1 = vox_flat[idx_z1_y0_x1.long()]
    i_z1_y1_x0 = vox_flat[idx_z1_y1_x0.long()]
    i_z1_y1_x1 = vox_flat[idx_z1_y1_x1.long()]

    # Finally calculate interpolated values.
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()
    z0_f = z0.float()
    z1_f = z1.float()

    x0_valid = torch.ones_like(x0, dtype=torch.float32)
    y0_valid = torch.ones_like(y0, dtype=torch.float32)
    z0_valid = torch.ones_like(z0, dtype=torch.float32)
    x1_valid = torch.ones_like(x1, dtype=torch.float32)
    y1_valid = torch.ones_like(y1, dtype=torch.float32)
    z1_valid = torch.ones_like(z1, dtype=torch.float32)

    w_z0_y0_x0 = ((x1_f - x) * (y1_f - y) *
                  (z1_f - z) * x1_valid * y1_valid * z1_valid).unsqueeze(2)
    w_z0_y0_x1 = ((x - x0_f) * (y1_f - y) *
                  (z1_f - z) * x0_valid * y1_valid * z1_valid).unsqueeze(2)
    w_z0_y1_x0 = ((x1_f - x) * (y - y0_f) *
                  (z1_f - z) * x1_valid * y0_valid * z1_valid).unsqueeze(2)
    w_z0_y1_x1 = ((x - x0_f) * (y - y0_f) *
                  (z1_f - z) * x0_valid * y0_valid * z1_valid).unsqueeze(2)
    w_z1_y0_x0 = ((x1_f - x) * (y1_f - y) *
                  (z - z0_f) * x1_valid * y1_valid * z0_valid).unsqueeze(2)
    w_z1_y0_x1 = ((x - x0_f) * (y1_f - y) *
                  (z - z0_f) * x0_valid * y1_valid * z0_valid).unsqueeze(2)
    w_z1_y1_x0 = ((x1_f - x) * (y - y0_f) *
                  (z - z0_f) * x1_valid * y0_valid * z0_valid).unsqueeze(2)
    w_z1_y1_x1 = ((x - x0_f) * (y - y0_f) *
                  (z - z0_f) * x0_valid * y0_valid * z0_valid).unsqueeze(2)

    output = w_z0_y0_x0 * i_z0_y0_x0 + w_z0_y0_x1 * i_z0_y0_x1 + \
             w_z0_y1_x0 * i_z0_y1_x0 + w_z0_y1_x1 * i_z0_y1_x1 + \
             w_z1_y0_x0 * i_z1_y0_x0 + w_z1_y0_x1 * i_z1_y0_x1 + \
             w_z1_y1_x0 * i_z1_y1_x0 + w_z1_y1_x1 * i_z1_y1_x1
    # output is B*N x C
    output = output.view(B, -1, C)
    output = output.permute(0, 2, 1)
    # output is B x C x N

    if return_inbounds:
        x_valid = (x>-0.5).byte() & (x<float(W_f-0.5)).byte()
        y_valid = (y>-0.5).byte() & (y<float(H_f-0.5)).byte()
        z_valid = (z>-0.5).byte() & (z<float(D_f-0.5)).byte()
        inbounds = (x_valid & y_valid & z_valid).float()
        inbounds = inbounds.reshape(B, N)
        return output, inbounds
    
    return output

def bilinear_sample2d(im, x, y, return_inbounds=False):
    # x and y are each B, N
    # output is B, C, N
    B, C, H, W = list(im.shape)
    N = list(x.shape)[1]

    x = x.float()
    y = y.float()
    H_f = torch.tensor(H, dtype=torch.float32)
    W_f = torch.tensor(W, dtype=torch.float32)
    
    # inbound_mask = (x>-0.5).float()*(y>-0.5).float()*(x<W_f+0.5).float()*(y<H_f+0.5).float()

    max_y = (H_f - 1).int()
    max_x = (W_f - 1).int()

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1
    
    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    dim2 = W
    dim1 = W * H

    base = torch.arange(0, B, dtype=torch.int64).cuda()*dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N])

    base_y0 = base + y0_clip * dim2
    base_y1 = base + y1_clip * dim2

    idx_y0_x0 = base_y0 + x0_clip
    idx_y0_x1 = base_y0 + x1_clip
    idx_y1_x0 = base_y1 + x0_clip
    idx_y1_x1 = base_y1 + x1_clip

    # use the indices to lookup pixels in the flat image
    # im is B x C x H x W
    # move C out to last dim
    im_flat = (im.permute(0, 2, 3, 1)).reshape(B*H*W, C)
    i_y0_x0 = im_flat[idx_y0_x0.long()]
    i_y0_x1 = im_flat[idx_y0_x1.long()]
    i_y1_x0 = im_flat[idx_y1_x0.long()]
    i_y1_x1 = im_flat[idx_y1_x1.long()]

    # Finally calculate interpolated values.
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()

    w_y0_x0 = ((x1_f - x) * (y1_f - y)).unsqueeze(2)
    w_y0_x1 = ((x - x0_f) * (y1_f - y)).unsqueeze(2)
    w_y1_x0 = ((x1_f - x) * (y - y0_f)).unsqueeze(2)
    w_y1_x1 = ((x - x0_f) * (y - y0_f)).unsqueeze(2)

    output = w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + \
             w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1
    # output is B*N x C
    output = output.view(B, -1, C)
    output = output.permute(0, 2, 1)
    # output is B x C x N

    if return_inbounds:
        x_valid = (x > -0.5).byte() & (x < float(W_f - 0.5)).byte()
        y_valid = (y > -0.5).byte() & (y < float(H_f - 0.5)).byte()
        inbounds = (x_valid & y_valid).float()
        inbounds = inbounds.reshape(B, N) # something seems wrong here for B>1; i'm getting an error here (or downstream if i put -1)
        return output, inbounds

    return output # B, C, N

def bilinear_sample_single(im, x, y, return_mask=False):
    C, H, W = list(im.shape)
    N = list(x.shape)
    N2 = list(y.shape)
    assert(N==N2)

    x = x.float()
    y = y.float()
    h_f = torch.tensor(H, dtype=torch.float32).cuda()
    w_f = torch.tensor(W, dtype=torch.float32).cuda()

    inbound_mask = (x>-0.5).float()*(y>-0.5).float()*(x<w_f+0.5).float()*(y<h_f+0.5).float()

    x = torch.clamp(x, 0, w_f-1)
    y = torch.clamp(y, 0, h_f-1)

    x0_f = torch.floor(x)
    y0_f = torch.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1
    x0 = x0_f.int()
    y0 = y0_f.int()
    x1 = torch.min(x1_f, w_f-1).int()
    y1 = torch.min(y1_f, h_f-1).int()
    dim2 = W
    dim1 = W*H
    idx_a = sub2ind(H, W, y0, x0)
    idx_b = sub2ind(H, W, y1, x0)
    idx_c = sub2ind(H, W, y0, x1)
    idx_d = sub2ind(H, W, y1, x1)

    # use the indices to lookup pixels in the flat image
    im_flat = (im.permute(1, 2, 0)).view(H*W, C)
    Ia = im_flat[idx_a.long()]
    Ib = im_flat[idx_b.long()]
    Ic = im_flat[idx_c.long()]
    Id = im_flat[idx_d.long()]
    # calculate interpolated values
    wa = ((x1_f-x) * (y1_f-y)).unsqueeze(1)
    wb = ((x1_f-x) * (y-y0_f)).unsqueeze(1)
    wc = ((x-x0_f) * (y1_f-y)).unsqueeze(1)
    wd = ((x-x0_f) * (y-y0_f)).unsqueeze(1)
    interp = wa*Ia+wb*Ib+wc*Ic+wd*Id
    
    interp = interp*inbound_mask.unsqueeze(1)
    # interp is N x C
    interp = interp.permute(1, 0)
    # interp is C x N

    if not return_mask:
        return interp
    else:
        mask = torch.zeros_like(im_flat[:,0:1])
        mask[idx_a.long()] = 1
        mask[idx_b.long()] = 1
        mask[idx_c.long()] = 1
        mask[idx_d.long()] = 1
        return interp, mask

def backwarp_using_3d_flow(vox1, flow0, binary_feat=False):
    # flow points from 0 to 1
    # vox1 is in coords1
    # returns vox0 
    # print('backwarping...')
    # print_shape(vox1)
    # print_shape(flow0)
    B, C, Z, Y, X = list(vox1.shape)
    cloud0 = gridcloud3d(B, Z, Y, X)
    cloud0_displacement = flow0.reshape(B, 3, Z*Y*X).permute(0, 2, 1)
    resampling_coords = cloud0 + cloud0_displacement
    return resample3d(vox1, resampling_coords, binary_feat=binary_feat)

def backwarp_using_2d_flow(im1, flow0, binary_feat=False, device='cuda'):
    # flow points from 0 to 1
    # im1 is in coords1
    # returns im0 
    B, C, Y, X = list(im1.shape)
    cloud0 = utils.basic.gridcloud2d(B, Y, X, device=device)
    cloud0_displacement = flow0.reshape(B, 2, Y*X).permute(0, 2, 1)
    resampling_coords = cloud0 + cloud0_displacement
    return resample2d(im1, resampling_coords, binary_feat=binary_feat)

def get_backwarp_mask(flow0):
    # flow points from 0 to 1
    # im1 is in coords1
    # returns im0 
    B, C, Y, X = list(flow0.shape)
    cloud0 = utils.basic.gridcloud2d(B, Y, X)
    cloud0_displacement = flow0.reshape(B, 2, Y*X).permute(0, 2, 1)
    resampling_coords = cloud0 + cloud0_displacement
    # resampling_coords = resampling_coords.long()
    mask = torch.zeros_like(flow0[:,0:1])
    for b in range(B):
        _, mask_ = bilinear_sample_single(mask[b].reshape(-1, Y, X), resampling_coords[b,:,0], resampling_coords[b,:,1], return_mask=True)
        mask[b] = mask_.reshape(1, Y, X)
    # out = empty.scatter_(0, resampling_coords, torch.ones_like(empty))
    # return out
    return mask

def resample3d(vox, xyz, binary_feat=False):
    # vox is some voxel feats
    # xyz is some 3d coordinates, e.g., from gridcloud3d
    B, C, Z, Y, X = list(vox.shape)
    xyz = normalize_gridcloud3d(xyz, Z, Y, X)
    xyz = torch.reshape(xyz, [B, Z, Y, X, 3])
    vox = F.grid_sample(vox, xyz)
    if binary_feat:
        vox = vox.round()
    return vox

def resample2d(im, xy, binary_feat=False):
    # im is some image feats
    # xy is some 2d coordinates, e.g., from gridcloud2d
    B, C, Y, X = list(im.shape)
    xy = normalize_gridcloud2d(xy, Y, X)
    xy = torch.reshape(xy, [B, Y, X, 2])
    im = F.grid_sample(im, xy)
    if binary_feat:
        im = im.round()
    return im

def crop_and_resize_box2d(im, box2d, Y, X):
    B, C, H, W = list(im.shape)
    B2, D = list(box2d.shape)
    assert(B==B2)
    assert(D==4)
    grid_y, grid_x = utils.basic.meshgrid2d(B, Y, X, stack=False, norm=True)
    # now the range is [-1,1]

    grid_y = (grid_y+1.0)/2.0
    grid_x = (grid_x+1.0)/2.0
    # now the range is [0,1]

    h, w = utils.geom.get_size_from_box2d(box2d)
    
    ymin, xmin, ymax, xmax = torch.unbind(box2d, dim=1)
    
    grid_y = grid_y*h.reshape(B, 1, 1) + ymin.reshape(B, 1, 1)
    grid_x = grid_x*w.reshape(B, 1, 1) + xmin.reshape(B, 1, 1)
    # now the range is (0,1)
    
    grid_y = (grid_y*2.0)-1.0
    grid_x = (grid_x*2.0)-1.0
    # now the range is (-1,1)

    xy = torch.stack([grid_x, grid_y], dim=3)
    samp = F.grid_sample(im, xy)
    return samp

def paste_crop_on_canvas(crop, box2d, H, W, fast=True, mask=None, canvas=None):
    # this is the inverse of crop_and_resize_box2d
    B, C, Y, X = list(crop.shape)
    B2, D = list(box2d.shape)
    assert(B == B2)
    assert(D == 4)

    # here, we want to place the crop into a bigger image,
    # at the location specified by the box2d.
    # since i don't care about differentiability,
    # the smart thing to do here might be to figure out the desired size,
    # do an F.interpolate,
    # then go to those pixels and overwrite them

    if canvas is None:
        canvas = torch.zeros((B, C, H, W), device=crop.device)
    else:
        B2, C2, H2, W2 = canvas.shape
        assert(B==B2)
        assert(C==C2)
        assert(H==H2)
        assert(W==W2)

    box2d_unnorm = utils.geom.unnormalize_box2d(box2d, H, W)

    if fast:
        ymin = box2d_unnorm[:, 0].long()
        xmin = box2d_unnorm[:, 1].long()
        ymax = box2d_unnorm[:, 2].long()
        xmax = box2d_unnorm[:, 3].long()
        w = (xmax - xmin).float()
        h = (ymax - ymin).float()

        grids = utils.basic.gridcloud2d(B, H, W)
        grids_flat = grids.reshape(B, -1, 2)
        grids_flat[:, :, 0] = (grids_flat[:, :, 0] - xmin.float().unsqueeze(1)) / w.unsqueeze(1) * X
        grids_flat[:, :, 1] = (grids_flat[:, :, 1] - ymin.float().unsqueeze(1)) / h.unsqueeze(1) * Y

        if mask is None:
            crop_resamp, inb = bilinear_sample2d(crop, grids_flat[:, :, 0], grids_flat[:, :, 1], return_inbounds=True)
            crop_resamp = crop_resamp.reshape(B, C, H, W)
            inb = inb.reshape(B, 1, H, W)
            canvas = canvas * (1 - inb) + crop_resamp * inb
        else:
            full_resamp = bilinear_sample2d(torch.cat([crop, mask], dim=1), grids_flat[:, :, 0], grids_flat[:, :, 1])
            full_resamp = full_resamp.reshape(B, C+1, H, W)
            crop_resamp = full_resamp[:,:3]
            mask_resamp = full_resamp[:,3:4]
            canvas = canvas * (1 - mask_resamp) + crop_resamp * mask_resamp
    else:
        for b in range(B):
            ymin = box2d_unnorm[b, 0].long()
            xmin = box2d_unnorm[b, 1].long()
            ymax = box2d_unnorm[b, 2].long()
            xmax = box2d_unnorm[b, 3].long()

            crop_b = F.interpolate(crop[b:b + 1], (ymax - ymin, xmax - xmin)).squeeze(0)

            # print('canvas[b,:,...', canvas[b,:,ymin:ymax,xmin:xmax].shape)
            # print('crop_b', crop_b.shape)

            canvas[b, :, ymin:ymax, xmin:xmax] = crop_b
    return canvas
    
        
def sample3d(vox, xyz, D, H, W, mode='bilinear'):
    # vox is the thing we are sampling from
    # xyz indicates the places to sample
    # D, H, W is the shape we want to end up with
    B, E, Z, Y, X = list(vox.shape)
    B, N, C = list(xyz.shape)
    assert(C==3)
    assert(N==(D*H*W))

    if (0):
        # our old func
        x, y, z = torch.unbind(xyz, dim=2)
        samp = trilinear_sample3d(vox, x, y, z)
    else:
        # pytorch's native func
        xyz = normalize_gridcloud3d(xyz, Z, Y, X)
        xyz = torch.reshape(xyz, [B, D, H, W, 3])
        samp = F.grid_sample(vox, xyz, mode=mode)
    
    samp = torch.reshape(samp, [B, E, D, H, W])
    return samp


def cuda_grid_sample(im, grid, use_native=False):
    assert(False) # this was disabled on oct15,2019, since torch has its own cuda resampler.
    
    gridshape = tuple(grid.shape)
    
    num_batch, channels, depth, height, width = list(im.shape)
    out_size = list(grid.shape)[1:-1]
    # grid = grid.view(-1, 3)
    #old - not using x, y, z = tf.unstack(grid, axis = -1)
    # z, y, x = tf.unstack(grid, axis = -1)
    
    # grid = tf.stack([z,y,x], axis=-1)
    grid = torch.reshape(grid, gridshape)

    if use_native:
        interpolate_func = interpolate_cuda.GridInterpolateFunction.apply

        raw_out = interpolate_func(im.permute(0,2,3,4,1), grid, True)
        raw_out = raw_out.permute(0,4,1,2,3)
        # return grid_interpolate3d(im, grid)
    else:
        # assert(False) # need to edit this to also return inbounds
        raw_out = non_cuda_grid_sample(im, grid)
    B,C,D,H,W = list(im.shape)
    inbounds = torch.cat([grid>=-0.5,
                          grid<=torch.tensor([D-0.5,H-0.5,W-0.5])],
                         dim=-1).float()
    inbounds = torch.sum(1.0-inbounds, dim=-1, keepdim=True)
    inbounds = inbounds < 0.5
    inbounds = inbounds.float()
    im_interp = torch.reshape(raw_out, tuple(im.shape))
    im_interp *= inbounds.permute(0,4,1,2,3)
    return im_interp, inbounds

def non_cuda_grid_sample(im, grid):
    #rename some variables, do some reshaping
    
    out_size = list(grid.shape)[1:-1]    
    grid = torch.reshape(grid, (-1, 3))
    z, y, x = grid[:,0], grid[:,1], grid[:,2]
    BS = list(im.shape)[0]

    #################
    
    num_batch, channels, depth, height, width = list(im.shape)

    x = x.float()
    y = y.float()
    z = z.float()
    
    depth_f = torch.tensor(depth, dtype=torch.float32)
    height_f = torch.tensor(height, dtype=torch.float32)
    width_f = torch.tensor(width, dtype=torch.float32)
    
    # Number of disparity interpolated.o
    out_depth = out_size[0]
    out_height = out_size[1]
    out_width = out_size[2]
    
    # 0 <= z < depth, 0 <= y < height & 0 <= x < width.
    max_z = depth - 1
    max_y = height - 1
    max_x = width - 1

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1
    z0 = torch.floor(z).int()
    z1 = z0 + 1

    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    z0_clip = torch.clamp(z0, 0, max_z)
    z1_clip = torch.clamp(z1, 0, max_z)
    
    dim3 = width
    dim2 = width * height
    dim1 = width * height * depth
    dim1, dim2, dim3 = torch.tensor(dim1), torch.tensor(dim2), torch.tensor(dim3), 


    base = torch.tensor(np.concatenate([np.array([i*dim1] * out_depth * out_height * out_width)
                        for i in list(range(BS))]).astype(np.int32))

    base_z0_y0 = base + z0_clip * dim2 + y0_clip * dim3
    base_z0_y1 = base + z0_clip * dim2 + y1_clip * dim3
    base_z1_y0 = base + z1_clip * dim2 + y0_clip * dim3
    base_z1_y1 = base + z1_clip * dim2 + y1_clip * dim3

    idx_z0_y0_x0 = base_z0_y0 + x0_clip
    idx_z0_y0_x1 = base_z0_y0 + x1_clip
    idx_z0_y1_x0 = base_z0_y1 + x0_clip
    idx_z0_y1_x1 = base_z0_y1 + x1_clip
    idx_z1_y0_x0 = base_z1_y0 + x0_clip
    idx_z1_y0_x1 = base_z1_y0 + x1_clip
    idx_z1_y1_x0 = base_z1_y1 + x0_clip
    idx_z1_y1_x1 = base_z1_y1 + x1_clip

    # Use indices to lookup pixels in the flat image and restore
    # channels dim
    im = im.permute(0,2,3,4,1)
    im_flat = torch.reshape(im, (-1, channels))
    im_flat = im_flat.float()
    i_z0_y0_x0 = im_flat[idx_z0_y0_x0.long()]
    i_z0_y0_x1 = im_flat[idx_z0_y0_x1.long()]
    i_z0_y1_x0 = im_flat[idx_z0_y1_x0.long()]
    i_z0_y1_x1 = im_flat[idx_z0_y1_x1.long()]
    i_z1_y0_x0 = im_flat[idx_z1_y0_x0.long()]
    i_z1_y0_x1 = im_flat[idx_z1_y0_x1.long()]
    i_z1_y1_x0 = im_flat[idx_z1_y1_x0.long()]
    i_z1_y1_x1 = im_flat[idx_z1_y1_x1.long()]

    # Finally calculate interpolated values.
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()
    z0_f = z0.float()
    z1_f = z1.float()
    
    if True: #out of range mode "boundary"
        x0_valid = torch.ones_like(x0_f)
        x1_valid = torch.ones_like(x1_f)
        y0_valid = torch.ones_like(y0_f)
        y1_valid = torch.ones_like(y1_f)
        z0_valid = torch.ones_like(z0_f)
        z1_valid = torch.ones_like(z1_f)

    w_z0_y0_x0 = ((x1_f - x) * (y1_f - y) *
                                 (z1_f - z) * x1_valid * y1_valid * z1_valid).unsqueeze(
                                1)
    w_z0_y0_x1 = ((x - x0_f) * (y1_f - y) *
                                 (z1_f - z) * x0_valid * y1_valid * z1_valid).unsqueeze(
                                1)
    w_z0_y1_x0 = ((x1_f - x) * (y - y0_f) *
                                 (z1_f - z) * x1_valid * y0_valid * z1_valid).unsqueeze(
                                1)
    w_z0_y1_x1 = ((x - x0_f) * (y - y0_f) *
                                 (z1_f - z) * x0_valid * y0_valid * z1_valid).unsqueeze(
                                1)
    w_z1_y0_x0 = ((x1_f - x) * (y1_f - y) *
                                 (z - z0_f) * x1_valid * y1_valid * z0_valid).unsqueeze(
                                1)
    w_z1_y0_x1 = ((x - x0_f) * (y1_f - y) *
                                 (z - z0_f) * x0_valid * y1_valid * z0_valid).unsqueeze(
                                1)
    w_z1_y1_x0 = ((x1_f - x) * (y - y0_f) *
                                 (z - z0_f) * x1_valid * y0_valid * z0_valid).unsqueeze(
                                1)
    w_z1_y1_x1 = ((x - x0_f) * (y - y0_f) *
                                 (z - z0_f) * x0_valid * y0_valid * z0_valid).unsqueeze(
                                1)

    weights_summed = (
        w_z0_y0_x0 +
        w_z0_y0_x1 +
        w_z0_y1_x0 +
        w_z0_y1_x1 +
        w_z1_y0_x0 +
        w_z1_y0_x1 +
        w_z1_y1_x0 +
        w_z1_y1_x1
    )

    output = (
        w_z0_y0_x0 * i_z0_y0_x0+w_z0_y0_x1 * i_z0_y0_x1+
        w_z0_y1_x0 * i_z0_y1_x0+w_z0_y1_x1 * i_z0_y1_x1+
        w_z1_y0_x0 * i_z1_y0_x0+w_z1_y0_x1 * i_z1_y0_x1+
        w_z1_y1_x0 * i_z1_y1_x0+w_z1_y1_x1 * i_z1_y1_x1
    )
    
    return output
 
def linear_sample1d(vec, y):
    B, C, H = list(vec.shape)
    N = list(y.shape)[1]

    y = y.float()
    H_f = torch.tensor(H, dtype=torch.float32)
    
    # print('vec', vec, vec.shape)
    # print('y', y, y.shape)
    # print('vec', vec.shape)
    # print('y', y.shape)

    max_y = (H_f - 1).int()

    y0 = torch.floor(y).int()
    y1 = y0 + 1
    
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    dim1 = H

    # print('dim1', dim1)
    # print('y0_clip', y0_clip.shape)
    # print('y1_clip', y1_clip.shape)

    base = torch.arange(0, B, dtype=torch.int32).cuda()*dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N])

    idx_y0 = base + y0_clip
    idx_y1 = base + y1_clip

    # print('idx_y0', idx_y0, idx_y0.shape)
    # print('idx_y1', idx_y1, idx_y1.shape)
    
    # use the indices to lookup pixels in the flattened vec
    # vec is B x C x H
    # move C out to last dim
    vec_flat = (vec.permute(0, 2, 1)).reshape(B*H, C)
    i_y0 = vec_flat[idx_y0.long()]
    i_y1 = vec_flat[idx_y1.long()]

    # print('i_y0', i_y0.shape)
    # print('i_y1', i_y1.shape)

    # Finally calculate interpolated values.
    y0_f = y0.float()
    y1_f = y1.float()

    # print('y0_f', y0_f.shape)
    # print('y1_f', y1_f.shape)
    
    w_y0 = ((y1_f - y)).unsqueeze(2)
    w_y1 = ((y - y0_f)).unsqueeze(2)

    # print('w_y0', w_y0.shape)
    # print('w_y1', w_y1.shape)
    
    output = w_y0 * i_y0 + w_y1 * i_y1
    # output is B*N x C
    output = output.view(B, -1, C)
    output = output.permute(0, 2, 1)
    # output is B x C x N

    return output
