import os
import numpy as np
from os.path import isfile
import torch
import torch.nn.functional as F
EPS = 1e-6
import copy

def get_lr_str(lr):
    lrn = "%.1e" % lr # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1] # e.g., 5e-4
    return lrn
    
def strnum(x):
    s = '%g' % x
    if '.' in s:
        if x < 1.0:
            s = s[s.index('.'):]
    return s

def assert_same_shape(t1, t2):
    for (x, y) in zip(list(t1.shape), list(t2.shape)):
        assert(x==y)

# def print_stats_py(name, tensor):
#     print('%s (%s) min = %.2f, mean = %.2f, max = %.2f' % (name, tensor.dtype, np.min(tensor), np.mean(tensor), np.max(tensor)))
def print_stats(name, tensor):
    shape = tensor.shape
    tensor = tensor.detach().cpu().numpy()
    print('%s (%s) min = %.2f, mean = %.2f, max = %.2f' % (name, tensor.dtype, np.min(tensor), np.mean(tensor), np.max(tensor)), shape)

def print_(name, tensor):
    tensor = tensor.detach().cpu().numpy()
    print(name, tensor, tensor.shape)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def tensor2summ(tensor, permute_dim=False):
    # if permute_dim = True: 
    # for 2d tensor, assume input is torch format B x S x C x H x W, we want B x S x H x W x C
    # for 3d tensor, assume input is torch format B x S x C x H x W x D, we want B x S x H x W x C x D
    # and finally unbind the sequeence dimension and return a list of [B x H x W x C].
    assert(tensor.ndim == 5 or tensor.ndim == 6)
    assert(tensor.size()[1] == 2) #sequense length should be 2
    if permute_dim:
        if tensor.ndim == 6: #3d tensor
            tensor = tensor.permute(0, 1, 3, 4, 5, 2)
        elif tensor.ndim == 5: #2d tensor
            tensor = tensor.permute(0, 1, 3, 4, 2)

    tensor = torch.unbind(tensor, dim=1)
    return tensor

def normalize_single(d):
    # d is a whatever shape torch tensor
    dmin = torch.min(d)
    dmax = torch.max(d)
    d = (d-dmin)/(EPS+(dmax-dmin))
    return d

def normalize(d):
    # d is B x whatever. normalize within each element of the batch
    out = torch.zeros(d.size())
    if d.is_cuda:
        out = out.cuda()
    B = list(d.size())[0]
    for b in list(range(B)):
        out[b] = normalize_single(d[b])
    return out

def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    # x and mask are the same shape, or at least broadcastably so < actually it's safer if you disallow broadcasting
    # returns shape-1
    # axis can be a list of axes
    for (a,b) in zip(x.size(), mask.size()):
        # if not b==1: 
        assert(a==b) # some shape mismatch!
    # assert(x.size() == mask.size())
    prod = x*mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS+torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS+torch.sum(mask, dim=dim, keepdim=keepdim)
        
    mean = numer/denom
    return mean

def reduce_masked_min(x, mask, dim=None, keepdim=False):
    # x and mask are the same shape, or at least broadcastably so
    # returns shape-1
    # axis can be a list of axes
    for (a,b) in zip(x.size(), mask.size()):
        if not b==1:
            assert(a==b) # some shape mismatch!
    mask = mask.expand_as(x)
    x[mask==0] = torch.max(x)
    masked_min = torch.min(x, dim=dim)[0]
    return masked_min

def reduce_masked_median(x, mask, keep_batch=False):
    # x and mask are the same shape
    # returns shape-1
    # axis can be a list of axes
    assert(x.size() == mask.size())
    prod = x*mask

    B = list(x.shape)[0]
    x = x.cpu().numpy()
    mask = mask.cpu().numpy()

    if keep_batch:
        x = np.reshape(x, [B, -1])
        mask = np.reshape(mask, [B, -1])
        meds = np.zeros([B], np.float32)
        for b in list(range(B)):
            xb = x[b]
            mb = mask[b]
            if np.sum(mb) > 0:
                xb = xb[mb > 0]
                meds[b] = np.median(xb)
            else:
                meds[b] = np.nan
        meds = torch.from_numpy(meds).cuda()
        return meds
    else:
        x = np.reshape(x, [-1])
        mask = np.reshape(mask, [-1])
        if np.sum(mask) > 0:
            x = x[mask > 0]
            med = np.median(x)
        else:
            med = np.nan
        med = np.array([med], np.float32)
        med = torch.from_numpy(med).cuda()
        return med

def pack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    B_, S = shapelist[:2]
    assert(B==B_)
    otherdims = shapelist[2:]
    tensor = torch.reshape(tensor, [B*S]+otherdims)
    return tensor

def unpack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    BS = shapelist[0]
    assert(BS%B==0)
    otherdims = shapelist[1:]
    S = int(BS/B)
    tensor = torch.reshape(tensor, [B,S]+otherdims)
    return tensor

def gridcloud3d(B, Z, Y, X, norm=False, device='cuda'):
    # we want to sample for each location in the grid
    grid_z, grid_y, grid_x = meshgrid3d(B, Z, Y, X, norm=norm, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])
    # these are B x N
    xyz = torch.stack([x, y, z], dim=2)
    # this is B x N x 3
    return xyz

def gridcloud2d(B, Y, X, norm=False, device='cuda'):
    # we want to sample for each location in the grid
    grid_y, grid_x = meshgrid2d(B, Y, X, norm=norm, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    # these are B x N
    xy = torch.stack([x, y], dim=2)
    # this is B x N x 2
    return xy

def gridcloud3d_py(Z, Y, X):
    # we want to sample for each location in the grid
    grid_z, grid_y, grid_x = meshgrid3d_py(Z, Y, X)
    x = np.reshape(grid_x, [-1])
    y = np.reshape(grid_y, [-1])
    z = np.reshape(grid_z, [-1])
    # these are N
    xyz = np.stack([x, y, z], axis=1)
    # this is N x 3
    return xyz

def meshgrid2d_py(Y, X):
    grid_y = np.linspace(0.0, Y-1, Y)
    grid_y = np.reshape(grid_y, [Y, 1])
    grid_y = np.tile(grid_y, [1, X])

    grid_x = np.linspace(0.0, X-1, X)
    grid_x = np.reshape(grid_x, [1, X])
    grid_x = np.tile(grid_x, [Y, 1])

    return grid_y, grid_x

def gridcloud2d_py(Y, X):
    # we want to sample for each location in the grid
    grid_y, grid_x = meshgrid2d_py(Y, X)
    x = np.reshape(grid_x, [-1])
    y = np.reshape(grid_y, [-1])
    # these are N
    xy = np.stack([x, y], axis=1)
    # this is N x 2
    return xy

def normalize_grid3d(grid_z, grid_y, grid_x, Z, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    grid_z = 2.0*(grid_z / float(Z-1)) - 1.0
    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0
    
    if clamp_extreme:
        grid_z = torch.clamp(grid_z, min=-2.0, max=2.0)
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)
    
    return grid_z, grid_y, grid_x

def normalize_grid2d(grid_y, grid_x, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0
    
    if clamp_extreme:
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)
        
    return grid_y, grid_x

def normalize_gridcloud3d(xyz, Z, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    x = xyz[...,0]
    y = xyz[...,1]
    z = xyz[...,2]
    
    z = 2.0*(z / float(Z-1)) - 1.0
    y = 2.0*(y / float(Y-1)) - 1.0
    x = 2.0*(x / float(X-1)) - 1.0

    xyz = torch.stack([x,y,z], dim=-1)
    
    if clamp_extreme:
        xyz = torch.clamp(xyz, min=-2.0, max=2.0)
    return xyz

def normalize_gridcloud2d(xy, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    x = xy[...,0]
    y = xy[...,1]
    
    y = 2.0*(y / float(Y-1)) - 1.0
    x = 2.0*(x / float(X-1)) - 1.0

    xy = torch.stack([x,y], dim=-1)
    
    if clamp_extreme:
        xy = torch.clamp(xy, min=-2.0, max=2.0)
    return xy

def meshgrid3d_yxz(B, Y, X, Z):
    # returns a meshgrid sized B x Y x X x Z
    # this ordering makes sense since usually Y=height, X=width, Z=depth

	grid_y = torch.linspace(0.0, Y-1, Y)
	grid_y = torch.reshape(grid_y, [1, Y, 1, 1])
	grid_y = grid_y.repeat(B, 1, X, Z)
	
	grid_x = torch.linspace(0.0, X-1, X)
	grid_x = torch.reshape(grid_x, [1, 1, X, 1])
	grid_x = grid_x.repeat(B, Y, 1, Z)

	grid_z = torch.linspace(0.0, Z-1, Z)
	grid_z = torch.reshape(grid_z, [1, 1, 1, Z])
	grid_z = grid_z.repeat(B, Y, X, 1)
	
	return grid_y, grid_x, grid_z

def meshgrid2d(B, Y, X, stack=False, norm=False, device='cuda'):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y-1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2d(
            grid_y, grid_x, Y, X)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x
    
def meshgrid3d(B, Z, Y, X, stack=False, norm=False, device='cuda'):
    # returns a meshgrid sized B x Z x Y x X
    
    grid_z = torch.linspace(0.0, Z-1, Z, device=device)
    grid_z = torch.reshape(grid_z, [1, Z, 1, 1])
    grid_z = grid_z.repeat(B, 1, Y, X)

    grid_y = torch.linspace(0.0, Y-1, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, 1, Y, 1])
    grid_y = grid_y.repeat(B, Z, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, 1, X])
    grid_x = grid_x.repeat(B, Z, Y, 1)

    # if cuda:
    #     grid_z = grid_z.cuda()
    #     grid_y = grid_y.cuda()
    #     grid_x = grid_x.cuda()
        
    if norm:
        grid_z, grid_y, grid_x = normalize_grid3d(
            grid_z, grid_y, grid_x, Z, Y, X)

    if stack:
        # note we stack in xyz order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_z, grid_y, grid_x

def meshgrid3dr(B, rots, Z, Y, X, stack=False, norm=False, cuda=True):
    N = len(rots)
    # returns a meshgrid sized B x N x Z x Y x X
    
    grid_r = torch.reshape(rots, [1, N, 1, 1, 1])
    grid_r = grid_r.repeat(B, 1, Z, Y, X)

    grid_z = torch.linspace(0.0, Z-1, Z)
    grid_z = torch.reshape(grid_z, [1, 1, Z, 1, 1])
    grid_z = grid_z.repeat(B, N, 1, Y, X)

    grid_y = torch.linspace(0.0, Y-1, Y)
    grid_y = torch.reshape(grid_y, [1, 1, 1, Y, 1])
    grid_y = grid_y.repeat(B, N, Z, 1, X)

    grid_x = torch.linspace(0.0, X-1, X)
    grid_x = torch.reshape(grid_x, [1, 1, 1, 1, X])
    grid_x = grid_x.repeat(B, N, Z, Y, 1)

    if cuda:
        grid_z = grid_z.cuda()
        grid_y = grid_y.cuda()
        grid_x = grid_x.cuda()
        grid_r = grid_r.cuda()
        
    if norm:
        assert(False) # not ready yet
        grid_z, grid_y, grid_x = normalize_grid3d(
            grid_z, grid_y, grid_x, Z, Y, X)

    if stack:
        assert(False) # not ready yet
        
        # note we stack in xyz order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_r, grid_z, grid_y, grid_x


def meshgrid3dr3(B, rots, Z, Y, X, stack=False, norm=False, cuda=True):
    N, _ = rots.shape
    # returns a meshgrid sized B x N x Z x Y x X

    grid_rx = torch.reshape(rots[:, 0], [1, N, 1, 1, 1])
    grid_rx = grid_rx.repeat(B, 1, Z, Y, X)

    grid_ry = torch.reshape(rots[:, 1], [1, N, 1, 1, 1])
    grid_ry = grid_ry.repeat(B, 1, Z, Y, X)

    grid_rz = torch.reshape(rots[:, 2], [1, N, 1, 1, 1])
    grid_rz = grid_rz.repeat(B, 1, Z, Y, X)

    grid_z = torch.linspace(0.0, Z - 1, Z)
    grid_z = torch.reshape(grid_z, [1, 1, Z, 1, 1])
    grid_z = grid_z.repeat(B, N, 1, Y, X)

    grid_y = torch.linspace(0.0, Y - 1, Y)
    grid_y = torch.reshape(grid_y, [1, 1, 1, Y, 1])
    grid_y = grid_y.repeat(B, N, Z, 1, X)

    grid_x = torch.linspace(0.0, X - 1, X)
    grid_x = torch.reshape(grid_x, [1, 1, 1, 1, X])
    grid_x = grid_x.repeat(B, N, Z, Y, 1)

    if cuda:
        grid_z = grid_z.cuda()
        grid_y = grid_y.cuda()
        grid_x = grid_x.cuda()
        grid_rx = grid_rx.cuda()
        grid_ry = grid_ry.cuda()
        grid_rz = grid_rz.cuda()

    if norm:
        assert (False)  # not ready yet
        grid_z, grid_y, grid_x = normalize_grid3d(
            grid_z, grid_y, grid_x, Z, Y, X)

    if stack:
        assert (False)  # not ready yet

        # note we stack in xyz order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_rx, grid_ry, grid_rz, grid_z, grid_y, grid_x

def meshgrid3d_py(Z, Y, X, stack=False, norm=False):
    grid_z = np.linspace(0.0, Z-1, Z)
    grid_z = np.reshape(grid_z, [Z, 1, 1])
    grid_z = np.tile(grid_z, [1, Y, X])

    grid_y = np.linspace(0.0, Y-1, Y)
    grid_y = np.reshape(grid_y, [1, Y, 1])
    grid_y = np.tile(grid_y, [Z, 1, X])

    grid_x = np.linspace(0.0, X-1, X)
    grid_x = np.reshape(grid_x, [1, 1, X])
    grid_x = np.tile(grid_x, [Z, Y, 1])

    if norm:
        grid_z, grid_y, grid_x = normalize_grid3d(
            grid_z, grid_y, grid_x, Z, Y, X)

    if stack:
        # note we stack in xyz order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = np.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_z, grid_y, grid_x

def sub2ind(height, width, y, x):
    return y*width + x

def sql2_on_axis(x, axis, keepdim=True):
    return torch.sum(x**2, axis, keepdim=keepdim)

def l2_on_axis(x, axis, keepdim=True):
    return torch.sqrt(EPS + sql2_on_axis(x, axis, keepdim=keepdim))

def l1_on_axis(x, axis, keepdim=True):
    return torch.sum(torch.abs(x), axis, keepdim=keepdim)

def sub2ind3d(depth, height, width, d, h, w):
    # when gathering/scattering with these inds, the tensor should be Z x Y x X
    return d*height*width + h*width + w

def gradient3d(x, absolute=False, square=False):
    # x should be B x C x D x H x W
    dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]

    zeros = torch.zeros_like(x)
    zero_z = zeros[:, :, 0:1, :, :]
    zero_y = zeros[:, :, :, 0:1, :]
    zero_x = zeros[:, :, :, :, 0:1]
    dz = torch.cat([dz, zero_z], axis=2)
    dy = torch.cat([dy, zero_y], axis=3)
    dx = torch.cat([dx, zero_x], axis=4)
    if absolute:
        dz = torch.abs(dz)
        dy = torch.abs(dy)
        dx = torch.abs(dx)
    if square:
        dz = dz ** 2
        dy = dy ** 2
        dx = dx ** 2
    return dz, dy, dx

def gradient2d(x, absolute=False, square=False):
    # x should be B x C x H x W
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]

    # zeros = tf.zeros_like(x)
    zeros = torch.zeros_like(x)
    zero_h = zeros[:, :, 0:1, :]
    zero_w = zeros[:, :, :, 0:1]
    dh = torch.cat([dh, zero_h], axis=2)
    dw = torch.cat([dw, zero_w], axis=3)
    if absolute:
        dh = torch.abs(dh)
        dw = torch.abs(dw)
    if square:
        dh = dh ** 2
        dw = dw ** 2
    return dh, dw

def matmul2(mat1, mat2):
    return torch.matmul(mat1, mat2)

def matmul3(mat1, mat2, mat3):
    return torch.matmul(mat1, torch.matmul(mat2, mat3))

def matmul4(mat1, mat2, mat3, mat4):
    return torch.matmul(torch.matmul(mat1, torch.matmul(mat2, mat3)), mat4)

def downsample(img, factor):
    down = torch.nn.AvgPool2d(factor)
    img = down(img)
    return img

def downsample3d(vox, factor):
    down = torch.nn.AvgPool3d(factor)
    vox = down(vox)
    return vox

def downsample3dflow(flow, factor):
    down = torch.nn.AvgPool3d(factor)
    flow = down(flow) * 1./factor
    return flow

def l2_normalize(x, dim=1):
    # dim1 is the channel dim
    return F.normalize(x, p=2, dim=dim)

def hard_argmax3d(tensor, stack_xyz=False):
    B, C, Z, Y, X = list(tensor.shape)
    assert(C==1)

    flat_tensor = tensor.reshape(B, -1)
    argmax = torch.argmax(flat_tensor, dim=1)

    # convert the indices into 3d coordinates
    argmax_z = argmax // (Y*X)
    argmax_y = (argmax % (Y*X)) // X
    argmax_x = (argmax % (Y*X)) % X

    argmax_z = argmax_z.reshape(B)
    argmax_y = argmax_y.reshape(B)
    argmax_x = argmax_x.reshape(B)

    if stack_xyz:
        return torch.stack([argmax_x, argmax_y, argmax_z], dim=1)
    else:
        return argmax_z, argmax_y, argmax_x

def hard_argmax3dr(tensor, rots):
    # this func has not yet been checked
    B, N, Z, Y, X = list(tensor.shape)

    # first find the slice with the true argmax
    tensor_ = torch.sum(tensor, dim=[2, 3, 4])
    argmax = torch.argmax(tensor_, dim=1)

    argmax_z = torch.zeros(B).float().cuda()
    argmax_y = torch.zeros(B).float().cuda()
    argmax_x = torch.zeros(B).float().cuda()
    argmax_r = torch.zeros(B).float().cuda()
    for b in list(range(B)):
        tensor_b = tensor[b]
        argmax_b = argmax[b]
        tensor_b = tensor_b[argmax_b]
        argmax_r[b] = rots[argmax_b]

        # now the spatial part
        flat_tensor = tensor_b.reshape(-1)
        argmax_here = torch.argmax(flat_tensor, dim=0)
        # convert the indices into 3d coordinates
        argmax_z[b] = argmax_here // (Y*X)
        argmax_y[b] = (argmax_here % (Y*X)) // X
        argmax_x[b] = (argmax_here % (Y*X)) % X

    return argmax_r, argmax_z, argmax_y, argmax_x

def hard_argmax2d(tensor):
    B, C, Y, X = list(tensor.shape)
    assert(C==1)

    # flatten the Tensor along the height and width axes
    flat_tensor = tensor.reshape(B, -1)
    # argmax of the flat tensor
    argmax = torch.argmax(flat_tensor, dim=1)

    # convert the indices into 2d coordinates
    argmax_y = argmax // X # row
    argmax_x = argmax % X # col

    argmax_y = argmax_y.reshape(B)
    argmax_x = argmax_x.reshape(B)
    return argmax_y, argmax_x

def argmax2d(heat, hard=True):
    B, C, Y, X = list(heat.shape)
    assert(C==1)

    if hard:
        # hard argmax
        loc_y, loc_x = hard_argmax2d(heat)
        loc_y = loc_y.float()
        loc_x = loc_x.float()
    else:
        heat = heat.reshape(B, Y*X)
        prob = torch.nn.functional.softmax(heat, dim=1)

        grid_y, grid_x = meshgrid2d(B, Y, X)

        grid_y = grid_y.reshape(B, -1)
        grid_x = grid_x.reshape(B, -1)
        
        loc_y = torch.sum(grid_y*prob, dim=1)
        loc_x = torch.sum(grid_x*prob, dim=1)
        # these are B
        
    return loc_y, loc_x

def argmax3d(heat, hard=True, stack_xyz=False, cuda=True):
    B, C, Z, Y, X = list(heat.shape)
    assert(C==1)

    if hard:
        # hard argmax
        loc_z, loc_y, loc_x = hard_argmax3d(heat)
        loc_z = loc_z.float()
        loc_y = loc_y.float()
        loc_x = loc_x.float()
    else:
        heat = heat.reshape(B, Z*Y*X)
        prob = torch.nn.functional.softmax(heat, dim=1)

        grid_z, grid_y, grid_x = meshgrid3d(B, Z, Y, X, cuda=cuda)

        grid_z = grid_z.reshape(B, -1)
        grid_y = grid_y.reshape(B, -1)
        grid_x = grid_x.reshape(B, -1)
        
        loc_z = torch.sum(grid_z*prob, dim=1)
        loc_y = torch.sum(grid_y*prob, dim=1)
        loc_x = torch.sum(grid_x*prob, dim=1)
        # these are B
        
    if stack_xyz:
        xyz = torch.stack([loc_x, loc_y, loc_z], dim=-1)
        return xyz
    else:
        return loc_z, loc_y, loc_x

def argmax3dr(heat, rots, hard=True, stack=False, cuda=True, grid=None):
    B, N, Z, Y, X = list(heat.shape)
    # N is the number of rotations

    if hard:
        # hard argmax
        loc_r, loc_z, loc_y, loc_x = hard_argmax3dr(heat)
        loc_z = loc_z.float()
        loc_y = loc_y.float()
        loc_x = loc_x.float()
        loc_r = loc_r.float()
    else:
        one_shot = True
        if one_shot:
            heat = heat.reshape(B, N*Z*Y*X)
            prob = torch.nn.functional.softmax(heat, dim=1)

            if grid is not None:
                grid_r, grid_z, grid_y, grid_x = grid
            else:
                grid_r, grid_z, grid_y, grid_x = meshgrid3dr(B, rots, Z, Y, X, cuda=cuda)
            # these are each B x N x Z x Y x X

            grid_r = grid_r.reshape(B, -1)
            grid_z = grid_z.reshape(B, -1)
            grid_y = grid_y.reshape(B, -1)
            grid_x = grid_x.reshape(B, -1)

            loc_r = torch.sum(grid_r*prob, dim=1)
            loc_z = torch.sum(grid_z*prob, dim=1)
            loc_y = torch.sum(grid_y*prob, dim=1)
            loc_x = torch.sum(grid_x*prob, dim=1)
            # these are B
        else:
            heat_ = torch.sum(heat, dim=[2,3,4])
            # this is B x N
            rot_prob = torch.nn.functional.softmax(heat_, dim=1)
            loc_r = torch.sum(rot_prob * rots.reshape(1, N), dim=1)

            heat = heat.reshape(B, N*Z*Y*X)
            prob = torch.nn.functional.softmax(heat, dim=1)

            grid_r, grid_z, grid_y, grid_x = meshgrid3dr(B, rots, Z, Y, X, cuda=cuda)
            # these are each B x N x Z x Y x X

            # grid_r = grid_r.reshape(B, -1)
            grid_z = grid_z.reshape(B, -1)
            grid_y = grid_y.reshape(B, -1)
            grid_x = grid_x.reshape(B, -1)

            # loc_r = torch.sum(grid_r*prob, dim=1)
            loc_z = torch.sum(grid_z*prob, dim=1)
            loc_y = torch.sum(grid_y*prob, dim=1)
            loc_x = torch.sum(grid_x*prob, dim=1)
            # these are B
            
            
    if stack:
        xyz = torch.stack([loc_x, loc_y, loc_z], dim=-1)
        return loc_r, xyz
    else:
        return loc_r, loc_z, loc_y, loc_x


def argmax3dr3(heat, rots, hard=True, stack=False, cuda=True, grid=None):
    # this function is to compute the argmax of 3-dim rotations
    B, N, Z, Y, X = list(heat.shape)
    # N is the number of rotations

    assert(not hard) # not implemented yet

    if hard:
        # hard argmax
        loc_r, loc_z, loc_y, loc_x = hard_argmax3dr(heat)
        loc_z = loc_z.float()
        loc_y = loc_y.float()
        loc_x = loc_x.float()
        loc_r = loc_r.float()
    else:
        one_shot = True
        if one_shot:
            heat = heat.reshape(B, N * Z * Y * X)
            prob = torch.nn.functional.softmax(heat, dim=1)

            if grid is not None:
                grid_rx, grid_ry, grid_rz, grid_z, grid_y, grid_x = grid
            else:
                grid_rx, grid_ry, grid_rz, grid_z, grid_y, grid_x = meshgrid3dr3(B, rots, Z, Y, X, cuda=cuda)
            # these are each B x N x Z x Y x X

            grid_rx = grid_rx.reshape(B, -1)
            grid_ry = grid_ry.reshape(B, -1)
            grid_rz = grid_rz.reshape(B, -1)
            grid_z = grid_z.reshape(B, -1)
            grid_y = grid_y.reshape(B, -1)
            grid_x = grid_x.reshape(B, -1)

            loc_rx = torch.sum(grid_rx * prob, dim=1)
            loc_ry = torch.sum(grid_ry * prob, dim=1)
            loc_rz = torch.sum(grid_rz * prob, dim=1)
            loc_z = torch.sum(grid_z * prob, dim=1)
            loc_y = torch.sum(grid_y * prob, dim=1)
            loc_x = torch.sum(grid_x * prob, dim=1)
            # these are B
        else:
            heat_ = torch.sum(heat, dim=[2, 3, 4])
            # this is B x N
            rot_prob = torch.nn.functional.softmax(heat_, dim=1)
            loc_r = torch.sum(rot_prob * rots.reshape(1, N), dim=1)

            heat = heat.reshape(B, N * Z * Y * X)
            prob = torch.nn.functional.softmax(heat, dim=1)

            grid_r, grid_z, grid_y, grid_x = meshgrid3dr(B, rots, Z, Y, X, cuda=cuda)
            # these are each B x N x Z x Y x X

            # grid_r = grid_r.reshape(B, -1)
            grid_z = grid_z.reshape(B, -1)
            grid_y = grid_y.reshape(B, -1)
            grid_x = grid_x.reshape(B, -1)

            # loc_r = torch.sum(grid_r*prob, dim=1)
            loc_z = torch.sum(grid_z * prob, dim=1)
            loc_y = torch.sum(grid_y * prob, dim=1)
            loc_x = torch.sum(grid_x * prob, dim=1)
            # these are B

    if stack:
        xyz = torch.stack([loc_x, loc_y, loc_z], dim=-1)
        rxyz = torch.stack([loc_rx, loc_ry, loc_rz], dim=-1)
        return rxyz, xyz
    else:
        return loc_rx, loc_ry, loc_rz, loc_z, loc_y, loc_x

def get_params(model_part):
    return [copy.deepcopy(p) for p in model_part.parameters()]

def check_equal(a, b):
    # first check that the length of the two list are equal
    assert len(a) == len(b), "the list sizes are not equal for sure failing"
    res = [torch.equal(p1, p2) for p1, p2 in zip(a, b)]
    return all(res)

def check_notequal(a, b):
    # here I still check that the lists are equal in length, since same subnet
    # params are being checked for not equality here
    assert len(a) == len(b), "same network params should have same length"
    res = [torch.equal(p1, p2) for p1, p2 in zip(a, b)]
    return not all(res)

def inner_prod(a, b):
    A, N, C = list(a.shape)
    B, M, D = list(b.shape)
    assert(A==B)
    assert(C==D)
    # we want the ans shaped B x N x M
    b = b.permute(0, 2, 1)
    prod = torch.matmul(a, b)
    return prod
    
def get_gaussian_kernel_3d(channels, kernel_size=3, sigma=2.0, mid_one=False):
    C = channels
    xyz_grid = gridcloud3d(C, kernel_size, kernel_size, kernel_size) # C x N x 3

    mean = (kernel_size - 1)/2.0
    variance = sigma**2.0

    gaussian_kernel = (1.0/(2.0*np.pi*variance)**1.5) * torch.exp(-torch.sum((xyz_grid - mean)**2.0, dim=-1) / (2.0*variance)) # C X N
    gaussian_kernel = gaussian_kernel.view(C, 1, kernel_size, kernel_size, kernel_size) # C x 1 x 3 x 3 x 3
    kernel_sum = torch.sum(gaussian_kernel, dim=(2,3,4), keepdim=True)

    gaussian_kernel = gaussian_kernel / kernel_sum # normalize

    if mid_one:
        # normalize so that the middle element is 1
        maxval = gaussian_kernel[:,:,(kernel_size//2),(kernel_size//2),(kernel_size//2)].reshape(C, 1, 1, 1, 1)
        gaussian_kernel = gaussian_kernel / maxval

    return gaussian_kernel

def gaussian_blur_3d(input, kernel_size=3, sigma=2.0, mid_one=False):
    B, C, Z, Y, X = input.shape
    kernel = get_gaussian_kernel_3d(C, kernel_size, sigma, mid_one=mid_one)
    out = F.conv3d(input, kernel, padding=(kernel_size - 1)//2, groups=C)
    return out

def get_gaussian_kernel_2d(channels, kernel_size=3, sigma=2.0, mid_one=False):
    C = channels
    xy_grid = gridcloud2d(C, kernel_size, kernel_size) # C x N x 2

    mean = (kernel_size - 1)/2.0
    variance = sigma**2.0

    gaussian_kernel = (1.0/(2.0*np.pi*variance)**1.5) * torch.exp(-torch.sum((xy_grid - mean)**2.0, dim=-1) / (2.0*variance)) # C X N
    gaussian_kernel = gaussian_kernel.view(C, 1, kernel_size, kernel_size) # C x 1 x 3 x 3
    kernel_sum = torch.sum(gaussian_kernel, dim=(2,3), keepdim=True)

    gaussian_kernel = gaussian_kernel / kernel_sum # normalize

    if mid_one:
        # normalize so that the middle element is 1
        maxval = gaussian_kernel[:,:,(kernel_size//2),(kernel_size//2)].reshape(C, 1, 1, 1)
        gaussian_kernel = gaussian_kernel / maxval

    return gaussian_kernel

def gaussian_blur_2d(input, kernel_size=3, sigma=2.0, reflect_pad=False, mid_one=False):
    B, C, Z, X = input.shape
    kernel = get_gaussian_kernel_2d(C, kernel_size, sigma, mid_one=mid_one)
    if reflect_pad:
        pad = (kernel_size - 1)//2
        out = F.pad(input, (pad, pad, pad, pad), mode='reflect')
        out = F.conv2d(out, kernel, padding=0, groups=C)
    else:
        out = F.conv2d(input, kernel, padding=(kernel_size - 1)//2, groups=C)
    return out

# def cross_blur_3d(input, kernel_size=3, sigma=2.0):
#     B, C, Z, Y, X = input.shape
#     kernel = get_cross_kernel_3d(C, kernel_size, sigma)
#     out = F.conv3d(input, kernel, padding=(kernel_size - 1)//2, groups=C)

#     return out

def dot_prod(mat1, mat2):
    B1, C1 = mat1.shape
    B2, C2 = mat2.shape
    assert(B1==B2)
    assert(C1==C2)
    mat1 = torch.reshape(mat1, (B1, 1, C1))
    mat2 = torch.reshape(mat2, (B2, C2, 1))
    prod = torch.reshape(torch.bmm(mat1, mat2), (B1, )) # this is a size-B tensor

    return prod

import re
def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data    
