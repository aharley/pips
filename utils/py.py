import glob, math
import numpy as np
# from scipy import misc
# from scipy import linalg
from PIL import Image
import io
import matplotlib.pyplot as plt
EPS = 1e-6


XMIN = -64.0 # right (neg is left)
XMAX = 64.0 # right
YMIN = -64.0 # down (neg is up)
YMAX = 64.0 # down
ZMIN = -64.0 # forward
ZMAX = 64.0 # forward

def print_stats(name, tensor):
    tensor = tensor.astype(np.float32)
    print('%s min = %.2f, mean = %.2f, max = %.2f' % (name, np.min(tensor), np.mean(tensor), np.max(tensor)), tensor.shape)
    
def reduce_masked_mean(x, mask, axis=None, keepdims=False):
    # x and mask are the same shape
    # returns shape-1
    # axis can be a list of axes
    prod = x*mask
    numer = np.sum(prod, axis=axis, keepdims=keepdims)
    denom = EPS+np.sum(mask, axis=axis, keepdims=keepdims)
    mean = numer/denom
    return mean

def reduce_masked_sum(x, mask, axis=None, keepdims=False):
    # x and mask are the same shape
    # returns shape-1
    # axis can be a list of axes
    prod = x*mask
    numer = np.sum(prod, axis=axis, keepdims=keepdims)
    return numer

def reduce_masked_median(x, mask, keep_batch=False):
    # x and mask are the same shape
    # returns shape-1
    # axis can be a list of axes

    if not (x.shape == mask.shape):
        print('reduce_masked_median: these shapes should match:', x.shape, mask.shape)
        assert(False)
    # assert(x.shape == mask.shape)

    B = list(x.shape)[0]

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
        return med
    
def get_nFiles(path):
    return len(glob.glob(path))

def get_file_list(path):
    return glob.glob(path)

def rotm2eul(R):
    # R is 3x3
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    if sy > 1e-6: # singular
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return x, y, z
            
def rad2deg(rad):
    return rad*180.0/np.pi

def deg2rad(deg):
    return deg/180.0*np.pi
            
def eul2rotm(rx, ry, rz):
    # copy of matlab, but order of inputs is different
    # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
    #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
    #        -sy            cy*sx             cy*cx]
    sinz = np.sin(rz)
    siny = np.sin(ry)
    sinx = np.sin(rx)
    cosz = np.cos(rz)
    cosy = np.cos(ry)
    cosx = np.cos(rx)
    r11 = cosy*cosz
    r12 = sinx*siny*cosz - cosx*sinz
    r13 = cosx*siny*cosz + sinx*sinz
    r21 = cosy*sinz
    r22 = sinx*siny*sinz + cosx*cosz
    r23 = cosx*siny*sinz - sinx*cosz
    r31 = -siny
    r32 = sinx*cosy
    r33 = cosx*cosy
    r1 = np.stack([r11,r12,r13],axis=-1)
    r2 = np.stack([r21,r22,r23],axis=-1)
    r3 = np.stack([r31,r32,r33],axis=-1)
    r = np.stack([r1,r2,r3],axis=0)
    return r

def wrap2pi(rad_angle):
    # puts the angle into the range [-pi, pi]
    return np.arctan2(np.sin(rad_angle), np.cos(rad_angle))

def rot2view(rx,ry,rz,x,y,z):
    # takes rot angles and 3d position as input
    # returns viewpoint angles as output
    # (all in radians)
    # it will perform strangely if z <= 0
    az = wrap2pi(ry - (-np.arctan2(z, x) - 1.5*np.pi))
    el = -wrap2pi(rx - (-np.arctan2(z, y) - 1.5*np.pi))
    th = -rz
    return az, el, th

def invAxB(a,b):
    """
    Compute the relative 3d transformation between a and b.
    
    Input:
    a -- first pose (homogeneous 4x4 matrix)
    b -- second pose (homogeneous 4x4 matrix)
    
    Output:
    Relative 3d transformation from a to b.
    """
    return np.dot(np.linalg.inv(a),b)
        
def merge_rt(r, t):
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

def split_rt(rt):
    r = rt[:3,:3]
    t = rt[:3,3]
    r = np.reshape(r, [3, 3])
    t = np.reshape(t, [3, 1])
    return r, t

def merge_lrt(l, rt):
    # l is 3
    # rt is 4 x 4
    # merges these into a 19 vector
    D = len(l)
    assert(D==3)
    E, F = list(rt.shape)
    assert(E==4 and F==4)
    rt = rt.reshape(16)
    lrt = np.concatenate([l, rt], axis=0)
    return lrt

def split_intrinsics(K):
    # K is 3 x 4 or 4 x 4
    fx = K[0,0]
    fy = K[1,1]
    x0 = K[0,2]
    y0 = K[1,2]
    return fx, fy, x0, y0
                    
def merge_intrinsics(fx, fy, x0, y0):
    # inputs are shaped []
    K = np.eye(4)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = x0
    K[1,2] = y0
    # K is shaped 4 x 4
    return K
                            
def scale_intrinsics(K, sx, sy):
    fx, fy, x0, y0 = split_intrinsics(K)
    fx *= sx
    fy *= sy
    x0 *= sx
    y0 *= sy
    return merge_intrinsics(fx, fy, x0, y0)

# def meshgrid(H, W):
#     x = np.linspace(0, W-1, W)
#     y = np.linspace(0, H-1, H)
#     xv, yv = np.meshgrid(x, y)
#     return xv, yv

def compute_distance(transform):
    """
    Compute the distance of the translational component of a 4x4 homogeneous matrix.
    """
    return numpy.linalg.norm(transform[0:3,3])

def radian_l1_dist(e, g):
    # if our angles are in [0, 360] we can follow this stack overflow answer:
    # https://gamedev.stackexchange.com/questions/4467/comparing-angles-and-working-out-the-difference
    # wrap2pi brings the angles to [-180, 180]; adding pi puts them in [0, 360]
    e = wrap2pi(e)+np.pi
    g = wrap2pi(g)+np.pi
    l = np.abs(np.pi - np.abs(np.abs(e-g) - np.pi))
    return l

def apply_pix_T_cam(pix_T_cam, xyz):
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    # xyz is shaped B x H*W x 3
    # returns xy, shaped B x H*W x 2
    N, C = xyz.shape
    x, y, z = np.split(xyz, 3, axis=-1)
    EPS = 1e-4
    z = np.clip(z, EPS, None)
    x = (x*fx)/(z)+x0
    y = (y*fy)/(z)+y0
    xy = np.concatenate([x, y], axis=-1)
    return xy

def apply_4x4(RT, XYZ):
    # RT is 4 x 4
    # XYZ is N x 3

    # put into homogeneous coords
    X, Y, Z = np.split(XYZ, 3, axis=1)
    ones = np.ones_like(X)
    XYZ1 = np.concatenate([X, Y, Z, ones], axis=1)
    # XYZ1 is N x 4

    XYZ1_t = np.transpose(XYZ1)
    # this is 4 x N

    XYZ2_t = np.dot(RT, XYZ1_t)
    # this is 4 x N
    
    XYZ2 = np.transpose(XYZ2_t)
    # this is N x 4
    
    XYZ2 = XYZ2[:,:3]
    # this is N x 3
    
    return XYZ2

def Ref2Mem(xyz, Z, Y, X):
    # xyz is N x 3, in ref coordinates
    # transforms ref coordinates into mem coordinates
    N, C = xyz.shape
    assert(C==3)
    mem_T_ref = get_mem_T_ref(Z, Y, X)
    xyz = apply_4x4(mem_T_ref, xyz)
    return xyz

# def Mem2Ref(xyz_mem, MH, MW, MD):
#     # xyz is B x N x 3, in mem coordinates
#     # transforms mem coordinates into ref coordinates
#     B, N, C = xyz_mem.get_shape().as_list()
#     ref_T_mem = get_ref_T_mem(B, MH, MW, MD)
#     xyz_ref = utils_geom.apply_4x4(ref_T_mem, xyz_mem)
#     return xyz_ref

def get_mem_T_ref(Z, Y, X):
    # sometimes we want the mat itself
    # note this is not a rigid transform
    
    # for interpretability, let's construct this in two steps...

    # translation
    center_T_ref = np.eye(4, dtype=np.float32)
    center_T_ref[0,3] = -XMIN
    center_T_ref[1,3] = -YMIN
    center_T_ref[2,3] = -ZMIN

    VOX_SIZE_X = (XMAX-XMIN)/float(X)
    VOX_SIZE_Y = (YMAX-YMIN)/float(Y)
    VOX_SIZE_Z = (ZMAX-ZMIN)/float(Z)
    
    # scaling
    mem_T_center = np.eye(4, dtype=np.float32)
    mem_T_center[0,0] = 1./VOX_SIZE_X
    mem_T_center[1,1] = 1./VOX_SIZE_Y
    mem_T_center[2,2] = 1./VOX_SIZE_Z
    
    mem_T_ref = np.dot(mem_T_center, center_T_ref)
    return mem_T_ref

def matmul2(mat1, mat2):
    return np.matmul(mat1, mat2)

def safe_inverse(a):
    r, t = split_rt(a)
    t = np.reshape(t, [3, 1])
    r_transpose = r.T
    inv = np.concatenate([r_transpose, -np.matmul(r_transpose, t)], 1)
    bottom_row = a[3:4, :] # this is [0, 0, 0, 1]
    inv = np.concatenate([inv, bottom_row], 0)
    return inv

def get_ref_T_mem(Z, Y, X):
    mem_T_ref = get_mem_T_ref(X, Y, X)
    # note safe_inverse is inapplicable here,
    # since the transform is nonrigid
    ref_T_mem = np.linalg.inv(mem_T_ref)
    return ref_T_mem

def voxelize_xyz(xyz_ref, Z, Y, X):
    # xyz_ref is N x 3
    xyz_mem = Ref2Mem(xyz_ref, Z, Y, X)
    # this is N x 3
    voxels = get_occupancy(xyz_mem, Z, Y, X)
    voxels = np.reshape(voxels, [Z, Y, X, 1])
    return voxels

def get_inbounds(xyz, Z, Y, X, already_mem=False):
    # xyz is H*W x 3
    
    if not already_mem:
        xyz = Ref2Mem(xyz, Z, Y, X)
    
    x_valid = np.logical_and(
        np.greater_equal(xyz[:,0], -0.5), 
        np.less(xyz[:,0], float(X)-0.5))
    y_valid = np.logical_and(
        np.greater_equal(xyz[:,1], -0.5), 
        np.less(xyz[:,1], float(Y)-0.5))
    z_valid = np.logical_and(
        np.greater_equal(xyz[:,2], -0.5), 
        np.less(xyz[:,2], float(Z)-0.5))
    inbounds = np.logical_and(np.logical_and(x_valid, y_valid), z_valid)
    return inbounds

def sub2ind3d_zyx(depth, height, width, d, h, w):
    # same as sub2ind3d, but inputs in zyx order
    # when gathering/scattering with these inds, the tensor should be Z x Y x X
    return d*height*width + h*width + w

def sub2ind3d_yxz(height, width, depth, h, w, d):
    return h*width*depth + w*depth + d

# def ind2sub(height, width, ind):
#     # int input
#     y = int(ind / height)
#     x = ind % height
#     return y, x

def get_occupancy(xyz_mem, Z, Y, X):
    # xyz_mem is N x 3
    # we want to fill a voxel tensor with 1's at these inds

    inbounds = get_inbounds(xyz_mem, Z, Y, X, already_mem=True)
    inds = np.where(inbounds)

    xyz_mem = np.reshape(xyz_mem[inds], [-1, 3])
    # xyz_mem is N x 3

    # this is more accurate than a cast/floor, but runs into issues when Y==0
    xyz_mem = np.round(xyz_mem).astype(np.int32)
    x = xyz_mem[:,0]
    y = xyz_mem[:,1]
    z = xyz_mem[:,2]

    voxels = np.zeros([Z, Y, X], np.float32)
    voxels[z, y, x] = 1.0

    return voxels

def Pixels2Camera(x,y,z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, z is a depth image in meters
    # their shapes are H x W
    # fx, fy, x0, y0 are scalar camera intrinsics
    # returns xyz, sized [B,H*W,3]
    
    H, W = z.shape
    
    fx = np.reshape(fx, [1,1])
    fy = np.reshape(fy, [1,1])
    x0 = np.reshape(x0, [1,1])
    y0 = np.reshape(y0, [1,1])
    
    # unproject
    x = ((z+EPS)/fx)*(x-x0)
    y = ((z+EPS)/fy)*(y-y0)
    
    x = np.reshape(x, [-1])
    y = np.reshape(y, [-1])
    z = np.reshape(z, [-1])
    xyz = np.stack([x,y,z], axis=1)
    return xyz

def depth2pointcloud(z, pix_T_cam):
    H = z.shape[0]
    W = z.shape[1]
    y, x = meshgrid2d(H, W)
    z = np.reshape(z, [H, W])
    
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = Pixels2Camera(x, y, z, fx, fy, x0, y0)
    return xyz

def meshgrid2d(Y, X):
    grid_y = np.linspace(0.0, Y-1, Y)
    grid_y = np.reshape(grid_y, [Y, 1])
    grid_y = np.tile(grid_y, [1, X])

    grid_x = np.linspace(0.0, X-1, X)
    grid_x = np.reshape(grid_x, [1, X])
    grid_x = np.tile(grid_x, [Y, 1])

    # outputs are Y x X
    return grid_y, grid_x

def gridcloud3d(Y, X, Z):
    x_ = np.linspace(0, X-1, X)
    y_ = np.linspace(0, Y-1, Y)
    z_ = np.linspace(0, Z-1, Z)
    y, x, z = np.meshgrid(y_, x_, z_, indexing='ij')
    x = np.reshape(x, [-1])
    y = np.reshape(y, [-1])
    z = np.reshape(z, [-1])
    xyz = np.stack([x,y,z], axis=1).astype(np.float32)
    return xyz

def gridcloud2d(Y, X):
    x_ = np.linspace(0, X-1, X)
    y_ = np.linspace(0, Y-1, Y)
    y, x = np.meshgrid(y_, x_, indexing='ij')
    x = np.reshape(x, [-1])
    y = np.reshape(y, [-1])
    xy = np.stack([x,y], axis=1).astype(np.float32)
    return xyz

def normalize(im):
    im = im - np.min(im)
    im = im / np.max(im)
    return im

def wrap2pi(rad_angle):
    # rad_angle can be any shape
    # puts the angle into the range [-pi, pi]
    return np.arctan2(np.sin(rad_angle), np.cos(rad_angle))

def convert_occ_to_height(occ):
    Z, Y, X, C = occ.shape
    assert(C==1)
    
    height = np.linspace(float(Y), 1.0, Y)
    height = np.reshape(height, [1, Y, 1, 1])
    height = np.max(occ*height, axis=1)/float(Y)
    height = np.reshape(height, [Z, X, C])
    return height

def create_depth_image(xy, Z, H, W):

    # turn the xy coordinates into image inds
    xy = np.round(xy)

    # lidar reports a sphere of measurements
    # only use the inds that are within the image bounds
    # also, only use forward-pointing depths (Z > 0)
    valid = (xy[:,0] < W-1) & (xy[:,1] < H-1) & (xy[:,0] >= 0) & (xy[:,1] >= 0) & (Z[:] > 0)

    # gather these up
    xy = xy[valid]
    Z = Z[valid]
    
    inds = sub2ind(H,W,xy[:,1],xy[:,0])
    depth = np.zeros((H*W), np.float32)

    for (index, replacement) in zip(inds, Z):
        depth[index] = replacement
    depth[np.where(depth == 0.0)] = 70.0
    depth = np.reshape(depth, [H, W])

    return depth

def vis_depth(depth, maxdepth=80.0, log_vis=True):
    depth[depth<=0.0] = maxdepth
    if log_vis:
        depth = np.log(depth)
        depth = np.clip(depth, 0, np.log(maxdepth))
    else:
        depth = np.clip(depth, 0, maxdepth)
    depth = (depth*255.0).astype(np.uint8)
    return depth

def preprocess_color(x):
    return x.astype(np.float32) * 1./255 - 0.5

def convert_box_to_ref_T_obj(boxes):
    shape = boxes.shape
    boxes = boxes.reshape(-1,9)
    rots = [eul2rotm(rx,ry,rz)
            for rx,ry,rz in boxes[:,6:]]
    rots = np.stack(rots,axis=0)
    trans = boxes[:,:3]
    ref_T_objs = [merge_rt(rot,tran)
                  for rot,tran in zip(rots,trans)]
    ref_T_objs = np.stack(ref_T_objs,axis=0)
    ref_T_objs = ref_T_objs.reshape(shape[:-1]+(4,4))
    ref_T_objs = ref_T_objs.astype(np.float32)
    return ref_T_objs

def get_rot_from_delta(delta, yaw_only=False):
    dx = delta[:,0]
    dy = delta[:,1]
    dz = delta[:,2]

    bot_hyp = np.sqrt(dz**2 + dx**2)
    # top_hyp = np.sqrt(bot_hyp**2 + dy**2)

    pitch = -np.arctan2(dy, bot_hyp)
    yaw = np.arctan2(dz, dx)

    if yaw_only:
        rot = [eul2rotm(0,y,0) for y in yaw]
    else:
        rot = [eul2rotm(0,y,p) for (p,y) in zip(pitch,yaw)]
        
    rot = np.stack(rot)
    # rot is B x 3 x 3
    return rot

def im2col(im, psize):
    n_channels = 1 if len(im.shape) == 2 else im.shape[0]
    (n_channels, rows, cols) = (1,) * (3 - len(im.shape)) + im.shape

    im_pad = np.zeros((n_channels,
                       int(math.ceil(1.0 * rows / psize) * psize),
                       int(math.ceil(1.0 * cols / psize) * psize)))
    im_pad[:, 0:rows, 0:cols] = im

    final = np.zeros((im_pad.shape[1], im_pad.shape[2], n_channels,
                      psize, psize))
    for c in np.arange(n_channels):
        for x in np.arange(psize):
            for y in np.arange(psize):
                im_shift = np.vstack(
                    (im_pad[c, x:], im_pad[c, :x]))
                im_shift = np.column_stack(
                    (im_shift[:, y:], im_shift[:, :y]))
                final[x::psize, y::psize, c] = np.swapaxes(
                    im_shift.reshape(int(im_pad.shape[1] / psize), psize,
                                     int(im_pad.shape[2] / psize), psize), 1, 2)

    return np.squeeze(final[0:rows - psize + 1, 0:cols - psize + 1])

def filter_discontinuities(depth, filter_size=9, thresh=10):
    H, W = list(depth.shape)

    # Ensure that filter sizes are okay
    assert filter_size % 2 == 1, "Can only use odd filter sizes."

    # Compute discontinuities
    offset = int((filter_size - 1) / 2)
    patches = 1.0 * im2col(depth, filter_size)
    mids = patches[:, :, offset, offset]
    mins = np.min(patches, axis=(2, 3))
    maxes = np.max(patches, axis=(2, 3))

    discont = np.maximum(np.abs(mins - mids),
                         np.abs(maxes - mids))
    mark = discont > thresh

    # Account for offsets
    final_mark = np.zeros((H, W), dtype=np.uint16)
    final_mark[offset:offset + mark.shape[0],
               offset:offset + mark.shape[1]] = mark

    return depth * (1 - final_mark)

def argmax2d(tensor):
    Y, X = list(tensor.shape)
    # flatten the Tensor along the height and width axes
    flat_tensor = tensor.reshape(-1)
    # argmax of the flat tensor
    argmax = np.argmax(flat_tensor)
    
    # convert the indices into 2d coordinates
    argmax_y = argmax // X # row
    argmax_x = argmax % X # col

    return argmax_y, argmax_x

def plot_traj_3d(traj):
    # traj is S x 3
    
    # print('traj', traj.shape)
    S, C = list(traj.shape)
    assert(C==3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = [plt.cm.RdYlBu(i) for i in np.linspace(0,1,S)]
    # print('colors', colors)
    
    xs = traj[:,0]
    ys = -traj[:,1]
    zs = traj[:,2]

    ax.scatter(xs, zs, ys, s=30, c=colors, marker='o', alpha=1.0, edgecolors=(0,0,0))#, color=color_map[n])
        
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    ax.set_xlim(0,1)
    ax.set_ylim(0,1) # this is really Z
    ax.set_zlim(-1,0) # this is really Y

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = np.array(Image.open(buf)) # H x W x 4
    image = image[:,:,:3]

    plt.close()
    return image

def farthest_point_sample(xyz, npoint, deterministic=False):
    N, C = xyz.shape
    inds = np.zeros((npoint), dtype=np.int32)
    distance = np.ones((N), dtype=np.float32) * 1e10
    if deterministic:
        farthest = np.zeros((1), dtype=np.int32)
    else:
        farthest = np.random.randint(N)
    for i in range(npoint):
        inds[i] = farthest
        centroid = xyz[farthest, :].reshape(1, C)
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return inds
