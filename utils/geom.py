import torch
import utils.basic
# import utils.box
# import utils.vox
import numpy as np
import torchvision.ops as ops
from utils.basic import print_, print_stats

def eye_2x2(B, device='cuda'):
    rt = torch.eye(2, device=torch.device(device)).view(1,2,2).repeat([B, 1, 1])
    return rt

def eye_3x3(B, device='cuda'):
    rt = torch.eye(3, device=torch.device(device)).view(1,3,3).repeat([B, 1, 1])
    return rt

def eye_3x3s(B, S, device='cuda'):
    rt = torch.eye(3, device=torch.device(device)).view(1,1,3,3).repeat([B, S, 1, 1])
    return rt

def eye_4x4(B, device='cuda'):
    rt = torch.eye(4, device=torch.device(device)).view(1,4,4).repeat([B, 1, 1])
    return rt

def eye_4x4s(B, S, device='cuda'):
    rt = torch.eye(4, device=torch.device(device)).view(1,1,4,4).repeat([B, S, 1, 1])
    return rt

def merge_rt(r, t):
    # r is B x 3 x 3
    # t is B x 3
    B, C, D = list(r.shape)
    B2, D2 = list(t.shape)
    assert(C==3)
    assert(D==3)
    assert(B==B2)
    assert(D2==3)
    t = t.view(B, 3)
    rt = eye_4x4(B, device=t.device)
    rt[:,:3,:3] = r
    rt[:,:3,3] = t
    return rt

def merge_rt_single(r, t):
    # r is 3 x 3
    # t is 3
    C, D = list(r.shape)
    assert(C==3)
    assert(D==3)
    t = t.view(3)
    rt = eye_4x4(1).squeeze(0)
    rt[:3,:3] = r
    rt[:3,3] = t
    return rt

def merge_rt_py(r, t):
    # r is B x 3 x 3
    # t is B x 3

    if r is None and t is None:
        assert(False) # you have to provide either r or t
        
    if r is None:
        shape = t.shape
        B = int(shape[0])
        r = np.tile(np.eye(3)[np.newaxis,:,:], (B,1,1))
    elif t is None:
        shape = r.shape
        B = int(shape[0])
        
        t = np.zeros((B, 3))
    else:
        shape = r.shape
        B = int(shape[0])
        
    bottom_row = np.tile(np.reshape(np.array([0.,0.,0.,1.], dtype=np.float32),[1,1,4]),
                         [B,1,1])
    rt = np.concatenate([r,np.expand_dims(t,2)], axis=2)
    rt = np.concatenate([rt,bottom_row], axis=1)
    return rt

def get_rays(pix_T_cam, H, W, target_T_cam=None):
    B, C, D = pix_T_cam.shape
    assert(C==4)
    assert(D==4)
    xy = utils.basic.gridcloud2d(B, H, W) # B, H*W, 2
    
    z = torch.ones_like(xy[:,:,0:1])
    xyz = torch.cat([xy, z], dim=2)
    xyz = pixels2camera3(xyz, pix_T_cam)
    if target_T_cam is not None:
        xyz = utils.geom.apply_4x4(target_T_cam, xyz)

    # print_('xy[:,::301,0]', xy[:,::301,0])

    # sanity check (see page3 of "Input-level Inductive Biases for 3D Reconstruction")
    if target_T_cam is not None:
        cam_T_target = safe_inverse(target_T_cam)
    else:
        cam_T_target = eye_4x4(B)
    xy_new = utils.geom.apply_pix_T_cam(pix_T_cam, utils.geom.apply_4x4(cam_T_target,xyz))
    # print_('xy_new[:,::301,0]', xy_new[:,::301,0])
    
    # assert(torch.all(torch.isclose(xy_new, xy, atol=1e-3))) # zeroth el is a bit sensitive, but 1e-3 seems ok
    assert(torch.all(torch.isclose(xy_new, xy, atol=1))) # 1px error seems reasonable
    # i think errors have to do with slight mismatch between rounded H,W and precise pix_T_cam
    
    ray = xyz / torch.norm(xyz, dim=2, keepdim=True)
    return ray
    
    

def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    # xyz2 = xyz2 / xyz2[:,:,3:4]
    xyz2 = xyz2[:,:,:3]
    return xyz2

def apply_4x4_single(RT, xyz):
    N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,0:1])
    xyz1 = torch.cat([xyz, ones], 1)
    xyz1_t = torch.transpose(xyz1, 0, 1)
    # this is 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 0, 1)
    xyz2 = xyz2[:,:3]
    return xyz2

def apply_4x4_py(RT, XYZ):
    # RT is B x 4 x 4
    # XYZ is B x N x 3

    # put into homogeneous coords
    X, Y, Z = np.split(XYZ, 3, axis=2)
    ones = np.ones_like(X)
    XYZ1 = np.concatenate([X, Y, Z, ones], axis=2)
    # XYZ1 is B x N x 4

    XYZ1_t = np.transpose(XYZ1, (0,2,1))
    # this is B x 4 x N

    XYZ2_t = np.matmul(RT, XYZ1_t)
    # this is B x 4 x N
    
    XYZ2 = np.transpose(XYZ2_t, (0,2,1))
    # this is B x N x 4
    
    XYZ2 = XYZ2[:,:,:3]
    # this is B x N x 3
    
    return XYZ2

def split_rt_single(rt):
    r = rt[:3, :3]
    t = rt[:3, 3].view(3)
    return r, t

def split_rt(rt):
    r = rt[:, :3, :3]
    t = rt[:, :3, 3].view(-1, 3)
    return r, t

def safe_inverse_single(a):
    r, t = split_rt_single(a)
    t = t.view(3,1)
    r_transpose = r.t()
    inv = torch.cat([r_transpose, -torch.matmul(r_transpose, t)], 1)
    bottom_row = a[3:4, :] # this is [0, 0, 0, 1]
    # bottom_row = torch.tensor([0.,0.,0.,1.]).view(1,4) 
    inv = torch.cat([inv, bottom_row], 0)
    return inv

# def safe_inverse(a):
#     B, _, _ = list(a.shape)
#     inv = torch.zeros(B, 4, 4).cuda()
#     for b in list(range(B)):
#         inv[b] = safe_inverse_single(a[b])
#     return inv

def safe_inverse(a): #parallel version
    B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1,2) #inverse of rotation matrix

    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])

    return inv

def get_camM_T_camXs(origin_T_camXs, ind=0):
    B, S = list(origin_T_camXs.shape)[0:2]
    camM_T_camXs = torch.zeros_like(origin_T_camXs)
    for b in list(range(B)):
        camM_T_origin = safe_inverse_single(origin_T_camXs[b,ind])
        for s in list(range(S)):
            camM_T_camXs[b,s] = torch.matmul(camM_T_origin, origin_T_camXs[b,s])
    return camM_T_camXs

def get_cami_T_camXs(origin_T_cami, origin_T_camXs):
    B, S = list(origin_T_camXs.shape)[0:2]
    cami_T_camXs = torch.zeros_like(origin_T_camXs)
    cami_T_origin = safe_inverse(origin_T_cami)
    for b in list(range(B)):
        for s in list(range(S)):
            cami_T_camXs[b,s] = torch.matmul(cami_T_origin[b], origin_T_camXs[b,s])
    return cami_T_camXs

def scale_intrinsics(K, sx, sy):
    fx, fy, x0, y0 = split_intrinsics(K)
    fx = fx*sx
    fy = fy*sy
    x0 = x0*sx
    y0 = y0*sy
    K = pack_intrinsics(fx, fy, x0, y0)
    return K

def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def pack_intrinsics(fx, fy, x0, y0):
    B = list(fx.shape)[0]
    K = torch.zeros(B, 4, 4, dtype=torch.float32, device=torch.device('cuda'))
    K[:,0,0] = fx
    K[:,1,1] = fy
    K[:,0,2] = x0
    K[:,1,2] = y0
    K[:,2,2] = 1.0
    K[:,3,3] = 1.0
    return K

def depth2pointcloud(z, pix_T_cam):
    B, C, H, W = list(z.shape)
    device = z.device
    y, x = utils.basic.meshgrid2d(B, H, W, device=device)
    z = torch.reshape(z, [B, H, W])
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = pixels2camera(x, y, z, fx, fy, x0, y0)
    return xyz

def xyd2pointcloud(xyd, pix_T_cam):
    # xyd is like a pointcloud but in pixel coordinates;
    # this means xy comes from a meshgrid with bounds H, W, 
    # and d comes from a depth map
    B, N, C = list(xyd.shape)
    assert(C==3)
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = pixels2camera(xyd[:,:,0], xyd[:,:,1], xyd[:,:,2], fx, fy, x0, y0)
    return xyz

def pixels2camera3(xyz,pix_T_cam):
    x,y,z = xyz[:,:,0],xyz[:,:,1],xyz[:,:,2]
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    return pixels2camera(x,y,z,fx,fy,x0,y0)

def pixels2camera2(x,y,z,pix_T_cam):
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = pixels2camera(x,y,z,fx,fy,x0,y0)
    return xyz

def pixels2camera(x,y,z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, z is a depth in meters
    # they can be images or pointclouds
    # fx, fy, x0, y0 are camera intrinsics
    # returns xyz, sized B x N x 3

    B = x.shape[0]
    
    fx = torch.reshape(fx, [B,1])
    fy = torch.reshape(fy, [B,1])
    x0 = torch.reshape(x0, [B,1])
    y0 = torch.reshape(y0, [B,1])

    x = torch.reshape(x, [B,-1])
    y = torch.reshape(y, [B,-1])
    z = torch.reshape(z, [B,-1])
    
    # unproject
    x = (z/fx)*(x-x0)
    y = (z/fy)*(y-y0)
    
    xyz = torch.stack([x,y,z], dim=2)
    # B x N x 3
    return xyz

def camera2pixels(xyz, pix_T_cam):
    # xyz is shaped B x H*W x 3
    # returns xy, shaped B x H*W x 2
    
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    x, y, z = torch.unbind(xyz, dim=-1)
    B = list(z.shape)[0]

    fx = torch.reshape(fx, [B,1])
    fy = torch.reshape(fy, [B,1])
    x0 = torch.reshape(x0, [B,1])
    y0 = torch.reshape(y0, [B,1])
    x = torch.reshape(x, [B,-1])
    y = torch.reshape(y, [B,-1])
    z = torch.reshape(z, [B,-1])

    EPS = 1e-4
    z = torch.clamp(z, min=EPS)
    x = (x*fx)/z + x0
    y = (y*fy)/z + y0
    xy = torch.stack([x, y], dim=-1)
    return xy

def eul2rotm(rx, ry, rz):
    # inputs are shaped B
    # this func is copied from matlab
    # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
    #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
    #        -sy            cy*sx             cy*cx]
    rx = torch.unsqueeze(rx, dim=1)
    ry = torch.unsqueeze(ry, dim=1)
    rz = torch.unsqueeze(rz, dim=1)
    # these are B x 1
    sinz = torch.sin(rz)
    siny = torch.sin(ry)
    sinx = torch.sin(rx)
    cosz = torch.cos(rz)
    cosy = torch.cos(ry)
    cosx = torch.cos(rx)
    r11 = cosy*cosz
    r12 = sinx*siny*cosz - cosx*sinz
    r13 = cosx*siny*cosz + sinx*sinz
    r21 = cosy*sinz
    r22 = sinx*siny*sinz + cosx*cosz
    r23 = cosx*siny*sinz - sinx*cosz
    r31 = -siny
    r32 = sinx*cosy
    r33 = cosx*cosy
    r1 = torch.stack([r11,r12,r13],dim=2)
    r2 = torch.stack([r21,r22,r23],dim=2)
    r3 = torch.stack([r31,r32,r33],dim=2)
    r = torch.cat([r1,r2,r3],dim=1)
    return r

def eul2rotm_py(rx, ry, rz):
    # inputs are shaped B
    # this func is copied from matlab
    # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
    #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
    #        -sy            cy*sx             cy*cx]
    rx = rx[:,np.newaxis]
    ry = ry[:,np.newaxis]
    rz = rz[:,np.newaxis]
    # these are B x 1
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
    r1 = np.stack([r11,r12,r13],axis=2)
    r2 = np.stack([r21,r22,r23],axis=2)
    r3 = np.stack([r31,r32,r33],axis=2)
    r = np.concatenate([r1,r2,r3],axis=1)
    return r

def rotm2eul(r):
    # r is Bx3x3, or Bx4x4
    r00 = r[:,0,0]
    r10 = r[:,1,0]
    r11 = r[:,1,1]
    r12 = r[:,1,2]
    r20 = r[:,2,0]
    r21 = r[:,2,1]
    r22 = r[:,2,2]
    
    ## python guide:
    # if sy > 1e-6: # singular
    #     x = math.atan2(R[2,1] , R[2,2])
    #     y = math.atan2(-R[2,0], sy)
    #     z = math.atan2(R[1,0], R[0,0])
    # else:
    #     x = math.atan2(-R[1,2], R[1,1])
    #     y = math.atan2(-R[2,0], sy)
    #     z = 0
    
    sy = torch.sqrt(r00*r00 + r10*r10)
    
    cond = (sy > 1e-6)
    rx = torch.where(cond, torch.atan2(r21, r22), torch.atan2(-r12, r11))
    ry = torch.where(cond, torch.atan2(-r20, sy), torch.atan2(-r20, sy))
    rz = torch.where(cond, torch.atan2(r10, r00), torch.zeros_like(r20))

    # rx = torch.atan2(r21, r22)
    # ry = torch.atan2(-r20, sy)
    # rz = torch.atan2(r10, r00)
    # rx[cond] = torch.atan2(-r12, r11)
    # ry[cond] = torch.atan2(-r20, sy)
    # rz[cond] = 0.0
    return rx, ry, rz

def get_random_rt(B,
                  rx_amount=5.0,
                  ry_amount=5.0,
                  rz_amount=5.0,
                  t_amount=1.0,
                  sometimes_zero=False,
                  return_pieces=False,
                  y_zero=False):
    # t_amount is in meters
    # r_amount is in degrees
    
    rx_amount = np.pi/180.0*rx_amount
    ry_amount = np.pi/180.0*ry_amount
    rz_amount = np.pi/180.0*rz_amount

    ## translation
    tx = np.random.uniform(-t_amount, t_amount, size=B).astype(np.float32)
    ty = np.random.uniform(-t_amount/2.0, t_amount/2.0, size=B).astype(np.float32)
    tz = np.random.uniform(-t_amount, t_amount, size=B).astype(np.float32)

    if y_zero:
        ty = ty * 0
    
    ## rotation
    rx = np.random.uniform(-rx_amount, rx_amount, size=B).astype(np.float32)
    ry = np.random.uniform(-ry_amount, ry_amount, size=B).astype(np.float32)
    rz = np.random.uniform(-rz_amount, rz_amount, size=B).astype(np.float32)

    if sometimes_zero:
        rand = np.random.uniform(0.0, 1.0, size=B).astype(np.float32)
        prob_of_zero = 0.5
        rx = np.where(np.greater(rand, prob_of_zero), rx, np.zeros_like(rx))
        ry = np.where(np.greater(rand, prob_of_zero), ry, np.zeros_like(ry))
        rz = np.where(np.greater(rand, prob_of_zero), rz, np.zeros_like(rz))
        tx = np.where(np.greater(rand, prob_of_zero), tx, np.zeros_like(tx))
        ty = np.where(np.greater(rand, prob_of_zero), ty, np.zeros_like(ty))
        tz = np.where(np.greater(rand, prob_of_zero), tz, np.zeros_like(tz))
        
    t = np.stack([tx, ty, tz], axis=1)
    t = torch.from_numpy(t)
    rx = torch.from_numpy(rx)
    ry = torch.from_numpy(ry)
    rz = torch.from_numpy(rz)
    r = eul2rotm(rx, ry, rz)
    rt = merge_rt(r, t).cuda()

    if return_pieces:
        return t.cuda(), rx.cuda(), ry.cuda(), rz.cuda()
    else:
        return rt

def get_random_scale(B, low=0.5, high=1.5):
    # return a scale matrix
    scale = torch.rand(B, 1, 1, device=torch.device('cuda')) * (high  - low) + low
    scale_matrix = scale * eye_4x4(B)
    scale_matrix[:, 3, 3] = 1.0 # fix the last element

    return scale_matrix

def convert_boxlist_to_lrtlist(boxlist):
    B, N, D = list(boxlist.shape)
    assert(D==9)
    boxlist_ = boxlist.view(B*N, D)
    rtlist_ = convert_box_to_ref_T_obj(boxlist_)
    rtlist = rtlist_.view(B, N, 4, 4)
    lenlist = boxlist[:,:,3:6].reshape(B, N, 3)
    lenlist = lenlist.clamp(min=0.01)
    lrtlist = merge_lrtlist(lenlist, rtlist)
    return lrtlist
    
def convert_box_to_ref_T_obj(box3d):
    # turn the box into obj_T_ref (i.e., obj_T_cam)
    B = list(box3d.shape)[0]

    # box3d is B x 9
    x, y, z, lx, ly, lz, rx, ry, rz = torch.unbind(box3d, axis=1)
    rot0 = eye_3x3(B, device=box3d.device)
    tra = torch.stack([x, y, z], axis=1)
    center_T_ref = merge_rt(rot0, -tra)
    # center_T_ref is B x 4 x 4
    
    t0 = torch.zeros([B, 3], device=box3d.device)
    rot = eul2rotm(rx, ry, rz)
    rot = torch.transpose(rot, 1, 2) # other dir
    obj_T_center = merge_rt(rot, t0)
    # this is B x 4 x 4

    # we want obj_T_ref
    # first we to translate to center,
    # and then rotate around the origin
    obj_T_ref = utils.basic.matmul2(obj_T_center, center_T_ref)

    # return the inverse of this, so that we can transform obj corners into cam coords
    ref_T_obj = obj_T_ref.inverse()
    return ref_T_obj

def get_offsetlist_from_lenlist(lenlist):
    B, N, D = list(lenlist.shape)
    assert(D==3)
    lx, ly, lz = torch.unbind(lenlist, axis=2)
    # B x N

    xs = []
    ys = []
    zs = []
    for a in [-1, 0, 1]:
        for b in [-1, 0, 1]:
            for c in [-1, 0, 1]:
                xs.append(a*lx/2.0)
                ys.append(b*ly/2.0)
                zs.append(c*lz/2.0)
    xs = torch.stack(xs, dim=2)
    ys = torch.stack(ys, dim=2)
    zs = torch.stack(zs, dim=2)
    # B x N x 27
    xyzlist = torch.stack([xs, ys, zs], axis=3)
    # this is B x N x 27 x 3
    return xyzlist

def get_xyzlist_from_lenlist(lenlist):
    B, N, D = list(lenlist.shape)
    assert(D==3)
    lx, ly, lz = torch.unbind(lenlist, axis=2)

    # frustum/train/provider.py
    # x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    # y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    # z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    
    # xs = torch.stack([-lx/2., -lx/2., -lx/2., -lx/2., lx/2., lx/2., lx/2., lx/2.], axis=2)
    # ys = torch.stack([-ly/2., -ly/2., ly/2., ly/2., -ly/2., -ly/2., ly/2., ly/2.], axis=2)
    # zs = torch.stack([-lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2.], axis=2)


    xs = torch.stack([lx/2., lx/2., -lx/2., -lx/2., lx/2., lx/2., -lx/2., -lx/2.], axis=2)
    ys = torch.stack([ly/2., ly/2., ly/2., ly/2., -ly/2., -ly/2., -ly/2., -ly/2.], axis=2)
    zs = torch.stack([lz/2., -lz/2., -lz/2., lz/2., lz/2., -lz/2., -lz/2., lz/2.], axis=2)
    
    # these are B x N x 8
    xyzlist = torch.stack([xs, ys, zs], axis=3)
    # this is B x N x 8 x 3
    return xyzlist


def get_xyzlist_from_lrtlist(lrtlist, include_clist=False):
    B, N, D = list(lrtlist.shape)
    assert(D==19)

    lenlist, rtlist = split_lrtlist(lrtlist)
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4

    xyzlist_obj = get_xyzlist_from_lenlist(lenlist)
    # xyzlist_obj is B x N x 8 x 3

    rtlist_ = rtlist.reshape(B*N, 4, 4)
    xyzlist_obj_ = xyzlist_obj.reshape(B*N, 8, 3)
    xyzlist_cam_ = apply_4x4(rtlist_, xyzlist_obj_)
    xyzlist_cam = xyzlist_cam_.reshape(B, N, 8, 3)
    
    if include_clist:
        clist_cam = get_clist_from_lrtlist(lrtlist).unsqueeze(2)
        xyzlist_cam = torch.cat([xyzlist_cam, clist_cam], dim=2)
    return xyzlist_cam

def get_clist_from_lrtlist(lrtlist):
    B, N, D = list(lrtlist.shape)
    assert(D==19)

    lenlist, rtlist = split_lrtlist(lrtlist)
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4

    xyzlist_obj = torch.zeros((B, N, 1, 3), device=lrtlist.device)
    # xyzlist_obj is B x N x 8 x 3

    rtlist_ = rtlist.reshape(B*N, 4, 4)
    xyzlist_obj_ = xyzlist_obj.reshape(B*N, 1, 3)
    xyzlist_cam_ = apply_4x4(rtlist_, xyzlist_obj_)
    xyzlist_cam = xyzlist_cam_.reshape(B, N, 3)
    return xyzlist_cam

def split_rtlist(rtlist):
    B, N, D, E = list(rtlist.shape)
    assert(D==4)
    assert(E==4)

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)
    rlist_, tlist_ = split_rt(__p(rtlist))
    rlist, tlist = __u(rlist_), __u(tlist_)
    return rlist, tlist

def merge_rtlist(rlist, tlist):
    B, N, D, E = list(rlist.shape)
    assert(D==3)
    assert(E==3)
    B, N, F = list(tlist.shape)
    assert(F==3)

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)
    rlist_, tlist_ = __p(rlist), __p(tlist)
    rtlist_ = merge_rt(rlist_, tlist_)
    rtlist = __u(rtlist_)
    return rtlist

def get_rlist_from_lrtlist(lrtlist):
    B, N, D = list(lrtlist.shape)
    assert(D==19)

    lenlist, rtlist = split_lrtlist(lrtlist)
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)
    rlist_, tlist_ = split_rt(__p(rtlist))
    # rlist, tlist = __u(rlist_), __u(tlist_)

    rx_, ry_, rz_ = rotm2eul(rlist_)
    rx, ry, rz = __u(rx_), __u(ry_), __u(rz_)

    # ok now, the funny thing is that these rotations may be wrt the camera origin, not wrt the object
    # so, an object parallel to our car but to the right of us will have a different pose than an object parallel in front

    # maybe that's entirely false

    rlist = torch.stack([rx, ry, rz], dim=2)

    return rlist

def get_carriedlist_from_lrtlist(lrtlist_1_camR, lrtlist_2_camR, scorelist=None, thresh=0.25, ignore_y=False):
    # if ignore_y, we ignore the distance in y direction when comparing
    # lrtlist_1_camR is the potential object BEING carried
    # lrtlist_2_camR is the potential CARRIER
    # set the thresh to be half smallest obj size (0.5/2)
    # lrtlist_camR is B x N x 19
    # assume R is the coord where we want to check inbound-ness
    # scorelist is B x N
    B, N1, D = list(lrtlist_1_camR.shape)
    B2, N2, D2 = list(lrtlist_2_camR.shape)
    assert(D==19)
    assert(B==B2)
    assert(D==D2)
    
    # validlist = scorelist
    # # this is B x N1
    
    xyzlist_1 = get_clist_from_lrtlist(lrtlist_1_camR) # B x N1 x 3
    lenlist_1, _ = split_lrtlist(lrtlist_1_camR) # B x N1 x 3
    center_bottom_face_1 = xyzlist_1
    center_bottom_face_1[:, :, 1] = center_bottom_face_1[:, :, 1] + lenlist_1[:, :, 1] / 2.0 # the y should be near 0 for objects on table
    
    xyzlist_2 = get_clist_from_lrtlist(lrtlist_2_camR) # B x N2 x 3
    lenlist_2, _ = split_lrtlist(lrtlist_2_camR) # B x N2 x 3
    center_bottom_face_2 = xyzlist_2
    center_bottom_face_2[:, :, 1] = center_bottom_face_2[:, :, 1] + lenlist_2[:, :, 1] / 2.0 # the y should be near 0 for objects on table
    
    # first create a table comparing the distance between each pair of objects
    center_bottom_face_row = center_bottom_face_1[:, :, None, :].repeat(1, 1, N2, 1)
    center_bottom_face_col = center_bottom_face_2[:, None, :, :].repeat(1, N1, 1, 1)
    if ignore_y:
        dist_table = torch.all(torch.abs(
            center_bottom_face_col[:, :, :, [0,2]] - center_bottom_face_row[:, :, :, [0,2]]
        ) < thresh, dim=3).float() # B x N x N, true if a object is close enough to another
    else:
        dist_table = torch.all(torch.abs(
            center_bottom_face_col - center_bottom_face_row
        ) < thresh, dim=3).float() # B x N x N, true if a object is close enough to another
        # B x N1 x N2
        
    # then create a table to compare size
    len_list_row = lenlist_1[:, :, None, :].repeat(1, 1, N2, 1)
    len_list_col = lenlist_2[:, None, :, :].repeat(1, N1, 1, 1)
    size_table = torch.all(len_list_row < len_list_col, dim=3).float() # the diagnal terms are 0s, B x N x N
    # B x N1 x N2
    
    final_table = dist_table * size_table
    carried_list = torch.max(final_table, dim=2)[0] # B x N1, this is the object BEING CARRIED (smaller one)
    
    if scorelist is not None:
        carried_list = carried_list * scorelist
        
    return carried_list#, final_table # B x N1 x N2

def transform_boxes_to_corners_single(boxes, legacy_format=False):
    N, D = list(boxes.shape)
    assert(D==9)
    
    xc,yc,zc,lx,ly,lz,rx,ry,rz = torch.unbind(boxes, axis=1)
    # these are each shaped N

    ref_T_obj = convert_box_to_ref_T_obj(boxes)
    
    if legacy_format:
        xs = torch.stack([-lx/2., -lx/2., -lx/2., -lx/2., lx/2., lx/2., lx/2., lx/2.], axis=1)
        ys = torch.stack([-ly/2., -ly/2., ly/2., ly/2., -ly/2., -ly/2., ly/2., ly/2.], axis=1)
        zs = torch.stack([-lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2.], axis=1)
    else:
        xs = torch.stack([lx/2., lx/2., -lx/2., -lx/2., lx/2., lx/2., -lx/2., -lx/2.], axis=1)
        ys = torch.stack([ly/2., ly/2., ly/2., ly/2., -ly/2., -ly/2., -ly/2., -ly/2.], axis=1)
        zs = torch.stack([lz/2., -lz/2., -lz/2., lz/2., lz/2., -lz/2., -lz/2., lz/2.], axis=1)
    
    xyz_obj = torch.stack([xs, ys, zs], axis=2)
    # centered_box is N x 8 x 3

    xyz_ref = apply_4x4(ref_T_obj, xyz_obj)
    # xyz_ref is N x 8 x 3
    return xyz_ref

def transform_boxes_to_corners(boxes, legacy_format=False):
    # returns corners, shaped B x N x 8 x 3
    B, N, D = list(boxes.shape)
    assert(D==9)
    
    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    boxes_ = __p(boxes)
    corners_ = transform_boxes_to_corners_single(boxes_, legacy_format=legacy_format)
    corners = __u(corners_)
    return corners

def transform_boxes3d_to_corners_py(boxes3d):
    N, D = list(boxes3d.shape)
    assert(D==9)
    
    xc,yc,zc,lx,ly,lz,rx,ry,rz = boxes3d[:,0], boxes3d[:,1], boxes3d[:,2], boxes3d[:,3], boxes3d[:,4], boxes3d[:,5], boxes3d[:,6], boxes3d[:,7], boxes3d[:,8]

    # these are each shaped N

    rotation_mat = eul2rotm_py(rx, ry, rz)
    translation = np.stack([xc, yc, zc], axis=1) 
    ref_T_obj = merge_rt_py(rotation_mat, translation)

    xs = np.stack([lx/2., lx/2., -lx/2., -lx/2., lx/2., lx/2., -lx/2., -lx/2.], axis=1)
    ys = np.stack([ly/2., ly/2., ly/2., ly/2., -ly/2., -ly/2., -ly/2., -ly/2.], axis=1)
    zs = np.stack([lz/2., -lz/2., -lz/2., lz/2., lz/2., -lz/2., -lz/2., lz/2.], axis=1)

    # xs = tf.stack([-lx/2., -lx/2., lx/2., lx/2., -lx/2., -lx/2., lx/2., lx/2.], axis=1)
    # ys = tf.stack([ly/2., -ly/2., ly/2., -ly/2., ly/2., -ly/2., ly/2., -ly/2.], axis=1)
    # zs = tf.stack([-lz/2., -lz/2., -lz/2., -lz/2., lz/2., lz/2., lz/2., lz/2.], axis=1)

    xyz_obj = np.stack([xs, ys, zs], axis=2)
    # centered_box is N x 8 x 3

    xyz_ref = apply_4x4_py(ref_T_obj, xyz_obj)
    # xyz_ref is N x 8 x 3
    return xyz_ref

def apply_pix_T_cam(pix_T_cam, xyz):

    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    
    # xyz is shaped B x H*W x 3
    # returns xy, shaped B x H*W x 2
    
    B, N, C = list(xyz.shape)
    assert(C==3)
    
    x, y, z = torch.unbind(xyz, axis=-1)

    fx = torch.reshape(fx, [B, 1])
    fy = torch.reshape(fy, [B, 1])
    x0 = torch.reshape(x0, [B, 1])
    y0 = torch.reshape(y0, [B, 1])

    EPS = 1e-4
    z = torch.clamp(z, min=EPS)
    x = (x*fx)/(z)+x0
    y = (y*fy)/(z)+y0
    xy = torch.stack([x, y], axis=-1)
    return xy

# def apply_4x4_to_boxes(Y_T_X, boxes_X):
#     B, N, C = boxes_X.get_shape().as_list()
#     assert(C==9)
#     corners_X = transform_boxes_to_corners(boxes_X) # corners is B x N x 8 x 3
#     corners_X_ = tf.reshape(corners_X, [B, N*8, 3])
#     corners_Y_ = apply_4x4(Y_T_X, corners_X_)
#     corners_Y = tf.reshape(corners_Y_, [B, N, 8, 3])
#     boxes_Y = corners_to_boxes(corners_Y)
#     return boxes_Y

def apply_4x4_to_corners(Y_T_X, corners_X):
    B, N, C, D = list(corners_X.shape)
    assert(C==8)
    assert(D==3)
    corners_X_ = torch.reshape(corners_X, [B, N*8, 3])
    corners_Y_ = apply_4x4(Y_T_X, corners_X_)
    corners_Y = torch.reshape(corners_Y_, [B, N, 8, 3])
    return corners_Y

def split_lrt(lrt):
    # splits a B x 19 tensor
    # into B x 3 (lens)
    # and B x 4 x 4 (rts)
    B, D = list(lrt.shape)
    assert(D==19)
    lrt = lrt.unsqueeze(1)
    l, rt = split_lrtlist(lrt)
    l = l.squeeze(1)
    rt = rt.squeeze(1)
    return l, rt

def split_lrtlist(lrtlist):
    # splits a B x N x 19 tensor
    # into B x N x 3 (lens)
    # and B x N x 4 x 4 (rts)
    B, N, D = list(lrtlist.shape)
    assert(D==19)
    lenlist = lrtlist[:,:,:3].reshape(B, N, 3)
    ref_T_objs_list = lrtlist[:,:,3:].reshape(B, N, 4, 4)
    return lenlist, ref_T_objs_list

def merge_lrtlist(lenlist, rtlist):
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4
    # merges these into a B x N x 19 tensor
    B, N, D = list(lenlist.shape)
    assert(D==3)
    B2, N2, E, F = list(rtlist.shape)
    assert(B==B2)
    assert(N==N2)
    assert(E==4 and F==4)
    rtlist = rtlist.reshape(B, N, 16)
    lrtlist = torch.cat([lenlist, rtlist], axis=2)
    return lrtlist

def merge_lrt(l, rt):
    # l is B x 3
    # rt is B x 4 x 4
    # merges these into a B x 19 tensor
    B, D = list(l.shape)
    assert(D==3)
    B2, E, F = list(rt.shape)
    assert(B==B2)
    assert(E==4 and F==4)
    rt = rt.reshape(B, 16)
    lrt = torch.cat([l, rt], axis=1)
    return lrt

def apply_4x4_to_lrtlist(Y_T_X, lrtlist_X):
    B, N, D = list(lrtlist_X.shape)
    assert(D==19)
    B2, E, F = list(Y_T_X.shape)
    assert(B2==B)
    assert(E==4 and F==4)
    
    lenlist, rtlist_X = split_lrtlist(lrtlist_X)
    # rtlist_X is B x N x 4 x 4

    Y_T_Xs = Y_T_X.unsqueeze(1).repeat(1, N, 1, 1)
    Y_T_Xs_ = Y_T_Xs.view(B*N, 4, 4)
    rtlist_X_ = rtlist_X.reshape(B*N, 4, 4)
    rtlist_Y_ = utils.basic.matmul2(Y_T_Xs_, rtlist_X_)
    rtlist_Y = rtlist_Y_.reshape(B, N, 4, 4)
    lrtlist_Y = merge_lrtlist(lenlist, rtlist_Y)
    return lrtlist_Y

def apply_scaling_to_lrt(Y_T_X, lrt_X):
    return apply_scaling_to_lrtlist(Y_T_X, lrt_X.unsqueeze(1)).squeeze(1)

def apply_scaling_to_lrtlist(Y_T_X, lrtlist_X): 
    B, N, D = list(lrtlist_X.shape)
    assert(D==19)
    B2, E, F = list(Y_T_X.shape)
    assert(B2==B)
    assert(E==4 and F==4)

    # Y_T_X is a scaling matrix, i.e. all off-diagnol terms are 0
    lenlist_X, rtlist_X = split_lrtlist(lrtlist_X)
    # rtlist_X is B x N x 4 x 4

    # lenlist is B x N x 3
    rtlist_X_ = rtlist_X.reshape(B*N, 4, 4)
    rlist_X_, tlist_X_ = split_rt(rtlist_X_) # B*N x 3 x 3 and B*N x 3

    lenlist_Y_ = apply_4x4(Y_T_X, lenlist_X).reshape(B*N, 3)
    tlist_Y_ = apply_4x4(Y_T_X, tlist_X_.reshape(B, N, 3)).reshape(B*N, 3)
    rlist_Y_ = rlist_X_ 

    rtlist_Y = merge_rt(rlist_Y_, tlist_Y_).reshape(B, N, 4, 4)
    lenlist_Y = lenlist_Y_.reshape(B, N, 3)
    lrtlist_Y = merge_lrtlist(lenlist_Y, rtlist_Y)

    return lrtlist_Y

def apply_4x4_to_lrt(Y_T_X, lrt_X):
    B, D = list(lrt_X.shape)
    assert(D==19)
    B2, E, F = list(Y_T_X.shape)
    assert(B2==B)
    assert(E==4 and F==4)

    return apply_4x4_to_lrtlist(Y_T_X, lrt_X.unsqueeze(1)).squeeze(1)

def apply_4x4s_to_lrts(Ys_T_Xs, lrt_Xs):
    B, S, D = list(lrt_Xs.shape)
    assert(D==19)
    B2, S2, E, F = list(Ys_T_Xs.shape)
    assert(B2==B)
    assert(S2==S)
    assert(E==4 and F==4)
    
    lenlist, rtlist_X = split_lrtlist(lrt_Xs)
    # rtlist_X is B x N x 4 x 4

    Ys_T_Xs_ = Ys_T_Xs.view(B*S, 4, 4)
    rtlist_X_ = rtlist_X.view(B*S, 4, 4)
    rtlist_Y_ = utils.basic.matmul2(Ys_T_Xs_, rtlist_X_)
    rtlist_Y = rtlist_Y_.view(B, S, 4, 4)
    lrtlist_Y = merge_lrtlist(lenlist, rtlist_Y)
    return lrtlist_Y

# import time
# if __name__ == "__main__":
#     input = torch.rand(10, 4, 4).cuda()
#     cur_time = time.time()
#     out_1 = safe_inverse(input)
#     print('time for non-parallel:{}'.format(time.time() - cur_time))

#     print(out_1[0])

#     cur_time = time.time()
#     out_2 = safe_inverse_parallel(input)
#     print('time for parallel:{}'.format(time.time() - cur_time))

#     print(out_2[0])

def create_depth_image_single(xy, z, H, W, force_positive=True, max_val=100.0, serial=False, slices=20):
    # turn the xy coordinates into image inds
    xy = torch.round(xy).long()
    depth = torch.zeros(H*W, dtype=torch.float32, device=xy.device)
    depth[:] = max_val
    
    # lidar reports a sphere of measurements
    # only use the inds that are within the image bounds
    # also, only use forward-pointing depths (z > 0)
    valid_inds = (xy[:,0] <= W-1) & (xy[:,1] <= H-1) & (xy[:,0] >= 0) & (xy[:,1] >= 0) & (z[:] > 0)

    # gather these up
    xy = xy[valid_inds]
    z = z[valid_inds]

    inds = utils.basic.sub2ind(H, W, xy[:,1], xy[:,0]).long()
    if not serial:
        depth[inds] = z
    else:
        if False:
            for (index, replacement) in zip(inds, z):
                if depth[index] > replacement:
                    depth[index] = replacement
        # ok my other idea is:
        # sort the depths by distance
        # create N depth maps
        # merge them back-to-front

        # actually maybe you don't even need the separate maps

        sort_inds = torch.argsort(z, descending=True)
        xy = xy[sort_inds]
        z = z[sort_inds]
        N = len(sort_inds)
        def split(a, n):
            k, m = divmod(len(a), n)
            return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

        slice_inds = split(range(N), slices)
        for si in slice_inds:
            mini_z = z[si]
            mini_xy = xy[si]
            inds = utils.basic.sub2ind(H, W, mini_xy[:,1], mini_xy[:,0]).long()
            depth[inds] = mini_z
        # cool; this is rougly as fast as the parallel, and as accurate as the serial
        
        if False:
            print('inds', inds.shape)
            unique, inverse, counts = torch.unique(inds, return_inverse=True, return_counts=True)
            print('unique', unique.shape)

            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

            # new_inds = inds[inverse_inds]
            # depth[new_inds] = z[unique_inds]

            depth[unique] = z[perm]

            # now for the duplicates...

            dup = counts > 1
            dup_unique = unique[dup]
            print('dup_unique', dup_unique.shape)
            depth[dup_unique] = 0.5
        
    if force_positive:
        # valid = (depth > 0.0).float()
        depth[torch.where(depth == 0.0)] = max_val
    # else:
    #     valid = torch.ones_like(depth)

    valid = (depth > 0.0).float() * (depth < max_val).float()
    
    depth = torch.reshape(depth, [1, H, W])
    valid = torch.reshape(valid, [1, H, W])
    return depth, valid

def create_depth_image(pix_T_cam, xyz_cam, H, W, offset_amount=0, max_val=100.0, serial=False, slices=20):
    B, N, D = list(xyz_cam.shape)
    assert(D==3)
    B2, E, F = list(pix_T_cam.shape)
    assert(B==B2)
    assert(E==4)
    assert(F==4)
    xy = apply_pix_T_cam(pix_T_cam, xyz_cam)
    z = xyz_cam[:,:,2]

    depth = torch.zeros(B, 1, H, W, dtype=torch.float32, device=xyz_cam.device)
    valid = torch.zeros(B, 1, H, W, dtype=torch.float32, device=xyz_cam.device)
    for b in list(range(B)):
        xy_b, z_b = xy[b], z[b]
        ind = z_b > 0
        xy_b = xy_b[ind]
        z_b = z_b[ind]
        depth_b, valid_b = create_depth_image_single(xy_b, z_b, H, W, max_val=max_val, serial=serial, slices=slices)
        if offset_amount:
            depth_b = depth_b.reshape(-1)
            valid_b = valid_b.reshape(-1)
            
            for off_x in range(offset_amount):
                for off_y in range(offset_amount):
                    for sign in [-1,1]:
                        offset = np.array([sign*off_x,sign*off_y]).astype(np.float32)
                        offset = torch.from_numpy(offset).reshape(1, 2).to(xyz_cam.device)
                        # offsets.append(offset)
                        depth_, valid_ = create_depth_image_single(xy_b + offset, z_b, H, W, max_val=max_val)
                        depth_ = depth_.reshape(-1)
                        valid_ = valid_.reshape(-1)
                        # at invalid locations, use this new value
                        depth_b[valid_b==0] = depth_[valid_b==0]
                        valid_b[valid_b==0] = valid_[valid_b==0]
                    
            depth_b = depth_b.reshape(1, H, W)
            valid_b = valid_b.reshape(1, H, W)
        depth[b] = depth_b
        valid[b] = valid_b
    return depth, valid


def create_feat_image_single(xy, z, feat_pts, H, W, force_positive=True, max_val=100.0, serial=False, slices=20):
    # turn the xy coordinates into image inds
    C, N = list(feat_pts.shape)
    xy = torch.round(xy).long()
    depth = torch.zeros(H * W, dtype=torch.float32, device=torch.device('cuda'))
    depth[:] = max_val
    feat = torch.zeros(C, H * W, dtype=torch.float32, device=torch.device('cuda'))

    # lidar reports a sphere of measurements
    # only use the inds that are within the image bounds
    # also, only use forward-pointing depths (z > 0)
    valid = (xy[:, 0] <= W - 1) & (xy[:, 1] <= H - 1) & (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (z[:] > 0)

    # gather these up
    xy = xy[valid]
    z = z[valid]
    feat_pts = feat_pts[:, valid]

    inds = utils.basic.sub2ind(H, W, xy[:, 1], xy[:, 0]).long()
    if not serial:
        depth[inds] = z
        feat[:, inds] = feat_pts
    else:
        sort_inds = torch.argsort(z, descending=True)
        xy = xy[sort_inds]
        z = z[sort_inds]
        feat_pts = feat_pts[:, sort_inds]
        N = len(sort_inds)

        def split(a, n):
            k, m = divmod(len(a), n)
            return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

        slice_inds = split(range(N), slices)
        for si in slice_inds:
            mini_z = z[si]
            mini_xy = xy[si]
            mini_feat = feat_pts[:, si]
            inds = utils.basic.sub2ind(H, W, mini_xy[:, 1], mini_xy[:, 0]).long()
            depth[inds] = mini_z
            feat[:, inds] = mini_feat
    if force_positive:
        depth[torch.where(depth == 0.0)] = max_val
        valid = (depth > 0.0).float() * (depth < max_val).float()
    else:
        valid = torch.ones_like(depth)
    depth = torch.reshape(depth, [1, H, W])
    feat = torch.reshape(feat, [C, H, W])
    valid = torch.reshape(valid, [1, H, W])
    return feat, valid


def create_feat_image(pix_T_cam, xyz_cam, feat_cam, H, W, offset_amount=0, max_val=100.0, serial=False, slices=20):
    B, N, D = list(xyz_cam.shape)
    assert (D == 3)
    B2, E, F = list(pix_T_cam.shape)
    assert (B == B2)
    assert (E == 4)
    assert (F == 4)
    B3, C, N2 = list(feat_cam.shape)
    assert (B3 == B)
    assert (N2 == N)
    xy = apply_pix_T_cam(pix_T_cam, xyz_cam)
    z = xyz_cam[:, :, 2]

    feat = torch.zeros(B, C, H, W, dtype=torch.float32, device=torch.device('cuda'))
    valid = torch.zeros(B, 1, H, W, dtype=torch.float32, device=torch.device('cuda'))
    for b in list(range(B)):
        feat_b, valid_b = create_feat_image_single(xy[b], z[b], feat_cam[b], H, W, max_val=max_val, serial=serial, slices=slices)
        if offset_amount:
            feat_b = feat_b.reshape(-1)
            valid_b = valid_b.reshape(-1)

            for off_x in range(offset_amount):
                for off_y in range(offset_amount):
                    for sign in [-1, 1]:
                        offset = np.array([sign * off_x, sign * off_y]).astype(np.float32)
                        offset = torch.from_numpy(offset).reshape(1, 2).cuda()
                        # offsets.append(offset)
                        feat_, valid_ = create_feat_image_single(xy[b] + offset, z[b], feat_cam[b], H, W, max_val=max_val, serial=serial, slices=slices)
                        feat_ = feat_.reshape(-1)
                        valid_ = valid_.reshape(-1)
                        # at invalid locations, use this new value
                        feat_b[valid_b == 0] = feat_[valid_b == 0]
                        valid_b[valid_b == 0] = valid_[valid_b == 0]

            feat_b = feat_b.reshape(C, H, W)
            valid_b = valid_b.reshape(1, H, W)
        feat[b] = feat_b
        valid[b] = valid_b
    return feat, valid

def get_iou_from_corresponded_lrtlists(lrtlist_a, lrtlist_b):
    B, N, D = list(lrtlist_a.shape)
    assert(D==19)
    B2, N2, D2 = list(lrtlist_b.shape)
    assert(B2==B)
    assert(N2==N)
    
    xyzlist_a = get_xyzlist_from_lrtlist(lrtlist_a)
    xyzlist_b = get_xyzlist_from_lrtlist(lrtlist_b)
    # these are B x N x 8 x 3

    xyzlist_a = xyzlist_a.detach().cpu().numpy()
    xyzlist_b = xyzlist_b.detach().cpu().numpy()

    # ious = np.zeros((B, N), np.float32)
    ioulist_3d = torch.zeros(B, N, dtype=torch.float32, device=torch.device('cuda'))
    ioulist_2d = torch.zeros(B, N, dtype=torch.float32, device=torch.device('cuda'))
    for b in list(range(B)):
        for n in list(range(N)):
            iou_3d =  utils.box.new_box3d_iou(lrtlist_a[b:b+1,n:n+1],lrtlist_b[b:b+1,n:n+1])
            _, iou_2d = utils.box.box3d_iou(xyzlist_a[b,n], xyzlist_b[b,n]+1e-4)
            # print('computed iou %d,%d: %.2f' % (b, n, iou))
            ioulist_3d[b,n] = iou_3d
            ioulist_2d[b,n] = iou_2d
            
    # print('ioulist_3d', ioulist_3d)
    # print('ioulist_2d', ioulist_2d)
    return ioulist_3d, ioulist_2d

def get_centroid_from_box2d(box2d):
    ymin = box2d[:,0]
    xmin = box2d[:,1]
    ymax = box2d[:,2]
    xmax = box2d[:,3]
    x = (xmin+xmax)/2.0
    y = (ymin+ymax)/2.0
    return y, x

def normalize_boxlist2d(boxlist2d, H, W):
    boxlist2d = boxlist2d.clone()
    ymin, xmin, ymax, xmax = torch.unbind(boxlist2d, dim=2)
    ymin = ymin / float(H)
    ymax = ymax / float(H)
    xmin = xmin / float(W)
    xmax = xmax / float(W)
    boxlist2d = torch.stack([ymin, xmin, ymax, xmax], dim=2)
    return boxlist2d

def unnormalize_boxlist2d(boxlist2d, H, W):
    boxlist2d = boxlist2d.clone()
    ymin, xmin, ymax, xmax = torch.unbind(boxlist2d, dim=2)
    ymin = ymin * float(H)
    ymax = ymax * float(H)
    xmin = xmin * float(W)
    xmax = xmax * float(W)
    boxlist2d = torch.stack([ymin, xmin, ymax, xmax], dim=2)
    return boxlist2d

def unnormalize_box2d(box2d, H, W):
    return unnormalize_boxlist2d(box2d.unsqueeze(1), H, W).squeeze(1)

def normalize_box2d(box2d, H, W):
    return normalize_boxlist2d(box2d.unsqueeze(1), H, W).squeeze(1)

def get_size_from_box2d(box2d):
    ymin = box2d[:,0]
    xmin = box2d[:,1]
    ymax = box2d[:,2]
    xmax = box2d[:,3]
    height = ymax-ymin
    width = xmax-xmin
    return height, width

def get_centroidlist_from_boxlist2d(boxlist2d):
    ymin = boxlist2d[:,:,0]
    xmin = boxlist2d[:,:,1]
    ymax = boxlist2d[:,:,2]
    xmax = boxlist2d[:,:,3]
    x = (xmin+xmax)/2.0
    y = (ymin+ymax)/2.0
    return y, x

def get_sizelist_from_boxlist2d(boxlist2d):
    ymin = boxlist2d[:,:,0]
    xmin = boxlist2d[:,:,1]
    ymax = boxlist2d[:,:,2]
    xmax = boxlist2d[:,:,3]
    height = ymax-ymin
    width = xmax-xmin
    return height, width

def get_box2d_from_centroid_and_size(cy, cx, h, w, clip=True):
    # centroids is B x N x 2
    # dims is B x N x 2
    # both are in normalized coords
    
    ymin = cy - h/2
    ymax = cy + h/2
    xmin = cx - w/2
    xmax = cx + w/2

    box = torch.stack([ymin, xmin, ymax, xmax], dim=1)
    if clip:
        box = torch.clamp(box, 0, 1)
    return box

def get_box2d_from_mask(mask, normalize=False):
    # mask is B, 1, H, W

    B, C, H, W = mask.shape
    assert(C==1)
    xy = utils.basic.gridcloud2d(B, H, W, norm=False) # B, H*W, 2
    print('xy', xy.shape)

    box = torch.zeros((B, 4), dtype=torch.float32, device=mask.device)
    for b in range(B):
        xy_b = xy[b] # H*W, 2
        mask_b = mask[b].reshape(H*W)
        xy_ = xy_b[mask_b > 0]
        x_ = xy_[:,0]
        y_ = xy_[:,1]
        ymin = torch.min(y_)
        ymax = torch.max(y_)
        xmin = torch.min(x_)
        xmax = torch.max(x_)
        box[b] = torch.stack([ymin, xmin, ymax, xmax], dim=0)
    box = normalize_boxlist2d(box.unsqueeze(1), H, W).squeeze(1)
    return box

def convert_box2d_to_intrinsics(box2d, pix_T_cam, H, W, use_image_aspect_ratio=True, mult_padding=1.0):
    # box2d is B x 4, with ymin, xmin, ymax, xmax in normalized coords
    # ymin, xmin, ymax, xmax = torch.unbind(box2d, dim=1)
    # H, W is the original size of the image
    # mult_padding is relative to object size in pixels

    # i assume we're rendering an image the same size as the original (H, W)

    if not mult_padding==1.0:
        y, x = get_centroid_from_box2d(box2d)
        h, w = get_size_from_box2d(box2d)
        box2d = get_box2d_from_centroid_and_size(
            y, x, h*mult_padding, w*mult_padding, clip=False)
        
    if use_image_aspect_ratio:
        h, w = get_size_from_box2d(box2d)
        y, x = get_centroid_from_box2d(box2d)

        # note h,w are relative right now
        # we need to undo this, to see the real ratio

        h = h*float(H)
        w = w*float(W)
        box_ratio = h/w
        im_ratio = H/float(W)

        # print('box_ratio:', box_ratio)
        # print('im_ratio:', im_ratio)

        if box_ratio >= im_ratio:
            w = h/im_ratio
            # print('setting w:', h/im_ratio)
        else:
            h = w*im_ratio
            # print('setting h:', w*im_ratio)
            
        box2d = get_box2d_from_centroid_and_size(
            y, x, h/float(H), w/float(W), clip=False)

    assert(h > 1e-4)
    assert(w > 1e-4)
        
    ymin, xmin, ymax, xmax = torch.unbind(box2d, dim=1)

    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)

    # the topleft of the new image will now have a different offset from the center of projection
    
    new_x0 = x0 - xmin*W
    new_y0 = y0 - ymin*H

    pix_T_cam = pack_intrinsics(fx, fy, new_x0, new_y0)
    # this alone will give me an image in original resolution,
    # with its topleft at the box corner

    box_h, box_w = get_size_from_box2d(box2d)
    # these are normalized, and shaped B. (e.g., [0.4], [0.3])

    # we are going to scale the image by the inverse of this,
    # since we are zooming into this area

    sy = 1./box_h
    sx = 1./box_w

    pix_T_cam = scale_intrinsics(pix_T_cam, sx, sy)
    return pix_T_cam, box2d

def rad2deg(rad):
    return rad*180.0/np.pi

def deg2rad(deg):
    return deg/180.0*np.pi

def wrap2pi(rad_angle):
    # puts the angle into the range [-pi, pi]
    return torch.atan2(torch.sin(rad_angle), torch.cos(rad_angle))

def corners_to_boxes(corners, legacy_format=False):
    # corners is B x N x 8 x 3
    B, N, C, D = list(corners.shape)
    assert(C==8)
    assert(D==3)
    assert(legacy_format) # you need to the corners in legacy (non-clockwise) format and acknowledge this
    # do them all at once
    corners_ = corners.reshape(B*N, 8, 3)
    boxes_ = corners_to_boxes_py(corners_.detach().cpu().numpy(), legacy_format=legacy_format)
    boxes_ = torch.from_numpy(boxes_).float().to('cuda')
    # reshape
    boxes = boxes_.reshape(B, N, 9)
    return boxes

def corners_to_boxes_py(corners, legacy_format=False):
    # corners is B x 8 x 3

    assert(legacy_format) # you need to the corners in legacy (non-clockwise) format and acknowledge this
 
    # assert(False) # this function has a flaw; use rigid_transform_boxes instead, or fix it.
    # # i believe you can fix it using what i noticed in rigid_transform_boxes:
    # # if we are looking at the box backwards, the rx/rz dirs flip

    # we want to transform each one to a box
    # note that the rotation may flip 180deg, since corners do not have this info
    
    boxes = []
    for ind, corner_set in enumerate(corners):
        xs = corner_set[:,0]
        ys = corner_set[:,1]
        zs = corner_set[:,2]
        # these are 8 each
        
        xc = np.mean(xs)
        yc = np.mean(ys)
        zc = np.mean(zs)

        # we constructed the corners like this:
        # xs = tf.stack([-lx/2., -lx/2., -lx/2., -lx/2., lx/2., lx/2., lx/2., lx/2.], axis=1)
        # ys = tf.stack([-ly/2., -ly/2., ly/2., ly/2., -ly/2., -ly/2., ly/2., ly/2.], axis=1)
        # zs = tf.stack([-lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2.], axis=1)
        # # so we can recover lengths like this:
        # lx = np.linalg.norm(xs[-1] - xs[0])
        # ly = np.linalg.norm(ys[-1] - ys[0])
        # lz = np.linalg.norm(zs[-1] - zs[0])
        # but that's a noisy estimate apparently. let's try all pairs
        
        # rotations are a bit more interesting...

        # defining the corners as: clockwise backcar face, clockwise frontcar face:
        #   E -------- F
        #  /|         /|
        # A -------- B .
        # | |        | |
        # . H -------- G
        # |/         |/
        # D -------- C

        # the ordered eight indices are:
        # A E D H B F C G

        # unstack on first dim
        A, E, D, H, B, F, C, G = corner_set

        back = [A, B, C, D] # back of car is closer to us
        front = [E, F, G, H]
        top = [A, E, B, F]
        bottom = [D, C, H, G]

        front = np.stack(front, axis=0)
        back = np.stack(back, axis=0)
        top = np.stack(top, axis=0)
        bottom = np.stack(bottom, axis=0)
        # these are 4 x 3

        back_z = np.mean(back[:,2])
        front_z = np.mean(front[:,2])
        # usually the front has bigger coords than back
        backwards = not (front_z > back_z)

        front_y = np.mean(front[:,1])
        back_y = np.mean(back[:,1])
        # someetimes the front dips down
        dips_down = front_y > back_y
        
        
        # the bottom should have bigger y coords than the bottom (since y increases down)
        top_y = np.mean(top[:,2])
        bottom_y = np.mean(bottom[:,2])
        upside_down = not (top_y < bottom_y)
        
        # rx: i need anything but x-aligned bars
        # there are 8 of these
        # atan2 wants the y part then the x part; here this means y then z

        x_bars = [[A, B], [D, C], [E, F], [H, G]]
        y_bars = [[A, D], [B, C], [E, H], [F, G]]
        z_bars = [[A, E], [B, F], [D, H], [C, G]]

        lx = 0.0
        for x_bar in x_bars:
            x0, x1 = x_bar
            lx += np.linalg.norm(x1-x0)
        lx /= 4.0
        
        ly = 0.0
        for y_bar in y_bars:
            y0, y1 = y_bar
            ly += np.linalg.norm(y1-y0)
        ly /= 4.0
        
        lz = 0.0
        for z_bar in z_bars:
            z0, z1 = z_bar
            lz += np.linalg.norm(z1-z0)
        lz /= 4.0
        
        # rx = 0.0
        # for pair in [z_bar:
            # rx += np.arctan2(A[1] - E[1], A[2] - E[2])
        # rx = rx / 8.0

        # x: we want atan2(y,z)
        # rx = np.arctan2(A[1] - E[1], A[2] - E[2])
        rx = 0.0
        for bar in z_bars:
            pt1, pt2 = bar
            # intermed = np.arctan2(np.abs(pt1[1] - pt2[1]), np.abs(pt1[2] - pt2[2]))
            intermed = np.arctan2((pt1[1] - pt2[1]), (pt1[2] - pt2[2]))
            rx += intermed
            # if ind==0:
            #     print 'temp rx = %.2f' % intermed
        # for bar in y_bars:
        #     pt1, pt2 = bar
        #     rx += np.arctan2(pt1[1] - pt2[1], pt1[2] - pt2[2])
        # rx /= 8.0
        rx /= 4.0

        ry = 0.0
        for bar in z_bars:
            pt1, pt2 = bar
            # intermed = np.arctan2(np.abs(pt1[2] - pt2[2]), np.abs(pt1[0] - pt2[0]))
            intermed = np.arctan2((pt1[2] - pt2[2]), (pt1[0] - pt2[0]))
            ry += intermed
            # if ind==0:
            #     print 'temp ry = %.2f' % intermed
        # for bar in x_bars:
        #     pt1, pt2 = bar
        #     ry += np.arctan2(pt1[2] - pt2[2], pt1[0] - pt2[0])
        #     if ind==0:
        #         print 'temp ry = %.2f' % np.arctan2(pt1[2] - pt2[2], pt1[0] - pt2[0])
        ry /= 4.0
        
        rz = 0.0
        for bar in x_bars:
            pt1, pt2 = bar
            # intermed = np.arctan2(np.abs(pt1[1] - pt2[1]), np.abs(pt1[0] - pt2[0]))
            intermed = np.arctan2((pt1[1] - pt2[1]), (pt1[0] - pt2[0]))
            rz += intermed
            # if ind==0:
            #     print 'temp rz = %.2f' % intermed
        # for bar in y_bars:
        #     pt1, pt2 = bar
        #     rz += np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0])
        # rz /= 8.0
        rz /= 4.0


        # # ry: i need anything but y-aligned bars
        # # y: we want atan2(z,x)
        # ry = np.arctan2(A[2] - E[2], A[0] - E[0])

        # rz: anything but z-aligned bars
        # z: we want atan2(y,x)
        # rz = np.arctan2(A[1] - B[1], A[0] - B[0])

        ry += np.pi/2.0

        # handle axis flips
            
        # if ind==0 or ind==1:
        #     # print 'rx = %.2f' % rx
        #     # print 'ry = %.2f' % ry
        #     # print 'rz = %.2f' % rz
        #     print 'rx = %.2f; ry = %.2f; rz = %.2f, backwards = %s; dips_down = %s, front %.2f, back %.2f, upside_down = %s' % (
        #         rx, ry, rz, backwards, dips_down,
        #         front_y, back_y, upside_down)
        if backwards:
            ry = -ry
        if not backwards:
            ry = ry - np.pi

        # rx = 0.0
        # rz = 0.0
        
        #     # rx = rx - np.pi
        #     rz = -rz 
           
        # if np.abs(rz) > np.pi/2.0:
        #     # rx = -rx
        #     rx = wrap2halfpi_single_py(rx)
        #     rz = wrap2halfpi_single_py(rz)

        # # hack
        # if np.abs(ry) < np.pi/2.0:
        #     rx = -rx

        
        #     rx = rx - np.pi
        # else:
        #     ry = ry - np.pi
        # # rx = -rx
        # if dips_down:
        #     rx = -rx
            
            # ry = -(ry - np.pi)
            # ry = -(ry - np.pi)
            # ry = -(ry - np.pi)
        # ry = wrap2pi_py(ry)
        #     if not dips_down:
        #         rx = -rx
        # if dips_down and not backwards:
        #     rx = -rx
        # if dips_down:
        #     rx = -rx
            
            # rx = -rx
            # rz = -rz
        # if backwards_x:
        #     rz = -rz
            
        box = np.array([xc, yc, zc, lx, ly, lz, rx, ry, rz])
        boxes.append(box)
    boxes = np.stack(boxes, axis=0).astype(np.float32)
    return boxes
    
    
def corners_to_box3d_single_py(corners):
    # corners is N x 8 x 3

    # boxes_new, tids_new, scores_new = tf.py_function(sink_invalid_boxes_py, (boxes, tids, scores),
    #                                                  (tf.float32, tf.int32, tf.float32))
    
    
    # (N, 8, 3) -> (N, 7) x,y,z,h,w,l,ry or rz
    if coordinate == 'lidar':
        for idx in list(range(len(boxes_corner))):
            boxes_corner[idx] = lidar_to_camera_point(boxes_corner[idx], rect_T_cam, cam_T_velo)
    ret = []

    
    for roi in boxes_corner:
        roi = np.array(roi)
        h = abs(np.sum(roi[:4, 1] - roi[4:, 1]) / 4.0)
        w = np.sum(
            np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
            np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
            np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
            np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
        ) / 4
        l = np.sum(
            np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
            np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
            np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
            np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
        ) / 4
        x = np.sum(roi[:, 0], axis=0) / 8.0
        y = np.sum(roi[0:4, 1], axis=0) / 4.0
        z = np.sum(roi[:, 2], axis=0) / 8.0
        ry = np.sum(
            np.arctan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
            np.arctan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
            np.arctan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
            np.arctan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
            np.arctan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
            np.arctan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
            np.arctan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
            np.arctan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
        ) / 8.0
        if w > l:
            w, l = l, w
            ry = angle_in_limit(ry + np.pi / 2.0)
        ret.append([x, y, z, h, w, l, ry])

    return np.array(ret).astype(np.float32)
    
def inflate_to_axis_aligned_boxlist(boxlist):
    B, N, D = list(boxlist.shape)
    assert(D==9)

    corners = transform_boxes_to_corners(boxlist) # corners is B x N x 8 x 3
    corners_max = torch.max(corners, dim=2)[0]
    corners_min = torch.min(corners, dim=2)[0]

    centers = (corners_max + corners_min)/2.0
    sizes = corners_max - corners_min
    rots = torch.zeros_like(sizes)

    # xc, yc, zc, lx, ly, lz, rx, ry, rz
    boxlist_norot = torch.cat([centers, sizes, rots], dim=2)
    # boxlist_norot is B x N x 9

    return boxlist_norot

def depthrt2flow(depth_cam0, cam1_T_cam0, pix_T_cam):
    B, C, H, W = list(depth_cam0.shape)
    assert(C==1)

    # get the two pointclouds
    xyz_cam0 = depth2pointcloud(depth_cam0, pix_T_cam)
    xyz_cam1 = apply_4x4(cam1_T_cam0, xyz_cam0)

    # project, and get 2d flow
    flow = pointcloud2flow(xyz_cam1, pix_T_cam, H, W)
    return flow

def pointcloud2flow(xyz1, pix_T_cam, H, W):
    # project xyz1 down, so that we get the 2d location of all of these pixels,
    # then subtract these 2d locations from the original ones to get optical flow
    
    B, N, C = list(xyz1.shape)
    assert(N==H*W)
    assert(C==3)
    
    # we assume xyz1 is the unprojection of the regular grid
    grid_y0, grid_x0 = utils.basic.meshgrid2d(B, H, W)

    xy1 = camera2pixels(xyz1, pix_T_cam)
    x1, y1 = torch.unbind(xy1, dim=2)
    x1 = x1.reshape(B, H, W)
    y1 = y1.reshape(B, H, W)

    flow_x = x1 - grid_x0
    flow_y = y1 - grid_y0
    flow = torch.stack([flow_x, flow_y], axis=1)
    # flow is B x 2 x H x W
    return flow

def get_boxlist2d_from_lrtlist(pix_T_cam, lrtlist_cam, H, W, pad=0, clamp=False):
    B, N, D = list(lrtlist_cam.shape)
    assert(D==19)
    corners_cam = get_xyzlist_from_lrtlist(lrtlist_cam)
    # this is B x N x 8 x 3
    corners_cam_ = torch.reshape(corners_cam, [B, N*8, 3])
    corners_pix_ = apply_pix_T_cam(pix_T_cam, corners_cam_)
    corners_pix = torch.reshape(corners_pix_, [B, N, 8, 2])

    xmin = torch.min(corners_pix[:,:,:,0], dim=2)[0]
    xmax = torch.max(corners_pix[:,:,:,0], dim=2)[0]
    ymin = torch.min(corners_pix[:,:,:,1], dim=2)[0]
    ymax = torch.max(corners_pix[:,:,:,1], dim=2)[0]
    # these are B x N

    if pad > 0:
        xmin = xmin - pad
        ymin = ymin - pad
        xmax = xmax + pad
        ymax = ymax + pad

    boxlist2d = torch.stack([ymin, xmin, ymax, xmax], dim=2)
    boxlist2d = normalize_boxlist2d(boxlist2d, H, W)

    if clamp:
        boxlist2d = boxlist2d.clamp(0,1)
    return boxlist2d

def get_masklist_from_lrtlist(pix_T_cam, lrtlist_cam, H, W, pad=0):
    B, N, D = lrtlist_cam.shape
    boxlist2d_norm = get_boxlist2d_from_lrtlist(pix_T_cam, lrtlist_cam, H, W, pad=pad, clamp=True)
    return get_masklist_from_boxlist2d(boxlist2d_norm, H, W, pad=pad)
    
def get_masklist_from_boxlist2d(boxlist2d_norm, H, W, pad=0):
    B, N, D = boxlist2d_norm.shape
    assert(D==4)
    boxlist2d = unnormalize_boxlist2d(boxlist2d_norm, H, W)
    masklist = torch.zeros((B, N, 1, H, W), dtype=torch.float32, device=boxlist2d_norm.device)
    for b in range(B):
        for n in range(N):
            box2d = boxlist2d[b,n].round().long() # 4
            ymin, xmin, ymax, xmax = box2d.unbind(0)
            ymin = ymin.clamp(0, H-1)
            ymax = ymax.clamp(0, H-1)
            xmin = xmin.clamp(0, W-1)
            xmax = xmax.clamp(0, W-1)
            masklist[b,n,0,ymin:ymax,xmin:xmax] = 1
    return masklist
    
def sincos_norm(sin, cos):
    both = torch.stack([sin, cos], dim=-1)
    both = utils.basic.l2_normalize(both, dim=-1)
    sin, cos = torch.unbind(both, dim=-1)
    return sin, cos
                
def sincos2rotm(sinz, siny, sinx, cosz, cosy, cosx):
    # copy of matlab
    # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
    #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
    #        -sy            cy*sx             cy*cx]
    r11 = cosy*cosz
    r12 = sinx*siny*cosz - cosx*sinz
    r13 = cosx*siny*cosz + sinx*sinz
    r21 = cosy*sinz
    r22 = sinx*siny*sinz + cosx*cosz
    r23 = cosx*siny*sinz - sinx*cosz
    r31 = -siny
    r32 = sinx*cosy
    r33 = cosx*cosy
    r1 = torch.stack([r11,r12,r13],dim=-1)
    r2 = torch.stack([r21,r22,r23],dim=-1)
    r3 = torch.stack([r31,r32,r33],dim=-1)
    r = torch.stack([r1,r2,r3],dim=-2)
    return r
                                                        
def convert_clist_to_lrtlist(clist, len0, angle0=None, smooth=3):
    B, S, D = list(clist.shape)
    B, E = list(len0.shape)
    assert(D==3)
    assert(E==3)

    boxlist = torch.zeros(B, S, 9).float().cuda()
    for s in list(range(S)):
        s_a = max(s-smooth, 0)
        s_b = min(s+smooth, S)
        xyz0 = torch.mean(clist[:,s_a:s+1], dim=1)
        xyz1 = torch.mean(clist[:,s:s_b+1], dim=1)

        delta = xyz1-xyz0
        delta_norm = torch.norm(delta, dim=1)
        invalid_NY = delta_norm < 0.0001

        if invalid_NY.sum() > 0:
            yaw = torch.zeros_like(delta[:,0])
            # B 
        else:
            delta = delta.detach().cpu().numpy()
            dx = delta[:,0]
            dy = delta[:,1]
            dz = delta[:,2]
            yaw = -np.arctan2(dz, dx) + np.pi/2.0
            yaw = torch.from_numpy(yaw).float().cuda()

        zero = torch.zeros_like(yaw)
        angles = torch.stack([zero, yaw, zero], dim=1)
        # this is B x 3

        boxlist[:,s] = torch.cat([clist[:,s], len0, angles], dim=1)

    lrtlist = convert_boxlist_to_lrtlist(boxlist)
    return lrtlist   

def angular_l1_norm(e, g, dim=1, keepdim=False):
    # inputs are shaped B x N
    # returns a tensor sized B x N, with the dist in every slot
    
    # if our angles are in [0, 360] we can follow this stack overflow answer:
    # https://gamedev.stackexchange.com/questions/4467/comparing-angles-and-working-out-the-difference
    # wrap2pi brings the angles to [-180, 180]; adding pi puts them in [0, 360]
    e = wrap2pi(e)+np.pi
    g = wrap2pi(g)+np.pi
    # now our angles are in [0, 360]
    l = torch.abs(np.pi - torch.abs(torch.abs(e-g) - np.pi))
    return torch.sum(l, dim=dim, keepdim=keepdim)

def angular_l1_dist(e, g):
    # inputs are shaped B x N
    # returns a tensor sized B x N, with the dist in every slot
    
    # if our angles are in [0, 360] we can follow this stack overflow answer:
    # https://gamedev.stackexchange.com/questions/4467/comparing-angles-and-working-out-the-difference
    # wrap2pi brings the angles to [-180, 180]; adding pi puts them in [0, 360]
    e = wrap2pi(e)+np.pi
    g = wrap2pi(g)+np.pi
    # now our angles are in [0, 360]
    l = torch.abs(np.pi - torch.abs(torch.abs(e-g) - np.pi))
    return l

def get_arrowheadlist_from_lrtlist(lrtlist):
    B, N, D = list(lrtlist.shape)
    assert(D==19)

    lenlist, rtlist = split_lrtlist(lrtlist)
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4

    xyzlist_obj = torch.zeros((B, N, 1, 3), device=lrtlist.device)
    # xyzlist_obj is B x N x 8 x 3

    # unit vector in Z direction
    arrow_head_init = torch.Tensor([[0,0,1]]).to(lrtlist.device).repeat(B*N,1,1)
    # arrow_head_init = torch.Tensor([[1,0,0]]).cuda().repeat(B*N,1,1) 
    lenlist_ = lenlist.reshape(B*N,1,3)

    arrow_head_ = arrow_head_init * lenlist_

    rtlist_ = rtlist.reshape(B*N, 4, 4)
    xyzlist_obj_ = xyzlist_obj.reshape(B*N, 1, 3) + arrow_head_

    xyzlist_cam_ = apply_4x4(rtlist_, xyzlist_obj_)
    xyzlist_cam = xyzlist_cam_.reshape(B, N, 3)
    return xyzlist_cam

def crop_and_resize(im, box2d, PH, PW, box2d_is_normalized=True):
    B, C, H, W = im.shape
    B2, D = box2d.shape
    assert(B==B2)
    assert(D==4)
    # PH, PW is the size to resize to

    # output is B x C x PH x PW

    # pt wants xy xy, unnormalized
    if box2d_is_normalized:
        box2d_unnorm = unnormalize_boxlist2d(box2d.unsqueeze(1), H, W).squeeze(1)
    else:
        box2d_unnorm = box2d
        
    ymin, xmin, ymax, xmax = box2d_unnorm.unbind(1)
    # box2d_pt = torch.stack([box2d_unnorm[:,1], box2d_unnorm[:,0], box2d_unnorm[:,3], box2d_unnorm[:,2]], dim=1)
    box2d_pt = torch.stack([xmin, ymin, xmax, ymax], dim=1)
    # we want a B-len list of K x 4 arrays
    box2d_list = list(box2d_pt.unsqueeze(1).unbind(0))
    rgb_crop = ops.roi_align(im, box2d_list, output_size=(PH, PW))

    return rgb_crop
    
def get_image_inbounds(pix_T_cam, xyz_cam, H, W, padding=0.0):
    # pix_T_cam is B x 4 x 4
    # xyz_cam is B x N x 3
    # padding should be 0 unless you are trying to account for some later cropping
    
    xy_pix = utils.geom.apply_pix_T_cam(pix_T_cam, xyz_cam)

    x = xy_pix[:,:,0]
    y = xy_pix[:,:,1]
    z = xyz_cam[:,:,2]

    # print('x', x.detach().cpu().numpy())
    # print('y', y.detach().cpu().numpy())
    # print('z', z.detach().cpu().numpy())

    x_valid = ((x-padding)>-0.5).bool() & ((x+padding)<float(W-0.5)).bool()
    y_valid = ((y-padding)>-0.5).bool() & ((y+padding)<float(H-0.5)).bool()
    z_valid = ((z>0.0)).bool()

    inbounds = x_valid & y_valid & z_valid
    return inbounds.bool()

def zero_rotations_from_lrtlist(lrtlist):
    B, N, D = list(lrtlist.shape)
    # now that we are in the target coords, let's eliminate the rotation,
    # since we are not estimating it anyway

    lenlist, rtlist = split_lrtlist(lrtlist)
    rtlist_ = rtlist.reshape(B*N, 4, 4)
    rlist_, tlist_ = split_rt(rtlist_)
    # rx_, ry_, rz_ = rotm2eul(rlist_)
    rtlist_ = merge_rt(eye_3x3(B*N), tlist_)
    rtlist = rtlist_.reshape(B, N, 4, 4)
    lrtlist = merge_lrtlist(lenlist, rtlist)
    return lrtlist

def get_NY_transforms_from_clist(clist, retain_pitch=False, use_endpoint=False, return_in_case=False, use_endpoint_in_case=False):
    B, S, _ = list(clist.shape)
    rot0 = utils.geom.eye_3x3(B)
    t0 = torch.zeros(B, 3).float().cuda()

    # let's go to noyaw, nopitch, nospeed,
    # then compare to the lib
    # then return top5,
    # then restore speed, pitch, yaw

    xyz0 = clist[:,0]
    if use_endpoint:
        xyz1 = clist[:,-1]
    else:
        xyz1 = clist[:,1]
    # these are B x 3
    # print('xyz0', xyz0.shape)

    delta = xyz1 - xyz0

    # ok_mask = (torch.norm(delta, dim=1) > 0.01).float()

    EPS = 1e-2

    # if return_in_case:
    #     if torch.norm(delta, dim=1) < EPS:
    #         print('returning from NY early')
    #         return None, None
    
    if use_endpoint_in_case:
        if torch.norm(delta, dim=1) < EPS:
            print('falling back to endpoint')
            # use endpoint after all
            xyz1 = clist[:,-1]
            delta = xyz1 - xyz0
            

    dx = delta[:,0]
    dy = delta[:,1]
    dz = delta[:,2]
    
    use_torch = True
    if use_torch:
        bot_hyp = torch.sqrt(dz**2 + dx**2)

        if torch.abs(dy) < EPS and torch.abs(bot_hyp) < EPS:
            pitch = torch.zeros_like(dy)
            # print('-- setting pitch to 0 --')
        else:
            pitch = -torch.atan2(dy, bot_hyp)

        if torch.abs(dz) < EPS and torch.abs(dx) < EPS:
            yaw = torch.zeros_like(dz)
            # print('-- setting yaw to 0 --')
            print('returning from NY early2')
            return None, None
        else:
            yaw = torch.atan2(dz, dx)

        # not_ok = torch.norm(delta, dim=1) < 0.01
        # pitch[not_ok] = 0.0
        # yaw[not_ok] = 0.0

        # pitch[torch.abs(dy) < 0.01] = 0.0
        # yaw[torch.logical_or(torch.abs(dx) < 0.01, torch.abs(dz) < 0.01)] = 0.0

        if retain_pitch:
            pitch = pitch*0.0

        # rot = utils.geom.eul2rotm(yaw*0.0, yaw, pitch)
        rot = utils.geom.eul2rotm(torch.zeros_like(yaw), yaw, pitch)
    else:
        delta = delta.detach().cpu().numpy()
        if np.linalg.norm(delta) > 0.01:
            dx = delta[:,0]
            dy = delta[:,1]
            dz = delta[:,2]

            bot_hyp = np.sqrt(dz**2 + dx**2)
            # top_hyp = np.sqrt(bot_hyp**2 + dy**2)

            pitch = -np.arctan2(dy, bot_hyp)
            yaw = np.arctan2(dz, dx)
        else:
            pitch = delta[:,0]*0.0
            yaw = delta[:,0]*0.0

        rot = [utils.py.eul2rotm(0,y,p) for (p,y) in zip(pitch,yaw)]
        rot = np.stack(rot)
        rot = torch.from_numpy(rot).float().cuda()

    # rot is B x 3 x 3
    t = -xyz0
    # t is B x 3
    zero_T_camX0 = utils.geom.merge_rt(rot0, t)
    camNY_T_zero = utils.geom.merge_rt(rot, t0)
    camNY_T_camX0 = utils.basic.matmul2(camNY_T_zero, zero_T_camX0)
    camNYs_T_camX0s = camNY_T_camX0.unsqueeze(1).repeat(1, S, 1, 1)
    
    return camNY_T_camX0, camNYs_T_camX0s


def get_NY_transforms_from_endpoints(xyz0, xyz1, retain_pitch=False, eps=0.001):
    B, D = list(xyz0.shape)
    B2, D2 = list(xyz1.shape)
    assert(B==B2)
    assert(D==3)
    assert(D2==3)
    
    rot0 = utils.geom.eye_3x3(B)
    t0 = torch.zeros(B, 3).float().cuda()

    # let's go to noyaw, nopitch, nospeed,
    # then compare to the lib
    # then return top5,
    # then restore speed, pitch, yaw

    delta = xyz1 - xyz0

    dx = delta[:,0]
    dy = delta[:,1]
    dz = delta[:,2]

    # ok_mask = (torch.norm(delta, dim=1) > 0.01).float()
    
    use_torch = True
    if use_torch:
        bot_hyp = torch.sqrt(dz**2 + dx**2)

        if torch.abs(dy) < eps and torch.abs(bot_hyp) < eps:
            pitch = torch.zeros_like(dy)
            # print('-- setting pitch to 0 --')
        else:
            pitch = -torch.atan2(dy, bot_hyp)

        if torch.abs(dz) < eps and torch.abs(dx) < eps:
            yaw = torch.zeros_like(dz)
            # print('-- setting yaw to 0 --')
        else:
            yaw = torch.atan2(dz, dx)

        # not_ok = torch.norm(delta, dim=1) < 0.01
        # pitch[not_ok] = 0.0
        # yaw[not_ok] = 0.0

        # pitch[torch.abs(dy) < 0.01] = 0.0
        # yaw[torch.logical_or(torch.abs(dx) < 0.01, torch.abs(dz) < 0.01)] = 0.0

        if retain_pitch:
            pitch = pitch*0.0

        rot = utils.geom.eul2rotm(yaw*0.0, yaw, pitch)
    else:
        delta = delta.detach().cpu().numpy()
        if np.linalg.norm(delta) > 0.01:
            dx = delta[:,0]
            dy = delta[:,1]
            dz = delta[:,2]

            bot_hyp = np.sqrt(dz**2 + dx**2)
            # top_hyp = np.sqrt(bot_hyp**2 + dy**2)

            pitch = -np.arctan2(dy, bot_hyp)
            yaw = np.arctan2(dz, dx)
        else:
            pitch = delta[:,0]*0.0
            yaw = delta[:,0]*0.0

        rot = [utils.py.eul2rotm(0,y,p) for (p,y) in zip(pitch,yaw)]
        rot = np.stack(rot)
        rot = torch.from_numpy(rot).float().cuda()

    # rot is B x 3 x 3
    t = -xyz0
    # t is B x 3
    zero_T_camX0 = utils.geom.merge_rt(rot0, t)
    camNY_T_zero = utils.geom.merge_rt(rot, t0)
    camNY_T_camX0 = utils.basic.matmul2(camNY_T_zero, zero_T_camX0)
    
    return camNY_T_camX0

def get_point_correspondence_from_flow(xyz0, xyz1, flow_f, pix_T_cam, H, W, flow_valid=None):
    # flow_f is the forward flow, from frame0 to frame1
    # xyz0 and xyz1 are pointclouds, in cam coords
    # we want to get a new xyz1, with points that correspond to xyz0
    B, N, D = list(xyz0.shape)

    # discard depths that are beyond this distance, since they are probably fake
    max_dist = 200.0
    
    # now sample the 2d flow vectors at the xyz0 locations
    # ah wait!:
    # it's important here to only use positions in front of the camera
    xy = apply_pix_T_cam(pix_T_cam, xyz0)
    z0 = xyz0[:, :, 2] # B x N
    x0 = xy[:, :, 0] # B x N
    y0 = xy[:, :, 1] # B x N
    uv = utils.samp.bilinear_sample2d(flow_f, x0, y0) # B x 2 x N

    frustum0_valid = get_image_inbounds(pix_T_cam, xyz0, H, W)

    # next we want to use these to sample into the depth of the next frame 
    # depth0, valid0 = create_depth_image(pix_T_cam, xyz0, H, W)
    depth1, valid1 = create_depth_image(pix_T_cam, xyz1, H, W)
    # valid0 = valid0 * (depth0 < max_dist).float()
    valid1 = valid1 * (depth1 < max_dist).float()
    
    u = uv[:, 0] # B x N
    v = uv[:, 1] # B x N
    x1 = x0 + u
    y1 = y0 + v

    # round to the nearest pixel, since the depth image has holes
    # x0 = torch.clamp(torch.round(x0), 0, W-1).long()
    # y0 = torch.clamp(torch.round(y0), 0, H-1).long()
    x1 = torch.clamp(torch.round(x1), 0, W-1).long()
    y1 = torch.clamp(torch.round(y1), 0, H-1).long()
    z1 = torch.zeros(B, N, dtype=torch.float32, device=torch.device('cuda'))
    valid = torch.zeros(B, N, dtype=torch.float32, device=torch.device('cuda'))
    # since we rounded and clamped, we can index directly, instead of bilinear sampling

    for b in range(B):
        # depth0_b = depth0[b] # 1 x H x W
        # valid0_b = valid0[b]
        # valid0_b_ = valid0_b[0, y0[b], x0[b]] # N
        # z0_b_ = depth0_b[0, y1[b], x1[b]] # N
        
        depth1_b = depth1[b] # 1 x H x W
        valid1_b = valid1[b]
        valid1_b_ = valid1_b[0, y1[b], x1[b]] # N
        z1_b_ = depth1_b[0, y1[b], x1[b]] # N
        
        z1[b] = z1_b_
        # valid[b] = valid0_b_ * valid1_b_ * frustum0_valid[b]
        valid[b] = valid1_b_ * frustum0_valid[b]

        if flow_valid is not None:
            validf_b = flow_valid[b] # 1 x H x W
            validf_b_ = validf_b[0, y1[b], x1[b]] # N
            valid[b] = valid[b] * validf_b_

    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz1 = pixels2camera(x1,y1,z1,fx,fy,x0,y0)
    # xyz1 is B x N x 3
    # valid is B x N
    return xyz1, valid

def get_pts_inbound_lrt(xyz, lrt, mult_pad=1.0, add_pad=0.0):
    B, N, D = list(xyz.shape)
    B1, C = lrt.shape
    assert(B == B1)
    assert(C == 19)
    assert(D == 3)

    lens, cam_T_obj = split_lrt(lrt)
    obj_T_cam = safe_inverse(cam_T_obj)

    xyz_obj = apply_4x4(obj_T_cam, xyz) # B x N x 3
    x = xyz_obj[:, :, 0] # B x N
    y = xyz_obj[:, :, 1]
    z = xyz_obj[:, :, 2]
    lx = lens[:, 0:1] * mult_pad + add_pad # B
    ly = lens[:, 1:2] * mult_pad + add_pad # B
    lz = lens[:, 2:3] * mult_pad + add_pad # B

    x_valid = (x >= -lx/2.0).bool() & (x <= lx/2.0).bool()
    y_valid = (y >= -ly/2.0).bool() & (y <= ly/2.0).bool()
    z_valid = (z >= -lz/2.0).bool() & (z <= lz/2.0).bool()
    inbounds = x_valid.bool() & y_valid.bool() & z_valid.bool() # B x N

    return inbounds

def random_occlusion(xyz, lrtlist, scorelist, pix_T_cam, H, W, mask_size=20, occ_prob=0.5, occlude_bkg_too=False):
    # with occ_prob, we create a random mask. else no operation
    num_try = 10
    max_dist = 200.0
    # lrtlist is B x 19
    B, N, D = list(xyz.shape)
    B, N_obj, C = lrtlist.shape
    assert(C == 19)
    depth, valid = create_depth_image(pix_T_cam, xyz, H, W) # B x 1 x H x W

    clist_cam = get_clist_from_lrtlist(lrtlist) # B x N_obj x 3
    clist_pix = camera2pixels(clist_cam, pix_T_cam) # B x N_obj x 2
    clist_pix = torch.round(clist_pix).long()
    # we create a mask around the center of the box
    xyz_new_s = torch.zeros(B, H*W, 3, device=torch.device('cuda'))

    # print(N_obj)

    mask = torch.ones_like(depth)

    for b in range(B):
        for n in range(N_obj):
            if np.random.uniform() < occ_prob and scorelist[b, n]:
                inbound = get_pts_inbound_lrt(xyz[b:b+1], lrtlist[b:b+1, n]) # 1 x N
                inb_pts_cnt = torch.sum(inbound)

                # print('inb_ori:', inb_pts_cnt)

                for _ in range(num_try):
                    rand_offset = torch.randint(-mask_size//2, mask_size//2, size=(1, 2), device=torch.device('cuda'))
                    mask_center = clist_pix[b, n:n+1] + rand_offset # 1 x 2
                    mask_lower_bound = mask_center - mask_size // 2
                    mask_upper_bound = mask_center + mask_size // 2
                    mask_lower_bound_x = mask_lower_bound[:, 0]
                    mask_lower_bound_y = mask_lower_bound[:, 1]
                    mask_upper_bound_x = mask_upper_bound[:, 0]
                    mask_upper_bound_y = mask_upper_bound[:, 1]

                    mask_lower_bound_x = torch.clamp(mask_lower_bound_x, 0, W-1) # each shape 1
                    mask_upper_bound_x = torch.clamp(mask_upper_bound_x, 0, W-1)
                    mask_lower_bound_y = torch.clamp(mask_lower_bound_y, 0, H-1)
                    mask_upper_bound_y = torch.clamp(mask_upper_bound_y, 0, H-1)

                    # do the masking
                    depth_b = depth[b:b+1].clone() # 1 x 1 x H x W
                    mask_b = torch.ones_like(depth_b)
                    mask_b[:, :, mask_lower_bound_y:mask_upper_bound_y, mask_lower_bound_x:mask_upper_bound_x] = 0
                    depth_b[:, :, mask_lower_bound_y:mask_upper_bound_y, mask_lower_bound_x:mask_upper_bound_x] = max_dist
                    # set to a large value, i.e. mask out these area

                    if occlude_bkg_too:
                        bkg_mask_size = mask_size * 2
                        mask_center_x = torch.randint(bkg_mask_size//2, W - bkg_mask_size//2, size=(1,), device=torch.device('cuda'))
                        mask_center_y = torch.randint(bkg_mask_size//2, H - bkg_mask_size//2, size=(1,), device=torch.device('cuda'))
                        mask_center = torch.stack([mask_center_x, mask_center_y], dim=1)
                        mask_lower_bound = mask_center - bkg_mask_size // 2
                        mask_upper_bound = mask_center + bkg_mask_size // 2
                        mask_lower_bound_x = mask_lower_bound[:, 0]
                        mask_lower_bound_y = mask_lower_bound[:, 1]
                        mask_upper_bound_x = mask_upper_bound[:, 0]
                        mask_upper_bound_y = mask_upper_bound[:, 1]

                        mask_lower_bound_x = torch.clamp(mask_lower_bound_x, 0, W-1) # each shape 1
                        mask_upper_bound_x = torch.clamp(mask_upper_bound_x, 0, W-1)
                        mask_lower_bound_y = torch.clamp(mask_lower_bound_y, 0, H-1)
                        mask_upper_bound_y = torch.clamp(mask_upper_bound_y, 0, H-1)

                        # do the additional masking
                        mask_b[:, :, mask_lower_bound_y:mask_upper_bound_y, mask_lower_bound_x:mask_upper_bound_x] = 0
                        depth_b[:, :, mask_lower_bound_y:mask_upper_bound_y, mask_lower_bound_x:mask_upper_bound_x] = max_dist
                        # set to a large value, i.e. mask out these area

                    xyz_new = depth2pointcloud(depth_b, pix_T_cam[b:b+1]) # 1 x N x 3
                    inbound_new = get_pts_inbound_lrt(xyz_new, lrtlist[b:b+1, n]) # 1 x N
                    inb_pts_cnt_new = torch.sum(inbound_new)

                    # print(inb_pts_cnt_new)

                    if (inb_pts_cnt_new < inb_pts_cnt and
                        inb_pts_cnt_new > (inb_pts_cnt / 8.0) and
                        inb_pts_cnt_new >= 3): # if we occlude part but not all of the obj, they we are good
                        depth[b:b+1] = depth_b
                        mask[b:b+1] = mask_b
                        # all good
                        break
                

        # convert back to pointcloud
        xyz_new = depth2pointcloud(depth[b:b+1], pix_T_cam[b:b+1]) # 1 x N x 3
        xyz_new_s[b:b+1] = xyz_new

    return xyz_new_s, mask

def random_pointcloud_occlusion(xyz, pix_T_cam, H, W, mask_min, mask_max, n_occs=5, occ_prob=0.5, occlude_bkg_too=False):
    # with occ_prob, we create a random mask. else no operation
    num_try = 10
    max_dist = 200.0
    # lrtlist is B x 19
    B, N, D = list(xyz.shape)
    depth, valid = create_depth_image(pix_T_cam, xyz, H, W) # B x 1 x H x W

    # we create a mask around the center of the box
    xyz_new_s = torch.zeros(B, H*W, 3, device=torch.device('cuda'))

    xyz_rem = xyz.clone()

    mask = torch.ones_like(depth)

    for b in range(B):
        mask_b = torch.ones_like(depth[b:b+1].clone())
        for n in range(n_occs):
            if np.random.uniform() > occ_prob:
                continue

            # random mask size
            mask_size = np.random.randint(10, 100)

            # random mask location
            mask_lower_bound_x = torch.randint(0, W+1-mask_size, size=(1,), device=torch.device('cuda'))
            mask_lower_bound_y = torch.randint(0, H+1-mask_size, size=(1,), device=torch.device('cuda'))
            mask_upper_bound_x = mask_lower_bound_x + mask_size
            mask_upper_bound_y = mask_lower_bound_y + mask_size

            # do the masking
            depth_b = depth[b:b+1].clone() # 1 x 1 x H x W
            mask_b[:, :, mask_lower_bound_y:mask_upper_bound_y, mask_lower_bound_x:mask_upper_bound_x] = 0
            depth_b[:, :, mask_lower_bound_y:mask_upper_bound_y, mask_lower_bound_x:mask_upper_bound_x] = max_dist

            if occlude_bkg_too:
                bkg_mask_size = mask_size * 2
                mask_center_x = torch.randint(bkg_mask_size//2, W - bkg_mask_size//2, size=(1,), device=torch.device('cuda'))
                mask_center_y = torch.randint(bkg_mask_size//2, H - bkg_mask_size//2, size=(1,), device=torch.device('cuda'))
                mask_center = torch.stack([mask_center_x, mask_center_y], dim=1)
                mask_lower_bound = mask_center - bkg_mask_size // 2
                mask_upper_bound = mask_center + bkg_mask_size // 2
                mask_lower_bound_x = mask_lower_bound[:, 0]
                mask_lower_bound_y = mask_lower_bound[:, 1]
                mask_upper_bound_x = mask_upper_bound[:, 0]
                mask_upper_bound_y = mask_upper_bound[:, 1]

                mask_lower_bound_x = torch.clamp(mask_lower_bound_x, 0, W-1) # each shape 1
                mask_upper_bound_x = torch.clamp(mask_upper_bound_x, 0, W-1)
                mask_lower_bound_y = torch.clamp(mask_lower_bound_y, 0, H-1)
                mask_upper_bound_y = torch.clamp(mask_upper_bound_y, 0, H-1)

                # do the additional masking
                mask_b[:, :, mask_lower_bound_y:mask_upper_bound_y, mask_lower_bound_x:mask_upper_bound_x] = 0
                depth_b[:, :, mask_lower_bound_y:mask_upper_bound_y, mask_lower_bound_x:mask_upper_bound_x] = max_dist
                # set to a large value, i.e. mask out these area

            depth[b:b+1] = depth_b
            mask[b:b+1] = mask_b

            xyz_new = depth2pointcloud(depth_b, pix_T_cam[b:b+1]) # 1 x N x 3   

        # convert back to pointcloud
        xyz_new = depth2pointcloud(depth[b:b+1], pix_T_cam[b:b+1]) # 1 x N x 3
        xyz_new_s[b:b+1] = xyz_new

    return xyz_new_s, mask



def random_image_occlusion(B, H, W, size_min=0.25, size_max=1.0, occ_prob=0.5):
    # with occ_prob, we create a random mask. else no operation
    
    mask = torch.ones((B, 1, H, W), dtype=torch.float32)

    for b in range(B):
        if np.random.uniform() < occ_prob:
            ymin = torch.randint(low=0, high=H-1, size=[])
            xmin = torch.randint(low=0, high=W-1, size=[])
            size_factor = np.random.uniform(size_min, size_max)
            ymax = (ymin + size_factor*H).long().clamp(0, H-1)
            xmax = (xmin + size_factor*W).long().clamp(0, W-1)
            mask[:,:,ymin:ymax,xmin:xmax] = 0.0
    return mask
    
def random_border_occlusion(B, H, W, size_min=1, size_max=50, occ_prob=0.5):
    # with occ_prob, we create a random mask. else no operation
    
    mask = torch.ones((B, 1, H, W), dtype=torch.float32)

    for b in range(B):
        # top
        if np.random.uniform() < occ_prob:
            size = torch.randint(low=size_min, high=size_max, size=[]).long()
            mask[:,:,0:size] = 0.0
        # left
        if np.random.uniform() < occ_prob:
            size = torch.randint(low=size_min, high=size_max, size=[]).long()
            mask[:,:,:,0:size] = 0.0
        # bottom
        if np.random.uniform() < occ_prob:
            size = torch.randint(low=size_min, high=size_max, size=[]).long()
            mask[:,:,-size:] = 0.0
        # right
        if np.random.uniform() < occ_prob:
            size = torch.randint(low=size_min, high=size_max, size=[]).long()
            mask[:,:,:,-size:] = 0.0
    return mask
    
    
def farthest_point_sample(xyz, npoint, deterministic=False):
    """
    Input:
        xyz: pointcloud data, [B, N, C], where C is probably 3
        npoint: number of samples
    Return:
        inds: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    inds = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if deterministic:
        farthest = torch.randint(0, 1, (B,), dtype=torch.long).to(device)
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        inds[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return inds

def get_nearest_xyz_from_xy(xy, xy_pix, xyz_cam):
    B, N, D = xy.shape
    assert(D==2)
    B2, N2, D2 = xy_pix.shape
    B3, N3, D3 = xyz_cam.shape
    assert(B==B2)
    assert(B2==B3)
    assert(N2==N3)
    assert(D2==2)
    assert(D3==3)

    device = xy.device
    # now, for each keypoint, i want its nearest neighbor in xy_pix
    xyz = torch.zeros((B, N, 3), dtype=torch.float32, device=device)
    for b in range(B):
        for n in range(N):
            xy_ = xy[b,n:n+1] # 1 x 2
            dist = torch.norm(xy_pix[b] - xy_, dim=1) # M
            ind = torch.argmin(dist)
            xyz_ = xyz_cam[b,ind]
            xyz[b,n:n+1] = xyz_
    return xyz
    # print_('xyz', xyz)

def fill_ray_single(xyz, target_T_given=None, ray_add=0.0, num_points=100):
    # xyz is N x 3, and in cam coords
    # we want to return a big fake pointcloud with points along the rays leading to each point

    # target_T_given, if it exists, takes us to the coords we want to be in; it is 4 x 4

    xyz = torch.reshape(xyz, (-1, 3))
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    # these are N

    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    z = z.unsqueeze(1)
    # these are N x 1

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

    # for each proportional distance in [0.0, 1.0], generate a new hypotenuse
    dists = torch.linspace(0.0, 1.0, num_points, device=xyz.device)
    dists = torch.reshape(dists, (1, num_points))
    v_ = dists * v.repeat(1, num_points)
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
    xyz = torch.squeeze(xyz, dim=0)
    
    # these are the coordinates we can observe
    return xyz

def fill_ray(xyz, target_T_given=None, ray_add=0.0, num_points=100):
    # xyz is B x N x 3, and in cam coords
    # we want to return a big fake pointcloud with points along the rays leading to each point

    # target_T_given, if it exists, takes us to the coords we want to be in; it is 4 x 4

    B, N, D = xyz.shape
    assert(D==3)
    
    x, y, z = xyz[:,:,0], xyz[:,:,1], xyz[:,:,2]
    # these are B x N

    x = x.unsqueeze(2)
    y = y.unsqueeze(2)
    z = z.unsqueeze(2)
    # these are B x N x 1
    # print('x', x.shape)

    
    # get the hypotenuses
    u = torch.sqrt(x**2+z**2) # flat to ground
    v = torch.sqrt(x**2+y**2+z**2)
    w = torch.sqrt(x**2+y**2)

    # print('v', v.shape)
    
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

    # for each proportional distance in [0.0, 1.0], generate a new hypotenuse
    dists = torch.linspace(0.0, 1.0, num_points, device=xyz.device)
    dists = torch.reshape(dists, (1, 1, num_points))
    v_ = dists * v.repeat(1, 1, num_points)
    v_ = v_ + ray_add

    # print('v_', v_.shape)

    # now, for each of these v_, we want to generate the xyz
    y_ = sin_theta*v_
    u_ = torch.abs(cos_theta*v_)
    z_ = sin_alpha*u_
    x_ = cos_alpha*u_
    # these are the ref coordinates we want to fill

    # print('x_', x_.shape)
    x = x_.reshape(B, N*num_points)
    y = y_.reshape(B, N*num_points)
    z = z_.reshape(B, N*num_points)
    # print('x', x.shape)
    
    # x = x_.flatten()
    # y = y_.flatten()
    # z = z_.flatten()

    xyz = torch.stack([x,y,z], dim=2)
    if target_T_given is not None:
        xyz = utils.geom.apply_4x4(target_T_given, xyz)
    
    # these are the coordinates we can observe
    return xyz

def get_plane_mask(H, W, pix_T_cam, xyz_cam, max_dist=1):
    B, N, D = list(xyz_cam.shape)
    assert(D==3)
    device = xyz_cam.device
    plane_mask = torch.zeros((B, 1, H, W)).to(device)
    for b in range(B):
        max_p0 = None
        max_norm = None

        xyz_cam_ = xyz_cam[b].clone() # N x 3

        z_cam_ = xyz_cam_[:,2]
        xyz_cam_ = xyz_cam_[z_cam_ < max_dist]

        N = xyz_cam_.shape[0]

        # do ransac
        num_tries = 100
        dist_thresh = 0.001

        max_num_inliers = 0

        for _ in range(num_tries):
            rand_id = (torch.randperm(N)[:3]).to(device)
            pts_to_fit = xyz_cam_[rand_id, :] # 3 x 3
            p0, p1, p2 = torch.unbind(pts_to_fit, dim=0) # each is a point
            plane_norm = torch.cross(p0-p1, p0-p2) # norm of the plane 
            plane_norm = plane_norm / torch.norm(plane_norm) # normalize it

            # compute the dist of all points to our plane
            vec_t_p0 = xyz_cam_ - p0.reshape(1, 3) # N x 3
            # take dot product with the norm

            dist_to_plane = torch.abs(torch.matmul(vec_t_p0, plane_norm.reshape(3, 1)).squeeze(1)) # N

            # count the number of inliers
            num_inliers = torch.sum(dist_to_plane < dist_thresh)
            if num_inliers > max_num_inliers: # update
                max_num_inliers = num_inliers
                max_p0 = p0
                max_norm = plane_norm

        # convert the inliers into a mask
        vec_t_p0 = xyz_cam_ - max_p0.reshape(1, 3) # N x 3
        # take dot product with the norm
        dist_to_plane = torch.abs(torch.matmul(vec_t_p0, max_norm.reshape(3, 1)).squeeze(1)) # N

        xyz_cam_inl = xyz_cam_[dist_to_plane < dist_thresh, :] # num_inl x 3
        # project into 2d

        xy_inl = utils.geom.camera2pixels(xyz_cam_inl.unsqueeze(0), pix_T_cam[b:b+1]).squeeze(0) # num_inl x 2
        x0, y0 = torch.unbind(xy_inl, dim=1) # N
        x0 = x0.clamp(0,W-1)
        y0 = y0.clamp(0,H-1)
        plane_mask[b, :, torch.round(y0).long(), torch.round(x0).long()] = 1.0

    return plane_mask

def get_plane_inds(xyz_cam, dist_thresh=0.001):
    B, N, D = list(xyz_cam.shape)
    assert(D==3)
    device = xyz_cam.device
    assert(B==1)

    plane_inds = []
    for b in range(B):
        max_p0 = None
        max_norm = None

        xyz_cam_ = xyz_cam[b].clone() # N x 3

        N = xyz_cam_.shape[0]

        # do ransac
        num_tries = 100

        max_num_inliers = 0

        for _ in range(num_tries):
            rand_id = (torch.randperm(N)[:3]).to(device)
            pts_to_fit = xyz_cam_[rand_id, :] # 3 x 3
            p0, p1, p2 = torch.unbind(pts_to_fit, dim=0) # each is a point
            plane_norm = torch.cross(p0-p1, p0-p2) # norm of the plane 
            plane_norm = plane_norm / torch.norm(plane_norm) # normalize it

            # compute the dist of all points to our plane
            vec_t_p0 = xyz_cam_ - p0.reshape(1, 3) # N x 3
            # take dot product with the norm

            dist_to_plane = torch.abs(torch.matmul(vec_t_p0, plane_norm.reshape(3, 1)).squeeze(1)) # N

            # count the number of inliers
            num_inliers = torch.sum(dist_to_plane < dist_thresh)
            if num_inliers > max_num_inliers: # update
                max_num_inliers = num_inliers
                max_p0 = p0
                max_norm = plane_norm

        # convert the inliers into a mask
        vec_t_p0 = xyz_cam_ - max_p0.reshape(1, 3) # N x 3
        # take dot product with the norm
        dist_to_plane = torch.abs(torch.matmul(vec_t_p0, max_norm.reshape(3, 1)).squeeze(1)) # N

        xyz_cam_inl = xyz_cam_[dist_to_plane < dist_thresh, :] # num_inl x 3
        # project into 2d

        inds = (dist_to_plane < dist_thresh).reshape(-1)
        plane_inds.append(inds)
    plane_inds = torch.stack(plane_inds, dim=0)

    return plane_inds

def get_random_affine_2d(B, rot_min=-5.0, rot_max=5.0, tx_min=-0.1, tx_max=0.1, ty_min=-0.1, ty_max=0.1, sx_min=-0.05, sx_max=0.05, sy_min=-0.05, sy_max=0.05, shx_min=-0.05, shx_max=0.05, shy_min=-0.05, shy_max=0.05, square_amount=0.75):
    '''
    Params:
        rot_min: rotation amount min
        rot_max: rotation amount max

        tx_min: translation x min
        tx_max: translation x max

        ty_min: translation y min
        ty_max: translation y max

        sx_min: scaling x min
        sx_max: scaling x max

        sy_min: scaling y min
        sy_max: scaling y max

        shx_min: shear x min
        shx_max: shear x max

        shy_min: shear y min
        shy_max: shear y max

    Returns:
        transformation matrix: (B, 3, 3)
    '''
    # rotation
    if rot_max - rot_min != 0:
        rot_amount = np.random.uniform(low=rot_min, high=rot_max, size=B)
        rot_amount = np.pi/180.0*rot_amount
    else:
        rot_amount = rot_min
    rotation = np.zeros((B, 3, 3)) # B, 3, 3
    rotation[:, 2, 2] = 1
    rotation[:, 0, 0] = np.cos(rot_amount)
    rotation[:, 0, 1] = -np.sin(rot_amount)
    rotation[:, 1, 0] = np.sin(rot_amount)
    rotation[:, 1, 1] = np.cos(rot_amount)

    # translation
    translation = np.zeros((B, 3, 3)) # B, 3, 3
    translation[:, [0,1,2], [0,1,2]] = 1 
    if tx_max - tx_min != 0:
        trans_x = np.random.uniform(low=tx_min, high=tx_max, size=B)
        translation[:, 0, 2] = trans_x
    else:
        translation[:, 0, 2] = tx_max
    if ty_max - ty_min != 0:
        trans_y = np.random.uniform(low=ty_min, high=ty_max, size=B)
        translation[:, 1, 2] = trans_y
    else:
        translation[:, 1, 2] = ty_max

    # scaling
    scaling = np.zeros((B, 3, 3)) # B, 3, 3
    scaling[:, [0,1,2], [0,1,2]] = 1 
    if sx_max - sx_min != 0:
        scale_x = 1 + np.random.uniform(low=sx_min, high=sx_max, size=B)
        scaling[:, 0, 0] = scale_x
    else:
        scaling[:, 0, 0] = sx_max
    if sy_max - sy_min != 0:
        scale_y = 1 + np.random.uniform(low=sy_min, high=sy_max, size=B)
        scaling[:, 1, 1] = scale_y
    else:
        scaling[:, 1, 1] = sy_max

    if square_amount > 0:
        # take things closer to square
        mean_scaling = (scaling[:,0,0] + scaling[:,1,1])/2.0
        scaling[:,0,0] = mean_scaling*square_amount + (1.0-square_amount)*scaling[:,0,0]
        scaling[:,1,1] = mean_scaling*square_amount + (1.0-square_amount)*scaling[:,1,1]

    # shear
    shear = np.zeros((B, 3, 3)) # B, 3, 3
    shear[:, [0,1,2], [0,1,2]] = 1 
    if shx_max - shx_min != 0:
        shear_x = np.random.uniform(low=shx_min, high=shx_max, size=B)
        shear[:, 0, 1] = shear_x
    else:
        shear[:, 0, 1] = shx_max
    if shy_max - shy_min != 0:
        shear_y = np.random.uniform(low=shy_min, high=shy_max, size=B)
        shear[:, 1, 0] = shear_y
    else:
        shear[:, 1, 0] = shy_max

    # compose all those
    rt = np.einsum("ijk,ikl->ijl", rotation, translation)
    ss = np.einsum("ijk,ikl->ijl", scaling, shear)
    trans = np.einsum("ijk,ikl->ijl", rt, ss)

    return trans

def convert_box_to_theta(box_mem, Z, Y, X):
    B, D = box_mem.shape
    assert(D==9)
    x, y, z, w, h, d, rx, ry, rz = torch.unbind(box_mem, axis=1)
    
    # we'll ignore rotations for now
    
    # we want things normalized to [0,1]
    x = x/(X-1)
    y = y/(Y-1)
    z = z/(Z-1)
    w = w/(X-1)
    h = h/(Y-1)
    d = d/(Z-1)
    theta = torch.zeros((B, 3, 4), dtype=torch.float32, device='cuda')
    theta[:,0,0] = 1/w
    theta[:,1,1] = 1/h
    theta[:,2,2] = 1/d
    theta[:,0,3] = -2*(x-0.5)/w
    theta[:,1,3] = -2*(y-0.5)/h
    theta[:,2,3] = -2*(z-0.5)/d
    return theta
    
