import torch
import torchvision.transforms
import cv2
import os
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import numpy as np
from matplotlib import cm
# import hyperparams as hyp
import matplotlib
# import imageio
from itertools import combinations
from tensorboardX import SummaryWriter
from sklearn.decomposition import PCA
# import utils.geom
# import utils.py
# import utils.basic
import torch.nn.functional as F

import utils.py
import utils.basic

from PIL import Image
import io
import matplotlib.pyplot as plt
EPS = 1e-6

# color conversion libs, for flow vis
from skimage.color import (
    rgb2lab, rgb2yuv, rgb2ycbcr, lab2rgb, yuv2rgb, ycbcr2rgb,
    rgb2hsv, hsv2rgb, rgb2xyz, xyz2rgb, rgb2hed, hed2rgb)

def _convert(input_, type_):
    return {
        'float': input_.float(),
        'double': input_.double(),
    }.get(type_, input_)


def _generic_transform_sk_4d(transform, in_type='', out_type=''):
    def apply_transform(input_):
        to_squeeze = (input_.dim() == 3)
        device = input_.device
        input_ = input_.cpu()
        input_ = _convert(input_, in_type)

        if to_squeeze:
            input_ = input_.unsqueeze(0)

        input_ = input_.permute(0, 2, 3, 1).numpy()
        transformed = transform(input_)
        output = torch.from_numpy(transformed).float().permute(0, 3, 1, 2)
        if to_squeeze:
            output = output.squeeze(0)
        output = _convert(output, out_type)
        return output.to(device)
    return apply_transform


def _generic_transform_sk_3d(transform, in_type='', out_type=''):
    def apply_transform_individual(input_):
        device = input_.device
        input_ = input_.cpu()
        input_ = _convert(input_, in_type)

        input_ = input_.permute(1, 2, 0).detach().numpy()
        transformed = transform(input_)
        output = torch.from_numpy(transformed).float().permute(2, 0, 1)
        output = _convert(output, out_type)
        return output.to(device)

    def apply_transform(input_):
        to_stack = []
        for image in input_:
            to_stack.append(apply_transform_individual(image))
        return torch.stack(to_stack)
    return apply_transform


# # --- Cie*LAB ---
rgb_to_lab = _generic_transform_sk_4d(rgb2lab)
lab_to_rgb = _generic_transform_sk_3d(lab2rgb, in_type='double', out_type='float')
# # --- YUV ---
# rgb_to_yuv = _generic_transform_sk_4d(rgb2yuv)
# yuv_to_rgb = _generic_transform_sk_4d(yuv2rgb)
# # --- YCbCr ---
# rgb_to_ycbcr = _generic_transform_sk_4d(rgb2ycbcr)
# ycbcr_to_rgb = _generic_transform_sk_4d(ycbcr2rgb, in_type='double', out_type='float')
# # --- HSV ---
# rgb_to_hsv = _generic_transform_sk_3d(rgb2hsv)
hsv_to_rgb = _generic_transform_sk_3d(hsv2rgb)
# # --- XYZ ---
# rgb_to_xyz = _generic_transform_sk_4d(rgb2xyz)
# xyz_to_rgb = _generic_transform_sk_3d(xyz2rgb, in_type='double', out_type='float')
# # --- HED ---
# rgb_to_hed = _generic_transform_sk_4d(rgb2hed)
# hed_to_rgb = _generic_transform_sk_3d(hed2rgb, in_type='double', out_type='float')

'''end color conversion in torch'''

def get_n_colors(N, sequential=False):
    label_colors = []
    for ii in range(N):
        # hue = ii/(N-1)
        # # sat = 0.8 + np.random.random()*0.2
        # # lit = 0.5 + np.random.random()*0.5
        # sat = 1.0
        # val = 1.0
        # hsv = np.stack([hue, sat, val])
        # rgb = hsv2rgb(hsv)
        # rgb = (rgb * 255).astype(np.uint8)
        # # somehow this is never giving blues/purples

        if sequential:
            rgb = cm.winter(ii/(N-1))
            rgb = (np.array(rgb) * 255).astype(np.uint8)[:3]
        else:
            rgb = np.zeros(3)
            # while np.sum(rgb) < 64: # ensure min brightness
            while np.sum(rgb) < 128: # ensure min brightness
                rgb = np.random.randint(0,256,3)

            
        label_colors.append(rgb)
    return label_colors


def rgb2lab(rgb):
    # rgb is in -0.5,0.5
    rgb = rgb + 0.5 # put it into [0,1] for my tool
    lab = rgb_to_lab(rgb) # this is in -100,100
    lab = lab / 100.0 # this is in -1,1
    return lab
def lab2rgb(lab):
    # lab is in -1,1
    lab = lab * 100.0 # this is in -100,100
    rgb = lab_to_rgb(lab) # this is in [0,1]
    rgb = rgb - 0.5 # this is in -0.5,0.5
    return rgb


COLORMAP_FILE = "./utils/bremm.png"
class ColorMap2d:
    def __init__(self, filename=None):
        self._colormap_file = filename or COLORMAP_FILE
        self._img = plt.imread(self._colormap_file)
        
        self._height = self._img.shape[0]
        self._width = self._img.shape[1]

    def __call__(self, X):
        assert len(X.shape) == 2

        # self._range_x = (X[:, 0].min(), X[:, 0].max())
        # self._range_y = (X[:, 1].min(), X[:, 1].max())
        output = np.zeros((X.shape[0], 3))
        for i in range(X.shape[0]):
            x, y = X[i, :]
            xp = int(self._width * x)
            yp = int(self._height * y)
            output[i, :] = self._img[yp, xp]
        return output


def preprocess_color_tf(x):
    import tensorflow as tf
    return tf.cast(x,tf.float32) * 1./255 - 0.5

def preprocess_color(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32) * 1./255 - 0.5
    else:
        return x.float() * 1./255 - 0.5

def pca_embed(emb, keep, valid=None):
    ## emb -- [S,H/2,W/2,C]
    ## keep is the number of principal components to keep
    ## Helper function for reduce_emb.
    emb = emb + EPS
    #emb is B x C x H x W
    emb = emb.permute(0, 2, 3, 1).cpu().detach().numpy() #this is B x H x W x C

    if valid:
        valid = valid.cpu().detach().numpy().reshape((H*W))

    emb_reduced = list()

    B, H, W, C = np.shape(emb)
    for img in emb:
        if np.isnan(img).any():
            emb_reduced.append(np.zeros([H, W, keep]))
            continue

        pixels_kd = np.reshape(img, (H*W, C))
        
        if valid:
            pixels_kd_pca = pixels_kd[valid]
        else:
            pixels_kd_pca = pixels_kd

        P = PCA(keep)
        P.fit(pixels_kd_pca)

        if valid:
            pixels3d = P.transform(pixels_kd)*valid
        else:
            pixels3d = P.transform(pixels_kd)

        out_img = np.reshape(pixels3d, [H,W,keep]).astype(np.float32)
        if np.isnan(out_img).any():
            emb_reduced.append(np.zeros([H, W, keep]))
            continue

        emb_reduced.append(out_img)

    emb_reduced = np.stack(emb_reduced, axis=0).astype(np.float32)

    return torch.from_numpy(emb_reduced).permute(0, 3, 1, 2)

def pca_embed_together(emb, keep):
    ## emb -- [S,H/2,W/2,C]
    ## keep is the number of principal components to keep
    ## Helper function for reduce_emb.
    emb = emb + EPS
    #emb is B x C x H x W
    emb = emb.permute(0, 2, 3, 1).cpu().detach().numpy() #this is B x H x W x C

    B, H, W, C = np.shape(emb)
    if np.isnan(emb).any():
        return torch.zeros(B, keep, H, W)
    
    pixelskd = np.reshape(emb, (B*H*W, C))
    P = PCA(keep)
    P.fit(pixelskd)
    pixels3d = P.transform(pixelskd)
    out_img = np.reshape(pixels3d, [B,H,W,keep]).astype(np.float32)
        
    if np.isnan(out_img).any():
        return torch.zeros(B, keep, H, W)
    
    return torch.from_numpy(out_img).permute(0, 3, 1, 2)

def reduce_emb(emb, valid=None, inbound=None, together=False):
    ## emb -- [S,C,H/2,W/2], inbound -- [S,1,H/2,W/2]
    ## Reduce number of chans to 3 with PCA. For vis.
    # S,H,W,C = emb.shape.as_list()
    S, C, H, W = list(emb.size())
    keep = 3

    if together:
        reduced_emb = pca_embed_together(emb, keep)
    else:
        reduced_emb = pca_embed(emb, keep, valid) #not im

    reduced_emb = utils.basic.normalize(reduced_emb) - 0.5
    if inbound is not None:
        emb_inbound = emb*inbound
    else:
        emb_inbound = None

    return reduced_emb, emb_inbound

def get_feat_pca(feat, valid=None):
    B, C, D, W = list(feat.size())
    # feat is B x C x D x W. If 3D input, average it through Height dimension before passing into this function.

    pca, _ = reduce_emb(feat, valid=valid,inbound=None, together=True)
    # pca is B x 3 x W x D
    return pca

def convert_occ_to_height(occ, reduce_axis=3):
    B, C, D, H, W = list(occ.shape)
    assert(C==1)
    # note that height increases DOWNWARD in the tensor
    # (like pixel/camera coordinates)
    
    G = list(occ.shape)[reduce_axis]
    values = torch.linspace(float(G), 1.0, steps=G, dtype=torch.float32, device=occ.device)
    if reduce_axis==2:
        # fro view
        values = values.view(1, 1, G, 1, 1)
    elif reduce_axis==3:
        # top view
        values = values.view(1, 1, 1, G, 1)
    elif reduce_axis==4:
        # lateral view
        values = values.view(1, 1, 1, 1, G)
    else:
        assert(False) # you have to reduce one of the spatial dims (2-4)
    values = torch.max(occ*values, dim=reduce_axis)[0]/float(G)
    # values = values.view([B, C, D, W])
    return values

def gif_and_tile(ims, just_gif=False):
    S = len(ims) 
    # each im is B x H x W x C
    # i want a gif in the left, and the tiled frames on the right
    # for the gif tool, this means making a B x S x H x W tensor
    # where the leftmost part is sequential and the rest is tiled
    gif = torch.stack(ims, dim=1)
    if just_gif:
        return gif
    til = torch.cat(ims, dim=2)
    til = til.unsqueeze(dim=1).repeat(1, S, 1, 1, 1)
    im = torch.cat([gif, til], dim=3)
    return im

def back2color(i, blacken_zeros=False):
    if blacken_zeros:
        const = torch.tensor([-0.5])
        i = torch.where(i==0.0, const.cuda() if i.is_cuda else const, i)
        return back2color(i)
    else:
        return ((i+0.5)*255).type(torch.ByteTensor)

def rgb2bgr(i):
    r = i[:,0]
    g = i[:,1]
    b = i[:,2]
    bgr = torch.stack([b,g,r], dim=1)
    return bgr

def colorize(d):
    # this does not work properly yet
    
    # # d is C x H x W or H x W
    # if d.ndim==3:
    #     d = d.squeeze(dim=0)
    # else:
    #     assert(d.ndim==2)

    if d.ndim==2:
        d = d.unsqueeze(dim=0)
    else:
        assert(d.ndim==3)
    # copy to the three chans
    d = d.repeat(3, 1, 1)
    return d
    
    # d = d.cpu().detach().numpy()
    # # move channels out to last dim
    # # d = np.transpose(d, [0, 2, 3, 1])
    # # d = np.transpose(d, [1, 2, 0])
    # print(d.shape)
    # d = cm.inferno(d)[:, :, 1:] # delete the alpha channel
    # # move channels into dim0
    # d = np.transpose(d, [2, 0, 1])
    # print_stats(d, 'colorize_out')
    # d = torch.from_numpy(d)
    # return d

def seq2color(im, norm=True, colormap='coolwarm'):
    B, S, H, W = list(im.shape)
    # S is sequential

    # prep a mask of the valid pixels, so we can blacken the invalids later
    mask = torch.max(im, dim=1, keepdim=True)[0]

    # turn the S dim into an explicit sequence
    coeffs = np.linspace(1.0, float(S), S).astype(np.float32)/float(S)
    
    # # increase the spacing from the center
    # coeffs[:int(S/2)] -= 2.0
    # coeffs[int(S/2)+1:] += 2.0
    
    coeffs = torch.from_numpy(coeffs).float().cuda()
    coeffs = coeffs.reshape(1, S, 1, 1).repeat(B, 1, H, W)
    # scale each channel by the right coeff
    im = im * coeffs
    # now im is in [1/S, 1], except for the invalid parts which are 0
    # keep the highest valid coeff at each pixel
    im = torch.max(im, dim=1, keepdim=True)[0]

    out = []
    for b in range(B):
        im_ = im[b]
        # move channels out to last dim_
        im_ = im_.detach().cpu().numpy()
        im_ = np.squeeze(im_)
        # im_ is H x W
        if colormap=='coolwarm':
            im_ = cm.coolwarm(im_)[:, :, :3]
        elif colormap=='PiYG':
            im_ = cm.PiYG(im_)[:, :, :3]
        elif colormap=='winter':
            im_ = cm.winter(im_)[:, :, :3]
        elif colormap=='spring':
            im_ = cm.spring(im_)[:, :, :3]
        elif colormap=='onediff':
            im_ = np.reshape(im_, (-1))
            im0_ = cm.spring(im_)[:, :3]
            im1_ = cm.winter(im_)[:, :3]
            im1_[im_==1/float(S)] = im0_[im_==1/float(S)]
            im_ = np.reshape(im1_, (H, W, 3))
        else:
            assert(False) # invalid colormap
        # move channels into dim 0
        im_ = np.transpose(im_, [2, 0, 1])
        im_ = torch.from_numpy(im_).float().cuda()
        out.append(im_)
    out = torch.stack(out, dim=0)
    
    # blacken the invalid pixels, instead of using the 0-color
    out = out*mask
    # out = out*255.0

    # put it in [-0.5, 0.5]
    out = out - 0.5
    
    return out

def oned2inferno(d, norm=True):
    # convert a 1chan input to a 3chan image output

    # if it's just B x H x W, add a C dim
    if d.ndim==3:
        d = d.unsqueeze(dim=1)
    # d should be B x C x H x W, where C=1
    B, C, H, W = list(d.shape)
    assert(C==1)

    if norm:
        d = utils.basic.normalize(d)
        
    rgb = torch.zeros(B, 3, H, W)
    for b in list(range(B)):
        rgb[b] = colorize(d[b])

    rgb = (255.0*rgb).type(torch.ByteTensor)

    # rgb = tf.cast(255.0*rgb, tf.uint8)
    # rgb = tf.reshape(rgb, [-1, hyp.H, hyp.W, 3])
    # rgb = tf.expand_dims(rgb, axis=0)
    return rgb

def xy2mask(xy, H, W, norm=False):
    # xy is B x N x 2, in either pixel coords or normalized coordinates (depending on norm)
    # returns a mask shaped B x 1 x H x W, with a 1 at each specified xy
    B = list(xy.shape)[0]
    if norm:
        # convert to pixel coords
        x, y = torch.unbind(xy, axis=2)
        x = x*float(W)
        y = y*float(H)
        xy = torch.stack(xy, axis=2)
        
    mask = torch.zeros([B, 1, H, W], dtype=torch.float32, device=torch.device('cuda'))
    for b in list(range(B)):
        mask[b] = xy2mask_single(xy[b], H, W)
    return mask

def xy2masks(xy, H, W, norm=False):
    # xy is B x N x 2, in either pixel coords or normalized coordinates (depending on norm)
    # returns masks, shaped B x N x H x W
    
    B, N, D = list(xy.shape)
    assert(D==2)
    if norm:
        # the xy's are normalized; convert them to pixel coords
        x, y = torch.unbind(xy, axis=2)
        x = x*float(W)
        y = y*float(H)
        xy = torch.stack(xy, axis=2)
        
    mask = torch.zeros([B, N, H, W], dtype=torch.float32, device=torch.device('cuda'))
    for b in list(range(B)):
        for n in list(range(N)):
            mask[b,n:n+1] = xy2mask_single(xy[b,n:n+1], H, W)
    return mask

def xy2mask_single(xy, H, W):
    # xy is N x 2
    x, y = torch.unbind(xy, axis=1)
    x = x.long()
    y = y.long()

    x = torch.clamp(x, 0, W-1)
    y = torch.clamp(y, 0, H-1)
    
    inds = utils.basic.sub2ind(H, W, y, x)

    valid = (inds > 0).byte() & (inds < H*W).byte()
    inds = inds[torch.where(valid)]

    mask = torch.zeros(H*W, dtype=torch.float32, device=torch.device('cuda'))
    mask[inds] = 1.0
    mask = torch.reshape(mask, [1,H,W])
    return mask

def xy2heatmap(xy, sigma, grid_xs, grid_ys, norm=False):
    # xy is B x N x 2, containing float x and y coordinates of N things
    # grid_xs and grid_ys are B x N x Y x X

    B, N, Y, X = list(grid_xs.shape)

    mu_x = xy[:,:,0].clone()
    mu_y = xy[:,:,1].clone()

    x_valid = (mu_x>-0.5) & (mu_x<float(X+0.5))
    y_valid = (mu_y>-0.5) & (mu_y<float(Y+0.5))
    not_valid = ~(x_valid & y_valid)

    mu_x[not_valid] = -10000
    mu_y[not_valid] = -10000

    mu_x = mu_x.reshape(B, N, 1, 1).repeat(1, 1, Y, X)
    mu_y = mu_y.reshape(B, N, 1, 1).repeat(1, 1, Y, X)

    sigma_sq = sigma*sigma
    # sigma_sq = (sigma*sigma).reshape(B, N, 1, 1)
    sq_diff_x = (grid_xs - mu_x)**2
    sq_diff_y = (grid_ys - mu_y)**2

    term1 = 1./2.*np.pi*sigma_sq
    term2 = torch.exp(-(sq_diff_x+sq_diff_y)/(2.*sigma_sq))
    gauss = term1*term2

    if norm:
        # normalize so each gaussian peaks at 1
        gauss_ = gauss.reshape(B*N, Y, X)
        gauss_ = utils.basic.normalize(gauss_)
        gauss = gauss_.reshape(B, N, Y, X)

    return gauss
    
def xyz2heatmap(xyz, sigma, grid_xs, grid_ys, grid_zs, norm=False):
    # xyz is B x N x 3, containing float x and y coordinates of N things
    # grid_xs and grid_ys and grid_zs are B x N x Z x Y x X

    B, N, Z, Y, X = list(grid_xs.shape)

    mu_x = xyz[:,:,0].clone()
    mu_y = xyz[:,:,1].clone()
    mu_z = xyz[:,:,2].clone()

    x_valid = (mu_x>-0.5) & (mu_x<float(X+0.5))
    y_valid = (mu_y>-0.5) & (mu_y<float(Y+0.5))
    z_valid = (mu_z>-0.5) & (mu_z<float(Z+0.5))
    not_valid = ~(x_valid & y_valid & z_valid)

    mu_x[not_valid] = -10000
    mu_y[not_valid] = -10000
    mu_z[not_valid] = -10000

    mu_x = mu_x.reshape(B, N, 1, 1, 1).repeat(1, 1, Z, Y, X)
    mu_y = mu_y.reshape(B, N, 1, 1, 1).repeat(1, 1, Z, Y, X)
    mu_z = mu_z.reshape(B, N, 1, 1, 1).repeat(1, 1, Z, Y, X)

    sigma_sq = sigma*sigma
    sq_diff_x = (grid_xs - mu_x)**2
    sq_diff_y = (grid_ys - mu_y)**2
    sq_diff_z = (grid_zs - mu_z)**2

    term1 = 1./2.*np.pi*sigma_sq
    term2 = torch.exp(-(sq_diff_x+sq_diff_y+sq_diff_z)/(2.*sigma_sq))
    gauss = term1*term2

    if norm:
        # normalize so each gaussian peaks at 1
        gauss_ = gauss.reshape(B*N, Z, Y, X)
        gauss_ = utils.basic.normalize(gauss_)
        gauss = gauss_.reshape(B, N, Z, Y, X)

    return gauss
    
def xy2heatmaps(xy, Y, X, sigma=30.0):
    # xy is B x N x 2

    B, N, D = list(xy.shape)
    assert(D==2)

    device = xy.device
    
    grid_y, grid_x = utils.basic.meshgrid2d(B, Y, X, device=device)
    # grid_x and grid_y are B x Y x X
    grid_xs = grid_x.unsqueeze(1).repeat(1, N, 1, 1)
    grid_ys = grid_y.unsqueeze(1).repeat(1, N, 1, 1)
    heat = xy2heatmap(xy, sigma, grid_xs, grid_ys, norm=True)
    return heat

def xyz2heatmaps(xyz, Z, Y, X, sigma=30.0):
    # xyz is B x N x 2

    B, N, D = list(xyz.shape)
    assert(D==3)

    device = xyz.get_device()
    
    grid_z, grid_y, grid_x = utils.basic.meshgrid3d(B, Z, Y, X)#, device=device)
    # grid_x and grid_y are B x Y x X
    grid_xs = grid_x.unsqueeze(1).repeat(1, N, 1, 1, 1)
    grid_ys = grid_y.unsqueeze(1).repeat(1, N, 1, 1, 1)
    grid_zs = grid_z.unsqueeze(1).repeat(1, N, 1, 1, 1)
    heat = xyz2heatmap(xyz, sigma, grid_xs, grid_ys, grid_zs, norm=True)
    return heat

def draw_circles_at_xy(xy, Y, X, sigma=12.5):
    B, N, D = list(xy.shape)
    assert(D==2)
    prior = xy2heatmaps(xy, Y, X, sigma=sigma)
    # prior is B x N x Y x X
    prior = (prior > 0.5).float()
    return prior


def draw_rect_on_image(rgb_torch, box, scale,negative= False):
    assert(False) # this requires hyp for some reason 
    C, H, W = list(rgb_torch.shape)
    assert(C==3)
    rgb_torch = back2color(rgb_torch)

    box = np.array([int(i) for i in box])

    rgb = rgb_torch.cpu().numpy()

    rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    start_point = box*scale
    end_point = start_point + hyp.max.searchRegion*scale 
    
    if negative:
        # red
        color = (0, 255, 0)     
    else:
        # blue
        color = (255, 0, 0) 

    thickness = 0

    rgb = rgb.astype(np.uint8)
    rgb = cv2.rectangle(rgb, tuple(start_point), tuple(end_point), color, thickness) 


    out = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
    out = torch.from_numpy(out).type(torch.ByteTensor).permute(2, 0, 1)
    out = torch.unsqueeze(out, dim=0)
    out = preprocess_color(out)
    out = torch.reshape(out, [1, C, H, W])
    return out

def draw_frame_id_on_vis(vis, frame_id, scale=0.5, left=5, top=20):

    rgb = vis.detach().cpu().numpy()[0]
    rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) 
    color = (255, 255, 255)
    # print('putting frame id', frame_id)

    frame_str = utils.basic.strnum(frame_id)
    
    cv2.putText(
        rgb,
        frame_str,
        (left, top), # from left, from top
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, # font scale (float)
        color, 
        1) # font thickness (int)
    rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
    vis = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return vis




class Summ_writer(object):
    def __init__(self, writer, global_step, log_freq=10, fps=8, scalar_freq=100, just_gif=False):
        self.writer = writer
        self.global_step = global_step
        self.log_freq = log_freq
        self.fps = fps
        self.just_gif = just_gif
        self.maxwidth = 10000
        self.save_this = (self.global_step % self.log_freq == 0)
        self.scalar_freq = max(scalar_freq,1)
        

    def summ_gif(self, name, tensor, blacken_zeros=False):
        # tensor should be in B x S x C x H x W
        
        assert tensor.dtype in {torch.uint8,torch.float32}
        shape = list(tensor.shape)

        if tensor.dtype == torch.float32:
            tensor = back2color(tensor, blacken_zeros=blacken_zeros)

        video_to_write = tensor[0:1]

        S = video_to_write.shape[1]
        if S==1:
            # video_to_write is 1 x 1 x C x H x W
            self.writer.add_image(name, video_to_write[0,0], global_step=self.global_step)
        else:
            self.writer.add_video(name, video_to_write, fps=self.fps, global_step=self.global_step)
            
        return video_to_write

    def summ_rgbs(self, name, ims, frame_ids=None, blacken_zeros=False, only_return=False):
        if self.save_this:

            ims = gif_and_tile(ims, just_gif=self.just_gif)
            vis = ims

            assert vis.dtype in {torch.uint8,torch.float32}

            if vis.dtype == torch.float32:
                vis = back2color(vis, blacken_zeros)           

            B, S, C, H, W = list(vis.shape)

            if frame_ids is not None:
                assert(len(frame_ids)==S)
                for s in range(S):
                    vis[:,s] = draw_frame_id_on_vis(vis[:,s], frame_ids[s])

            if int(W) > self.maxwidth:
                vis = vis[:,:,:,:self.maxwidth]

            if only_return:
                return vis
            else:
                return self.summ_gif(name, vis, blacken_zeros)

    def summ_rgb(self, name, ims, blacken_zeros=False, frame_id=None, only_return=False, halfres=False):
        if self.save_this:
            assert ims.dtype in {torch.uint8,torch.float32}

            if ims.dtype == torch.float32:
                ims = back2color(ims, blacken_zeros)

            #ims is B x C x H x W
            vis = ims[0:1] # just the first one
            B, C, H, W = list(vis.shape)

            if halfres:
                vis = F.interpolate(vis, scale_factor=0.5)

            if frame_id is not None:
                vis = draw_frame_id_on_vis(vis, frame_id)

            if int(W) > self.maxwidth:
                vis = vis[:,:,:,:self.maxwidth]

            if only_return:
                return vis
            else:
                return self.summ_gif(name, vis.unsqueeze(1), blacken_zeros)

    def summ_occs(self, name, occs, bev=False, fro=False, reduce_axes=[3], frame_ids=None):
        if self.save_this:
            B, C, D, H, W = list(occs[0].shape)
            if bev:
                reduce_axes = [3]
            elif fro:
                reduce_axes = [2]
            for reduce_axis in reduce_axes:
                heights = [convert_occ_to_height(occ, reduce_axis=reduce_axis) for occ in occs]
                self.summ_oneds(name=('%s_ax%d' % (name, reduce_axis)), ims=heights, norm=False, frame_ids=frame_ids)
            
    def summ_occ(self, name, occ, reduce_axes=[3], bev=False, fro=False, pro=False, frame_id=None, only_return=False):
        if self.save_this:
            B, C, D, H, W = list(occ.shape)
            if bev:
                reduce_axes = [3]
            elif fro:
                reduce_axes = [2]
            elif pro:
                reduce_axes = [4]
            for reduce_axis in reduce_axes:
                height = convert_occ_to_height(occ, reduce_axis=reduce_axis)
                # if only_return:
                #     return height
                if reduce_axis == reduce_axes[-1]:
                    return self.summ_oned(name=('%s_ax%d' % (name, reduce_axis)), im=height, norm=False, frame_id=frame_id, only_return=only_return)
                else:
                    self.summ_oned(name=('%s_ax%d' % (name, reduce_axis)), im=height, norm=False, frame_id=frame_id, only_return=only_return)

    def summ_oneds(self, name, ims, frame_ids=None, bev=False, fro=False, logvis=False, reduce_max=False, max_val=0.0, norm=True, only_return=False):
        if self.save_this:
            if bev: 
                B, C, H, _, W = list(ims[0].shape)
                if reduce_max:
                    ims = [torch.max(im, dim=3)[0] for im in ims]
                else:
                    ims = [torch.mean(im, dim=3) for im in ims]
            elif fro: 
                B, C, _, H, W = list(ims[0].shape)
                if reduce_max:
                    ims = [torch.max(im, dim=2)[0] for im in ims]
                else:
                    ims = [torch.mean(im, dim=2) for im in ims]


            if len(ims) != 1: # sequence
                im = gif_and_tile(ims, just_gif=self.just_gif)
            else:
                im = torch.stack(ims, dim=1) # single frame

            B, S, C, H, W = list(im.shape)
            
            if logvis and max_val:
                max_val = np.log(max_val)
                im = torch.log(torch.clamp(im, 0)+1.0)
                im = torch.clamp(im, 0, max_val)
                im = im/max_val
                norm = False
            elif max_val:
                im = torch.clamp(im, 0, max_val)
                im = im/max_val
                norm = False
                
            if norm:
                # normalize before oned2inferno,
                # so that the ranges are similar within B across S
                im = utils.basic.normalize(im)

            im = im.view(B*S, C, H, W)
            vis = oned2inferno(im, norm=norm)
            vis = vis.view(B, S, 3, H, W)

            if frame_ids is not None:
                assert(len(frame_ids)==S)
                for s in range(S):
                    vis[:,s] = draw_frame_id_on_vis(vis[:,s], frame_ids[s])

            if W > self.maxwidth:
                vis = vis[...,:self.maxwidth]

            if only_return:
                return vis
            else:
                self.summ_gif(name, vis)

    def summ_oned(self, name, im, bev=False, fro=False, logvis=False, max_val=0, max_along_y=False, norm=True, frame_id=None, only_return=False):
        if self.save_this:

            if bev: 
                B, C, H, _, W = list(im.shape)
                if max_along_y:
                    im = torch.max(im, dim=3)[0]
                else:
                    im = torch.mean(im, dim=3)
            elif fro:
                B, C, _, H, W = list(im.shape)
                if max_along_y:
                    im = torch.max(im, dim=2)[0]
                else:
                    im = torch.mean(im, dim=2)
            else:
                B, C, H, W = list(im.shape)
                
            im = im[0:1] # just the first one
            assert(C==1)
            
            if logvis and max_val:
                max_val = np.log(max_val)
                im = torch.log(im)
                im = torch.clamp(im, 0, max_val)
                im = im/max_val
                norm = False
            elif max_val:
                im = torch.clamp(im, 0, max_val)/max_val
                norm = False

            vis = oned2inferno(im, norm=norm)
            # vis = vis.view(B, 3, H, W)
            if W > self.maxwidth:
                vis = vis[...,:self.maxwidth]
            # self.writer.add_images(name, vis, global_step=self.global_step, dataformats='NCHW')
            return self.summ_rgb(name, vis, blacken_zeros=False, frame_id=frame_id, only_return=only_return)
            # writer.add_images(name + "_R", vis[:,0:1], global_step=global_step, dataformats='NCHW')
            # writer.add_images(name + "_G", vis[:,1:2], global_step=global_step, dataformats='NCHW')
            # writer.add_images(name + "_B", vis[:,2:3], global_step=global_step, dataformats='NCHW')

    def summ_unps(self, name, unps, occs):
        if self.save_this:
            unps = torch.stack(unps, dim=1)
            occs = torch.stack(occs, dim=1)
            B, S, C, D, H, W = list(unps.shape)
            occs = occs.repeat(1, 1, C, 1, 1, 1)
            unps = utils.basic.reduce_masked_mean(unps, occs, dim=4)
            unps = torch.unbind(unps, dim=1) #should be S x B x W x D x C
            # unps = [unp.transpose(1, 2) for unp in unps] #rotate 90 degree counter-clockwise
            # return self.summ_rgbs(name=name, ims=unps, blacken_zeros=True) 
            return self.summ_rgbs(name=name, ims=unps, blacken_zeros=False) 


    def summ_unp(self, name, unp, occ, bev=False, fro=False, only_return=False):
        if self.save_this:
            B, C, D, H, W = list(unp.shape)
            occ = occ.repeat(1, C, 1, 1, 1)
            if bev:
                reduce_axis = 3
            elif fro:
                reduce_axis = 2
            else:
                # default to bev
                reduce_axis = 3
                
            unp = utils.basic.reduce_masked_mean(unp, occ, dim=reduce_axis)
            # unp = [unp.transpose(1, 2) for unp in unp] #rotate 90 degree counter-clockwise
            return self.summ_rgb(name=name, ims=unp, blacken_zeros=True, only_return=only_return)

    def summ_feats(self, name, feats, valids=None, pca=True, fro=False, only_return=False, frame_ids=None):
        if self.save_this:
            if valids is not None:
                valids = torch.stack(valids, dim=1)
            
            feats  = torch.stack(feats, dim=1)
            # feats leads with B x S x C

            if feats.ndim==6:

                # feats is B x S x C x D x H x W
                if fro:
                    reduce_dim = 3
                else:
                    reduce_dim = 4
                    
                if valids is None:
                    feats = torch.mean(feats, dim=reduce_dim)
                else: 
                    valids = valids.repeat(1, 1, feats.size()[2], 1, 1, 1)
                    feats = utils.basic.reduce_masked_mean(feats, valids, dim=reduce_dim)

            B, S, C, D, W = list(feats.size())

            if not pca:
                # feats leads with B x S x C
                feats = torch.mean(torch.abs(feats), dim=2, keepdims=True)
                # feats leads with B x S x 1
                
                # feats is B x S x D x W
                feats = torch.unbind(feats, dim=1)
                # feats is a len=S list, each element of shape B x W x D
                # # make "forward" point up, and make "right" point right
                # feats = [feat.transpose(1, 2) for feat in feats]
                return self.summ_oneds(name=name, ims=feats, norm=True, only_return=only_return, frame_ids=frame_ids)

            else:
                __p = lambda x: utils.basic.pack_seqdim(x, B)
                __u = lambda x: utils.basic.unpack_seqdim(x, B)

                feats_  = __p(feats)
                
                if valids is None:
                    feats_pca_ = get_feat_pca(feats_)
                else:
                    valids_ = __p(valids)
                    feats_pca_ = get_feat_pca(feats_, valids)

                feats_pca = __u(feats_pca_)

                return self.summ_rgbs(name=name, ims=torch.unbind(feats_pca, dim=1), only_return=only_return, frame_ids=frame_ids)

    def summ_feat(self, name, feat, valid=None, pca=True, only_return=False, bev=False, fro=False, frame_id=None):
        if self.save_this:
            if feat.ndim==5: # B x C x D x H x W

                if bev:
                    reduce_axis = 3
                elif fro:
                    reduce_axis = 2
                else:
                    # default to bev
                    reduce_axis = 3
                
                if valid is None:
                    feat = torch.mean(feat, dim=reduce_axis)
                else:
                    valid = valid.repeat(1, feat.size()[1], 1, 1, 1)
                    feat = utils.basic.reduce_masked_mean(feat, valid, dim=reduce_axis)
                    
            B, C, D, W = list(feat.shape)

            if not pca:
                feat = torch.mean(torch.abs(feat), dim=1, keepdims=True)
                # feat is B x 1 x D x W
                return self.summ_oned(name=name, im=feat, norm=True, only_return=only_return, frame_id=frame_id)
            else:
                feat_pca = get_feat_pca(feat, valid)
                return self.summ_rgb(name, feat_pca, only_return=only_return, frame_id=frame_id)

    def summ_scalar(self, name, value):
        if (not (isinstance(value, int) or isinstance(value, float) or isinstance(value, np.float32))) and ('torch' in value.type()):
            value = value.detach().cpu().numpy()
        if not np.isnan(value):
            if (self.log_freq == 1):
                self.writer.add_scalar(name, value, global_step=self.global_step)
            elif self.save_this or np.mod(self.global_step, self.scalar_freq)==0:
                self.writer.add_scalar(name, value, global_step=self.global_step)

    def summ_box(self, name, rgbR, boxes_camR, scores, tids, pix_T_cam, only_return=False):
        B, C, H, W = list(rgbR.shape)
        corners_camR = utils.geom.transform_boxes_to_corners(boxes_camR)
        return self.summ_box_by_corners(name, rgbR, corners_camR, scores, tids, pix_T_cam, only_return=only_return)

    def summ_boxlist2d(self, name, rgb, boxlist, scores=None, tids=None, frame_id=None, only_return=False):
        B, C, H, W = list(rgb.shape)
        boxlist_vis = self.draw_boxlist2d_on_image(rgb, boxlist, scores=scores, tids=tids)
        return self.summ_rgb(name, boxlist_vis, frame_id=frame_id, only_return=only_return)

    def summ_boxlist2ds(self, name, rgbs, boxlist_s, frame_ids=None, scorelist_s=None, tidlist_s=None, only_return=False):
        B, S, N, D = list(boxlist_s.shape)
        assert(D==4)
        boxlist_vis = []
        for s in range(S):
            if scorelist_s is None:
                scores = None
            else:
                scores = scorelist_s[:,s]
            if tidlist_s is None:
                tids = None
            else:
                tids = tidlist_s[:,s]
            boxlist_vis.append(self.draw_boxlist2d_on_image(rgbs[:,s], boxlist_s[:,s], scores=scores, tids=tids))
        if not only_return:
            self.summ_rgbs(name, boxlist_vis, frame_ids=frame_ids)
        return boxlist_vis

    def summ_box_by_corners(self, name, rgbR, corners, scores, tids, pix_T_cam, only_return=False, frame_id=None):
        # rgb is B x H x W x C
        # corners is B x N x 8 x 3
        # scores is B x N
        # tids is B x N
        # pix_T_cam is B x 4 x 4

        B, C, H, W = list(rgbR.shape)
        boxes_vis = self.draw_corners_on_image(rgbR,
                                               corners,
                                               torch.mean(corners, dim=2),
                                               scores,
                                               tids,
                                               pix_T_cam,
                                               frame_id=frame_id)
        if not only_return:
            self.summ_rgb(name, boxes_vis)
        return boxes_vis
    
    def summ_lrtlist(self, name, rgbR, lrtlist, scorelist, tidlist, pix_T_cam, only_return=False, frame_id=None, include_zeros=False, halfres=False, show_ids=False):
        # rgb is B x H x W x C
        # lrtlist is B x N x 19
        # scorelist is B x N
        # tidlist is B x N
        # pix_T_cam is B x 4 x 4

        if self.save_this:

            B, C, H, W = list(rgbR.shape)
            B, N, D = list(lrtlist.shape)

            xyzlist_cam = utils.geom.get_xyzlist_from_lrtlist(lrtlist)
            # this is B x N x 8 x 3

            clist_cam = utils.geom.get_clist_from_lrtlist(lrtlist)
            arrowlist_cam = utils.geom.get_arrowheadlist_from_lrtlist(lrtlist)

            boxes_vis = self.draw_corners_on_image(rgbR,
                                                   xyzlist_cam,
                                                   clist_cam, 
                                                   scorelist,
                                                   tidlist,
                                                   pix_T_cam,
                                                   arrowlist_cam=arrowlist_cam,
                                                   frame_id=frame_id,
                                                   include_zeros=include_zeros,
                                                   show_ids=show_ids)
            return self.summ_rgb(name, boxes_vis, only_return=only_return, halfres=halfres)
    
    
    def summ_lrtlist_bev(self, name, occ_memR, lrtlist, scorelist, tidlist, vox_util, lrt=None, already_mem=False, only_return=False, frame_id=None, include_zeros=False, show_ids=False):
        if self.save_this:
            # rgb is B x C x Z x Y x X
            # lrtlist is B x N x 19
            # scorelist is B x N
            # tidlist is B x N

            # print('occ_memR', occ_memR.shape)
            # print('lrtlist', lrtlist.shape)
            # print('scorelist', scorelist.shape)
            # print('tidlist', tidlist.shape)
            # if lrt is not None:
            #     print('lrt', lrt.shape)

            B, _, Z, Y, X = list(occ_memR.shape)
            B, N, D = list(lrtlist.shape)

            corners_cam = utils.geom.get_xyzlist_from_lrtlist(lrtlist)
            centers_cam = utils.geom.get_clist_from_lrtlist(lrtlist)
            arrowlist_cam = utils.geom.get_arrowheadlist_from_lrtlist(lrtlist)
            
            if lrt is None:
                if not already_mem:
                    corners_mem = vox_util.Ref2Mem(corners_cam.reshape(B, N*8, 3), Z, Y, X, assert_cube=False).reshape(B, N, 8, 3)
                    # this is B x N x 8 x 3
                    centers_mem = vox_util.Ref2Mem(centers_cam, Z, Y, X, assert_cube=False).reshape(B, N, 1, 3)
                    # this is B x N x 1 x 3
                    arrowlist_mem = vox_util.Ref2Mem(arrowlist_cam, Z, Y, X, assert_cube=False).reshape(B, N, 1, 3)
                else:
                    corners_mem = corners_cam.clone().reshape(B, N, 8, 3)
                    centers_mem = centers_cam.clone().reshape(B, N, 1, 3)
                    arrowlist_mem = arrowlist_cam.clone().reshape(B, N, 1, 3)
                    
            else:
                # use the lrt to know where to voxelize
                corners_mem = vox_util.Ref2Zoom(corners_cam.reshape(B, N*8, 3), lrt, Z, Y, X).reshape(B, N, 8, 3)
                centers_mem = vox_util.Ref2Zoom(centers_cam, lrt, Z, Y, X).reshape(B, N, 1, 3)
                arrowlist_mem = vox_util.Ref2Zoom(arrowlist_cam, lrt, Z, Y, X).reshape(B, N, 1, 3)

            # rgb = utils.basic.reduce_masked_mean(unp_memR, occ_memR.repeat(1, C, 1, 1, 1), dim=3)
            rgb_vis = self.summ_occ('', occ_memR, only_return=True)
            # utils.py.print_stats('rgb_vis', rgb_vis.cpu().numpy())
            # print('rgb', rgb.shape)
            # rgb_vis = back2color(rgb)
            # this is in [0, 255]

            # print('rgb_vis', rgb_vis.shape)

            if False:
                # alt method
                box_mem = torch.cat([centers_mem, corners_mem], dim=2).reshape(B, N*9, 3)
                box_vox = vox_util.voxelize_xyz(box_mem, Z, Y, X, already_mem=True)
                box_vis = self.summ_occ('', box_vox, reduce_axes=[3], only_return=True)

                box_vis = convert_occ_to_height(box_vox, reduce_axis=3)
                box_vis = utils.basic.normalize(box_vis)
                box_vis = oned2inferno(box_vis, norm=False)
                # this is in [0, 255]

                # replace black with occ vis
                box_vis[box_vis==0] = (rgb_vis[box_vis==0].float()*0.5).byte() # darken the bkg a bit
                box_vis = preprocess_color(box_vis)
                return self.summ_rgb(('%s' % (name)), box_vis, only_return=only_return)#, only_return=only_return)

            # take the xz part
            centers_mem = torch.stack([centers_mem[:,:,:,0], centers_mem[:,:,:,2]], dim=3)
            corners_mem = torch.stack([corners_mem[:,:,:,0], corners_mem[:,:,:,2]], dim=3)
            arrowlist_mem = torch.stack([arrowlist_mem[:,:,:,0], arrowlist_mem[:,:,:,2]], dim=3)

            if frame_id is not None:
                rgb_vis = draw_frame_id_on_vis(rgb_vis, frame_id)

            out = self.draw_boxes_on_image_py(rgb_vis[0].detach().cpu().numpy(),
                                              corners_mem[0].detach().cpu().numpy(),
                                              centers_mem[0].detach().cpu().numpy(),
                                              scorelist[0].detach().cpu().numpy(),
                                              tidlist[0].detach().cpu().numpy(),
                                              arrowlist_pix=arrowlist_mem[0].detach().cpu().numpy(),
                                              frame_id=frame_id,
                                              include_zeros=include_zeros,
                                              show_ids=show_ids)
            # utils.py.print_stats('py out', out)
            out = torch.from_numpy(out).type(torch.ByteTensor).permute(2, 0, 1)
            out = torch.unsqueeze(out, dim=0)
            out = preprocess_color(out)
            return self.summ_rgb(name, out, only_return=only_return)
            # box_vis = torch.reshape(out, [1, 3, Z, X]).byte()
            # out = torch.reshape(out, [1, C, X, Z])
            # out = out.permute(0, 1, 3, 2)

            # box_vis = preprocess_color(out)
            # utils.py.print_stats('box_vis', box_vis.cpu().numpy())

            # if not only_return:
            #     self.summ_rgb(name, box_vis)
            # return box_vis
            #     self.summ_rgb(name, box_vis)

    def draw_corners_on_image(self, rgb, corners_cam, centers_cam, scores, tids, pix_T_cam, frame_id=None, arrowlist_cam=None, include_zeros=False, show_ids=False):
        # first we need to get rid of invalid gt boxes
        # gt_boxes = trim_gt_boxes(gt_boxes)
        B, C, H, W = list(rgb.shape)
        assert(C==3)
        B2, N, D, E = list(corners_cam.shape)
        assert(B2==B)
        assert(D==8) # 8 corners
        assert(E==3) # 3D

        rgb = back2color(rgb)

        corners_cam_ = torch.reshape(corners_cam, [B, N*8, 3])
        centers_cam_ = torch.reshape(centers_cam, [B, N*1, 3])
        
        corners_pix_ = utils.geom.apply_pix_T_cam(pix_T_cam, corners_cam_)
        centers_pix_ = utils.geom.apply_pix_T_cam(pix_T_cam, centers_cam_)
        
        corners_pix = torch.reshape(corners_pix_, [B, N, 8, 2])
        centers_pix = torch.reshape(centers_pix_, [B, N, 1, 2])
        
        if arrowlist_cam is not None:
            arrowlist_cam_ = torch.reshape(arrowlist_cam, [B, N*1, 3])
            arrowlist_pix_ = utils.geom.apply_pix_T_cam(pix_T_cam, arrowlist_cam_)
            arrowlist_pix = torch.reshape(arrowlist_pix_, [B, N, 1, 2])
            
        if frame_id is not None:
            rgb = draw_frame_id_on_vis(rgb, frame_id)

        out = self.draw_boxes_on_image_py(rgb[0].detach().cpu().numpy(),
                                          corners_pix[0].detach().cpu().numpy(),
                                          centers_pix[0].detach().cpu().numpy(),
                                          scores[0].detach().cpu().numpy(),
                                          tids[0].detach().cpu().numpy(),
                                          frame_id=frame_id,
                                          arrowlist_pix=arrowlist_pix[0].detach().cpu().numpy(),
                                          include_zeros=include_zeros,
                                          show_ids=show_ids)
        out = torch.from_numpy(out).type(torch.ByteTensor).permute(2, 0, 1)
        out = torch.unsqueeze(out, dim=0)
        out = preprocess_color(out)
        out = torch.reshape(out, [1, C, H, W])
        return out
    
    def draw_boxes_on_image_py(self, rgb, corners_pix, centers_pix, scores, tids, boxes=None, thickness=1, frame_id=None, arrowlist_pix=None, include_zeros=False, show_ids=False):
        # all inputs are numpy tensors
        # rgb is H x W x 3
        # corners_pix is N x 8 x 2, in xy order
        # centers_pix is N x 1 x 2, in xy order
        # scores is N
        # tids is N
        # boxes is N x 9 < this is only here to print some rotation info

        # cv2.cvtColor seems to cause an Illegal instruction error on compute-0-38; no idea why

        rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) 

        H, W, C = rgb.shape
        assert(C==3)
        N, D, E = corners_pix.shape
        assert(D==8)
        assert(E==2)

        N2 = scores.shape
        N3 = tids.shape
        # assert(N==N2)
        # assert(N==N3)

        if boxes is not None:
            rx = boxes[:,6].clone()
            ry = boxes[:,7].clone()
            rz = boxes[:,8].clone()
        else:
            rx = 0
            ry = 0
            rz = 0

        color_map = matplotlib.cm.get_cmap('tab20')
        color_map = color_map.colors

        corners_pix = corners_pix.astype(np.int32)
        centers_pix = centers_pix.astype(np.int32)

        # else:
        #     print('frame_id is none')
            
        # draw
        for ind, corners in enumerate(corners_pix):
            # corners is 8 x 2

            if include_zeros or (not np.isclose(scores[ind], 0.0)):

                # print('ind', ind)
                # print('score = %.2f' % scores[ind])
                color_id = tids[ind] % 20
                # print('color_id', color_id)
                # print('color_map', color_map)
                color = color_map[color_id]
                color = np.array(color)*255.0
                color = color[::-1]
                # color = (0,191,255)
                # color = (255,191,0)
                # print 'tid = %d; score = %.3f' % (tids[ind], scores[ind])

                # # draw center
                # cv2.circle(rgb,(centers_pix[ind,0,0],centers_pix[ind,0,1]),1,color,-1)

                if False:
                    if arrowlist_pix is not None:
                        cv2.arrowedLine(rgb,(centers_pix[ind,0,0],centers_pix[ind,0,1]),(arrowlist_pix[ind,0,0],arrowlist_pix[ind,0,1]),color,
                                        thickness=1,line_type=cv2.LINE_AA,tipLength=0.25)


                # if scores[ind] < 1.0 and scores[ind] > 0.0:
                # if False:
                if scores[ind] < 1.0:
                    # print('for example, putting this one at', np.min(corners[:,0]), np.min(corners[:,1]))
                    cv2.putText(rgb,
                                '%.2f' % (scores[ind]), 
                                # '%.2f match' % (scores[ind]), 
                                # '%.2f IOU' % (scores[ind]), 
                                # '%d (%.2f)' % (tids[ind], scores[ind]), 
                                (np.min(corners[:,0]), np.min(corners[:,1])),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, # font scale (float)
                                color,
                                1) # font thickness (int)

                if show_ids: # write tid
                    cv2.putText(rgb,
                                '%d' % (tids[ind]),
                                # '%.2f match' % (scores[ind]), 
                                # '%.2f IOU' % (scores[ind]), 
                                # '%d (%.2f)' % (tids[ind], scores[ind]), 
                                (np.max(corners[:,0]), np.max(corners[:,1])),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, # font scale (float)
                                color,
                                1) # font thickness (int)
                        

                # for c in corners:
                #     # rgb[pt1[0], pt1[1], :] = 255
                #     # rgb[pt2[0], pt2[1], :] = 255
                #     # rgb[np.clip(int(c[0]), 0, W), int(c[1]), :] = 255
                #     c0 = np.clip(int(c[0]), 0,  W-1)
                #     c1 = np.clip(int(c[1]), 0,  H-1)
                #     rgb[c1, c0, :] = 255
                    
                    
                # we want to distinguish between in-plane edges and out-of-plane ones
                # so let's recall how the corners are ordered:

                # (new clockwise ordering)
                xs = np.array([1/2., 1/2., -1/2., -1/2., 1/2., 1/2., -1/2., -1/2.])
                ys = np.array([1/2., 1/2., 1/2., 1/2., -1/2., -1/2., -1/2., -1/2.])
                zs = np.array([1/2., -1/2., -1/2., 1/2., 1/2., -1/2., -1/2., 1/2.])

                # for ii in list(range(0,2)):
                #     cv2.circle(rgb,(corners_pix[ind,ii,0],corners_pix[ind,ii,1]),1,color,-1)
                # for ii in list(range(2,4)):
                #     cv2.circle(rgb,(corners_pix[ind,ii,0],corners_pix[ind,ii,1]),1,color,-1)

                xs = np.reshape(xs, [8, 1])
                ys = np.reshape(ys, [8, 1])
                zs = np.reshape(zs, [8, 1])
                offsets = np.concatenate([xs, ys, zs], axis=1)

                corner_inds = list(range(8))
                combos = list(combinations(corner_inds, 2))

                for combo in combos:
                    pt1 = offsets[combo[0]]
                    pt2 = offsets[combo[1]]
                    # draw this if it is an in-plane edge
                    eqs = pt1==pt2
                    if np.sum(eqs)==2:
                        i, j = combo
                        pt1 = (corners[i, 0], corners[i, 1])
                        pt2 = (corners[j, 0], corners[j, 1])
                        retval, pt1, pt2 = cv2.clipLine((0, 0, W, H), pt1, pt2)
                        if retval:
                            cv2.line(rgb, pt1, pt2, color, thickness, cv2.LINE_AA)

                        # rgb[pt1[0], pt1[1], :] = 255
                        # rgb[pt2[0], pt2[1], :] = 255
        rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
        # utils.basic.print_stats_py('rgb_uint8', rgb)
        # imageio.imwrite('boxes_rgb.png', rgb)
        return rgb

    def draw_boxlist2d_on_image(self, rgb, boxlist, scores=None, tids=None):
        B, C, H, W = list(rgb.shape)
        assert(C==3)
        B2, N, D = list(boxlist.shape)
        assert(B2==B)
        assert(D==4) # ymin, xmin, ymax, xmax

        rgb = back2color(rgb)
        if scores is None:
            scores = torch.ones(B2, N).float()
        if tids is None:
            tids = torch.zeros(B2, N).long()
        out = self.draw_boxlist2d_on_image_py(
            rgb[0].cpu().detach().numpy(),
            boxlist[0].cpu().detach().numpy(),
            scores[0].cpu().detach().numpy(),
            tids[0].cpu().detach().numpy())
        out = torch.from_numpy(out).type(torch.ByteTensor).permute(2, 0, 1)
        out = torch.unsqueeze(out, dim=0)
        out = preprocess_color(out)
        out = torch.reshape(out, [1, C, H, W])
        return out
    
    def draw_boxlist2d_on_image_py(self, rgb, boxlist, scores, tids, thickness=1):
        # all inputs are numpy tensors
        # rgb is H x W x 3
        # boxlist is N x 4
        # scores is N
        # tids is N

        rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        H, W, C = rgb.shape
        assert(C==3)
        N, D = boxlist.shape
        assert(D==4)

        color_map = matplotlib.cm.get_cmap('tab20')
        color_map = color_map.colors

        # draw
        for ind, box in enumerate(boxlist):
            # box is 4
            if not np.isclose(scores[ind], 0.0):
                # box = utils.geom.scale_box2d(box, H, W)
                ymin, xmin, ymax, xmax = box


                ymin, ymax = ymin*H, ymax*H
                xmin, xmax = xmin*W, xmax*W
                
                # print 'score = %.2f' % scores[ind]
                color_id = tids[ind] % 20
                color = color_map[color_id]
                color = np.array(color)*255.0
                color = color[::-1]
                # print 'tid = %d; score = %.3f' % (tids[ind], scores[ind])

                # if False:
                if scores[ind] < 1.0: # not gt
                    cv2.putText(rgb,
                                # '%d (%.2f)' % (tids[ind], scores[ind]), 
                                '%.2f' % (scores[ind]), 
                                (int(xmin), int(ymin)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, # font size
                                color),
                    #1) # font weight

                xmin = np.clip(int(xmin), 0,  W-1)
                xmax = np.clip(int(xmax), 0,  W-1)
                ymin = np.clip(int(ymin), 0,  H-1)
                ymax = np.clip(int(ymax), 0,  H-1)

                cv2.line(rgb, (xmin, ymin), (xmin, ymax), color, thickness, cv2.LINE_AA)
                cv2.line(rgb, (xmin, ymin), (xmax, ymin), color, thickness, cv2.LINE_AA)
                cv2.line(rgb, (xmax, ymin), (xmax, ymax), color, thickness, cv2.LINE_AA)
                cv2.line(rgb, (xmax, ymax), (xmin, ymax), color, thickness, cv2.LINE_AA)
                
        rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return rgb
    
    def summ_histogram(self, name, data):
        if self.save_this:
            data = data.flatten() 
            self.writer.add_histogram(name, data, global_step=self.global_step)

    def flow2color(self, flow, clip=50.0):
        """
        :param flow: Optical flow tensor.
        :return: RGB image normalized between 0 and 1.
        """

        # flow is B x C x H x W

        B, C, H, W = list(flow.size())

        flow = flow.clone().detach()
        
        abs_image = torch.abs(flow)
        flow_mean = abs_image.mean(dim=[1,2,3])
        flow_std = abs_image.std(dim=[1,2,3])

        if clip:
            flow = torch.clamp(flow, -clip, clip)/clip
        else:
            # Apply some kind of normalization. Divide by the perceived maximum (mean + std*2)
            flow_max = flow_mean + flow_std*2 + 1e-10
            for b in range(B):
                flow[b] = flow[b].clamp(-flow_max[b].item(), flow_max[b].item()) / flow_max[b].clamp(min=1)

        radius = torch.sqrt(torch.sum(flow**2, dim=1, keepdim=True)) #B x 1 x H x W
        radius_clipped = torch.clamp(radius, 0.0, 1.0)

        angle = torch.atan2(flow[:, 1:], flow[:, 0:1]) / np.pi #B x 1 x H x W

        hue = torch.clamp((angle + 1.0) / 2.0, 0.0, 1.0)
        saturation = torch.ones_like(hue) * 0.75
        value = radius_clipped
        hsv = torch.cat([hue, saturation, value], dim=1) #B x 3 x H x W

        #flow = tf.image.hsv_to_rgb(hsv)
        flow = hsv_to_rgb(hsv)
        flow = (flow*255.0).type(torch.ByteTensor)
        return flow

    def summ_flow(self, name, im, clip=0.0, only_return=False, frame_id=None):
        # flow is B x C x D x W
        if self.save_this:
            return self.summ_rgb(name, self.flow2color(im, clip=clip), only_return=only_return, frame_id=frame_id)
        else:
            return None

    def summ_3d_flow(self, name, flow, clip=0.0):
        if self.save_this:
            self.summ_histogram('%s_flow_x' % name, flow[:,0])
            self.summ_histogram('%s_flow_y' % name, flow[:,1])
            self.summ_histogram('%s_flow_z' % name, flow[:,2])

            # flow is B x 3 x D x H x W; inside the 3 it's XYZ
            # D->z, H->y, W->x
            flow_xz = torch.cat([flow[:, 0:1], flow[:, 2:]], dim=1) # grab x, z
            flow_xy = torch.cat([flow[:, 0:1], flow[:, 1:2]], dim=1) # grab x, y
            flow_yz = torch.cat([flow[:, 1:2], flow[:, 2:]], dim=1) # grab y, z
            # these are B x 2 x D x H x W

            flow_xz = torch.mean(flow_xz, dim=3) # reduce over H (y)
            flow_xy = torch.mean(flow_xy, dim=2) # reduce over D (z)
            flow_yz = torch.mean(flow_yz, dim=4) # reduce over W (x)

            return self.summ_flow('%s_flow_xz' % name, flow_xz, clip=clip) # rot90 for interp
            # self.summ_flow('%s_flow_xy' % name, flow_xy, clip=clip)
            # self.summ_flow('%s_flow_yz' % name, flow_yz, clip=clip) # not as interpretable
            
            # flow_mag = torch.mean(torch.sum(torch.sqrt(EPS+flow**2), dim=1, keepdim=True), dim=3)
            # self.summ_oned('%s_flow_mag' % name, flow_mag)
        else:
            return None
    
    def draw_circles_on_image_from_traj_pix(self, rgb, traj_pix, sizes, color_map, seq2color_colormap='coolwarm'):
        S, _ = list(traj_pix.shape) 
        H, W, C = rgb.shape
        rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        rgb_list = []
        for ind in range(S) :
            # color_id = tids[ind] % 20
            color_id = 1
            color = color_map[color_id]
            color = np.array(color)*255.0
            radius = int(sizes[0,ind]//8)
            rgb_circled = cv2.circle(rgb.copy(),(traj_pix[ind,0],traj_pix[ind,1]),radius,color,2)
            rgb_circled = cv2.cvtColor(rgb_circled.astype(np.uint8), cv2.COLOR_BGR2RGB)
            rgb_circled = torch.from_numpy(rgb_circled).permute(2, 0, 1)
            # C x H x W
            rgb_list.append(rgb_circled)

        rgb_stacked = torch.stack(rgb_list, axis=0).cuda().unsqueeze(0)
        # B x S x C x H x W
        rgb_stacked = preprocess_color(rgb_stacked)+0.5
        rgb_stacked[rgb_stacked>0.0] = 1.0
        rgb_stacked = torch.max(rgb_stacked, dim=2)[0]
        rgb_seq_colored = seq2color(rgb_stacked, colormap=seq2color_colormap).squeeze()
        rgb = torch.unsqueeze(rgb_seq_colored, dim=0)
        rgb = torch.reshape(rgb, [1, C, H, W])

        return rgb

    def summ_point_clusters_on_images(self, name, rgbs, points_matrix, valid_matrix, label_matrix, frame_ids=None, only_return=False):

        # trajs
        # rgb is a list of len-S

        # points_matrix is N x T x 2
        # valid_matrix is N x T
        # label_matrix is N

        # all numpy array

        B, C, H, W = rgbs[0].shape 
        S = len(rgbs)

        N = points_matrix.shape[0] # number of trajs
        assert(points_matrix.shape[0] == valid_matrix.shape[0])
        assert(points_matrix.shape[0] == label_matrix.shape[0])

        if N < 10:
            color_map = matplotlib.cm.get_cmap('tab10')
        else:
            color_map = matplotlib.cm.get_cmap('tab20')
            
        color_map = color_map.colors

        rgbs_color = []
        for rgb in rgbs:
            rgb = back2color(0.2*rgb)[0].detach().cpu().numpy()  # 0.5 to make the rgb darker
            rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
           
            rgbs_color.append(rgb) # each element 3 x H x W

        

        for s in range(S):
            points = points_matrix[:, s] # N x 2
            valid = valid_matrix[:, s] # N
            rgb = rgbs_color[s]
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for n in range(N):
                if valid[n]:
                    color_id = label_matrix[n] % 20
                    color = color_map[color_id]
                    color = np.array(color)*255.0    

                    cv2.circle(rgb, (points[n, 0], points[n, 1]), 1, color, -1)

            rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
            rgbs_color[s] = rgb

        rgbs = []
        for rgb in rgbs_color:
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            rgbs.append(preprocess_color(rgb))

        return self.summ_rgbs(name, rgbs, frame_ids=frame_ids, only_return=only_return)

    def summ_affinity_on_images(self, name, rgbs, points_matrix, valid_matrix, affinity_matrix, frame_ids=None, only_return=False):

        # trajs
        # rgb is a list of len-S

        # points_matrix is N x T x 2
        # valid_matrix is N x T
        # affinity_matrix is N x N

        # all numpy array

        B, C, H, W = rgbs[0].shape 
        S = len(rgbs)

        N = points_matrix.shape[0] # number of trajs
        assert(points_matrix.shape[0] == valid_matrix.shape[0])
        assert(points_matrix.shape[0] == affinity_matrix.shape[0])

        rgbs_color = []
        for rgb in rgbs:
            rgb = back2color(0.2*rgb)[0].detach().cpu().numpy()  # 0.5 to make the rgb darker
            rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
           
            rgbs_color.append(rgb) # each element 3 x H x W

        # randomly select a traj
        center_traj_id = np.random.randint(N)
        affinity = affinity_matrix[center_traj_id, :] # (N, )

        # color_map = matplotlib.cm.get_cmap('coolwarm')
        color_map = matplotlib.cm.get_cmap('jet')


        for s in range(S):
            points = points_matrix[:, s] # N x 2
            valid = valid_matrix[:, s] # N
            rgb = rgbs_color[s]
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if valid[center_traj_id]:
                for n in range(N):
                    if n == center_traj_id:
                        color = np.array(color_map(0.0)[:3]) * 255
                        cv2.circle(rgb, (points[n, 0], points[n, 1]), 3, color, -1)
                    else:
                        if valid[n]:
                            color = np.array(color_map(1.0-affinity[n])[:3]) * 255 # rgb
                            cv2.circle(rgb, (points[n, 0], points[n, 1]), 1, color, -1)
                    
            rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
            rgbs_color[s] = rgb

        rgbs = []
        for rgb in rgbs_color:
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            rgbs.append(preprocess_color(rgb))

        return self.summ_rgbs(name, rgbs, frame_ids=frame_ids, only_return=only_return)

    def summ_anchor_affinity_on_images(self, name, rgbs, points_matrix, valid_matrix, affinity, frame_ids=None, only_return=False):

        # rgbs is a list of length S

        # points_matrix is N x T x 2
        # valid_matrix is N x T
        # affinity is N

        # all numpy array

        B, C, H, W = rgbs[0].shape 
        S = len(rgbs)

        N = points_matrix.shape[0] # number of trajs
        assert(points_matrix.shape[0] == valid_matrix.shape[0])
        assert(points_matrix.shape[0] == affinity.shape[0])

        rgbs_color = []
        for rgb in rgbs:
            rgb = back2color(0.2*rgb)[0].detach().cpu().numpy()  # 0.5 to make the rgb darker
            rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
            rgbs_color.append(rgb) # each element 3 x H x W

        # color_map = matplotlib.cm.get_cmap('coolwarm')
        color_map = matplotlib.cm.get_cmap('jet')

        for s in range(S):
            points = points_matrix[:, s, :2] # N x 2
            valid = valid_matrix[:, s] # N
            rgb = rgbs_color[s]
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            for n in range(N):
                if valid[n]:
                    # print('affinity[%d] = %.2f' % (n, affinity[n]))
                    color = np.array(color_map(1.0-affinity[n])[:3]) * 255 # rgb
                    # print('color', color)
                    cv2.circle(rgb, (points[n, 0], points[n, 1]), 1, color, -1)
                    
            rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
            rgbs_color[s] = rgb

        rgbs = []
        for rgb in rgbs_color:
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            rgbs.append(preprocess_color(rgb))

        return self.summ_rgbs(name, rgbs, frame_ids=frame_ids, only_return=only_return)
    
    def summ_point_traj_on_images(self, name, rgbs, trajs_XYs, trajs_Ts, frame_ids=None, only_return=False, double_speed=False):

        # rgb is a list of len-S
        B, C, H, W = rgbs[0].shape 
        S = len(rgbs)

        rgbs_color = []
        for rgb in rgbs:
            rgb = back2color(rgb)[0].detach().cpu().numpy() 
            rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
            rgbs_color.append(rgb) # each element 3 x H x W

        num_trajs = len(trajs_XYs)
        for i in range(num_trajs):
            traj_XY = trajs_XYs[i].long().detach().cpu().numpy()
            traj_T = trajs_Ts[i].detach().cpu().numpy().astype(np.int32)
            num_lasting_frames = len(traj_T)
            if num_lasting_frames > 1:
                for t in range(num_lasting_frames):
                    cur_t = traj_T[t]
                    rgbs_color[cur_t] = self.draw_traj_on_image_py(
                        rgbs_color[cur_t],
                        traj_XY[:t+1],
                        S=S)

        rgbs = []
        for rgb in rgbs_color:
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            rgbs.append(preprocess_color(rgb))

        if double_speed:
            rgbs = rgbs[::2]

        return self.summ_rgbs(name, rgbs, only_return=only_return, frame_ids=frame_ids)

    def summ_xys_on_rgbs(self, name, xys, rgbs, frame_ids=None, only_return=False, thickness=2):
        # xys is B, S, 2
        # rgbs is B, S, C, H, W
        B, S, D = xys.shape
        B, S2, C, H, W = rgbs.shape
        assert(S==S2)

        rgbs = rgbs[0] # S, C, H, W
        xys = xys[0] # S, 2
        
        rgbs_color = []
        for rgb in rgbs:
            rgb = back2color(rgb).detach().cpu().numpy() 
            rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
            rgbs_color.append(rgb) # each element 3 x H x W

        xys = xys.long().detach().cpu().numpy() # S, 2
        for t in range(S):
            rgbs_color[t] = self.draw_traj_on_image_py(rgbs_color[t], xys[:t+1], S=S, thickness=thickness)

        rgbs = []
        for rgb in rgbs_color:
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            rgbs.append(preprocess_color(rgb))

        return self.summ_rgbs(name, rgbs, only_return=only_return, frame_ids=frame_ids)

    def summ_traj2ds_on_rgbs(self, name, trajs, rgbs, valids=None, frame_ids=None, only_return=False, show_dots=True, cmap='coolwarm', linewidth=1):
        # trajs is B, S, N, 2
        # rgbs is B, S, C, H, W
        B, S, C, H, W = rgbs.shape
        B, S2, N, D = trajs.shape
        assert(S==S2)

        rgbs = rgbs[0] # S, C, H, W
        trajs = trajs[0] # S, N, 2
        if valids is None:
            valids = torch.ones_like(trajs[:,:,0]) # S, N
        else:
            valids = valids[0]
        # print('trajs', trajs.shape)
        # print('valids', valids.shape)
        
        rgbs_color = []
        for rgb in rgbs:
            rgb = back2color(rgb).detach().cpu().numpy() 
            rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
            rgbs_color.append(rgb) # each element 3 x H x W

        for i in range(N):
            if cmap=='onediff' and i==0:
                cmap_ = 'spring'
            elif cmap=='onediff':
                cmap_ = 'winter'
            else:
                cmap_ = cmap
            traj = trajs[:,i].long().detach().cpu().numpy() # S, 2
            valid = valids[:,i].long().detach().cpu().numpy() # S

            # print('traj', traj.shape)
            # print('valid', valid.shape)
            
            for t in range(S):
                if valid[t]:
                    rgbs_color[t] = self.draw_traj_on_image_py(rgbs_color[t], traj[:t+1], S=S, show_dots=show_dots, cmap=cmap_, linewidth=linewidth)

        rgbs = []
        for rgb in rgbs_color:
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            rgbs.append(preprocess_color(rgb))

        return self.summ_rgbs(name, rgbs, only_return=only_return, frame_ids=frame_ids)

    def summ_traj2ds_on_rgbs2(self, name, trajs, visibles, rgbs, valids=None, frame_ids=None, only_return=False, show_dots=True, cmap='coolwarm', linewidth=1):
        # trajs is B, S, N, 2
        # rgbs is B, S, C, H, W
        B, S, C, H, W = rgbs.shape
        B, S2, N, D = trajs.shape
        assert(S==S2)

        rgbs = rgbs[0] # S, C, H, W
        trajs = trajs[0] # S, N, 2
        visibles = visibles[0] # S, N
        if valids is None:
            valids = torch.ones_like(trajs[:,:,0]) # S, N
        else:
            valids = valids[0]
        # print('trajs', trajs.shape)
        # print('valids', valids.shape)
        
        rgbs_color = []
        for rgb in rgbs:
            rgb = back2color(rgb).detach().cpu().numpy() 
            rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
            rgbs_color.append(rgb) # each element 3 x H x W

        trajs = trajs.long().detach().cpu().numpy() # S, N, 2
        visibles = visibles.float().detach().cpu().numpy() # S, N
        valids = valids.long().detach().cpu().numpy() # S, N

        for i in range(N):
            if cmap=='onediff' and i==0:
                cmap_ = 'spring'
            elif cmap=='onediff':
                cmap_ = 'winter'
            else:
                cmap_ = cmap
            traj = trajs[:,i] # S,2
            vis = visibles[:,i] # S
            valid = valids[:,i] # S

            # print('traj', traj.shape)
            # print('valid', valid.shape)
            
            # for t in range(S):
            #     if valid[t]:
            #         rgbs_color[t] = self.draw_traj_on_image_py(rgbs_color[t], traj[:t+1], S=S, show_dots=show_dots, cmap=cmap_, linewidth=linewidth)
            rgbs_color = self.draw_traj_on_images_py(rgbs_color, traj, S=S, show_dots=show_dots, cmap=cmap_, linewidth=linewidth)
            
        for i in range(N):
            if cmap=='onediff' and i==0:
                cmap_ = 'spring'
            elif cmap=='onediff':
                cmap_ = 'winter'
            else:
                cmap_ = cmap
            traj = trajs[:,i] # S,2
            vis = visibles[:,i] # S
            valid = valids[:,i] # S
            rgbs_color = self.draw_circ_on_images_py(rgbs_color, traj, vis, S=S, show_dots=show_dots, cmap=cmap_, linewidth=linewidth)

        # bak_color = np.array(color_map(1.0)[:3]) * 255 # rgb
        # cv2.circle(rgbs[s], (traj[s,0], traj[s,1]), linewidth*4, bak_color, -1)

        rgbs = []
        for rgb in rgbs_color:
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            rgbs.append(preprocess_color(rgb))

        return self.summ_rgbs(name, rgbs, only_return=only_return, frame_ids=frame_ids)

    
    def summ_attn_on_rgb_single(self, name, coords, attn_coords, attn_scores, rgbs, traj_id=0, frame_id=0, only_return=False, cmap='inferno'):
        # coords (B, S-1, N, 2)
        # attn_coords (B, S-1, N, K)
        # attn_scores (B, S-1, N, K)
        # rgbs (B, S, C, H, W)
        B1, SS1, N1, K1 = attn_coords.shape
        B2, SS2, N2, K2 = attn_scores.shape
        assert (B1==B2 and SS1==SS2 and N1==N2 and K1==K2)
        B, SS, N, K = B1, SS1, N1, K1
        
        self.maxwidth = 9999
        B, S, C, H, W = rgbs.shape
        sc = 1

        rgbs_bak = back2color(rgbs[0])
        rgbs_color = [rgbs_bak[i].permute(1,2,0) for i in range(SS)]
        rgbs_color = torch.cat(rgbs_color, dim=1).unsqueeze(0).permute(0,3,1,2)
        rgbs_color = F.interpolate(rgbs_color.float(), (H//sc, W*S//sc), mode='bilinear').squeeze(0).permute(1,2,0)
        rgbs_color = rgbs_color.detach().cpu().numpy() # H, W*S, C
        rgbs_color = rgbs_color.astype(np.uint8).copy()

        scores = attn_scores[0, frame_id, traj_id].detach().cpu().numpy() # K

        a_coords_xy = coords[:,1:].reshape(-1,2)[attn_coords.reshape(-1)].reshape(SS,N,K,2)[frame_id,traj_id] # K, 2
        a_coords_t = attn_coords[0, frame_id, traj_id].unsqueeze(-1) // N # K, 1
        a_coords = torch.cat([a_coords_xy, a_coords_t], dim=1) # K, 3
        a_coords = a_coords.detach().cpu().numpy()

        coords_all = coords.clone().detach().cpu().numpy() 
        coords = coords[0, frame_id+1, traj_id].detach().cpu().numpy() # 2

        from_x, from_y = int(float(coords[0]) / sc), int(float(coords[1]) / sc)
        from_x += (frame_id+1) * W // sc

        # draw trajs
        '''
        color_map = matplotlib.cm.get_cmap('coolwarm')
        for s in range(S-1):
            color = np.array(color_map((s)/max(1,float(S-2)))[:3]) * 255
            for n in range(N):
                cv2.line(rgbs_color, (int(coords_all[0,s,n,0].item()+s*W),int(coords_all[0,s,n,1].item())), (int(coords_all[0,s+1,n,0].item()+(s+1)*W),int(coords_all[0,s+1,n,1].item())), color, 1, cv2.LINE_AA)
        '''

        # draw attns
        color_map = matplotlib.cm.get_cmap(cmap)
        color_p = np.array(color_map(1.0)[:3]) * 255
        cv2.circle(rgbs_color, (from_x, from_y), 2, color_p, -1)

        color_map = matplotlib.cm.get_cmap(cmap)
        for k in range(K):
            color = np.array(color_map((scores[k]))[:3]) * 255 # rgb
            to_x = int(float(a_coords[k,0] + (a_coords[k,2]+1) * W) / sc)
            to_y = int(float(a_coords[k,1]) / sc)
            cv2.line(rgbs_color, (from_x, from_y), (to_x, to_y), color, 1, cv2.LINE_AA)

            cv2.circle(rgbs_color, (to_x, to_y), 2, color_p, -1)

        rgbs_color = torch.from_numpy(rgbs_color).permute(2,0,1).unsqueeze(0)
        rgbs_color = preprocess_color(rgbs_color)
        return self.summ_rgb(name, rgbs_color, only_return=only_return, frame_id=frame_id)

    def summ_attn_on_rgb(self, name, coords, attn_coords, attn_scores, rgb, traj_id=0, frame_id=None, only_return=False, cmap='inferno'):
        # coords (B, S-1, N, 2)
        # attn_coords (B, S-1, N, K, 2)
        # attn_scores (B, S-1, N, K)
        B1, SS1, N1, K1 = attn_coords.shape
        B2, SS2, N2, K2 = attn_scores.shape
        assert (B1==B2 and SS1==SS2 and N1==N2 and K1==K2)
        B, SS, N, K = B1, SS1, N1, K1

        rgb = rgb[0] # S, C, H, W
        rgb_color = back2color(rgb).detach().cpu().numpy()
        rgb_color = np.transpose(rgb_color, [1, 2, 0]) # put channels last

        # choose a random traj to draw attn on
        from_coords = coords[0, :, traj_id].long().detach().cpu().numpy() # S-1, 2
        to_coords = coords.reshape(-1,2)[attn_coords.reshape(-1)].reshape(SS, N, K, 2)[:,traj_id]
        #to_coords = attn_coords[0,:,traj_id].long().detach().cpu().numpy() # S-1, K
        scores = attn_scores[0,:,traj_id].detach().cpu().numpy() # S-1, K

        rgb_color = self.draw_attn_on_image_py(rgb_color, from_coords, to_coords, scores, cmap=cmap)
        rgb_color = torch.from_numpy(rgb_color).permute(2,0,1).unsqueeze(0)
        rgb = preprocess_color(rgb_color)
        return self.summ_rgb(name, rgb, only_return=only_return, frame_id=frame_id)

    def draw_attn_on_image_py(self, rgb, from_coords, to_coords, scores, thickness=1, cmap='inferno'):
        # rgb: 3xHxW
        # from_coords: S-1, 2
        # to_coords: S-1, K, 2
        # scores: S-1, K
        H, W, C = rgb.shape
        assert (C==3)

        SS, K, _ = to_coords.shape

        rgb = rgb.astype(np.uint8).copy()
        color_map = matplotlib.cm.get_cmap(cmap)
        for s in range(SS):
            for k in range(K):
                if scores[s,k] < 0.06:
                    continue
                color = np.array(color_map((scores[s,k]))[:3]) * 255 # rgb
                cv2.line(rgb, (int(from_coords[s,0]), int(from_coords[s,1])), (int(to_coords[s,k,0]), int(to_coords[s,k,1])), color, thickness, cv2.LINE_AA)

        return rgb

    def summ_traj2ds_on_rgb(self, name, trajs, rgb, valids=None, show_dots=True, frame_id=None, only_return=False, cmap='coolwarm', linewidth=1):
        # trajs is B, S, N, 2
        # rgb is B, C, H, W
        B, C, H, W = rgb.shape
        B, S, N, D = trajs.shape

        rgb = rgb[0] # S, C, H, W
        trajs = trajs[0] # S, N, 2
        
        if valids is None:
            valids = torch.ones_like(trajs[:,:,0])
        else:
            valids = valids[0]

        # print('trajs', trajs.shape)
        # print('valids', valids.shape)
        # print('sum(valids, dim=0)', torch.sum(valids, dim=0))
        
        rgb_color = back2color(rgb).detach().cpu().numpy() 
        rgb_color = np.transpose(rgb_color, [1, 2, 0]) # put channels last

        # using maxdist will dampen the colors for short motions
        norms = torch.sqrt(1e-4 + torch.sum((trajs[-1] - trajs[0])**2, dim=1)) # N
        maxdist = torch.quantile(norms, 0.95).detach().cpu().numpy()
        maxdist = None 
        trajs = trajs.long().detach().cpu().numpy() # S, N, 2
        valids = valids.long().detach().cpu().numpy() # S, N
        
        for i in range(N):
            if cmap=='onediff' and i==0:
                cmap_ = 'spring'
            elif cmap=='onediff':
                cmap_ = 'winter'
            else:
                cmap_ = cmap
            traj = trajs[:,i] # S, 2
            valid = valids[:,i] # S
            if valid[0]==1:
                traj = traj[valid>0]
                rgb_color = self.draw_traj_on_image_py(rgb_color, traj, S=S, show_dots=show_dots, cmap=cmap_, maxdist=maxdist, linewidth=linewidth)

        rgb_color = torch.from_numpy(rgb_color).permute(2, 0, 1).unsqueeze(0)
        rgb = preprocess_color(rgb_color)
        return self.summ_rgb(name, rgb, only_return=only_return, frame_id=frame_id)
                                
    def draw_traj_on_image_py(self, rgb, traj, S=50, linewidth=1, show_dots=False, cmap='coolwarm', maxdist=None):
        # all inputs are numpy tensors
        # rgb is 3 x H x W
        # traj is S x 2
        
        H, W, C = rgb.shape
        assert(C==3)

        rgb = rgb.astype(np.uint8).copy()

        S1, D = traj.shape
        assert(D==2)

        color_map = matplotlib.cm.get_cmap(cmap)
        S1, D = traj.shape

        for s in range(S1-1):
            if maxdist is not None:
                val = (np.sqrt(np.sum((traj[s]-traj[0])**2))/maxdist).clip(0,1)
                color = np.array(color_map(val)[:3]) * 255 # rgb
            else:
                color = np.array(color_map((s)/max(1,float(S-2)))[:3]) * 255 # rgb

            cv2.line(rgb,
                     (int(traj[s,0]), int(traj[s,1])),
                     (int(traj[s+1,0]), int(traj[s+1,1])),
                     color,
                     linewidth,
                     cv2.LINE_AA)
            if show_dots:
                cv2.circle(rgb, (traj[s,0], traj[s,1]), linewidth, color, -1)

        if maxdist is not None:
            val = (np.sqrt(np.sum((traj[-1]-traj[0])**2))/maxdist).clip(0,1)
            color = np.array(color_map(val)[:3]) * 255 # rgb
        else:
            # draw the endpoint of traj, using the next color (which may be the last color)
            color = np.array(color_map((S1-1)/max(1,float(S-2)))[:3]) * 255 # rgb
            
        # color = np.array(color_map(1.0)[:3]) * 255
        cv2.circle(rgb, (traj[-1,0], traj[-1,1]), linewidth*2, color, -1)

        return rgb


    def draw_traj_on_images_py(self, rgbs, traj, S=50, linewidth=1, show_dots=False, cmap='coolwarm', maxdist=None):
        # all inputs are numpy tensors
        # rgbs is a list of H,W,3
        # traj is S,2
        H, W, C = rgbs[0].shape
        assert(C==3)

        rgbs = [rgb.astype(np.uint8).copy() for rgb in rgbs]

        S1, D = traj.shape
        assert(D==2)
        
        x = int(np.clip(traj[0,0], 0, W-1))
        y = int(np.clip(traj[0,1], 0, H-1))
        color = rgbs[0][y,x]
        color = (int(color[0]),int(color[1]),int(color[2]))
        for s in range(S):
            # bak_color = np.array(color_map(1.0)[:3]) * 255 # rgb
            # cv2.circle(rgbs[s], (traj[s,0], traj[s,1]), linewidth*4, bak_color, -1)
            cv2.polylines(rgbs[s],
                          [traj[:s+1]],
                          False,
                          color,
                          linewidth,
                          cv2.LINE_AA)
            # cv2.circle(rgbs[s], (traj[s,0], traj[s,1]), linewidth*2, color, -1)
        
        # for s in range(S1-1):
        #     if maxdist is not None:
        #         val = (np.sqrt(np.sum((traj[s]-traj[0])**2))/maxdist).clip(0,1)
        #         color = np.array(color_map(val)[:3]) * 255 # rgb
        #     else:
        #         color = np.array(color_map((s)/max(1,float(S-2)))[:3]) * 255 # rgb

        #     if False:
        #         cv2.line(rgb,
        #                  (int(traj[s,0]), int(traj[s,1])),
        #                  (int(traj[s+1,0]), int(traj[s+1,1])),
        #                  color,
        #                  linewidth,
        #                  cv2.LINE_AA)
        #     if False and show_dots:
        #         cv2.circle(rgb, (traj[s,0], traj[s,1]), linewidth, color, -1)

        # if maxdist is not None:
        #     val = (np.sqrt(np.sum((traj[-1]-traj[0])**2))/maxdist).clip(0,1)
        #     color = np.array(color_map(val)[:3]) * 255 # rgb
        # else:
        #     # draw the endpoint of traj, using the next color (which may be the last color)
        #     color = np.array(color_map((S1-1)/max(1,float(S-2)))[:3]) * 255 # rgb
            
        # color = np.array(color_map(1.0)[:3]) * 255

        return rgbs
    
    def draw_circ_on_images_py(self, rgbs, traj, vis, S=50, linewidth=1, show_dots=False, cmap='coolwarm', maxdist=None):
        # all inputs are numpy tensors
        # rgbs is a list of 3,H,W
        # traj is S,2
        H, W, C = rgbs[0].shape
        assert(C==3)

        rgbs = [rgb.astype(np.uint8).copy() for rgb in rgbs]

        S1, D = traj.shape
        assert(D==2)

        bremm = ColorMap2d()
        traj_ = traj[0:1].astype(np.float32)
        traj_[:,0] /= float(W)
        traj_[:,1] /= float(H)
        color = bremm(traj_)
        # print('color', color)
        color = (color[0]*255).astype(np.uint8) 
        # print('color', color)
        color = (int(color[0]),int(color[1]),int(color[2]))
        # color = (int(color[2]),int(color[1]),int(color[0]))
        # label_colors = cmap(kp_xy_n[0])
        # label_colors = [(l*255).astype(np.uint8) for l in label_colors]
        # color_map = matplotlib.cm.get_cmap(cmap)
        # # print('sample', color_map(0))

        # color_map = matplotlib.cm.get_cmap(cmap)
        # # print('sample', color_map(0))

        x = int(np.clip(traj[0,0], 0, W-1))
        y = int(np.clip(traj[0,1], 0, H-1))
        color_ = rgbs[0][y,x]
        color_ = (int(color_[0]),int(color_[1]),int(color_[2]))
        color_ = (int(color_[0]),int(color_[1]),int(color_[2]))
        # print('color_', color_)
        # print('vis[0]', vis[0])
        for s in range(S):
            # bak_color = np.array(color_map(vis[s])[:3]) * 255 # rgb
            # print('bak_color', bak_color)
            # cv2.circle(rgbs[s], (traj[s,0], traj[s,1]), linewidth*4, bak_color, -1)
            cv2.circle(rgbs[s], (traj[s,0], traj[s,1]), linewidth*4, color, -1)
            # cv2.circle(rgbs[s], (traj[s,0], traj[s,1]), linewidth*2, color_, -1)
            # print('vis[s]', vis[s])
            vis_color = int(np.squeeze(vis[s])*255)
            vis_color = (vis_color,vis_color,vis_color)
            cv2.circle(rgbs[s], (traj[s,0], traj[s,1]), linewidth*2, vis_color, -1)
            # cv2.circle(rgbs[s], (traj[s,0], traj[s,1]), linewidth*4, bak_color)
            # cv2.circle(rgbs[s], (traj[s,0], traj[s,1]), linewidth*2, color, -1)
            # if vis[s] == 0:
            #     color = (0,0,0)
            # cv2.circle(rgbs[s], (traj[s,0], traj[s,1]), linewidth*2, bak_color, -1)
                
        return rgbs
    

    def summ_traj_on_image(self, name, traj, pix_T_cam, H, W, traj_g=None, size_factor=1.0, only_return=False, frame_id=None):
        # traj is B x S x 3

        B, S, D = list(traj.shape)
        B2, E, F = list(pix_T_cam.shape)
        assert(D==3)
        assert(E==4)
        assert(F==4)
        assert(B==B2)
        
        sizes = utils.misc.get_radius_in_pix_from_clist(traj, pix_T_cam)
        sizes = sizes * size_factor
        if traj_g is not None:
            sizes_g = utils.misc.get_radius_in_pix_from_clist(traj_g, pix_T_cam)
            sizes_g = sizes_g * size_factor

        traj_pix_ = utils.geom.apply_pix_T_cam(pix_T_cam, traj)

        if traj_g is not None:
            traj_g_pix_ = utils.geom.apply_pix_T_cam(pix_T_cam, traj_g)

        rgb = np.zeros([H, W, 3], dtype=np.uint8)
        # rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)

        color_map = matplotlib.cm.get_cmap('tab20')
        color_map = color_map.colors

        if frame_id is not None:
            rgb = draw_frame_id_on_vis(rgb, frame_id)

        traj_pix = traj_pix_[0].detach().cpu().numpy()

        rgb_e = self.draw_circles_on_image_from_traj_pix(rgb, traj_pix, sizes, color_map)

        if traj_g is not None:
            traj_g_pix = traj_g_pix_[0].detach().cpu().numpy()
            rgb_g = self.draw_circles_on_image_from_traj_pix(rgb, traj_g_pix, sizes_g, color_map, seq2color_colormap='winter')
            rgb_e_any = (torch.max(rgb_e, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            rgb_e[rgb_e_any == -0.5] = rgb_g[rgb_e_any == -0.5]
            rgb = rgb_e
        else:
            rgb = rgb_e

        return self.summ_rgb(name, rgb, only_return=only_return)

    def summ_2d_traj(self, name, traj, H, W, traj_g=None, size=1.0, only_return=False, frame_id=None):
        # traj is B x S x 3

        B, S, D = list(traj.shape)
        assert(D==2)
        
        sizes = size * np.ones((1, S))
        sizes_g = sizes
    
        # traj_pix_ = utils.geom.apply_pix_T_cam(pix_T_cam, traj)

        # if traj_g is not None:
        #     traj_g_pix_ = utils.geom.apply_pix_T_cam(pix_T_cam, traj_g)

        rgb = np.zeros([H, W, 3], dtype=np.uint8)
        # rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)

        color_map = matplotlib.cm.get_cmap('tab20')
        color_map = color_map.colors

        # if frame_id is not None:
        #     rgb = draw_frame_id_on_vis(rgb, frame_id)

        traj_pix = traj[0].long().detach().cpu().numpy()

        rgb_e = self.draw_circles_on_image_from_traj_pix(rgb, traj_pix, sizes, color_map)

        if traj_g is not None:
            traj_g_pix = traj_g[0].detach().cpu().numpy()
            rgb_g = self.draw_circles_on_image_from_traj_pix(rgb, traj_g_pix, sizes_g, color_map, seq2color_colormap='winter')
            rgb_e_any = (torch.max(rgb_e, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            rgb_e[rgb_e_any == -0.5] = rgb_g[rgb_e_any == -0.5]
            rgb = rgb_e
        else:
            rgb = rgb_e

        return self.summ_rgb(name, rgb, only_return=only_return, frame_id=frame_id)


    def summ_traj_on_occ(self, name, traj, occ_mem, vox_util, heightmap=False, traj_g=None, bev=False, fro=False, show_bkg=True, already_mem=False, sigma=2, only_return=False, frame_id=None):
        # traj is B x S x 3
        if heightmap:
            B, C, Z, X = list(occ_mem.shape)
            Y = X
        else:
            B, C, Z, Y, X = list(occ_mem.shape)
            
        B2, S, D = list(traj.shape)
        assert(D==3)
        assert(B==B2)
        
        if self.save_this:
            if already_mem:
                traj_mem = traj
                if traj_g is not None:
                    traj_g_mem = traj_g
            else:
                traj_mem = vox_util.Ref2Mem(traj, Z, Y, X, assert_cube=False)
                if traj_g is not None:
                    traj_g_mem = vox_util.Ref2Mem(traj_g, Z, Y, X, assert_cube=False)

            if show_bkg:
                if heightmap:
                    height_mem = occ_mem
                else:
                    if fro:
                        height_mem = convert_occ_to_height(occ_mem, reduce_axis=2)
                    else:
                        height_mem = convert_occ_to_height(occ_mem, reduce_axis=3)
                    # this is B x C x Z x X

                occ_vis = utils.basic.normalize(height_mem)
                occ_vis = oned2inferno(occ_vis, norm=False)
                # print(vis.shape)
            else:
                if fro:
                    occ_vis = torch.zeros(B, 3, Y, X).cpu().byte()
                else:  
                    occ_vis = torch.zeros(B, 3, Z, X).cpu().byte()



            if traj_g is not None:
                x, y, z = torch.unbind(traj_g_mem, dim=2)
                if fro:
                    xy = torch.stack([x,y], dim=2)
                    heats = draw_circles_at_xy(xy, Y, X, sigma=sigma)
                else:
                    xz = torch.stack([x,z], dim=2)
                    heats = draw_circles_at_xy(xz, Z, X, sigma=sigma)
                # this is B x S x 1 x Z x X
                heats = torch.squeeze(heats, dim=2)
                heat = seq2color(heats, colormap='winter')
                # make black 0
                heat = back2color(heat)

                # show e without overwriting g
                x, y, z = torch.unbind(traj_mem, dim=2)
                if fro:
                    xy = torch.stack([x,y], dim=2)
                    heats = draw_circles_at_xy(xy, Y, X, sigma=sigma)
                else:  
                    xz = torch.stack([x,z], dim=2)
                    heats = draw_circles_at_xy(xz, Z, X, sigma=sigma)
                # this is B x S x 1 x Z x X
                heats = torch.squeeze(heats, dim=2)
                heat_e = seq2color(heats)
                # make black 0
                heat_e = back2color(heat_e)
                heat_any = (torch.max(heat, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
                # replace black with g vis
                heat[heat_any==0] = (heat_e[heat_any==0].float()).byte()

            else:
                x, y, z = torch.unbind(traj_mem, dim=2)
                if fro:
                    xy = torch.stack([x,y], dim=2)
                    heats = draw_circles_at_xy(xy, Y, X, sigma=sigma)
                else:
                    xz = torch.stack([x,z], dim=2)
                    heats = draw_circles_at_xy(xz, Z, X, sigma=sigma)
                # this is B x S x 1 x Z x X
                heats = torch.squeeze(heats, dim=2)
                heat = seq2color(heats)
                # make black 0
                heat = back2color(heat)

            # print(heat.shape)
            # vis[heat > 0] = heat
            
            # replace black with occ vis
            # heat[heat==0] = (occ_vis[heat==0].float()*0.5).byte() # darken the bkg a bit
            # heat[heat==0] = (occ_vis[heat==0].float()*0.5).byte() # darken the bkg a bit
            heat_any = (torch.max(heat, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            heat[heat_any==0] = occ_vis[heat_any==0]

            if frame_id is not None:
                heat = draw_frame_id_on_vis(heat, frame_id)
            
            heat = preprocess_color(heat)
            
            
            return self.summ_rgb(('%s' % (name)), heat, only_return=only_return)

    def summ_seg(self, name, seg, only_return=False, frame_id=None, colormap='tab20', label_colors=None):
        if not self.save_this:
            return

        B,H,W = seg.shape

        if label_colors is None:
            custom_label_colors = False
            label_colors = matplotlib.cm.get_cmap(colormap).colors
            label_colors = [[int(i*255) for i in l] for l in label_colors]
        else:
            custom_label_colors = True
        # label_colors = matplotlib.cm.get_cmap(colormap).colors
        # label_colors = [[int(i*255) for i in l] for l in label_colors]
        # print('label_colors', label_colors)
        
        # label_colors = [
        #     (0, 0, 0),         # None
        #     (70, 70, 70),      # Buildings
        #     (190, 153, 153),   # Fences
        #     (72, 0, 90),       # Other
        #     (220, 20, 60),     # Pedestrians
        #     (153, 153, 153),   # Poles
        #     (157, 234, 50),    # RoadLines
        #     (128, 64, 128),    # Roads
        #     (244, 35, 232),    # Sidewalks
        #     (107, 142, 35),    # Vegetation
        #     (0, 0, 255),      # Vehicles
        #     (102, 102, 156),  # Walls
        #     (220, 220, 0)     # TrafficSigns
        # ]

        r = torch.zeros_like(seg,dtype=torch.uint8)
        g = torch.zeros_like(seg,dtype=torch.uint8)
        b = torch.zeros_like(seg,dtype=torch.uint8)
        
        for label in range(0,len(label_colors)):
            if (not custom_label_colors):# and (N > 20):
                label_ = label % 20
            else:
                label_ = label
            
            idx = (seg == label)
            r[idx] = label_colors[label_][0]
            g[idx] = label_colors[label_][1]
            b[idx] = label_colors[label_][2]
            
        rgb = torch.stack([r,g,b],axis=1)
        return self.summ_rgb(name,rgb,only_return=only_return, frame_id=frame_id)
        

    def summ_soft_seg(self, name, seg, bev=False, max_along_y=False, only_return=False, frame_id=None, colormap='tab20', label_colors=None):
        if not self.save_this:
            return

        if bev:
            B,N,D,H,W = seg.shape
            if max_along_y:
                seg = torch.max(seg, dim=3)[0]
            else:
                seg = torch.mean(seg, dim=3)
        B,N,H,W = seg.shape
            
        # the values along N should sum to 1
        
        if N > 10:
            colormap = 'tab20'

        if label_colors is None:
            custom_label_colors = False
            label_colors = matplotlib.cm.get_cmap(colormap).colors
            label_colors = [[int(i*255) for i in l] for l in label_colors]
        else:
            custom_label_colors = True
            
        color_map = torch.zeros([B, 3, N, H, W], dtype=torch.float32).cuda()
        seg_ = seg.unsqueeze(1)
        # this is B x 1 x N x H x W

        # print('label_colors', label_colors, len(label_colors))

        for label in range(0,N):
            if (not custom_label_colors) and (N > 20):
                label_ = label % 20
            else:
                label_ = label
            # print('label_colors[%d]' % (label_), label_colors[label_])
            color_map[:,0,label_] = label_colors[label_][0]
            color_map[:,1,label_] = label_colors[label_][1]
            color_map[:,2,label_] = label_colors[label_][2]

        out = torch.sum(color_map * seg_, dim=2)
        out = out.type(torch.ByteTensor)
        return self.summ_rgb(name, out, only_return=only_return, frame_id=frame_id)
    
    def summ_soft_seg_thr(self, name, seg_e_sig, thr=0.5, only_return=False, colormap='tab20', frame_id=None, label_colors=None):
        B, N, H, W = list(seg_e_sig.shape)
        assert(thr > 0.0)
        assert(thr < 1.0)
        seg_e_hard = (seg_e_sig > thr).float()
        single_class = (torch.sum(seg_e_hard, dim=1, keepdim=True)==1).float()
        seg_e_hard = seg_e_hard * single_class
        seg_e_sig = (seg_e_sig - thr).clamp(0, (1-thr))/(1-thr)
        return self.summ_soft_seg(name, seg_e_hard * seg_e_sig, only_return=only_return, colormap=colormap, frame_id=frame_id, label_colors=label_colors)

    def summ_ordered_seg_thr(self, name, seg_e_sig, thr=0.5, only_return=False, colormap='tab20', frame_id=None, label_colors=None):
        B, N, H, W = list(seg_e_sig.shape)
        assert(thr > 0.0)
        assert(thr < 1.0)
        prep = torch.zeros_like(seg_e_sig)
        for n in range(N):
            prep[:,n] = seg_e_sig[:,n]
            hard = (prep[:,n:n+1] > thr).float()
            prep[:,:n] = prep[:,:n] * (1-hard)
        return self.summ_soft_seg_thr(name, prep, only_return=only_return, colormap=colormap, frame_id=frame_id, label_colors=label_colors)

    def summ_heat_helper(self, name, kp_xyz_cam, vox_util, Z1, Y1, X1, sigma=3):
        kp_xyz_mem = vox_util.Ref2Mem(kp_xyz_cam, Z1, Y1, X1)
        kp_heat = utils.improc.xyz2heatmaps(kp_xyz_mem, Z1, Y1, X1, sigma=sigma)
        kp_heat = (kp_heat > 0.5).float()
        kp_heat_prime = utils.improc.xyz2heatmaps(kp_xyz_mem, Z1, Y1, X1, sigma=sigma)
        kp_heat_prime = (kp_heat_prime > 0.5).float()

        kp_bev = torch.max(kp_heat, dim=3)[0]
        kp_fro = torch.max(kp_heat, dim=2)[0]
        self.summ_soft_seg_thr('%s_bev' % name, kp_bev, thr=0.5)
        self.summ_soft_seg_thr('%s_fro' % name, kp_fro, thr=0.5)

    def summ_rotating_pc(self, name, zyx_pc, rgb_pc, blacken_zeros=False):
        my_dpi = 100
        fig = plt.figure(figsize = (480/my_dpi, 480/my_dpi), dpi=my_dpi)
        fig.add_subplot(111, projection='3d')
        fig.tight_layout(pad=0)
        ax = plt.gca()

        pc_x = zyx_pc[:,2].cpu().numpy()
        pc_y = zyx_pc[:,0].cpu().numpy()
        pc_z = -zyx_pc[:,1].cpu().numpy()
        pc_c = rgb_pc.cpu().numpy() + 0.5
        ax.scatter(pc_x, pc_y, pc_z, c = pc_c, s = 0.1)

        # collect views
        views = []
        for angle in range(0,360,20):
            ax.view_init(30, angle)
            buf = io.BytesIO()
            fig.savefig(buf, format='raw')
            buf.seek(0)
            data = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8), newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            data = data[:,:,:3]

            #print("angle", angle, data.shape)

            if angle > 0:
                views.append(data)

        views = np.stack(views, axis=0)
        rotate_seq = torch.from_numpy(views).permute(0,3,1,2).unsqueeze(0).cuda()
        return self.summ_gif(name, rotate_seq)
    

if __name__ == "__main__":
    logdir = './runs/my_test'
    writer = SummaryWriter(logdir = logdir)

    summ_writer = Summ_writer(writer, 0, 'my_test')

    '''test summ_rgbs'''
    # rand_input = torch.rand(1, 2, 128, 384, 3) - 0.5 #rand from -0.5 to 0.5
    # summ_rgbs(name = 'rgb', ims = torch.unbind(rand_input, dim=1), writer=writer, global_step=0)
    # rand_input = torch.rand(1, 2, 128, 384, 3) - 0.5 #rand from -0.5 to 0.5
    # summ_rgbs(name = 'rgb', ims = torch.unbind(rand_input, dim=1), writer=writer, global_step=1)

    '''test summ_occs'''
    # rand_input = torch.randint(low=0, high = 2, size=(1, 2, 32, 32, 32, 1)).type(torch.FloatTensor) #random 0 or 1
    # summ_occs(name='occs', occs=torch.unbind(rand_input, dim=1), writer=writer, global_step=0)
    # rand_input = torch.randint(low=0, high = 2, size=(1, 2, 32, 32, 32, 1)).type(torch.FloatTensor) #random 0 or 1
    # summ_occs(name='occs', occs=torch.unbind(rand_input, dim=1), writer=writer, global_step=1)

    '''test summ_unps'''
    # for global_step in [0, 1]:
    #     rand_occs = torch.randint(low=0, high = 2, size=(1, 2, 128, 128, 32, 1)).type(torch.FloatTensor) #random 0 or 1
    #     rand_unps = torch.rand(1, 2, 128, 128, 32, 3) - 0.5
    #     summ_unps(name='unps', unps=torch.unbind(rand_unps, dim=1), occs=torch.unbind(rand_occs, dim=1), writer=writer, global_step=global_step)

    '''test summ_feats'''
    # for global_step in [0, 1]:
    #     rand_feats = torch.rand(1, 2, 128, 128, 32, 3) - 0.5
    #     summ_feats(name='feats', feats=torch.unbind(rand_feats, dim=1), writer=writer, global_step=global_step)

    '''test summ_flow'''
    # rand_feats = torch.rand(2, 2, 128, 128) - 0.5
    # summ_writer.summ_flow('flow', rand_feats)

    '''test summ_flow'''
    rand_feats = torch.rand(2, 3, 128, 32, 128)
    summ_writer.summ_3D_flow(rand_feats)


    writer.close()

def convert_boxlist2d_to_masklist(boxlist2d, H, W):
    B, N, C = list(boxlist2d.shape)
    assert(C==4) # 2d boxes
    boxlist2d = utils.geom.unnormalize_boxlist2d(boxlist2d, H, W)
    yminlist, xminlist, ymaxlist, xmaxlist = torch.unbind(boxlist2d, dim=2)
    
    yminlist = yminlist.long()
    xminlist = xminlist.long()
    ymaxlist = ymaxlist.long()
    xmaxlist = xmaxlist.long()

    yminlist = yminlist.clamp(0, H-1)
    ymaxlist = ymaxlist.clamp(0, H-1)
    
    xminlist = xminlist.clamp(0, W-1)
    xmaxlist = xmaxlist.clamp(0, W-1)
    
    masklist = torch.zeros([B, N, H, W]).float().cuda()
    for b in list(range(B)):
        for n in list(range(N)):
            ymin = yminlist[b, n]
            xmin = xminlist[b, n]
            ymax = ymaxlist[b, n]
            xmax = xmaxlist[b, n]
            masklist[b,n,ymin:ymax,xmin:xmax] = 1.0
    return masklist

    
def plot_traj_3d(traj):
    # traj = batch['traj'] # B x S x 3

    # traj is B x S x 3
    traj = traj[0:1]
    
    # print('traj', traj.shape)
    B, S, C = list(traj.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = [plt.cm.RdYlBu(i) for i in np.linspace(0,1,S)]
    # print('colors', colors)
    
    for b in range(B):
        traj_b = traj[b]

        xs = traj_b[:,0].clone()
        ys = -traj_b[:,1].clone()
        zs = traj_b[:,2].clone()

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
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0) # 1 x 3 x H x W

    image = preprocess_color(image)
    
    plt.close()
    return image

def dilate2d(im, times=1, device='cuda'):
    weights2d = torch.ones(1, 1, 3, 3, device=device)
    for time in range(times):
        im = F.conv2d(im, weights2d, padding=1).clamp(0, 1)
    return im

def dilate3d(im, times=1, device='cuda'):
    weights3d = torch.ones(1, 1, 3, 3, 3, device=device)
    for time in range(times):
        im = F.conv3d(im, weights3d, padding=1).clamp(0, 1)
    return im

def erode2d(im, times=1, device='cuda'):
    weights2d = torch.ones(1, 1, 3, 3, device=device)
    for time in range(times):
        im = 1.0 - F.conv2d(1.0 - im, weights2d, padding=1).clamp(0, 1)
    return im

def erode3d(im, times=1, device='cuda'):
    weights3d = torch.ones(1, 1, 3, 3, 3, device=device)
    for time in range(times):
        im = 1.0 - F.conv3d(1.0 - im, weights3d, padding=1).clamp(0, 1)
    return im
