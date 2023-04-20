from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import supervisely as sly
from supervisely.geometry.geometry import Geometry
import sly_globals as g

import numpy as np
import torch
import torch.nn.functional as F

import saverloader
from nets.pips import Pips


FRAMES_PER_ITER = 8
H_RESIZED, W_RESIZED = 360, 640


def init_tracker(logger, stride: int = 4, frames_per_iter: int = FRAMES_PER_ITER):
    init_dir = "/workspaces/pips_tracker/reference_model"
    # init_dir = "/pips_tracker/reference_model"
    model = Pips(S = frames_per_iter, stride = stride).cuda()
    try:
        logger.info("Loading model.")
        if init_dir:
            _ = saverloader.load(init_dir, model)
        model.eval()
    except Exception as e:
        logger.error(f"Something goes wrong: {str(e)}")
        raise e
    logger.info("Model loaded.")
    return model


def pad_origin(h_origin: int, w_origin: int):
    h_pad, w_pad = 0, 0
    cur_h, cur_w = H_RESIZED, W_RESIZED

    while cur_h < h_origin:
        cur_h += H_RESIZED
    else:
        h_pad = cur_h - h_origin
    
    while cur_w < w_origin:
        cur_w += W_RESIZED
    else:
        w_pad = cur_w - w_origin
    
    top_pad, left_pad = h_pad // 2, w_pad // 2
    bot_pad, right_pad = h_pad - top_pad, w_pad - left_pad 
    
    return left_pad, right_pad, top_pad, bot_pad


def check_bounds(points: np.ndarray, h_max: int, w_max: int):
    points[:, 0] = np.clip(points[:, 0], a_max=w_max, a_min=0)
    points[:, 1] = np.clip(points[:, 1], a_max=h_max, a_min=0)
    return points


def run_model(model: Pips, frames: torch.Tensor, orig_points: torch.Tensor):
    points_number = len(orig_points)
    B, S, C, H_origin, W_origin = frames.shape
    rgbs_ = frames.reshape(B*S, C, H_origin, W_origin)
    H_resized, W_resized = H_RESIZED, W_RESIZED
    pads = pad_origin(H_origin, W_origin)
    rgbs_ = F.pad(rgbs_, pads, "constant", 0)
    rgbs_ = F.interpolate(rgbs_, (H_resized, W_resized), mode='bilinear')
    rgbs = rgbs_.reshape(B, S, C, H_resized, W_resized)

    Rx, Ry = W_origin / W_resized, H_origin / H_resized

    points = torch.clone(orig_points)
    points[:, 0] = (points[:, 0] + pads[0]) / Rx
    points[:, 1] = (points[:, 1] + pads[2]) / Ry
    xy0 = points.reshape(B, points_number, 2).int()

    trajs_e = torch.zeros((B, S, points_number, 2), dtype=torch.float32, device='cuda')

    for n in range(points_number):
        cur_frame = 0
        done = False
        traj_e = torch.zeros((B, S, 2), dtype=torch.float32, device='cuda')
        traj_e[:,0] = xy0[:,n] # B, 1, 2  # set first position 
        feat_init = None
        while not done:
            end_frame = cur_frame + FRAMES_PER_ITER

            rgb_seq: torch.Tensor = rgbs[:,cur_frame:end_frame]
            rgb_seq = rgb_seq.cuda().float() 
            S_local = rgb_seq.shape[1]  # may become less then FRAMES_PER_ITER

            # add new frames if S_local != FRAMES_PER_ITER
            rgb_seq = torch.cat([rgb_seq, rgb_seq[:,-1].unsqueeze(1).repeat(1,8-S_local,1,1,1)], dim=1) 

            outs = model(traj_e[:,cur_frame].reshape(1, -1, 2), rgb_seq, iters=6, feat_init=feat_init, return_feat=True)
            rgb_seq.cpu()
            torch.cuda.empty_cache()
            preds = outs[0]
            vis = outs[2] # B, S, 1
            feat_init = outs[3]
            
            vis = torch.sigmoid(vis) # visibility confidence
            xys = preds[-1].reshape(1, FRAMES_PER_ITER, 2)
            traj_e[:,cur_frame:end_frame] = xys[:,:S_local]

            found_skip = False
            thr = 0.9
            si_last = FRAMES_PER_ITER - 1 # last frame we are willing to take
            si_earliest = 1 # earliest frame we are willing to take
            si = si_last
            while not found_skip:
                if vis[0,si] > thr:
                    found_skip = True
                else:
                    si -= 1
                if si == si_earliest:
                    # print('decreasing thresh')
                    thr -= 0.02
                    si = si_last
            # # print('found skip at frame %d, where we have' % si, vis[0,si].detach().item())

            cur_frame = cur_frame + si_last

            if cur_frame >= S:
                done = True
        trajs_e[:,:,n] = traj_e

    trajs_e = trajs_e.squeeze(0) # .permute(1, 0, 2)  # sequence len, points_num, 2
    trajs_e[:, :, 0] = trajs_e[:, :, 0] * Rx - pads[0]
    trajs_e[:, :, 1] = trajs_e[:, :, 1] * Ry - pads[2]
    return trajs_e.detach().cpu().numpy()


def geometry_to_np(figure: Geometry):
    if isinstance(figure, sly.Rectangle):
        left_top = [figure.left, figure.top]
        right_bottom = [figure.right, figure.bottom]
        return np.array([left_top, right_bottom])
    if isinstance(figure, sly.Point):
        return np.array([[figure.col, figure.row]])
    if isinstance(figure, sly.Polygon):
        return figure.exterior_np[:, ::-1].copy()
    raise ValueError(f"Can't process figures with type `{figure.geometry_name()}`")

def np_to_geometry(points: np.ndarray, geom_type: str) -> Geometry:
    if geom_type == "rectangle":
        points = points.astype(int)
        left, right = min(points[:, 0]), max(points[:, 0])
        top, bottom = min(points[:, 1]), max(points[:, 1])
        fig = sly.Rectangle(
            top=top,
            left=left,
            bottom=bottom,
            right=right,
        )
        return fig
    if geom_type == "point":
        col, row = points.squeeze().astype(int)
        return sly.Point(row, col)
    if geom_type == "polygon":
        obj = points.astype(int)[:, ::-1]
        exterior = [sly.PointLocation(*obj_point) for obj_point in obj]
        return sly.Polygon(exterior=exterior)
    raise ValueError(f"Can't process figures with type `{geom_type}`")
