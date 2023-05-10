from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2

# import sly_globals as g
# from pathlib import Path
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from nets.pips import Pips


def pad_origin(h_origin: int, w_origin: int, new_size: Tuple[int, int]):
    H_RESIZED, W_RESIZED = new_size

    h_mult = h_origin // H_RESIZED + 1
    w_mult = w_origin // W_RESIZED + 1

    h_pad = abs(H_RESIZED * h_mult - h_origin)
    w_pad = abs(W_RESIZED * w_mult - w_origin)

    top_pad, left_pad = h_pad // 2, w_pad // 2
    bot_pad, right_pad = h_pad - top_pad, w_pad - left_pad

    return left_pad, right_pad, top_pad, bot_pad


def check_bounds(points: np.ndarray, h_max: int, w_max: int):
    points[:, 0] = np.clip(points[:, 0], a_max=w_max - 1, a_min=0)
    points[:, 1] = np.clip(points[:, 1], a_max=h_max - 1, a_min=0)
    return points


def run_model(
    model: Pips,
    frames: torch.Tensor,
    orig_points: torch.Tensor,
    resized_shape: Tuple[int, int],
    frames_per_iter: int = 8,
    device: str = "cpu",
):
    points_number = len(orig_points)
    B, S, C, H_origin, W_origin = frames.shape
    rgbs_ = frames.reshape(B * S, C, H_origin, W_origin)
    H_resized, W_resized = resized_shape
    pads = pad_origin(H_origin, W_origin, resized_shape)
    rgbs_ = F.pad(rgbs_, pads, "constant", 0)
    _, _, H_padded, W_padded = rgbs_.shape
    rgbs_ = F.interpolate(rgbs_, (H_resized, W_resized), mode="bilinear")
    rgbs = rgbs_.reshape(B, S, C, H_resized, W_resized)

    Rx, Ry = W_padded / W_resized, H_padded / H_resized

    points = torch.clone(orig_points)
    points[:, 0] = (points[:, 0] + pads[0]) / Rx
    points[:, 1] = (points[:, 1] + pads[2]) / Ry
    xy0 = points.reshape(B, points_number, 2).int()

    trajs_e = torch.zeros((B, S, points_number, 2), dtype=torch.float32, device=device)

    for n in range(points_number):
        cur_frame = 0
        done = False
        traj_e = torch.zeros((B, S, 2), dtype=torch.float32, device=device)
        traj_e[:, 0] = xy0[:, n]  # B, 1, 2  # set first position
        feat_init = None
        while not done:
            end_frame = cur_frame + frames_per_iter

            rgb_seq: torch.Tensor = rgbs[:, cur_frame:end_frame]
            rgb_seq = rgb_seq.to(torch.device(device)).float()
            S_local = rgb_seq.shape[1]  # may become less then frames_per_iter

            # add new frames if S_local != frames_per_iter
            rgb_seq = torch.cat(
                [rgb_seq, rgb_seq[:, -1].unsqueeze(1).repeat(1, 8 - S_local, 1, 1, 1)], dim=1
            )

            outs = model(
                traj_e[:, cur_frame].reshape(1, -1, 2),
                rgb_seq,
                iters=6,
                feat_init=feat_init,
                return_feat=True,
            )
            rgb_seq.cpu()
            torch.cuda.empty_cache()
            preds = outs[0]
            vis = outs[2]  # B, S, 1
            feat_init = outs[3]

            vis = torch.sigmoid(vis)  # visibility confidence
            xys = preds[-1].reshape(1, frames_per_iter, 2)
            traj_e[:, cur_frame:end_frame] = xys[:, :S_local]

            found_skip = False
            thr = 0.9
            si_last = frames_per_iter - 1  # last frame we are willing to take
            si_earliest = 1  # earliest frame we are willing to take
            si = si_last
            while not found_skip:
                if vis[0, si] > thr:
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
        trajs_e[:, :, n] = traj_e

    trajs_e = trajs_e.squeeze(0)  # .permute(1, 0, 2)  # sequence len, points_num, 2
    trajs_e[:, :, 0] = trajs_e[:, :, 0] * Rx - pads[0]
    trajs_e[:, :, 1] = trajs_e[:, :, 1] * Ry - pads[2]
    preds = check_bounds(
        trajs_e.detach().cpu().numpy().squeeze(),
        h_max=H_origin,
        w_max=W_origin,
    )
    return preds


def draw_and_save(img: Union[np.ndarray, torch.Tensor], cord: Tuple[int, int]):
    if isinstance(img, torch.Tensor):
        np_img = img.detach().cpu().numpy()
    else:
        np_img = img.copy()

    np_img = cv2.circle(np_img, cord, radius=1, color=(255, 0, 0), thickness=2)
    i = np.random.randint(1000)
    cv2.imwrite(f"img_{i}.jpg", np_img)
