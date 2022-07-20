import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils.improc
import utils.misc
import utils.basic
import utils.geom
import utils.samp

class Linker(nn.Module):
    def __init__(self):
        super(Linker, self).__init__()

    def meshgrid2d(self, B, Y, X, stride=1, margin=0, device='cpu'):
        # X_stepped, Y_stepped = (X-2*margin)//stride, (Y-2*margin)//stride

        # grid_y = torch.linspace(margin, Y-margin-1, Y_stepped, device=torch.device('cuda'))
        # grid_y = torch.reshape(grid_y, [1, Y_stepped, 1])
        # grid_y = grid_y.repeat(B, 1, X_stepped) # B x Y x X

        # grid_x = torch.linspace(margin, X-margin-1, X_stepped, device=torch.device('cuda'))
        # grid_x = torch.reshape(grid_x, [1, 1, X_stepped])
        # grid_x = grid_x.repeat(B, Y_stepped, 1)

        # return grid_y, grid_x
        grid_y = torch.arange(margin, Y-margin, stride, device=device, dtype=torch.float32)
        Y_stepped = grid_y.shape[0]
        grid_y = torch.reshape(grid_y, [1, Y_stepped, 1])

        grid_x = torch.arange(margin, X-margin, stride, device=device, dtype=torch.float32)
        X_stepped = grid_x.shape[0]
        grid_x = torch.reshape(grid_x, [1, 1, X_stepped])

        grid_y = grid_y.repeat(B, 1, X_stepped) # B x Y x X
        grid_x = grid_x.repeat(B, Y_stepped, 1)

        return grid_y, grid_x

    def prune_flow_field(self, flow_f, flow_b, mult_thresh=0.01, add_thresh=0.1, device='cpu'):
        B, C, H, W = flow_f.shape
        # xs_ori, ys_ori = self.meshgrid2d(1, H, W, stride=1, margin=0)
        flow_b_at_target_loc = utils.samp.backwarp_using_2d_flow(flow_b, flow_f, device=device)
        # utils.basic.print_stats('flow_b_at_target_loc', torch.norm(flow_b_at_target_loc, dim=1))
        diff = torch.norm((flow_f + flow_b_at_target_loc), dim=1) # B x H x W
        # off = (diff > 0.2*torch.norm(torch.cat([flow_f, flow_b_at_target_loc], dim=1), dim=1) + 1.0).float()
        # off = (diff > 0.1*torch.norm(torch.cat([flow_f, flow_b_at_target_loc], dim=1), dim=1) + 1.0).float()
        # off = (diff > 0.05*torch.norm(torch.cat([flow_f, flow_b_at_target_loc], dim=1), dim=1) + 0.5).float()
        # off = (diff > 0.05*torch.norm(torch.cat([flow_f, flow_b_at_target_loc], dim=1), dim=1) + 0.1).float()
        off = (diff > mult_thresh*torch.norm(torch.cat([flow_f, flow_b_at_target_loc], dim=1), dim=1) + add_thresh).float()
        # off = (diff**2 > 0.01*(torch.norm(flow_f, dim=1)**2 + torch.norm(flow_b_at_target_loc, dim=1)**2) + 0.5).float()
        on = 1.0 - off

        return on

    def terminate_at_t(self, inds_end, s, Start, X, Y, device='cpu'):
        # inds_end are the trajs to be terminate
        # inds_end is len-N
        # s is a scalar, indicating the current timestep
        traj_XYs = []
        traj_Ts = []

        for ind_end in inds_end:
            t_start = Start[ind_end] # the starting frame of this traj
            t_end = s
            time_interval = torch.arange(t_start, t_end+1).long().to(device)
            traj_Ts.append(time_interval)
            traj_XYs.append(torch.stack([X[time_interval, ind_end], Y[time_interval, ind_end]], dim=-1))


        return traj_XYs, traj_Ts

    def forward(self, rgbs, flows_f, flows_b,
                margin=10,
                stride=4,
                valids=None,
                summ_writer=None,
                born=True,
                mult_thresh=0.01,
                add_thresh=0.1,
                device='cpu'):
        # rgbs is B x S x 3 x H x W
        # flows_* is B x (S-1) x 2 x H x W
        B, S, C1, H, W = rgbs.shape
        B2, S2, C2, H2, W2 = flows_f.shape

        assert(B==B2)
        assert(S==S2+1)
        assert(H==H2)
        assert(W==W2)
        
        assert(B==1)

        ys, xs = self.meshgrid2d(B, H, W, stride, margin, device=device) # B x H x W H->Y, W->X

        # print(xs)

        # xs = xs.reshape(-1) + 0.5
        # ys = ys.reshape(-1) + 0.5
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)

        # print('xs', xs.shape)
        # print('ys', ys.shape)

        if valids is not None:
            valid_check = utils.samp.bilinear_sample2d(valids[:,0], xs.unsqueeze(0), ys.unsqueeze(0)).squeeze(1) # B x N
            print('valid_check', valid_check.shape)
            inds = (valid_check > 0.5).reshape(-1)
            print('inds', inds.shape)
            xs = xs[inds]
            ys = ys[inds]
        
        # print('xs', xs.shape)
        # print('ys', ys.shape)
        
        X = torch.zeros((S, 2*H*W//(stride**2))).to(device) # store all points
        Y = torch.zeros((S, 2*H*W//(stride**2))).to(device)
        Start = -torch.ones(2*H*W//(stride**2)).to(device) # initialize as -1, means invalid

        X[0, :len(xs)] = xs # initalize
        Y[0, :len(ys)] = ys #
        Start[:len(ys)] = 0 # start at frame 0

        trajs_XYs = []
        trajs_Ts = []

        if summ_writer is not None and summ_writer.save_this:
            # flow_b = flows_b[:, 0]
            # flow_f = flows_f[:, 0]
            # flow_f_norm = torch.norm(flow_f, dim=1, keepdim=True) # B x 1 x H x W
            # utils.basic.print_stats('flow_f_norm', flow_f_norm) 

            # summ_writer.summ_flow('flow/flow_f', flow_f, clip=20)
            # summ_writer.summ_flow('flow/flow_b', flow_b, clip=20)

            # thresholds = [0.1, 0.5, 1.0, 5.0]
            # for thres in thresholds:
            #     summ_writer.summ_oned('flow/flow_norm_{}'.format(thres), (flow_f_norm > thres).float())

            flow_f_vis = []
            flow_b_vis = []
            flow_f_blend_vis = []
            for s in range(S-1):
                flow_f = flows_f[:, s]
                flow_b = flows_b[:, s]
                flow_f_vis.append(summ_writer.summ_flow('', flow_f, clip=20, only_return=True))
                flow_b_vis.append(summ_writer.summ_flow('', flow_b, clip=20, only_return=True))
                im1 = 0.7*(flow_f_vis[-1].float())
                im2 = 0.3*(summ_writer.summ_rgb('', rgbs[:, s], only_return=True)).float()
                flow_f_blend_vis.append((im1+im2).type(torch.ByteTensor)) 


            summ_writer.summ_rgbs('flow/flow_f', flow_f_vis)
            summ_writer.summ_rgbs('flow/flow_f_rgb', flow_f_blend_vis)
            summ_writer.summ_rgbs('flow/flow_b', flow_b_vis)


        for s in range(S-1):
            rgb0 = rgbs[:, s] # B x 3 x H x W
            rgb1 = rgbs[:, s+1]
            flow_f = flows_f[:, s] # B x 2 x H x W
            flow_b = flows_b[:, s] # B x 2 x H x W
            # note we feed aligned flows_f and flows_b, so technically flows_b starts at frame1

            if valids is not None:
                valid0 = valids[:, s] # B x 1 x H x W
                valid1 = valids[:, s+1]

            # filter pts based on flow filtering
            on = self.prune_flow_field(flow_f, flow_b, mult_thresh=mult_thresh, add_thresh=add_thresh, device=device) # B x H x W
            
            inds = torch.where(Start >= 0)[0]
            xs_ori = X[s:s+1, inds] # B x N_p
            ys_ori = Y[s:s+1, inds] # B x N_p

            uv = utils.samp.bilinear_sample2d(flow_f, xs_ori, ys_ori) # B x 2 x N, forward flow at the discrete points
            u = uv[:, 0] # B x N
            v = uv[:, 1]
            
            xs_tar = xs_ori + u # B x N_p
            ys_tar = ys_ori + v

            print('s = %d' % s)
            if not s==(S-2):
                fb_check = utils.samp.bilinear_sample2d(on.unsqueeze(1), xs_ori, ys_ori).squeeze(1) # B x N
                # print(torch.sum(fb_check > 0.5))

                if valids is not None:
                    valid_check = utils.samp.bilinear_sample2d(valid0, xs_ori, ys_ori).squeeze(1) # B x N

                margin_check = ((xs_tar >= margin) &
                                (ys_tar >= margin) &
                                (xs_tar < W - margin) &
                                (ys_tar < H - margin)) # choose inbound pts & pass forward-backward check
                if valids is not None:
                    choose = margin_check & (fb_check > 0.5) & (valid_check > 0.5)
                else:
                    choose = margin_check & (fb_check > 0.5)
            else:
                print('last iter; not applying any checks')
                # choose =
                choose = torch.ones_like(inds) > 0
                
            choose = choose.squeeze(0) # N_p
            inds_on = inds[choose]
            inds_off = inds[~choose]

            X[s+1, inds_on] = xs_tar[0, choose]
            Y[s+1, inds_on] = ys_tar[0, choose]

            # terminate
            traj_XYs, traj_Ts = self.terminate_at_t(inds_off, s, Start, X, Y, device=device)
            trajs_XYs.extend(traj_XYs) # each element is len_history x 2
            trajs_Ts.extend(traj_Ts) # each element is len_history
            Start[inds_off] = -1

            # sample new points at t+1-> cover with 0 the area to be sampled and 1 the area to be left untouched
            map_occupied = torch.zeros(B, 1, H, W).to(device)
            map_occupied[:, :, ys_tar[0, choose].long(), xs_tar[0, choose].long()] = 1.0
            img_dilation_kernel = torch.ones(1, 1, 5, 5).to(device)
            map_occupied = F.conv2d(map_occupied, img_dilation_kernel, padding=2)
            map_occupied = (map_occupied > 0.0).float() # B x 1 x H x W

            map_occ = utils.samp.bilinear_sample2d(map_occupied, xs.unsqueeze(0), ys.unsqueeze(0)).squeeze()
            map_free = 1.0 - map_occ
            if valids is not None:
                valid_ok = utils.samp.bilinear_sample2d(valid0, xs.unsqueeze(0), ys.unsqueeze(0)).squeeze()
                map_free = map_free * valid_ok
            # ind_map_free = utils.samp.bilinear_sample2d(map_occupied, xs.unsqueeze(0), ys.unsqueeze(0)).squeeze() == 0.0
            ind_map_free = map_free==1.0
            xs_added = xs[ind_map_free]
            ys_added = ys[ind_map_free]
            

            if born: # add borning of new points
                num_added = len(xs_added)
                free_inds = torch.where(Start < 0)[0] # we re-use slots in Start
                min_added = min(num_added, len(free_inds))
                X[s+1, free_inds[:min_added]] = xs_added[:min_added]
                Y[s+1, free_inds[:min_added]] = ys_added[:min_added]
                Start[free_inds[:min_added]] = s+1

            # return torch.stack([X[s:s+1, inds_on], Y[s:s+1, inds_on]], dim=-1), torch.stack([X[s+1:s+2, inds_on], Y[s+1:s+2, inds_on]], dim=-1)

       
        # terminate all remaining trajs
        inds_on = torch.where(Start >= 0)[0]
        traj_XYs, traj_Ts = self.terminate_at_t(inds_on, S-1, Start, X, Y)
        trajs_XYs.extend(traj_XYs) # each element is len_history x 2
        trajs_Ts.extend(traj_Ts) # each element is len_history

        return trajs_XYs, trajs_Ts

if __name__ == "__main__":
    net = Linker()
    rgbs = torch.zeros((1, 10, 3, 64, 32)).cuda()
    flow_f = torch.zeros((1, 9, 3, 64, 32)).cuda()
    flow_b = torch.zeros((1, 9, 3, 64, 32)).cuda()
    net(rgbs, flow_f, flow_b)








