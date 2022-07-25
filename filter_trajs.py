import torch
import utils.samp

def filter_trajs(trajs_e, all_masks, all_flows_f, all_flows_b):
    B, S, N, D = trajs_e.shape
    B, S1, C, H, W = all_masks.shape
    assert(S1==S)
    B, S2, C, H, W = all_flows_f.shape
    assert(S2==S-1)
    # print('trajs_e', trajs_e.shape)

    # req inbounds the full way
    trajs_e_ = trajs_e.round() # B,S-1,N,2
    max_x = torch.max(trajs_e_[:,:,:,0], dim=1)[0]
    min_x = torch.min(trajs_e_[:,:,:,0], dim=1)[0]
    max_y = torch.max(trajs_e_[:,:,:,1], dim=1)[0]
    min_y = torch.min(trajs_e_[:,:,:,1], dim=1)[0]
    inb = (max_x <= W-1) & (min_x >= 0) & (max_y <= H-1) & (min_y >= 0)
    # print('inb', inb.shape, torch.sum(inb))
    trajs_e = trajs_e[:,:,inb.reshape(-1)]
    # print('trajs_e in', trajs_e.shape)

    # req that we stay in the same object id
    id0 = utils.samp.bilinear_sample2d(all_masks[:,0], trajs_e[:,0,:,0].round(), trajs_e[:,0,:,1].round()).reshape(-1) # N
    id_ok = torch.ones_like(id0) > 0
    for s in range(S):
        # the aliasing for masks and flow is slightly different,
        # so let's require the 3x3 neighborhood to match
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                idi = utils.samp.bilinear_sample2d(all_masks[:,s], trajs_e[:,s,:,0].round() + dx, trajs_e[:,s,:,1].round() + dy).reshape(-1) # N
                id_ok = id_ok & (idi == id0)
    trajs_e = trajs_e[:,:,id_ok]
    # print('trajs_e id', trajs_e.shape)

    # req forward-backward consistency
    fb_ok = torch.ones_like(trajs_e[0,0,:,0]) > 0
    for s in range(S-1):
        ff = utils.samp.bilinear_sample2d(all_flows_f[:,s], trajs_e[:,s,:,0].round(), trajs_e[:,s,:,1].round()).permute(0,2,1).reshape(-1, 2) # N,2
        bf = utils.samp.bilinear_sample2d(all_flows_b[:,s], trajs_e[:,s+1,:,0].round(), trajs_e[:,s+1,:,1].round()).permute(0,2,1).reshape(-1, 2) # N,2
        dist = torch.norm(ff+bf, dim=1)
        # print_stats('dist', dist)
        fb_ok = fb_ok & (dist < 0.5)
    trajs_e = trajs_e[:,:,fb_ok]
    # print('trajs_e fb', trajs_e.shape)

    return trajs_e
