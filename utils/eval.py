import io as sysio
import time
# import hyperparams as hyp
import random
# import numba
# import hardPositiveMiner

import numpy as np
import utils.geom
import utils.box
import utils.ap
import torch

EPS = 1e-6

def make_border_green(vis):
    vis = np.copy(vis)
    vis[0,:,0] = 0
    vis[0,:,1] = 255
    vis[0,:,2] = 0
    
    vis[-1,:,0] = 0
    vis[-1,:,1] = 255
    vis[-1,:,2] = 0

    vis[:,0,0] = 0
    vis[:,0,1] = 255
    vis[:,0,2] = 0
    
    vis[:,-1,0] = 0
    vis[:,-1,1] = 255
    vis[:,-1,2] = 0
    return vis

def make_border_black(vis):
    vis = np.copy(vis)
    vis[0,:,:] = 0
    vis[-1,:,:] = 0
    vis[:,0,:] = 0
    vis[:,-1,:] = 0
    return vis

def compute_precision(xxx_todo_changeme, xxx_todo_changeme1, recalls=[1,3,5], pool_size=100):
    # inputs are lists
    # list elements are H x W x C
    
    (emb_e, vis_e) = xxx_todo_changeme
    (emb_g, vis_g) = xxx_todo_changeme1
    assert(len(emb_e)==len(emb_g))
    B = len(emb_e)
    precision = np.zeros(len(recalls), np.float32)
    # print 'precision B = %d' % B

    if len(vis_e[0].shape)==4:
        # H x W x D x C
        # squish the height dim, and look at the birdview
        vis_e = [np.mean(vis, axis=0) for vis in vis_e]
        vis_g = [np.mean(vis, axis=0) for vis in vis_g]
        H = vis_e[0].shape[0]
        W = vis_e[0].shape[1]
    elif len(vis_e[0].shape)==3:
        # H x W x C
        H = vis_e[0].shape[0]
        W = vis_e[0].shape[1]
    else:
        assert(False) # vis_e shape is weird

    perm = np.random.permutation(B)
    vis_inds = perm[:10] # just vis 10 queries
    
    # print 'B = %d; pool_size = %d' % (B, pool_size)
    
    if B >= pool_size: # otherwise it's not going to be accurate
        emb_e = np.stack(emb_e, axis=0)
        emb_g = np.stack(emb_g, axis=0)
        # emb_e = np.concatenate(emb_e, axis=0)
        # emb_g = np.concatenate(emb_g, axis=0)
        vect_e = np.reshape(emb_e, [B, -1])
        vect_g = np.reshape(emb_g, [B, -1])
        scores = np.dot(vect_e, np.transpose(vect_g))
        ranks = np.flip(np.argsort(scores), axis=1)

        vis = []
        for i in vis_inds:
            minivis = []
            # first col: query
            # minivis.append(vis_e[i])
            minivis.append(make_border_black(vis_e[i]))
            
            # # second col: true answer
            # minivis.append(vis_g[i])
            
            # remaining cols: ranked answers
            for j in list(range(10)):
                v = vis_g[ranks[i, j]]
                if ranks[i, j]==i:
                    minivis.append(make_border_green(v))
                else:
                    minivis.append(v)
            # concat retrievals along width
            minivis = np.concatenate(minivis, axis=1)
            # print 'got this minivis:', 
            # print minivis.shape
            
            vis.append(minivis)
        # concat examples along height
        vis = np.concatenate(vis, axis=0)
        # print 'got this vis:', 
        # print vis.shape
            
        for recall_id, recall in enumerate(recalls):
            for emb_id in list(range(B)):
                if emb_id in ranks[emb_id, :recall]:
                    precision[recall_id] += 1
            # print("precision@", recall, float(precision[recall_id])/float(B))
        precision = precision/float(B)
    else:
        precision = np.nan*precision
        vis = np.zeros((H*10, W*11, 3), np.uint8)
    # print 'precision  %.2f' % np.mean(precision)
    
    return precision, vis

def compute_patch_based_vis(pool_e, pool_g, num_embeds,summ_writer):
    hpm = hardPositiveMiner.HardPositiveMiner()

    num_patches_per_emb = hyp.max.num_patches_per_emb
    scores = torch.zeros((num_embeds, num_embeds)).cuda()

    h_init_e = np.random.randint(2,14)
    d_init_e = np.random.randint(2,14)
    w_init_e = np.random.randint(2,14)
    _, unps_e = pool_e.fetch()
    _, unps_g = pool_e.fetch()
    featQuery_i , perm_i =  hpm.extractPatches_det(pool_e,d_init_e,h_init_e,w_init_e)
    featQuery_i = featQuery_i[:1]
    
    unp_e = unps_e[0]
    topkImg_i, topkScale, topkValue_i, topkW , topkH , topkD, topkR  = hpm.RetrievalResForExpectation(pool_g, featQuery_i)
    summ_writer.summ_evalmines("eval_mines",[[topkImg_i,topkD,topkH,topkW,topkR],[d_init_e,h_init_e,w_init_e],[unps_g,unp_e]])


def compute_precision_o_cuda(pool_e, pool_g, pool_size=100, summ_writer=None,steps_done=0):
    (emb_e, vis_e) = pool_e.fetch()
    (emb_g, vis_g) = pool_g.fetch()
    
    assert(len(emb_e)==len(emb_g))
    B = len(emb_e)
    if len(vis_e[0].shape)==4:
        vis_e = [np.mean(vis, axis=0) for vis in vis_e]
        vis_g = [np.mean(vis, axis=0) for vis in vis_g]
        H = vis_e[0].shape[0]
        W = vis_e[0].shape[1]
    elif len(vis_e[0].shape)==3:
        # H x W x C
        H = vis_e[0].shape[0]
        W = vis_e[0].shape[1]
    else:
        assert(False) 

    perm = np.random.permutation(B)
    vis_inds = perm[:10] # just vis 10 queries    
    exp_done = False
    if (B >= pool_size) and summ_writer.save_this : # otherwise it's not going to be accurate
        emb_e = torch.stack(emb_e, axis=0)
        emb_g = torch.stack(emb_g, axis=0)
        vect_e = torch.nn.functional.normalize(torch.reshape(emb_e, [B, -1]),dim=1)
        vect_g = torch.nn.functional.normalize(torch.reshape(emb_g, [B, -1]),dim=1)
        compute_patch_based_vis(pool_e, pool_g, len(emb_e),summ_writer)

def drop_invalid_lrts(lrtlist_e, lrtlist_g, scorelist_e, scorelist_g):
    B, N, D = lrtlist_e.shape
    assert(B==1)
    # unlike drop_invalid_boxes, this is all in pt
    # print('before:')
    # print(lrtlist_e.shape)
    # print(lrtlist_g.shape)
    # print(scorelist_e.shape)
    # print(scorelist_g.shape)
    # lrtlists are shaped B x N x 19
    # scorelists are shaped B x N
    lrtlist_e_, scorelist_e_, lrtlist_g_, scorelist_g_ = [], [], [], []
    lenlist_e, _ = utils.geom.split_lrtlist(lrtlist_e)
    for i in list(range(len(lrtlist_e))):
        lrt_e = lrtlist_e[i]
        score_e = scorelist_e[i]
        len_e = lenlist_e[i]
        valid_e = torch.where(len_e[:, 0] > 0.01)
        lrtlist_e_.append(lrt_e[valid_e])
        scorelist_e_.append(score_e[valid_e])
    for i in list(range(len(lrtlist_g))):
        lrt_g = lrtlist_g[i]
        score_g = scorelist_g[i]
        valid_g = torch.where(score_g > 0.5)
        lrtlist_g_.append(lrt_g[valid_g])
        scorelist_g_.append(score_g[valid_g])
    lrtlist_e, lrtlist_g, scorelist_e, scorelist_g = torch.stack(lrtlist_e_), torch.stack(lrtlist_g_), torch.stack(scorelist_e_), torch.stack(scorelist_g_)
    # print('after')
    # print(lrtlist_e.shape)
    # print(lrtlist_g.shape)
    return lrtlist_e, lrtlist_g, scorelist_e, scorelist_g

def get_mAP_from_lrtlist(lrtlist_e, scores, lrtlist_g, iou_thresholds):
    # lrtlist are 1 x N x 19
    # the returns are numpy
    B, Ne, _ = list(lrtlist_e.shape)
    B, Ng, _ = list(lrtlist_g.shape)
    assert(B==1)
    scores = scores.detach().cpu().numpy()
    # print("e", boxes_e, "g", boxes_g, "score", scores)
    scores = scores.flatten()
    # size [N, 8, 3]
    ious_3d = np.zeros((Ne, Ng), dtype=np.float32)
    ious_2d = np.zeros((Ne, Ng), dtype=np.float32)
    for i in list(range(Ne)):
        for j in list(range(Ng)):
            iou_3d, iou_2d = utils.geom.get_iou_from_corresponded_lrtlists(lrtlist_e[:, i:i+1], lrtlist_g[:, j:j+1])
            ious_3d[i, j] = iou_3d[0, 0]
            ious_2d[i, j] = iou_2d[0, 0]
    maps_3d = []
    maps_2d = []
    for iou_threshold in iou_thresholds:
        map3d, precision, recall, overlaps = utils.ap.compute_ap(
            "box3d_" + str(iou_threshold), scores, ious_3d, iou_threshold=iou_threshold)
        maps_3d.append(map3d)
        map2d, precision, recall, overlaps = utils.ap.compute_ap(
            "box2d_" + str(iou_threshold), scores, ious_2d, iou_threshold=iou_threshold)
        maps_2d.append(map2d)
    maps_3d = np.stack(maps_3d, axis=0).astype(np.float32)
    maps_2d = np.stack(maps_2d, axis=0).astype(np.float32)
    if np.isnan(maps_3d).any():
        # print('got these nans in maps; setting to zero:', maps)
        maps_3d[np.isnan(maps_3d)] = 0.0
    if np.isnan(maps_2d).any():
        # print('got these nans in maps; setting to zero:', maps)
        maps_2d[np.isnan(maps_2d)] = 0.0

    # print("maps_3d", maps_3d)
    return maps_3d, maps_2d
        
def drop_invalid_boxes_from_2d_boxlists(boxlist_e, boxlist_g, scorelist_e, scorelist_g):
    B, N, D = boxlist_e.shape
    assert(B==1)
    boxlist_e_, scorelist_e_, boxlist_g_, scorelist_g_ = [], [], [], []
    lenlist_e = utils.geom.get_lens_from_boxlist(boxlist_e)
    for i in list(range(len(boxlist_e))):
        box_e = boxlist_e[i]
        score_e = scorelist_e[i]
        len_e = lenlist_e[i]
        valid_e = torch.where(torch.sum(len_e > 0.01, dim=1) == 2)
        boxlist_e_.append(box_e[valid_e])
        scorelist_e_.append(score_e[valid_e])
    for i in list(range(len(boxlist_g))):
        box_g = boxlist_g[i]
        score_g = scorelist_g[i]
        valid_g = torch.where(score_g > 0.5)
        boxlist_g_.append(box_g[valid_g])
        scorelist_g_.append(score_g[valid_g])
    boxlist_e, boxlist_g, scorelist_e, scorelist_g = torch.stack(boxlist_e_), torch.stack(boxlist_g_), torch.stack(scorelist_e_), torch.stack(scorelist_g_)
    # print('after')
    # print(boxlist_e.shape)
    # print(boxlist_g.shape)
    return boxlist_e, boxlist_g, scorelist_e, scorelist_g

def get_mAP_from_2d_boxlists(boxlist_e, scores, boxlist_g, iou_thresholds):
    # boxlists are 1 x N x 4
    B, Ne, _ = list(boxlist_e.shape)
    B, Ng, _ = list(boxlist_g.shape)
    assert(B==1)
    
    boxlist_e = boxlist_e.detach().cpu().numpy()
    boxlist_g = boxlist_g.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    
    # print("e", boxes_e, "g", boxes_g, "score", scores)
    scores = scores.flatten()
    # size [N, 8, 3]
    ious_2d = np.zeros((Ne, Ng), dtype=np.float32)
    for i in list(range(Ne)):
        iou_2d = utils.box.boxlist_2d_iou(boxlist_e[:,i:i+1].repeat(Ng, axis=1), boxlist_g)
        ious_2d[i,:] = iou_2d[0].squeeze(axis=1)
    maps_2d = []
    for iou_threshold in iou_thresholds:
        map2d, precision, recall, overlaps = utils.ap.compute_ap(
            "box2d_" + str(iou_threshold), scores, ious_2d, iou_threshold=iou_threshold)
        maps_2d.append(map2d)
    maps_2d = np.stack(maps_2d, axis=0).astype(np.float32)
    if np.isnan(maps_2d).any():
        # print('got these nans in maps; setting to zero:', maps)
        maps_2d[np.isnan(maps_2d)] = 0.0
    return maps_2d

    

# def get_mAP(prec):
#     sums = 0
#     for i in list(range(0, len(prec), 4)):
#         sums += prec[i]
#     return sums / 11 * 100


# #@numba.jit
# def get_thresholds(scores, num_gt, num_sample_pts=41):
#     scores.sort()
#     scores = scores[::-1]
#     current_recall = 0
#     thresholds = []
#     for i, score in enumerate(scores):
#         l_recall = (i + 1) / num_gt
#         if i < (len(scores) - 1):
#             r_recall = (i + 2) / num_gt
#         else:
#             r_recall = l_recall
#         if (((r_recall - current_recall) < (current_recall - l_recall))
#                 and (i < (len(scores) - 1))):
#             continue
#         # recall = l_recall
#         thresholds.append(score)
#         current_recall += 1 / (num_sample_pts - 1.0)
#     # print(len(thresholds), len(scores), num_gt)
#     return thresholds


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van',
                   'person_sitting', 'car', 'tractor', 'trailer']
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in list(range(num_gt)):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == "Pedestrian".lower()
              and "Person_sitting".lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == "Car".lower() and "Van".lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
                or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
                or (height <= MIN_HEIGHT[difficulty])):
            # if gt_anno["difficulty"][i] > difficulty or gt_anno["difficulty"][i] == -1:
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
    # for i in list(range(num_gt)):
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    for i in list(range(num_dt)):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


# @numba.jit(nopython=True)
# def image_box_overlap(boxes, query_boxes, criterion=-1):
#     N = boxes.shape[0]
#     K = query_boxes.shape[0]
#     overlaps = np.zeros((N, K), dtype=boxes.dtype)
#     for k in list(range(K)):
#         qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
#                      (query_boxes[k, 3] - query_boxes[k, 1]))
#         for n in list(range(N)):
#             iw = (min(boxes[n, 2], query_boxes[k, 2]) -
#                   max(boxes[n, 0], query_boxes[k, 0]))
#             if iw > 0:
#                 ih = (min(boxes[n, 3], query_boxes[k, 3]) -
#                       max(boxes[n, 1], query_boxes[k, 1]))
#                 if ih > 0:
#                     if criterion == -1:
#                         ua = (
#                             (boxes[n, 2] - boxes[n, 0]) *
#                             (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
#                     elif criterion == 0:
#                         ua = ((boxes[n, 2] - boxes[n, 0]) *
#                               (boxes[n, 3] - boxes[n, 1]))
#                     elif criterion == 1:
#                         ua = qbox_area
#                     else:
#                         ua = 1.0
#                     overlaps[n, k] = iw * ih / ua
#     return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    assert(False) # rotate_iou_gpu_eval (from the old nms_gpu) seems to require cudatoolkit=7.5, which seems unavailable
    # riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


# @numba.jit(nopython=True, parallel=True)
# def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
#     # ONLY support overlap in CAMERA, not lider.
#     N, K = boxes.shape[0], qboxes.shape[0]
#     for i in list(range(N)):
#         for j in list(range(K)):
#             if rinc[i, j] > 0:
#                 iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
#                     boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

#                 if iw > 0:
#                     area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
#                     area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
#                     inc = iw * rinc[i, j]
#                     if criterion == -1:
#                         ua = (area1 + area2 - inc)
#                     elif criterion == 0:
#                         ua = area1
#                     elif criterion == 1:
#                         ua = area2
#                     else:
#                         ua = 1.0
#                     rinc[i, j] = inc / (EPS + ua)
#                 else:
#                     rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


# #@numba.jit(nopython=True)
# def compute_statistics_jit(overlaps,
#                            gt_datas,
#                            dt_datas,
#                            ignored_gt,
#                            ignored_det,
#                            dc_bboxes,
#                            metric,
#                            min_overlap,
#                            thresh=0,
#                            compute_fp=False,
#                            compute_aos=False):
#     det_size = dt_datas.shape[0]
#     gt_size = gt_datas.shape[0]
#     dt_scores = dt_datas[:, -1]
#     dt_alphas = dt_datas[:, 4]
#     gt_alphas = gt_datas[:, 4]
#     dt_bboxes = dt_datas[:, :4]
#     # gt_bboxes = gt_datas[:, :4]

#     assigned_detection = [False] * det_size
#     ignored_threshold = [False] * det_size
#     if compute_fp:
#         for i in list(range(det_size)):
#             if (dt_scores[i] < thresh):
#                 ignored_threshold[i] = True
#     NO_DETECTION = -10000000
#     tp, fp, fn, similarity = 0, 0, 0, 0
#     # thresholds = [0.0]
#     # delta = [0.0]
#     thresholds = np.zeros((gt_size, ))
#     thresh_idx = 0
#     delta = np.zeros((gt_size, ))
#     delta_idx = 0
#     for i in list(range(gt_size)):
#         if ignored_gt[i] == -1:
#             continue
#         det_idx = -1
#         valid_detection = NO_DETECTION
#         max_overlap = 0
#         assigned_ignored_det = False

#         for j in list(range(det_size)):
#             if (ignored_det[j] == -1):
#                 continue
#             if (assigned_detection[j]):
#                 continue
#             if (ignored_threshold[j]):
#                 continue
#             overlap = overlaps[j, i]
#             dt_score = dt_scores[j]
#             if (not compute_fp and (overlap > min_overlap)
#                     and dt_score > valid_detection):
#                 det_idx = j
#                 valid_detection = dt_score
#             elif (compute_fp and (overlap > min_overlap)
#                   and (overlap > max_overlap or assigned_ignored_det)
#                   and ignored_det[j] == 0):
#                 max_overlap = overlap
#                 det_idx = j
#                 valid_detection = 1
#                 assigned_ignored_det = False
#             elif (compute_fp and (overlap > min_overlap)
#                   and (valid_detection == NO_DETECTION)
#                   and ignored_det[j] == 1):
#                 det_idx = j
#                 valid_detection = 1
#                 assigned_ignored_det = True

#         if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
#             fn += 1
#         elif ((valid_detection != NO_DETECTION)
#               and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
#             assigned_detection[det_idx] = True
#         elif valid_detection != NO_DETECTION:
#             # only a tp add a threshold.
#             tp += 1
#             # thresholds.append(dt_scores[det_idx])
#             thresholds[thresh_idx] = dt_scores[det_idx]
#             thresh_idx += 1
#             if compute_aos:
#                 # delta.append(gt_alphas[i] - dt_alphas[det_idx])
#                 delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
#                 delta_idx += 1

#             assigned_detection[det_idx] = True
#     if compute_fp:
#         for i in list(range(det_size)):
#             if (not (assigned_detection[i] or ignored_det[i] == -1
#                      or ignored_det[i] == 1 or ignored_threshold[i])):
#                 fp += 1
#         nstuff = 0
#         if metric == 0:
#             overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
#             for i in list(range(dc_bboxes.shape[0])):
#                 for j in list(range(det_size)):
#                     if (assigned_detection[j]):
#                         continue
#                     if (ignored_det[j] == -1 or ignored_det[j] == 1):
#                         continue
#                     if (ignored_threshold[j]):
#                         continue
#                     if overlaps_dt_dc[j, i] > min_overlap:
#                         assigned_detection[j] = True
#                         nstuff += 1
#         fp -= nstuff
#         if compute_aos:
#             tmp = np.zeros((fp + delta_idx, ))
#             # tmp = [0] * fp
#             for i in list(range(delta_idx)):
#                 tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
#                 # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
#             # assert len(tmp) == fp + tp
#             # assert len(delta) == tp
#             if tp > 0 or fp > 0:
#                 similarity = np.sum(tmp)
#             else:
#                 similarity = -1
#     return tp, fp, fn, similarity, thresholds[:thresh_idx]


# def get_split_parts(num, num_part):
#     same_part = num // num_part
#     remain_num = num % num_part
#     if same_part == 0:
#         return [remain_num]
#     elif remain_num == 0:
#         return [same_part] * num_part
#     else:
#         return [same_part] * num_part + [remain_num]


# #@numba.jit(nopython=True)
# def fused_compute_statistics(overlaps,
#                              pr,
#                              gt_nums,
#                              dt_nums,
#                              dc_nums,
#                              gt_datas,
#                              dt_datas,
#                              dontcares,
#                              ignored_gts,
#                              ignored_dets,
#                              metric,
#                              min_overlap,
#                              thresholds,
#                              compute_aos=False):
#     gt_num = 0
#     dt_num = 0
#     dc_num = 0
#     for i in list(range(gt_nums.shape[0])):
#         for t, thresh in enumerate(thresholds):
#             overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
#                                gt_num + gt_nums[i]]

#             gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
#             dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
#             ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
#             ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
#             dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
#             tp, fp, fn, similarity, _ = compute_statistics_jit(
#                 overlap,
#                 gt_data,
#                 dt_data,
#                 ignored_gt,
#                 ignored_det,
#                 dontcare,
#                 metric,
#                 min_overlap=min_overlap,
#                 thresh=thresh,
#                 compute_fp=True,
#                 compute_aos=compute_aos)
#             pr[t, 0] += tp
#             pr[t, 1] += fp
#             pr[t, 2] += fn
#             if similarity != -1:
#                 pr[t, 3] += similarity
#         gt_num += gt_nums[i]
#         dt_num += dt_nums[i]
#         dc_num += dc_nums[i]


# def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
#     """fast iou algorithm. this function can be used independently to
#     do result analysis. Must be used in CAMERA coordinate system.
#     Args:
#         gt_annos: dict, must from get_label_annos() in kitti_common.py
#         dt_annos: dict, must from get_label_annos() in kitti_common.py
#         metric: eval type. 0: bbox, 1: bev, 2: 3d
#         num_parts: int. a parameter for fast calculate algorithm
#     """
#     assert len(gt_annos) == len(dt_annos)
#     total_dt_num = np.stack([len(a["location"]) for a in dt_annos], 0)
#     total_gt_num = np.stack([len(a["location"]) for a in gt_annos], 0)
#     num_examples = len(gt_annos)
#     split_parts = get_split_parts(num_examples, num_parts)
#     parted_overlaps = []
#     example_idx = 0

#     for num_part in split_parts:
#         gt_annos_part = gt_annos[example_idx:example_idx + num_part]
#         dt_annos_part = dt_annos[example_idx:example_idx + num_part]
#         if metric == 0:
#             gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
#             dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
#             overlap_part = image_box_overlap(gt_boxes, dt_boxes)
#         elif metric == 1:
#             loc = np.concatenate(
#                 [a["location"][:, [0, 2]] for a in gt_annos_part], 0)
#             dims = np.concatenate(
#                 [a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
#             rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
#             gt_boxes = np.concatenate(
#                 [loc, dims, rots[..., np.newaxis]], axis=1)
#             loc = np.concatenate(
#                 [a["location"][:, [0, 2]] for a in dt_annos_part], 0)
#             dims = np.concatenate(
#                 [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
#             rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
#             dt_boxes = np.concatenate(
#                 [loc, dims, rots[..., np.newaxis]], axis=1)
#             overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
#                 np.float64)
#         elif metric == 2:
#             loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
#             dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
#             rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
#             gt_boxes = np.concatenate(
#                 [loc, dims, rots[..., np.newaxis]], axis=1)
#             loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
#             dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
#             rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
#             dt_boxes = np.concatenate(
#                 [loc, dims, rots[..., np.newaxis]], axis=1)
#             overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(
#                 np.float64)
#         else:
#             raise ValueError("unknown metric")
#         parted_overlaps.append(overlap_part)
#         example_idx += num_part
#     overlaps = []
#     example_idx = 0
#     for j, num_part in enumerate(split_parts):
#         gt_annos_part = gt_annos[example_idx:example_idx + num_part]
#         dt_annos_part = dt_annos[example_idx:example_idx + num_part]
#         gt_num_idx, dt_num_idx = 0, 0
#         for i in list(range(num_part)):
#             gt_box_num = total_gt_num[example_idx + i]
#             dt_box_num = total_dt_num[example_idx + i]
#             overlaps.append(
#                 parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
#                                    dt_num_idx:dt_num_idx + dt_box_num])
#             gt_num_idx += gt_box_num
#             dt_num_idx += dt_box_num
#         example_idx += num_part

#     return overlaps, parted_overlaps, total_gt_num, total_dt_num


# def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
#     gt_datas_list = [np.zeros([f['location'].shape[0],5],
#                               dtype=np.float64)
#                      for f in gt_annos]
#     dt_datas_list = [np.concatenate([np.zeros([f['location'].shape[0],5],
#                                               dtype=np.float64),
#                                      f['score'].reshape(-1,1)],
#                                     axis=1)
#                      for f in dt_annos]
#     ignored_gts = [np.zeros(f['location'].shape[0], dtype=np.int64)
#                    for f in gt_annos]
#     ignored_dts = [np.zeros(f['location'].shape[0], dtype=np.int64)
#                    for f in dt_annos]
#     dontcares = [np.zeros([0,4], dtype=np.float64)
#                  for f in gt_annos]
#     total_dc_num = np.zeros(len(gt_annos), dtype=np.int64)
#     total_num_valid_gt = sum(f['location'].shape[0]
#                              for f in gt_annos)
#     return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dts,
#             dontcares, total_dc_num, total_num_valid_gt)

def drop_invalid_boxes(boxlist_e, boxlist_g, scorelist_e, scorelist_g):
    # print('before:')
    print(boxlist_e.shape)
    print(boxlist_g.shape)
    boxlist_e_, boxlist_g_, scorelist_e_, scorelist_g_ = [], [], [], []
    for i in list(range(len(boxlist_e))):
        box_e = boxlist_e[i]
        # print('box_e', box_e.shape)
        score_e = scorelist_e[i]
        valid_e = np.where(box_e[:,3] > 0.0) # lx
        boxlist_e_.append(box_e[valid_e])
        scorelist_e_.append(score_e[valid_e])
    # print('boxlist_e_', boxlist_e_)
    for i in list(range(len(boxlist_g))):
        box_g = boxlist_g[i]
        score_g = scorelist_g[i]
        valid_g = np.where(score_g > 0.5)
        boxlist_g_.append(box_g[valid_g])
        scorelist_g_.append(score_g[valid_g])
    boxlist_e, boxlist_g, scorelist_e, scorelist_g = np.array(boxlist_e_), np.array(boxlist_g_), np.array(scorelist_e_), np.array(scorelist_g_)
    # print('after')
    # print('boxlist_e', boxlist_e.shape)
    # print('boxlist_g', boxlist_g.shape)
    return boxlist_e, boxlist_g, scorelist_e, scorelist_g

def get_mAP(boxes_e, scores, boxes_g, iou_thresholds):
    # boxes are 1 x N x 9
    B, Ne, _ = list(boxes_e.shape)
    B, Ng, _ = list(boxes_g.shape)
    assert(B==1)
    boxes_e = np.reshape(boxes_e, (B*Ne, 9))
    boxes_g = np.reshape(boxes_g, (B*Ng, 9))
    corners_e = utils.geom.transform_boxes3d_to_corners_py(boxes_e)
    corners_g = utils.geom.transform_boxes3d_to_corners_py(boxes_g)
    # print("e", boxes_e, "g", boxes_g, "score", scores)
    scores = scores.flatten()
    # size [N, 8, 3]
    ious = np.zeros((Ne, Ng), dtype=np.float32)
    for i in list(range(Ne)):
        for j in list(range(Ng)):
            if(boxes_e[i,3]>0 and boxes_g[j,3]>0):
                iou_single, iou_2d_single = utils.box.box3d_iou(corners_e[i], corners_g[j])
                ious[i,j] = iou_single
    maps = []
    for iou_threshold in iou_thresholds:
        map3d, precision, recall, overlaps = utils.ap.compute_ap(
            "box3d_"+str(iou_threshold), scores, ious, iou_threshold=iou_threshold)
        maps.append(map3d)
    maps = np.stack(maps, axis=0).astype(np.float32)
    if np.isnan(maps).any():
        print('got these nans in maps; setting to zero:', maps)
        maps[np.isnan(maps)] = 0.0
        # assert(False)
    
    # print("maps", maps)
    return maps

def measure_semantic_retrieval_precision(feats, masks, debug=False):
    # feat_memXs is B x C x Z x Y x X
    # mask_memXs is B x 1 x Z x Y x X
    # mask_memXs is ones inside the object masks; zeros in the bkg
    B, C, Z, Y, X = list(feats.shape)
    assert(B>1)
    feats = feats[:2]
    masks = masks[:2]

    obj_feats = []
    bkg_feats = []
    for b in list(range(B)):
        feat = feats[b]
        mask = masks[b]
        # feat is C x Z x Y x X
        feat = feat.permute(1, 2, 3, 0).reshape(-1, C)
        mask = mask.permute(1, 2, 3, 0).reshape(-1)
        # feat is N x C
        # mask is N
        obj_inds = torch.where((mask).reshape(-1) > 0.5)
        bkg_inds = torch.where((mask).reshape(-1) < 0.5)
        # obj_inds is ?
        # bkg_inds is ?

        obj_feat = feat[obj_inds]
        bkg_feat = feat[bkg_inds]
        
        obj_feat = obj_feat.detach().cpu().numpy()
        bkg_feat = bkg_feat.detach().cpu().numpy()
        np.random.shuffle(obj_feat)
        np.random.shuffle(bkg_feat)

        def trim(feat, max_N=50):
            N, C = feat.shape
            if N > max_N:
                # print('trimming!')
                feat = feat[:max_N]
            return feat
        obj_feat = trim(obj_feat)
        bkg_feat = trim(bkg_feat)
        
        obj_feats.append(obj_feat)
        bkg_feats.append(bkg_feat)

    # let's just do this for b=0 for now

    source_obj_feat = obj_feats[0]
    source_bkg_feat = bkg_feats[0]
    other_obj_feats = np.concatenate(obj_feats[1:], axis=0)
    other_bkg_feats = np.concatenate(bkg_feats[1:], axis=0)
            # print('other_obj_feats', other_obj_feats.shape)
            # print('other_bkg_feats', other_bkg_feats.shape)
    
    # make it a fair game: even balance of obj vs bkg
    if other_bkg_feats.shape[0] > other_obj_feats.shape[0]:
        other_bkg_feats = other_bkg_feats[:other_obj_feats.shape[0]]
    else:
        other_obj_feats = other_obj_feats[:other_bkg_feats.shape[0]]

    other_feats = np.concatenate([other_obj_feats,
                                  other_bkg_feats], axis=0)
    labels = np.concatenate([np.ones_like(other_obj_feats[:,0]),
                             np.zeros_like(other_bkg_feats[:,0])], axis=0)
    # print('other_feats', other_feats.shape)

    precisions = []
    for n in list(range(len(source_obj_feat))):
        feat = source_obj_feat[n:n+1]
        dists = np.linalg.norm(other_feats - feat, axis=1)
        inds = np.argsort(dists)
        sorted_labels = labels[inds]
        sorted_dists = dists[inds]
        precision = np.mean(sorted_labels[:10])
        if not np.isnan(precision):
            precisions.append(precision)
        if np.isnan(precision) or debug:
            print('sorted dists and labels:')
            print(sorted_dists, sorted_dists.shape)
            print(sorted_labels, sorted_labels.shape)
            print('other_obj_feats', other_obj_feats.shape)
            print('other_bkg_feats', other_bkg_feats.shape)
            print('precision', precision)
            print('mean precision so far', np.mean(precisions))
            
    if len(precisions):
        mean_precision = np.mean(precisions)
    else:
        print('semantic retrieval bug!')
        print('some info:')
        print('source_obj_feat',source_obj_feat.shape)
        print('other_obj_feats', other_obj_feats.shape)
        print('other_bkg_feats', other_bkg_feats.shape)
        print('other_feats', other_feats.shape)
        assert(False)
        # input()
        # print(source_bkg_feat.shape)
        mean_precision = np.nan
        
    return mean_precision

def linmatch(labelpools, obj_inds, bkg_inds, codes_flat):
    num_embeddings = len(labelpools)
    # print('there seem to be %d embeddings')
    
    # now:
    # on every iter, take a balanced set of obj and bkg inds
    # for each index, see which codeword lights up
    # this is plinko for codewords
    # for each codeword, calculate if either "bkg" or "obj" dominates its inputs;
    # this means: max(mean(obj_yes), mean(1.0-obj_yes))

    np.random.shuffle(obj_inds)
    np.random.shuffle(bkg_inds)

    # print('before trim', len(obj_inds), len(bkg_inds))
    def trim(inds0, inds1):
        N0 = len(inds0)
        N1 = len(inds1)
        if N0 < N1:
            inds1 = inds1[:N0]
        else:
            inds0 = inds0[:N1]
        return inds0, inds1
    obj_inds, bkg_inds = trim(obj_inds, bkg_inds)

    # print('after trim', len(obj_inds), len(bkg_inds))

    print('making %d assignments' % (len(obj_inds) + len(bkg_inds)))
    # input()

    # print('ind_map', ind_map.shape)
    # print('expected:', self.H//8, self.W//8)
    if len(obj_inds) > 0:
        # at each index, i need to find the codeword used
        for ind in obj_inds:
            code = np.squeeze(codes_flat[ind])
            # print('obj: updating code', code)
            labelpools[code].update([1])
        for ind in bkg_inds:
            code = np.squeeze(codes_flat[ind])
            # print('bkg: updating code', code)
            labelpools[code].update([0])

    # print('updated the pools; got these:')

    accs = []
    pool_sizes = []
    for code in list(range(num_embeddings)):
        pool = labelpools[code].fetch()
        print('code', code, 'pool has %d items' % len(pool))
        pool_sizes.append(len(pool))
        # input()
        if len(pool) > 20:
            acc = np.mean(pool)
            acc = np.maximum(acc, 1.0-acc)
            accs.append(acc)
            print('acc = %.2f' % acc)
            # input()
    if len(accs) > 1:
        mean_acc = np.mean(accs)
        print('ok, have %d valid accs' % len(accs))
        print('overall acc: %.2f' % mean_acc)
    else:
        print('only have %d valid accs' % len(accs))
        mean_acc = np.nan
    num_codes_w_20 = len(accs)
    mean_pool_size = np.mean(pool_sizes)
    return mean_acc, mean_pool_size, num_codes_w_20

    # input()

    # an issue with that evaluation is that if most of the codewords devote themselves to
    # background, then they will dominate this accuracy measure, simply because they outnumber
    # the codes doing a mix.
    # maybe that's ok?

    # because of the way i am pooling, the data is evenly distributed between the classes


    # if this were empirically a problem,
    # then the more codewords i had, the higher this acc would be
    # but in my 2d experiments, this was not the case

def get_cater_classification_label(clist_camR_e, clist_camR_g, nrows=6, ncols=6):
    # this function is to evaluate 36-way classification and l1 dist the same as cater baseline.
    # nrows and ncols are grid size.
    # clist_camR_e and clist_camR_g are both B x S x N x 3
    B, S, N, D = clist_camR_e.shape
    B2, S2, N2, D2 = clist_camR_g.shape
    assert(B==B2)
    assert(S==S2)
    assert(N==N2)
    assert(D==D2==3)

    nrows = nrows // 2
    ncols = ncols // 2

    # project to birds-eye view. since they are in camR coordinate, just take the first and the last dims.
    xlist_camR_e, ylist_camR_e = torch.unbind(clist_camR_e[:, :, :, [0, 2]], dim=-1)
    xlist_camR_g, ylist_camR_g = torch.unbind(clist_camR_g[:, :, :, [0, 2]], dim=-1)

    # evaluate on the last frame
    x_camR_e, y_camR_e = xlist_camR_e[:, -1], ylist_camR_e[:, -1]
    x_camR_g, y_camR_g = xlist_camR_g[:, -1], ylist_camR_g[:, -1]
    # these are B x N

    # discretize x, y
    eps = 1e-5
    x_disc_e = x_camR_e.clamp(min=-3, max=3-eps) / 3.0 * ncols
    y_disc_e = y_camR_e.clamp(min=-3, max=3-eps) / 3.0 * nrows
    x_disc_g = x_camR_g.clamp(min=-3, max=3-eps) / 3.0 * ncols
    y_disc_g = y_camR_g.clamp(min=-3, max=3-eps) / 3.0 * nrows

    x_disc_e, y_disc_e = torch.floor(x_disc_e).long() + ncols, torch.floor(y_disc_e).long() + nrows
    x_disc_g, y_disc_g = torch.floor(x_disc_g).long() + ncols, torch.floor(y_disc_g).long() + nrows

    # these are [0, 2 * ncols), [0, 2 * nrows)

    def l1_dist(x_e, y_e, x_g, y_g):
        return torch.abs(x_e - x_g) + torch.abs(y_e - y_g)

    cls_e = y_disc_e * (ncols * 2) + x_disc_e
    cls_g = y_disc_g * (ncols * 2) + x_disc_g
    dist = l1_dist(x_disc_e, y_disc_e, x_disc_g, y_disc_g).float()

    return cls_e, cls_g, dist

if __name__ == '__main__':
    b1 = torch.tensor([[[0,0,2,2], [0,0,2,2], [0,0,2,2], [0,0,2,2], [0,0,2,2], [0,0,4,4], [0,0,4,4], [0,0,2,2], [0,0,2,2]]])
    b2 = torch.tensor([[[0,3,2,5], [3,0,5,2], [0,2,2,5], [2,0,5,2], [1,1,3,3], [1,1,3,3], [0,0,4,4], [0,1,2,3], [1,0,3,2]]])
    print(get_mAP_from_2d_boxlists(b1, torch.ones((1,9)), b2, [0., 0.5, 0.8, 0.9]))

    b1 = torch.tensor([[[0,0,2,2], [0,0,2,2], [0,0,2,2], [0,0,2,2], [0,0,2,2], [0,0,4,4], [0,0,4,4], [0,0,2,2], [0,0,2,2]]])
    b2 = torch.tensor([[[0,3,2,5], [3,0,5,2], [0,2,2,5], [2,0,5,2], [1,1,3,3], [1,1,3,3], [0,0,4,4], [0,1,2,3], [1,0,3,2]]])
    print(drop_invalid_boxes_from_2d_boxlists(b1, b2, torch.ones((1,9)), torch.ones((1,9))))
