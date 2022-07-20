from collections import defaultdict
import numpy as np

window_sz = 2
cumu_gt_match = defaultdict(list)
cumu_pred_match = defaultdict(list)
cumu_pred_scores = defaultdict(list)

def compute_matches(name, overlaps, pred_scores, 
                    iou_threshold=0.5, score_threshold=0.0,
                    oracle=False):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    n_boxes, n_gt_boxes = overlaps.shape
    # Sort predictions by score from high to low
    # indices = np.argsort(pred_scores)[::-1]
    # print('SORTING BY OVERLAPS[:,0]')
    if oracle:
        indices = np.argsort(np.sum(overlaps, axis=1))[::-1]
    else:
        indices = np.argsort(pred_scores)[::-1]
        
    pred_scores = pred_scores[indices]
    overlaps = overlaps[indices]

    # print('pred_scores', pred_scores)
    # print('overlaps', overlaps)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([n_boxes])
    gt_match = -1 * np.ones([n_gt_boxes])
    for i in list(range(n_boxes)):
        # Find best matching ground truth box
        
        # 1. Sort matches by overlap
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            #if pred_class_ids[i] == gt_class_ids[j]:
            match_count += 1
            gt_match[j] = i
            pred_match[i] = j
            break

    # print('adding %d gt matches' % len(gt_match))
    # print('adding %d pred matches' % len(pred_match))
    
    # cumu_gt_match[name].append(gt_match)
    # cumu_pred_match[name].append(pred_match)
    # cumu_pred_scores[name].append(pred_scores)
    # if window_sz is not None and len(cumu_gt_match[name])>window_sz:
    #     cumu_gt_match[name].pop(0)
    #     cumu_pred_match[name].pop(0)
    #     cumu_pred_scores[name].pop(0)
    # gt_match = np.concatenate(cumu_gt_match[name])
    # pred_match = np.concatenate(cumu_pred_match[name])
    # pred_scores = np.concatenate(cumu_pred_scores[name])
    # indices = np.argsort(pred_scores)[::-1]
    # pred_match = pred_match[indices]
    
    return gt_match, pred_match, overlaps


def compute_ap(name, pred_scores, overlaps, iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        name, overlaps, pred_scores, iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1).astype(np.float32) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)
    
    # if iou_threshold==0.1:
    #     print("pred_match", pred_match, "gt_match", gt_match)
    #     print('iou_threshold', iou_threshold)
    #     print('precisions', precisions)
    #     print('recalls', recalls)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in list(range(len(precisions) - 2, -1, -1)):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])
    
    # if iou_threshold==0.1:
    #     print('map', mAP)
    #     input()
        
    return mAP, precisions, recalls, overlaps


