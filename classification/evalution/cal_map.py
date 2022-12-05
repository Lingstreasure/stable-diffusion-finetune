import os
import numpy as np
import torch
import json


def json_map(cls_id, pred_json, ann_json, types):
    assert len(ann_json) == len(pred_json)
    num = len(ann_json)
    predict = np.zeros((num), dtype=np.float64)
    target = np.zeros((num), dtype=np.float64)

    for i in range(num):
        predict[i] = pred_json[i]["scores"][cls_id]
        target[i] = ann_json[i]["target"][cls_id]

    if types == 'wider':
        tmp = np.where(target != 99)[0]
        predict = predict[tmp]
        target = target[tmp]
        num = len(tmp)

    if types == 'voc07':
        tmp = np.where(target != 0)[0]
        predict = predict[tmp]
        target = target[tmp]
        neg_id = np.where(target == -1)[0]
        target[neg_id] = 0
        num = len(tmp)


    tmp = np.argsort(-predict)
    target = target[tmp]
    predict = predict[tmp]


    pre, obj = 0, 0
    for i in range(num):
        if target[i] == 1:
            obj += 1.0
            pre += obj / (i+1)
    pre /= obj
    return pre


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()










