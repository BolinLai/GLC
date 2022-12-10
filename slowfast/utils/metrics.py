#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""
import ipdb
import torch
import numpy as np
import math

from scipy import ndimage
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(0), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]


# for gaze estimation
def gaze_iou(preds, labels, threshold):
    # pytorch
    binary_preds = (preds.squeeze(1) > threshold).int()
    binary_labels = (labels > 0.001).int()
    intersection = (binary_preds * binary_labels).sum(dim=(2, 3))
    union = (binary_preds.sum(dim=(2, 3)) + binary_labels.sum(dim=(2, 3))) - intersection

    iou = intersection / (union + 1e-4)
    return float(iou.mean().cpu().numpy())  # need np.float64 in logging rather than np.float32

    # numpy
    # binary_preds = (preds.squeeze(1) > threshold).astype(np.int)
    # binary_labels = (labels > threshold).astype(np.int)
    # intersection = (binary_preds * binary_labels).sum(axis=(2, 3))
    # union = (binary_preds.sum(axis=(2, 3)) + binary_labels.sum(axis=(2, 3))) - intersection
    #
    # iou = intersection / (union + 1e-6)
    # return iou.mean()


# for gaze estimation
def pixel_f1(preds, labels, threshold):
    binary_preds = (preds.squeeze(1) > threshold).int()
    binary_labels = (labels > 0.001).int()
    tp = (binary_preds * binary_labels).sum(dim=(2, 3))
    fg_labels = binary_labels.sum(dim=(2, 3))
    fg_preds = binary_preds.sum(dim=(2, 3))

    # calculate per frame
    # recall = ((tp + 1e-6) / (fg_labels + 1e-6))
    # precision = ((tp + 1e-6) / (fg_preds + 1e-6))
    # f1 = ((2 * recall * precision) / (recall + precision + 1e-6)).mean()

    # calculate over average recall and precision
    recall = (tp / (fg_labels + 1e-6)).mean()
    precision = (tp / (fg_preds + 1e-6)).mean()
    f1 = (2 * recall * precision) / (recall + precision + 1e-6)

    return f1, recall, precision


# for gaze estimation
def adaptive_f1(preds, labels_hm, labels, dataset):
    """
    Automatically select the threshold getting the best f1 score.
    """
    # Numpy
    # # thresholds = np.linspace(0, 1.0, 51)
    # thresholds = np.linspace(0, 0.2, 11)
    # # thresholds = np.array([0.5])
    # preds, labels = preds.cpu().numpy(), labels.cpu().numpy()
    # all_preds = np.zeros(shape=(thresholds.shape + labels.shape))
    # all_labels = np.zeros(shape=(thresholds.shape + labels.shape))
    # binary_labels = (labels > 0.001).astype(np.int)
    # for i in range(thresholds.shape[0]):
    #     binary_preds = (preds.squeeze(1) > thresholds[i]).astype(np.int)
    #     all_preds[i, ...] = binary_preds
    #     all_labels[i, ...] = binary_labels
    # tp = (all_preds * all_labels).sum(axis=(3, 4))
    # fg_labels = all_labels.sum(axis=(3, 4))
    # fg_preds = all_preds.sum(axis=(3, 4))
    # recall = (tp / (fg_labels + 1e-6)).mean(axis=(1, 2))
    # precision = (tp / (fg_preds + 1e-6)).mean(axis=(1, 2))
    # f1 = (2 * recall * precision) / (recall + precision + 1e-6)
    # max_idx = np.argmax(f1)
    # return f1[max_idx], recall[max_idx], precision[max_idx], thresholds[max_idx]

    # PyTorch
    # thresholds = np.linspace(0.3, 0.5, 11)
    thresholds = np.linspace(0, 0.02, 11)  # the one we used for GLC
    # thresholds = np.array([0.5])
    all_preds = torch.zeros(size=(thresholds.shape + labels_hm.size()), device=labels_hm.device)
    all_labels = torch.zeros(size=(thresholds.shape + labels_hm.size()), device=labels_hm.device)
    binary_labels = (labels_hm > 0.001).int()  # change to 0.001
    for i in range(thresholds.shape[0]):  # There is some space for improvement. You can calculate f1 in the loop rather than save all preds. It consumes much memory.
        binary_preds = (preds.squeeze(1) > thresholds[i]).int()
        all_preds[i, ...] = binary_preds
        all_labels[i, ...] = binary_labels
    tp = (all_preds * all_labels).sum(dim=(3, 4))
    fg_labels = all_labels.sum(dim=(3, 4))
    fg_preds = all_preds.sum(dim=(3, 4))

    if dataset == 'egteagaze':
        fixation_idx = 1
    elif dataset == 'ego4dgaze' or dataset == 'ego4d_av_gaze':
        fixation_idx = 0
    else:
        raise NotImplementedError(f'Metrics of {dataset} is not implemented.')
    labels_flat = labels.view(labels.size(0) * labels.size(1), labels.size(2))
    tracked_idx = torch.where(labels_flat[:, 2] == fixation_idx)[0]
    tp = tp.view(tp.size(0), tp.size(1)*tp.size(2)).index_select(1, tracked_idx)
    fg_labels = fg_labels.view(fg_labels.size(0), fg_labels.size(1)*fg_labels.size(2)).index_select(1, tracked_idx)
    fg_preds = fg_preds.view(fg_preds.size(0), fg_preds.size(1)*fg_preds.size(2)).index_select(1, tracked_idx)
    recall = (tp / (fg_labels + 1e-6)).mean(dim=1)
    precision = (tp / (fg_preds + 1e-6)).mean(dim=1)
    f1 = (2 * recall * precision) / (recall + precision + 1e-6)
    max_idx = torch.argmax(f1)

    # recall = (tp / (fg_labels + 1e-6)).mean(dim=(1, 2))
    # precision = (tp / (fg_preds + 1e-6)).mean(dim=(1, 2))
    # f1 = (2 * recall * precision) / (recall + precision + 1e-6)
    # max_idx = torch.argmax(f1)
    # ipdb.set_trace()
    return float(f1[max_idx].cpu().numpy()), float(recall[max_idx].cpu().numpy()), \
           float(precision[max_idx].cpu().numpy()), thresholds[max_idx]  # need np.float64 in logging rather than np.float32


# for gaze estimation
def average_angle_error(preds, labels, dataset):
    if dataset == 'egteagaze':
        fixation_idx = 1
    elif dataset == 'ego4dgaze' or dataset == 'ego4d_av_gaze':
        fixation_idx = 0
    else:
        raise NotImplementedError(f'Metrics of {dataset} is not implemented.')
    labels = labels.view(labels.size(0) * labels.size(1), labels.size(2))
    tracked_idx = torch.where(labels[:, 2] == fixation_idx)[0]
    labels = labels.index_select(0, tracked_idx)
    preds = preds.squeeze(1)
    preds = preds.view(preds.size(0) * preds.size(1), preds.size(2), preds.size(3))
    preds = preds.index_select(0, tracked_idx).cpu().numpy()
    labels = labels.cpu().numpy()

    aae = list()
    for frame in range(preds.shape[0]):
        out_sq = preds[frame, :, :]
        predicted = ndimage.measurements.center_of_mass(out_sq)
        (i, j) = labels[frame, 1] * 64, labels[frame, 0] * 64
        d = 32 / math.tan(math.pi / 6)
        r1 = np.array([predicted[0] - 32, predicted[1] - 32, d])
        r2 = np.array([i - 32, j - 32, d])
        angle = math.atan2(np.linalg.norm(np.cross(r1, r2)), np.dot(r1, r2))
        aae.append(math.degrees(angle))

    return float(np.mean(aae))


# for gaze estimation
def auc(preds, labels_hm, labels, dataset):
    if dataset == 'egteagaze':
        fixation_idx = 1
    elif dataset == 'ego4dgaze' or 'ego4d_av_gaze':
        fixation_idx = 0
    else:
        raise NotImplementedError(f'Metrics of {dataset} is not implemented.')
    labels = labels.view(labels.size(0) * labels.size(1), labels.size(2))
    tracked_idx = torch.where(labels[:, 2] == fixation_idx)[0]
    labels = labels.index_select(0, tracked_idx)
    preds = preds.squeeze(1)
    preds = preds.view(preds.size(0) * preds.size(1), preds.size(2), preds.size(3))
    preds = preds.index_select(0, tracked_idx).cpu().numpy()
    labels = labels.cpu().numpy()

    auc = list()
    for frame in range(preds.shape[0]):
        out_sq = preds[frame, :, :]
        predicted = ndimage.measurements.center_of_mass(out_sq)
        (i, j) = round(labels[frame, 1] * 63), round(labels[frame, 0] * 63)

        z = np.zeros((labels_hm.size(-2), labels_hm.size(-1)))
        if np.isnan(predicted[0]) or np.isnan(predicted[1]):  # the prediction may be nan for some algorithms
            z[z.shape[0] // 2, z.shape[1] // 2] = 1
        else:
            z[int(predicted[0]), int(predicted[1])] = 1
        z = ndimage.filters.gaussian_filter(z, 3.2)
        z = z - np.min(z)
        z = z / np.max(z)
        atgt = z[i][j]
        fpbool = z > atgt
        auc1 = 1 - float(fpbool.sum()) / preds.shape[2] / preds.shape[1]
        auc.append(auc1)

    return float(np.mean(auc))


# for action recognition
def mean_class_accuracy(preds, labels):
    y_pred = preds.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    y_pred = np.argmax(y_pred, axis=1)
    cf = confusion_matrix(y_true, y_pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt

    return np.mean(cls_acc), cls_acc


def conf_matrix(preds, labels):
    y_pred = preds.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    if y_pred.ndim == 2:  # if preds are probability
        y_pred = np.argmax(y_pred, axis=1)
    elif y_pred.ndim == 1:  # if preds are categories
        pass
    else:
        raise NotImplementedError
    cf = confusion_matrix(y_true, y_pred)
    return cf


# for persuasion strategy classification
def mean_f1_for_multilabel(preds, labels):
    y_pred = preds.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=None)

    return f1, f1.mean()


# def auc(preds, labels_hm, labels):
#     preds = preds.squeeze(1)
#
#     labels_flat = labels.view(labels.size(0) * labels.size(1), 4)
#     tracked_idx = torch.where(labels_flat[:, 2] == 1)
#     tracked_labels_hm = labels_hm.view(labels_hm.size(0) * labels_hm.size(1), -1).index_select(0, tracked_idx[0])
#     tracked_preds = preds.view(preds.size(0) * preds.size(1), -1).index_select(0, tracked_idx[0])
#     # ipdb.set_trace()
#
#     p = tracked_preds.squeeze(1).view(-1).cpu().numpy()
#     binary_labels = (tracked_labels_hm > 0.001).int()
#     l = binary_labels.view(-1).cpu().numpy()
#     score = roc_auc_score(y_true=l, y_score=p)
#     # ipdb.set_trace()
#
#     return score


