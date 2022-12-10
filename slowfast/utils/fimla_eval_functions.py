from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import torch
import numpy as np
import cv2
"""functions for aggregating scores"""

import torch.nn as nn
from torchvision.utils import save_image

def noramlization(data):
    minVals = data.min()
    maxVals = data.max()
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    # print(m)
    normData = data - np.tile(minVals, (m, 1))
    normData = normData/np.tile(ranges, (m, 1))
    return normData, ranges, minVals

n_class = 2

def iou(preds, targets):
    ious = []
    # check hand
    p_hand, p_bg = preds[0], preds[1]
    l_hand, l_bg = targets[0], targets[1]
    p_hand, _, _ = noramlization(p_hand)
    p_bg, _, _ = noramlization(p_bg)
    preds = torch.cat((p_bg, p_hand), dim=0)
    targets = torch.cat((l_bg, l_hand), dim=0)

    preds = preds.argmax(axis=0)
    targets = targets.argmax(axis=0)

    for cls in range(n_class):
        pred_inds = preds == cls
        target_inds = targets == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('0'))
        else:
            ious.append(float(intersection) / max(union, 1))
    return np.mean(ious)


def pixel_acc(preds, targets):
    correct = (preds.argmax(axis=0) == targets.argmax(axis=0)).sum()
    total   = (targets.argmax(axis=0) == targets.argmax(axis=0)).sum()
    return (correct / total).tolist()

def to_onehot(indices, num_classes):
    """Convert a tensor of indices of any shape `(N, ...)` to a
    tensor of one-hot indicators of shape `(N, num_classes, ...)`.
    """
    onehot = torch.zeros(indices.shape[0],
                        num_classes,
                        *indices.shape[1:],
                        device=indices.device, dtype = torch.int64)
    return onehot.scatter_(1, indices.unsqueeze(1), 1)

def mean_class_accuracy(cm):
    """Compute mean class accuracy based on the input confusion matrix"""
    # Increase floating point precision
    cm = cm.type(torch.float64)
    cls_cnt = cm.sum(dim=1) + 1e-15
    cls_hit = cm.diag()
    print(cls_hit/cls_cnt)
    cls_acc = (cls_hit/cls_cnt).mean().item()
    return cls_acc

def confusion_matrix(pred, target):
    num_classes = pred.shape[1]
    assert pred.shape[0] == target.shape[0]
    with torch.no_grad():
      target_ohe = to_onehot(target, num_classes)
      target_ohe_t = target_ohe.transpose(0, 1).float()

      pred_idx = torch.argmax(pred, dim=1)
      pred_ohe = to_onehot(pred_idx.reshape(-1), num_classes)
      pred_ohe = pred_ohe.float()

      confusion_matrix = torch.matmul(target_ohe_t, pred_ohe)
    return confusion_matrix

m = nn.MaxPool3d(kernel_size=(1, 3, 3),
                            stride=(1, 2, 2),padding=(0, 1, 1))
def _compare_gaze_maps(gt, pred, all_thresh):
  """ compare two gaze maps """
  # return tp, tn, fp, fn

  N, Z = 1, gt.shape[1] 
  # X, Y = gt.shape[2], gt.shape[3]


  # # reshape both gaze maps
  # gt_map = np.reshape(np.squeeze(gt), (N, X*Z, Y))
  # pred_map = np.reshape(np.squeeze(pred), (N, X*Z, Y))

  gt_map, pred_map = gt, pred

  # for i in range(N):
  #   pred_map[i,:,:] = pred_map[i,:,:]/np.max(pred_map[i,:,:])

  # not technicallly correct, yet should be very similar to the true PR
  tp = np.zeros((all_thresh.shape[0],))
  fp = np.zeros((all_thresh.shape[0],))
  fn = np.zeros((all_thresh.shape[0],))
  tn = np.zeros((all_thresh.shape[0],))

  # # get valid gaze slices
  valid_slice = []
  for slice_idx in range(N):
    if np.max(gt_map[slice_idx, :, :]) > 0.01:
      valid_slice.append(slice_idx)
  # reslice the data
  valid_gt = gt_map[valid_slice, :, :]
  valid_gt = (valid_gt>0.01)
  valid_pred = pred_map[valid_slice, :, :]

  for idx, thresh in enumerate(all_thresh):
    mask = (valid_pred>=thresh)
    tp[idx] += np.sum(np.logical_and(mask==1, valid_gt==1))
    tn[idx] += np.sum(np.logical_and(mask==0, valid_gt==0))
    fp[idx] += np.sum(np.logical_and(mask==1, valid_gt==0))
    fn[idx] += np.sum(np.logical_and(mask==0, valid_gt==1))

  # print(tp, tn, fp, fn)
  return tp, tn, fp, fn


def main():
  output_file = "/nethome/wjia34/slowfast/output.pkl"

  """Evaluation of action localization"""
  with open(output_file, 'rb') as f:
    # attengt_list, attenpred_list, preds_list,labels_list, videoid_list = pickle.load(f)
    preds_list,labels_list, videoid_list = pickle.load(f)
    # preds_list = np.load('pred.npy', allow_pickle=True).tolist()
    # labels_list = np.load('label.npy', allow_pickle=True).tolist()

    # print(preds_list[0].shape)  # torch.Size([16, 2, 2, 224, 224])
    # print(labels_list[0].shape)  # torch.Size([16, 2, 2, 224, 224])
    # print(len(preds_list))  # 3
    # print(len(labels_list))  # 3


    # all_preds = torch.empty((len(preds_list)*8,2,1,224,224), dtype=torch.int64)
    # all_labels = torch.empty((len(preds_list)*8,2,1,224,224), dtype=torch.int64)
    # # all_preds = np.zeros((len(preds_list)*8,2,1,224,224))
    # # all_labels = np.zeros((len(labels_list)*8,2,1,224,224))
    # for idx in range(len(preds_list)):
    #   for i in range(len(preds_list[idx])):
    #     all_preds[idx*len(preds_list[idx])+i] = preds_list[idx][i].argmax(axis=0)
    #     all_labels[idx*len(preds_list[idx])+i] = labels_list[idx][i].argmax(axis=0)



    # all_preds = torch.empty((len(preds_list),8,2,1,224,224), dtype=torch.int64)
    # all_labels = torch.empty((len(preds_list),8,2,1,224,224), dtype=torch.int64)
    # for idx in range(len(preds_list)):
    #   for i in range(len(preds_list[idx])):
    #     all_preds[idx][i] = preds_list[idx][i].argmax(axis=0)
    #     all_labels[idx][i] = labels_list[idx][i].argmax(axis=0)
    # print(all_preds.shape)    
    # print(all_labels.shape)
    # print(all_preds.dtype)
    # print(all_labels.dtype)


  all_preds = torch.cat(preds_list)
  all_labels = torch.cat(labels_list)

  # cm =confusion_matrix(all_preds,all_labels)
  # cls_acc = mean_class_accuracy(cm)
  # print(cls_acc)
  # print(len(attengt_list))

  # # tp / fp / fn / tn for gaze estimation
  all_thresh = np.linspace(0, 1.0, 41)
  f_tp = np.zeros((all_thresh.shape[0],))
  f_fp = np.zeros((all_thresh.shape[0],))
  f_fn = np.zeros((all_thresh.shape[0],))
  f_tn = np.zeros((all_thresh.shape[0],))

  l_tp = np.zeros((all_thresh.shape[0],))
  l_fp = np.zeros((all_thresh.shape[0],))
  l_fn = np.zeros((all_thresh.shape[0],))
  l_tn = np.zeros((all_thresh.shape[0],))

  m_tp = np.zeros((all_thresh.shape[0],))
  m_fp = np.zeros((all_thresh.shape[0],))
  m_fn = np.zeros((all_thresh.shape[0],))
  m_tn = np.zeros((all_thresh.shape[0],))

  f_all_iou = []
  l_all_iou = []
  m_all_iou = []

  f_all_acc = []
  l_all_acc = []
  m_all_acc = []

  print(len(all_labels))
  # print(all_preds[0].shape)
  # os._exit(0)
  for idx in range(len(all_labels)):
    # score, pred_gaze, label, gt_gaze, video_id
    # gt_item = m(all_labels[idx])
    # pred_item = m(all_preds[idx])
    f_pred = torch.cat((torch.sigmoid(all_preds[idx][0][0]).unsqueeze(0), torch.sigmoid(all_preds[idx][1][0]).unsqueeze(0)), dim=0).unsqueeze(1)
    f_label = torch.cat((all_labels[idx][0][0].unsqueeze(0), all_labels[idx][1][0].unsqueeze(0)), dim=0).unsqueeze(1)

    m_pred = torch.cat((torch.sigmoid(all_preds[idx][0][1]).unsqueeze(0), torch.sigmoid(all_preds[idx][1][1]).unsqueeze(0)), dim=0).unsqueeze(1)
    m_label = torch.cat((all_labels[idx][0][1].unsqueeze(0), all_labels[idx][1][1].unsqueeze(0)), dim=0).unsqueeze(1)

    l_pred = torch.cat((torch.sigmoid(all_preds[idx][0][2]).unsqueeze(0), torch.sigmoid(all_preds[idx][1][2]).unsqueeze(0)), dim=0).unsqueeze(1)
    l_label = torch.cat((all_labels[idx][0][2].unsqueeze(0), all_labels[idx][1][2].unsqueeze(0)), dim=0).unsqueeze(1)


    f_all_iou.append(iou(f_pred, f_label))
    f_all_acc.append(pixel_acc(f_pred, f_label))

    m_all_iou.append(iou(m_pred, m_label))
    m_all_acc.append(pixel_acc(m_pred, m_label))

    l_all_iou.append(iou(l_pred, l_label))
    l_all_acc.append(pixel_acc(l_pred, l_label))

    f_ctp, f_ctn, f_cfp, f_cfn = _compare_gaze_maps(
    np.squeeze(f_label.numpy()), np.squeeze(f_pred.numpy()), all_thresh)
    f_tp = f_tp + f_ctp
    f_tn = f_tn + f_ctn
    f_fp = f_fp + f_cfp
    f_fn = f_fn + f_cfn

    # f_ctp, f_ctn, f_cfp, f_cfn = _compare_gaze_maps(
    # f_label.numpy()[0], f_pred.numpy()[0], all_thresh)
    # f_tp = f_tp + f_ctp
    # f_tn = f_tn + f_ctn
    # f_fp = f_fp + f_cfp
    # f_fn = f_fn + f_cfn


    m_ctp, m_ctn, m_cfp, m_cfn = _compare_gaze_maps(
    np.squeeze(m_label.numpy()), np.squeeze(m_pred.numpy()), all_thresh)
    m_tp = m_tp + m_ctp
    m_tn = m_tn + m_ctn
    m_fp = m_fp + m_cfp
    m_fn = m_fn + m_cfn

    l_ctp, l_ctn, l_cfp, l_cfn = _compare_gaze_maps(
    np.squeeze(l_label.numpy()), np.squeeze(l_pred.numpy()), all_thresh)
    l_tp = l_tp + l_ctp
    l_tn = l_tn + l_ctn
    l_fp = l_fp + l_cfp
    l_fn = l_fn + l_cfn
    
    '''
    gt_item = all_labels[idx]
    pred_item = all_preds[idx]

    hand_gt = gt_item[0]
    fgt_hand = hand_gt[0].unsqueeze(0)
    mgt_hand = hand_gt[1].unsqueeze(0)
    lgt_hand = hand_gt[2].unsqueeze(0)

    bg_gt = gt_item[1]
    fgt_bg = bg_gt[0].unsqueeze(0)
    mgt_bg = bg_gt[1].unsqueeze(0)
    lgt_bg = bg_gt[2].unsqueeze(0)


    fgt_item = torch.cat((fgt_hand, fgt_bg), dim=0).unsqueeze(1)
    mgt_item = torch.cat((mgt_hand, mgt_bg), dim=0).unsqueeze(1)
    lgt_item = torch.cat((lgt_hand, lgt_bg), dim=0).unsqueeze(1)


    hand_p = pred_item[0]
    fp_hand = hand_p[0]
    fp_hand = torch.sigmoid(fp_hand).unsqueeze(0)
    mp_hand = hand_p[1]
    mp_hand = torch.sigmoid(mp_hand).unsqueeze(0)
    lp_hand = hand_p[2]
    lp_hand = torch.sigmoid(lp_hand).unsqueeze(0)

    bg_p = pred_item[1]
    fp_bg = bg_p[0]
    fp_bg = torch.sigmoid(fp_bg).unsqueeze(0)
    mp_bg = bg_p[1]
    mp_bg = torch.sigmoid(mp_bg).unsqueeze(0)
    lp_bg = bg_p[2]
    lp_bg = torch.sigmoid(lp_bg).unsqueeze(0)

    fp_item = torch.cat((fp_hand, fp_bg), dim=0).unsqueeze(1)
    mp_item = torch.cat((mp_hand, mp_bg), dim=0).unsqueeze(1)
    lp_item = torch.cat((lp_hand, lp_bg), dim=0).unsqueeze(1)

    f_all_iou.append(iou(fp_item, fgt_item))
    f_all_acc.append(pixel_acc(fp_item, fgt_item))

    m_all_iou.append(iou(mp_item, mgt_item))
    m_all_acc.append(pixel_acc(mp_item, mgt_item))

    l_all_iou.append(iou(lp_item, lgt_item))
    l_all_acc.append(pixel_acc(lp_item, lgt_item))

    fgt = np.squeeze(fgt_item.numpy())
    fpred = np.squeeze(fp_item.numpy())

    mgt = np.squeeze(mgt_item.numpy())
    mpred = np.squeeze(mp_item.numpy())

    lgt = np.squeeze(lgt_item.numpy())
    lpred = np.squeeze(lp_item.numpy())

    # print(fgt.shape, fpred.shape)
    # print(lgt.shape, lpred.shape)
    
    f_ctp, f_ctn, f_cfp, f_cfn = _compare_gaze_maps(
      fgt, fpred, all_thresh)

    f_tp = f_tp + f_ctp
    f_tn = f_tn + f_ctn
    f_fp = f_fp + f_cfp
    f_fn = f_fn + f_cfn

    m_ctp, m_ctn, m_cfp, m_cfn = _compare_gaze_maps(
      mgt, mpred, all_thresh)

    m_tp = m_tp + m_ctp
    m_tn = m_tn + m_ctn
    m_fp = m_fp + m_cfp
    m_fn = m_fn + m_cfn

    l_ctp, l_ctn, l_cfp, l_cfn = _compare_gaze_maps(
      lgt, lpred, all_thresh)

    l_tp = l_tp + l_ctp
    l_tn = l_tn + l_ctn
    l_fp = l_fp + l_cfp
    l_fn = l_fn + l_cfn
    '''


  # # prec / recall for gaze
  f_prec = f_tp / (f_tp+f_fp+1e-6)
  f_recall = f_tp / (f_tp+f_fn+1e-6)
  f_f1 = 2*f_prec*f_recall / (f_prec + f_recall + 1e-6)
  f_idx = np.argmax(f_f1)
  print("Mean IoU of x+1: "+str(np.mean(f_all_iou)))
  print("Mean pixel accuracy of x+1: "+str(np.mean(f_all_acc)))
  print("F1 Score of x+1 {:0.4f} (P={:0.4f}, R={:0.4f}) at th={:0.4f}".format(
    f_f1[f_idx], f_prec[f_idx], f_recall[f_idx], all_thresh[f_idx]))
  print("****************************************")
  m_prec = m_tp / (m_tp+m_fp+1e-6)
  m_recall = m_tp / (m_tp+m_fn+1e-6)
  m_f1 = 2*m_prec*m_recall / (m_prec + m_recall + 1e-6)
  m_idx = np.argmax(m_f1)
  print("Mean IoU of x+6: "+str(np.mean(m_all_iou)))
  print("Mean pixel accuracy of x+6: "+str(np.mean(m_all_acc)))
  print("F1 Score of x+6 {:0.4f} (P={:0.4f}, R={:0.4f}) at th={:0.4f}".format(
    m_f1[m_idx], m_prec[m_idx], m_recall[m_idx], all_thresh[m_idx]))
  print("****************************************")
  l_prec = l_tp / (l_tp+l_fp+1e-6)
  l_recall = l_tp / (l_tp+l_fn+1e-6)
  l_f1 = 2*l_prec*l_recall / (l_prec + l_recall + 1e-6)
  l_idx = np.argmax(l_f1)
  print("Mean IoU of x+12: "+str(np.mean(l_all_iou)))
  print("Mean pixel accuracy of x+12: "+str(np.mean(l_all_acc)))
  print("F1 Score of x+12 {:0.4f} (P={:0.4f}, R={:0.4f}) at th={:0.4f}".format(
    l_f1[l_idx], l_prec[l_idx], l_recall[l_idx], all_thresh[l_idx]))
  


if __name__ == "__main__":
    main()
