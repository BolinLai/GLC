#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        ipdb.set_trace()
        loss = - (5 * y * F.logsigmoid(x) + (1 - y) * torch.log(1 - torch.sigmoid(x)))
        ipdb.set_trace()
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            raise NotImplementedError


class KLDiv(nn.Module):
    """
      KL divergence for 3D attention maps
    """
    def __init__(self):
        super(KLDiv, self).__init__()
        self.register_buffer('norm_scalar', torch.tensor(1, dtype=torch.float32))

    def forward(self, pred, target=None):
        # get output shape
        batch_size, T = pred.shape[0], pred.shape[2]
        H, W = pred.shape[3], pred.shape[4]

        # N T HW
        atten_map = pred.view(batch_size, T, -1)
        log_atten_map = torch.log(atten_map + 1e-10)

        if target is None:
            # uniform prior: this is really just neg entropy
            # we keep the loss scale the same here
            log_q = torch.log(self.norm_scalar / float(H * W))
            # \sum p logp - log(1/hw) -> N T
            kl_losses = (atten_map * log_atten_map).sum(dim=-1) - log_q
        else:
            log_q = torch.log(target.view(batch_size, T, -1) + 1e-10)
            # \sum p logp - \sum p logq -> N T
            kl_losses = (atten_map * log_atten_map).sum(dim=-1) - (atten_map * log_q).sum(dim=-1)
        # N T -> N
        norm_scalar = T * torch.log(self.norm_scalar * H * W)
        kl_losses = kl_losses.sum(dim=-1) / norm_scalar
        kl_loss = kl_losses.mean()
        return kl_loss


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "kldiv": KLDiv
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
