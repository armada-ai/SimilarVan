from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F

from pysot.models.iou_loss import linear_iou


def get_topk_cls_loss(pred, label, select, top_k=10):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    loss = F.nll_loss(pred, label, reduction="none")
    topk_mean_loss = loss[torch.topk(loss, k=min(top_k, len(loss)))[1]].mean()
    return topk_mean_loss


def select_topk_cross_entropy_loss(pred, label, top_k=10):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero(as_tuple=False).squeeze().cuda()
    neg = label.data.eq(0).nonzero(as_tuple=False).squeeze().cuda()
    loss_pos = get_topk_cls_loss(pred, label, pos, top_k)
    loss_neg = get_topk_cls_loss(pred, label, neg, top_k * 3)
    # print(loss_pos, loss_neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_topk_l1_loss(pred_loc, label_loc, loss_weight, topk=10):
    """
    :param pred_loc: (N, 4k, h, w)
    :param label_loc: (N, 4, k, h, w)
    :param loss_weight: (N, k, h, w)
    :return:
    """
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    topk_loss = torch.topk(loss, k=topk)[0].sum().div(b)
    return topk_loss
