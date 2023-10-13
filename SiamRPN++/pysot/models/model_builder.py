# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss, select_iou_loss
from pysot.models.topk_loss import select_topk_cross_entropy_loss, weight_topk_l1_loss


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        if cfg.DEBUG.SHOW_TEMPLATE or cfg.DEBUG.SHOW_SEARCH_REGION or \
                cfg.DEBUG.FINAL_HEATMAP.SHOW_CLS or cfg.DEBUG.FINAL_HEATMAP.SHOW_CLS:
            try:
                from pysot.utils.my_visdom import vis_util
                self.vis_util = vis_util
            except:
                print("please open the visdom server")
                import sys

                sys.exit(0)

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def template(self, z):
        if cfg.DEBUG.SHOW_TEMPLATE:
            self.vis_util.vis_img(z[0], "template")
        zf = self.backbone(z)
        # for i, f in enumerate(zf):
        #     self.vis_util.vis_feat(f, "template_%d" % i, mean=True)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)

        # for i, f in enumerate(zf):
        #     self.vis_util.vis_feat(f, "adj_template_%d" % i, mean=True)

        self.zf = zf

    def track(self, x):
        if cfg.DEBUG.SHOW_SEARCH_REGION:
            self.vis_util.vis_img(x[0], title="search_region")
        xf = self.backbone(x)
        # for i, f in enumerate(xf):
        #     self.vis_util.vis_feat(f, "search_region_%d" % i, mean=True)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        # for i, f in enumerate(xf):
        #     self.vis_util.vis_feat(f, "adj_search_region_%d" % i, mean=True)
        cls, loc = self.rpn_head(self.zf, xf)
        if cfg.DEBUG.FINAL_HEATMAP.SHOW_CLS:
            self.vis_util.vis_heatmap(cls, mean=cfg.DEBUG.FINAL_HEATMAP.MEAN, title="final_cls_heatmap")
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
            'cls': cls,
            'loc': loc,
            'mask': mask if cfg.MASK.MASK else None
        }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        if cfg.DEBUG.SHOW_TEMPLATE:
            self.vis_util.vis_img(template[0], "template[0]")

        if cfg.DEBUG.SHOW_SEARCH_REGION:
            self.vis_util.vis_img(search[0], title="search_region[0]")

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        cls, loc = self.rpn_head(zf, xf)

        if cfg.DEBUG.FINAL_HEATMAP.SHOW_CLS:
            self.vis_util.vis_heatmap(cls, mean=cfg.DEBUG.FINAL_HEATMAP.MEAN, title="final_cls_heatmap[0]")

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        # loc: (N, 4k, h, w) label_loc: (N, 4, k, h, w) label_loc_weight: (N, k, h, w)
        if cfg.TRAIN.LOC_LOSS_TYPE == "l1_loss":
            loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)
        else:
            loc_loss = select_iou_loss(loc, label_loc, label_cls)
        # loc_loss = loss_builder(loc, label_loc, cfg.TRAIN.LOC_LOSS_TYPE, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if cfg.TRAIN.TOPK.USE_TOPK_CLS:
            topk_cls_loss = select_topk_cross_entropy_loss(cls, label_cls, cfg.TRAIN.TOPK.TOPK_NUM_CLS)
            outputs['total_loss'] += cfg.TRAIN.TOPK.WEIGHT_CLS * topk_cls_loss
            outputs['topk_cls_loss'] = topk_cls_loss

        if cfg.TRAIN.TOPK.USE_TOPK_REG:
            topk_loc_loss = weight_topk_l1_loss(loc, label_loc, label_loc_weight, cfg.TRAIN.TOPK.TOPK_NUM_REG)
            outputs['total_loss'] += cfg.TRAIN.TOPK.WEIGHT_REG * topk_loc_loss
            outputs['topk_loc_loss'] = topk_loc_loss

        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        return outputs