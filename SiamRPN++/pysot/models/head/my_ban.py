from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import nn
import torch.nn.functional as F
from pysot.models.head.ban import BAN
from pysot.models.head.rpn import DepthwiseXCorr


class MyDepthwiseBAN(BAN):
    def __init__(self, in_channels, out_channels, cls_out_channels=2,
                 confusion=False, confusion_way="sum", confusion_num=4):
        super(MyDepthwiseBAN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, cls_out_channels)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4)
        self.confusion = confusion
        self.confusion_way = confusion_way
        self.confusion_num = confusion_num

    def forward(self, z_f, x_f):
        if x_f.shape[0] > 1 and self.confusion:
            x_f_cls = self.features_confusion(x_f, self.confusion_way, self.confusion_num)
        else:
            x_f_cls = x_f
        cls = self.cls(z_f, x_f_cls)
        loc = self.loc(z_f, x_f)
        return cls, loc

    def features_confusion(self, features, confusion_way, confusion_num=4):
        import numpy as np
        batch = features.shape[0]
        feats = features.clone()
        for j in range(batch // 2):
            index = np.random.randint(j, batch, confusion_num + 1)
            index[-1] = j
            if confusion_way.lower() == "mean":
                feats[j] = torch.mean(features[index], dim=0)
            elif confusion_way.lower() == "sum":
                feats[j] = torch.sum(features[index], dim=0)
            else:
                print("confusion_way must be in ['mean', 'sum']")
                raise NotImplementedError
        return feats


class MyMultiBAN(BAN):
    def __init__(self, in_channels, cls_out_channels,
                 confusion, confusion_way, confusion_num,
                 weighted=False):
        super(MyMultiBAN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('box' + str(i + 2),
                            MyDepthwiseBAN(in_channels[i], in_channels[i], cls_out_channels,
                                           confusion[i], confusion_way[i], confusion_num[i]))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))
        self.loc_scale = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            box = getattr(self, 'box' + str(idx))
            c, l = box(z_f, x_f)
            cls.append(c)
            loc.append(torch.exp(l * self.loc_scale[idx - 2]))

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)


if __name__ == '__main__':
    z_f = torch.rand(15, 256, 7, 7).cuda()
    x_f = torch.rand(15, 256, 31, 31).cuda()
    z_fs = [z_f, z_f, z_f]
    x_fs = [x_f, x_f, x_f]
    in_channels = [256, 256, 256]
    cls_out_channels = 2
    confusions = [True, True, True]
    confusion_ways = ["sum", "sum", "sum"]
    confusion_nums = [4, 4, 4]
    weighted = True
    # ban = MyDepthwiseBAN(256, 256).cuda()
    ban = MyMultiBAN(in_channels, cls_out_channels, confusions,
                     confusion_ways, confusion_nums, weighted).cuda()
    cls, loc = ban(z_fs, x_fs)
    print(cls.shape, loc.shape)
