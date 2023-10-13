# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import logging

logger = logging.getLogger('global')


class NewAdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7, mode="ori"):
        super(NewAdjustLayer, self).__init__()
        assert mode in ["ori", "avg", "max", "add"]
        self.mode = mode
        logger.info("using NewAdjustLayer, mode is '%s'" % mode)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if mode == "max":
            self.max = nn.AdaptiveMaxPool2d(center_size)
        elif mode == "avg":
            self.avg = nn.AdaptiveAvgPool2d(center_size)
        elif mode == "add":
            self.max = nn.AdaptiveMaxPool2d(center_size)
            self.avg = nn.AdaptiveAvgPool2d(center_size)
        self.center_size = center_size

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            if self.mode == "ori":
                l = (x.size(3) - self.center_size) // 2
                r = l + self.center_size
                x = x[:, :, l:r, l:r]
            elif self.mode == "avg":
                x = self.avg(x)
            elif self.mode == "max":
                x = self.max(x)
            elif self.mode == "add":
                x1 = self.max(x)
                x2 = self.avg(x)
                x = x1 + x2

        return x


class NewAdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7, mode=None):
        super(NewAdjustAllLayer, self).__init__()
        if mode is None:
            mode = ["ori", "ori", "ori"]
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = NewAdjustLayer(in_channels[0],
                                             out_channels[0],
                                             center_size,
                                             mode[0])
        else:
            for i in range(self.num):
                self.add_module('downsample' + str(i + 2),
                                NewAdjustLayer(in_channels[i],
                                               out_channels[i],
                                               center_size,
                                               mode[i]))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample' + str(i + 2))
                out.append(adj_layer(features[i]))
            return out


if __name__ == '__main__':
    import torch

    in_chann = 1024
    out_chann = 256
    mode = "max"
    adj = NewAdjustLayer(in_chann, out_chann, 7, mode).cuda()
    a = torch.rand(3, 1024, 15, 15).cuda()
    b = adj(a)
    print(b.shape)
