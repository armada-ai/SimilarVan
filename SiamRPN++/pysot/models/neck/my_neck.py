# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.center_size = center_size

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = (x.size(3) - self.center_size) // 2
            r = l + self.center_size
            x = x[:, :, l:r, l:r]
        return x


class MyAdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(MyAdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0],
                                          out_channels[0],
                                          center_size)
        else:
            for i in range(self.num):
                self.add_module('downsample' + str(i + 2),
                                AdjustLayer(in_channels[i],
                                            out_channels[i],
                                            center_size))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample' + str(i + 2))
                feat_i = adj_layer(features[i])
                if i > 0:
                    feat_i += out[i - 1]
                out.append(feat_i)

            return out


class MyAdjustAllLayer1(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(MyAdjustAllLayer1, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0],
                                          out_channels[0],
                                          center_size)
        else:
            for i in range(self.num):
                self.add_module('downsample' + str(i + 2),
                                AdjustLayer(in_channels[i],
                                            out_channels[i],
                                            center_size))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample' + str(i + 2))
                feat_i = adj_layer(features[i])
                if i > 0:
                    feat_i += out[i - 1]
                out.append(feat_i)
            for i in range(self.num - 2, -1, -1):
                out[i] += out[i + 1]
            # for i in range(self.num):
            #     print("max: ", torch.max(out[i]).item(), "min: ", torch.min(out[i]).item(),
            #           "mean: ", torch.mean(out[i]).item(), "std: ", torch.std(out[i]).item())
            return out


class MyAdjustAllLayer2(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(MyAdjustAllLayer2, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0],
                                          out_channels[0],
                                          center_size)
        else:
            for i in range(self.num):
                self.add_module('downsample' + str(i + 2),
                                AdjustLayer(in_channels[i],
                                            out_channels[i],
                                            center_size))
                self.add_module('BN' + str(i + 2),
                                nn.BatchNorm2d(out_channels[i]))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample' + str(i + 2))
                feat_i = adj_layer(features[i])
                if i > 0:
                    feat_i += out[i - 1]
                out.append(feat_i)
            for i in range(self.num - 2, -1, -1):
                out[i] += out[i + 1]
            # out_ = []
            for i in range(self.num):
                BN_layer = getattr(self, "BN" + str(i + 2))
                out[i] = BN_layer(out[i])
                # out_.append()
                # print("max: ", torch.max(out[i]).item(), "min: ", torch.min(out[i]).item(),
                #       "mean: ", torch.mean(out[i]).item(), "std: ", torch.std(out[i]).item())
            return out
