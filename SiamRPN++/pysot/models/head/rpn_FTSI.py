import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from pysot.core.xcorr import xcorr_depthwise
from pysot.models.head.rpn import RPN, DepthwiseXCorr


class DepthwiseXCorrFTSI(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3,
                 confusion=False, confusion_way="sum", confusion_num=4):
        super(DepthwiseXCorrFTSI, self).__init__()
        print("FTSI: ", confusion, "FTSI_way: ", confusion_way, "FTSI_num: ", confusion_num)
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

        self.confusion = confusion
        self.confusion_way = confusion_way
        self.confusion_num = confusion_num

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        if search.shape[0] > 1 and self.confusion:
            search = self.features_confusion(search, self.confusion_way, self.confusion_num)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out

    def features_confusion(self, features, confusion_way, confusion_num=4):
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


class MyDepthwiseRPNFTSI(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256,
                 confusion=False, confusion_way="sum", confusion_num=4):
        super(MyDepthwiseRPNFTSI, self).__init__()
        self.cls = DepthwiseXCorrFTSI(in_channels, out_channels, 2 * anchor_num,
                                      confusion=confusion, confusion_way=confusion_way,
                                      confusion_num=confusion_num)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class MyMultiRPNFTSI(RPN):
    def __init__(self, anchor_num, in_channels,
                 confusion, confusion_way, confusion_num,
                 weighted=False):
        super(MyMultiRPNFTSI, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('rpn' + str(i + 2),
                            MyDepthwiseRPNFTSI(anchor_num, in_channels[i], in_channels[i],
                                               confusion[i], confusion_way[i], confusion_num[i]))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            rpn = getattr(self, 'rpn' + str(idx))
            c, l = rpn(z_f, x_f)
            cls.append(c)
            loc.append(l)

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
    # r = MyDepthwiseRPN(layer_for_reg=[0], layer_for_cls=[0, 1]).cuda()
    # r = CrossDepthwiseRPN(layer_for_reg=[0], layer_for_cls=[0, 1]).cuda()
    in_channels = [256, 256, 256]
    anchor_num = 5
    confusion = [True, True, True]
    confusion_way = ["sum", "sum", "sum"]
    confusion_num = [4, 4, 4]
    weighted = True
    r = MyMultiRPNFTSI(anchor_num, in_channels, confusion, confusion_way, confusion_num, weighted)
    r = r.cuda()

    z_f = torch.rand(15, 256, 7, 7).cuda()
    x_f = torch.rand(15, 256, 31, 31).cuda()
    z_fs = [z_f, z_f, z_f]
    x_fs = [x_f, x_f, x_f]
    cls, reg = r(z_fs, x_fs)
    print(cls.shape, reg.shape)
