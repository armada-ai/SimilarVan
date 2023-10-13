import torch
from torch import nn
import torch.nn.functional as F
from pysot.models.head.rpn import RPN, DepthwiseXCorr


class CrossDepthwiseRPN(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256, layer_for_reg=None, layer_for_cls=None,
                 cls_coef=0.5, reg_coef=0.5):
        super(CrossDepthwiseRPN, self).__init__()
        if layer_for_cls is None:
            layer_for_cls = [0, 1]
        if layer_for_reg is None:
            layer_for_reg = [0]
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)
        self.layer_for_reg = layer_for_reg
        self.layer_for_cls = layer_for_cls
        self.cls_coef = cls_coef
        self.reg_coef = reg_coef

    def forward(self, z_f, x_f):
        if isinstance(z_f, list):
            assert len(z_f) >= max(len(self.layer_for_reg), len(self.layer_for_cls))
            z_f_cls = sum([z_f[layer] for layer in self.layer_for_cls])
            x_f_cls = sum([x_f[layer] for layer in self.layer_for_cls])
            z_f_reg = sum([z_f[layer] for layer in self.layer_for_reg])
            x_f_reg = sum([x_f[layer] for layer in self.layer_for_reg])
        else:
            z_f_cls = z_f
            x_f_cls = x_f
            z_f_reg = z_f
            x_f_reg = x_f

        # print("z_f_cls: ", torch.max(z_f_cls[0]), torch.min(z_f_cls[0]))
        # print("z_f_reg: ", torch.max(z_f_reg[0]), torch.min(z_f_reg[0]))
        # print("x_f_cls: ", torch.max(x_f_cls[0]), torch.min(x_f_cls[0]))
        # print("x_f_reg: ", torch.max(x_f_reg[0]), torch.min(x_f_reg[0]))
        # print("where: ", torch.where(z_f_cls[0] == torch.max(z_f_cls[0]).item()))
        # print("where: ", torch.where(z_f_reg[0] == torch.max(z_f_reg[0]).item()))
        z_f_cls_clone = [z_cls.clone() for z_cls in z_f_cls]
        z_f_reg_clone = [z_reg.clone() for z_reg in z_f_reg]
        x_f_cls_clone = [x_cls.clone() for x_cls in x_f_cls]
        x_f_reg_clone = [x_reg.clone() for x_reg in x_f_reg]
        # print("z_f_cls_clone: ", torch.max(z_f_cls_clone[0]), torch.min(z_f_cls_clone[0]))
        # print("z_f_reg_clone: ", torch.max(z_f_reg_clone[0]), torch.min(z_f_reg_clone[0]))
        # print("x_f_cls_clone: ", torch.max(x_f_cls_clone[0]), torch.min(x_f_cls_clone[0]))
        # print("x_f_reg_clone: ", torch.max(x_f_reg_clone[0]), torch.min(x_f_reg_clone[0]))
        for i in range(len(z_f_cls)):
            z_f_cls[i] += self.reg_coef * z_f_reg_clone[i]
            z_f_reg[i] += self.cls_coef * z_f_cls_clone[i]

        for i in range(len(x_f_cls)):
            x_f_cls[i] += self.reg_coef * x_f_reg_clone[i]
            x_f_reg[i] += self.cls_coef * x_f_cls_clone[i]

        # print("z_f_cls: ", torch.max(z_f_cls[0]), torch.min(z_f_cls[0]))
        # print("z_f_reg: ", torch.max(z_f_reg[0]), torch.min(z_f_reg[0]))
        # print("x_f_cls: ", torch.max(x_f_cls[0]), torch.min(x_f_cls[0]))
        # print("x_f_reg: ", torch.max(x_f_reg[0]), torch.min(x_f_reg[0]))
        cls = self.cls(z_f_cls, x_f_cls)
        loc = self.loc(z_f_reg, x_f_reg)
        return cls, loc


class MyDepthwiseRPN(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256, layer_for_reg=None, layer_for_cls=None,
                 confusion=False, confusion_way="sum", confusion_num=4):
        super(MyDepthwiseRPN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)
        self.layer_for_reg = layer_for_reg
        self.layer_for_cls = layer_for_cls
        self.confusion = confusion
        self.confusion_way = confusion_way
        self.confusion_num = confusion_num

    def forward(self, z_f, x_f):
        if isinstance(z_f, list):
            assert len(z_f) >= max(len(self.layer_for_reg), len(self.layer_for_cls))
            if self.layer_for_cls is not None:
                z_f_cls = sum([z_f[layer] for layer in self.layer_for_cls])
                x_f_cls = sum([x_f[layer] for layer in self.layer_for_cls])
            else:
                z_f_cls = z_f
                x_f_cls = x_f
            if self.layer_for_reg is not None:
                z_f_reg = sum([z_f[layer] for layer in self.layer_for_reg])
                x_f_reg = sum([x_f[layer] for layer in self.layer_for_reg])
            else:
                z_f_reg = z_f
                x_f_reg = x_f
        else:
            z_f_cls = z_f
            x_f_cls = x_f
            z_f_reg = z_f
            x_f_reg = x_f
        if x_f_cls.shape[0] > 1 and self.confusion:
            x_f_cls = self.features_confusion(x_f_cls, self.confusion_way, self.confusion_num)
        cls = self.cls(z_f_cls, x_f_cls)
        loc = self.loc(z_f_reg, x_f_reg)
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


class MyDepthwiseRPNNew(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256,
                 confusion=False, confusion_way="sum", confusion_num=4):
        super(MyDepthwiseRPNNew, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)
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


class MyMultiRPN(RPN):
    def __init__(self, anchor_num, in_channels,
                 confusion, confusion_way, confusion_num,
                 weighted=False):
        super(MyMultiRPN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('rpn' + str(i + 2),
                            MyDepthwiseRPNNew(anchor_num, in_channels[i], in_channels[i],
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
    r = MyMultiRPN(anchor_num, in_channels, confusion, confusion_way, confusion_num, weighted)
    r = r.cuda()

    z_f = torch.rand(15, 256, 7, 7).cuda()
    x_f = torch.rand(15, 256, 31, 31).cuda()
    z_fs = [z_f, z_f, z_f]
    x_fs = [x_f, x_f, x_f]
    cls, reg = r(z_fs, x_fs)
    print(cls.shape, reg.shape)
