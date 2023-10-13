import math

import torch
from torch import nn


class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """

    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class IBNBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1, ibn=None):
        super(IBNBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn == "a":
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        padding = 2 - stride
        if downsample is not None and dilation > 1:
            dilation = dilation // 2
            padding = dilation

        assert stride == 1 or dilation == 1, \
            "stride and dilation must have one equals to zero at least"

        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.IN = nn.InstanceNorm2d(planes * 4, affine=True) if ibn == 'b' else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class ResNet50Modify(nn.Module):
    def __init__(self, block, layers, used_layers, ibn_cfg=(None, None, None)):
        self.inplanes = 64
        self.used_layers = used_layers
        super(ResNet50Modify, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0,  # 3
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, ibn=ibn_cfg[1])

        self.feature_size = 128 * block.expansion
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, ibn=ibn_cfg[2])  # 15x15, 7x7
        self.feature_size = (256 + 128) * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, ibn=None):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=3, stride=stride, bias=False,
                              padding=padding, dilation=dd),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, dilation=dilation, ibn=ibn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, ibn=ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_ = self.relu(x)
        x = self.maxpool(x_)

        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        out = [x_, p1, p2, p3]
        out = [out[i] for i in self.used_layers]
        if len(out) == 1:
            return out[0]
        else:
            return out


def resnet50_modified(**kwargs):
    model = ResNet50Modify(IBNBottleneck, [3, 4, 6], ibn_cfg=(None, None, None), **kwargs)
    return model


def ibn_resnet50_modified_layer123(**kwargs):
    model = ResNet50Modify(IBNBottleneck, [3, 4, 6], ibn_cfg=("a", "a", "a"), **kwargs)
    return model


def ibn_resnet50_modified_layer23(**kwargs):
    model = ResNet50Modify(IBNBottleneck, [3, 4, 6], ibn_cfg=(None, "a", "a"), **kwargs)
    return model


def ibn_resnet50_modified_layer3(**kwargs):
    model = ResNet50Modify(IBNBottleneck, [3, 4, 6], ibn_cfg=(None, None, "a"), **kwargs)
    return model


def ibn_resnet50_modified_layer2(**kwargs):
    model = ResNet50Modify(IBNBottleneck, [3, 4, 6], ibn_cfg=(None, "a", None), **kwargs)
    return model


def ibn_b_resnet50_modified_layer23(**kwargs):
    model = ResNet50Modify(IBNBottleneck, [3, 4, 6], ibn_cfg=(None, "b", "b"), **kwargs)
    return model


if __name__ == '__main__':
    from pysot.utils.model_load import load_pretrain

    print("res50_mod")
    res50_mod = resnet50_modified(used_layers=[2, 3])
    total_params = sum(p.numel() for p in res50_mod.parameters())
    print(total_params)
    print(res50_mod.state_dict().keys())
    load_pretrain(res50_mod, "../../../debug/res50_mod_pretrain.model")
    res50_mod_ = torch.load("../../../debug/res50_mod_pretrain.model")
    keys = ["layer2.0.downsample.0.weight", "layer1.1.conv1.weight"]
    for key in keys:
        print((res50_mod_[key] == res50_mod.state_dict()[key]).all())

    print("\nibn_res50_mod_layer3")
    ibn_res50_mod_layer3 = ibn_resnet50_modified_layer3(used_layers=[2, 3])
    total_params = sum(p.numel() for p in ibn_res50_mod_layer3.parameters())
    print(total_params)
    print(ibn_res50_mod_layer3.state_dict().keys())
    load_pretrain(ibn_res50_mod_layer3, "../../../debug/res50_mod_pretrain.model")
    keys = ["layer3.0.bn1.BN.weight", "layer3.0.bn1.IN.weight",
            "layer2.0.downsample.0.weight", "layer1.1.conv1.weight"]
    ibn_res50_mod_layer3_ = torch.load("../../../debug/ibn_resnet50_modified.model")
    for key in keys:
        print((ibn_res50_mod_layer3_[key] == ibn_res50_mod_layer3.state_dict()[key]).all())

    print("\nibn_res50_mod_layer23")
    ibn_res50_mod_layer23 = ibn_resnet50_modified_layer23(used_layers=[2, 3])
    total_params = sum(p.numel() for p in ibn_res50_mod_layer23.parameters())
    print(total_params)
    print(ibn_res50_mod_layer23.state_dict().keys())
    load_pretrain(ibn_res50_mod_layer23, "../../../debug/res50_mod_pretrain.model")
    keys = ["layer2.0.bn1.BN.weight", "layer2.0.bn1.IN.weight",
            "layer2.0.downsample.0.weight", "layer1.1.conv1.weight"]
    ibn_res50_mod_layer23_ = torch.load("../../../debug/ibn_res50_modified_layer23.model")
    for key in keys:
        print((ibn_res50_mod_layer23_[key] == ibn_res50_mod_layer23.state_dict()[key]).all())

    print("\nibn_res50_mod_layer123")
    ibn_res50_mod_layer123 = ibn_resnet50_modified_layer123(used_layers=[2, 3])
    total_params = sum(p.numel() for p in ibn_res50_mod_layer123.parameters())
    print(total_params)
    print(ibn_res50_mod_layer123.state_dict().keys())
    load_pretrain(ibn_res50_mod_layer123, "../../../debug/res50_mod_pretrain.model")
    keys = ["layer1.0.bn1.BN.weight", "layer1.0.bn1.IN.weight",
            "layer2.0.downsample.0.weight", "layer1.1.conv1.weight"]
    ibn_res50_mod_layer123_ = torch.load("../../../debug/ibn_resnet50_modified.model")
    for key in keys:
        print((ibn_res50_mod_layer123_[key] == ibn_res50_mod_layer123.state_dict()[key]).all())

    print("\nibn_b_res50_mod_layer23")
    ibn_b_res50_mod_layer23 = ibn_b_resnet50_modified_layer23(used_layers=[2, 3])
    total_params = sum(p.numel() for p in ibn_b_res50_mod_layer23.parameters())
    print(total_params)
    print(ibn_b_res50_mod_layer23.state_dict().keys())
    load_pretrain(ibn_b_res50_mod_layer23, "../../../debug/res50_mod.model")
