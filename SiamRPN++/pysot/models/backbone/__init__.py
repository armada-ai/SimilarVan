# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.backbone.alexnet import alexnetlegacy, alexnet
from pysot.models.backbone.mobile_v2 import mobilenetv2
from pysot.models.backbone.resnet50_modified import *
from pysot.models.backbone.resnet_atrous import resnet18, resnet34, resnet50

BACKBONES = {
    'alexnetlegacy': alexnetlegacy,
    'mobilenetv2': mobilenetv2,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'alexnet': alexnet,
    'resnet50_modified': resnet50_modified,
    'ibn_resnet50_modified_layer123': ibn_resnet50_modified_layer123,
    'ibn_resnet50_modified_layer23': ibn_resnet50_modified_layer23,
    'ibn_resnet50_modified_layer3': ibn_resnet50_modified_layer3,
    'ibn_resnet50_modified_layer2': ibn_resnet50_modified_layer2,
    'ibn_b_resnet50_modified_layer23': ibn_b_resnet50_modified_layer23,
}


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
