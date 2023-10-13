# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.head.mask import MaskCorr, Refine
from pysot.models.head.my_ban import MyDepthwiseBAN, MyMultiBAN
from pysot.models.head.rpn import UPChannelRPN, DepthwiseRPN, MultiRPN
from pysot.models.head.my_rpn import MyDepthwiseRPN, CrossDepthwiseRPN, MyDepthwiseRPNNew, MyMultiRPN
from pysot.models.head.ban import UPChannelBAN, DepthwiseBAN, MultiBAN
from pysot.models.head.rpn_FTSI import MyMultiRPNFTSI

RPNS = {
    'UPChannelRPN': UPChannelRPN,
    'DepthwiseRPN': DepthwiseRPN,
    'MultiRPN': MultiRPN,
    'CrossDepthwiseRPN': CrossDepthwiseRPN,
    'MyDepthwiseRPN': MyDepthwiseRPN,
    'MyDepthwiseRPNNew': MyDepthwiseRPNNew,
    'MyMultiRPN': MyMultiRPN,

    'MyMultiRPNFTSI': MyMultiRPNFTSI,

    'UPChannelBAN': UPChannelBAN,
    'DepthwiseBAN': DepthwiseBAN,
    'MultiBAN': MultiBAN,
    'MyDepthwiseBAN': MyDepthwiseBAN,
    'MyMultiBAN': MyMultiBAN,
}

MASKS = {
    'MaskCorr': MaskCorr,
}

REFINE = {
    'Refine': Refine,
}


def get_rpn_head(name, **kwargs):
    return RPNS[name](**kwargs)


def get_mask_head(name, **kwargs):
    return MASKS[name](**kwargs)


def get_refine_head(name):
    return REFINE[name]()
