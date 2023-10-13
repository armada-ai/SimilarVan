# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.neck.neck import AdjustLayer, AdjustAllLayer
from pysot.models.neck.my_neck import MyAdjustAllLayer, MyAdjustAllLayer1, MyAdjustAllLayer2
from pysot.models.neck.new_adj import NewAdjustLayer, NewAdjustAllLayer

NECKS = {
    'AdjustLayer': AdjustLayer,
    'AdjustAllLayer': AdjustAllLayer,
    'MyAdjustAllLayer': MyAdjustAllLayer,
    'MyAdjustAllLayer1': MyAdjustAllLayer1,
    'MyAdjustAllLayer2': MyAdjustAllLayer2,

    'NewAdjustLayer': NewAdjustLayer,
    'NewAdjustAllLayer': NewAdjustAllLayer,
}


def get_neck(name, **kwargs):
    return NECKS[name](**kwargs)
