#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = 'loss'
__author__ = 'fangwudi'
__time__ = '18-2-28 19:13'

code is far away from bugs 
     ┏┓   ┏┓
    ┏┛┻━━━┛┻━┓
    ┃        ┃
    ┃ ┳┛  ┗┳ ┃
    ┃    ┻   ┃
    ┗━┓    ┏━┛
      ┃    ┗━━━━━┓
      ┃          ┣┓
      ┃          ┏┛
      ┗┓┓┏━━┳┓┏━━┛
       ┃┫┫  ┃┫┫
       ┗┻┛  ┗┻┛
with the god animal protecting
     
"""
from keras import backend as K


def myloss(slope=0.001):
    """modify squared loss, to punish close result
    :param slope:
    :return:
    """
    def _myloss(y_true, y_pred):
        x = K.abs(y_pred - y_true)
        x = K.switch(x > 1, x ** 2, K.maximum(slope * x, (2-slope) * x - (1-slope)))
        return K.mean(x)

    return _myloss
