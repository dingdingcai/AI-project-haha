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


def change_mask_loss():
    """loss for change_mask
    """
    def _myloss(y_true, y_pred):
        loss = binary_cross_entropy(y_true, y_pred)
        return 1000*loss

    return _myloss

def binary_cross_entropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred))
