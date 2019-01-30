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


def myloss(alpha=0.5):
    """modify loss, to punish unsimilarity
    :param alpha:
    :return:
    """
    def _myloss(y_true, y_pred):
        x = K.binary_crossentropy(y_true, y_pred)
        x = K.switch(y_true > 0.5, x, alpha * x)
        return K.sum(x)
    return _myloss
