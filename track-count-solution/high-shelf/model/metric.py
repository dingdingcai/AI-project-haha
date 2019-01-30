#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = 'metric'
__author__ = 'fangwudi'
__time__ = '18-1-10 14:27'

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


def count_accuracy():
    def _count_accuracy(y_true, y_pred):
        acc_all = K.cast(K.equal(y_true, K.round(y_pred)), 'int8')
        acc_batch = K.min(acc_all, axis=-1, keepdims=False)
        acc = K.mean(K.cast(acc_batch, 'float32'), axis=-1)
        return acc
    return _count_accuracy

