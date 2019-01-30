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

def all_accuracy():
    def _all_accuracy(y_true, y_pred):
        acc_all = K.cast(K.equal(y_true, K.round(y_pred)), 'int8')
        acc_batch = K.min(acc_all, axis=[1, 2, 3, 4, 5])
        acc = K.mean(K.cast(acc_batch, 'float32'), axis=-1)
        return acc
    return _all_accuracy


def any_accuracy():
    def _any_accuracy(y_true, y_pred):
        have = K.max(y_true, axis=[3, 4, 5])
        have_sum = K.sum(have)
        acc_all = K.cast(K.equal(y_true, K.round(y_pred)), 'float32')
        acc_batch = K.sum(K.cast(have, 'float32') * K.max(y_true * acc_all, axis=[3, 4, 5]))
        acc = acc_batch/(K.epsilon() + K.cast(have_sum, 'float32'))
        return acc
    return _any_accuracy
