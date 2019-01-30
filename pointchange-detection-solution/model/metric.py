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

def change_mask_all_accuracy():
    def _all_accuracy(y_true, y_pred):
        acc_all = K.cast(K.equal(y_true, K.round(y_pred)), 'int8')
        acc_batch = K.min(acc_all, axis=[1, 2, 3])
        acc = K.mean(K.cast(acc_batch, 'float32'))
        return acc
    return _all_accuracy

def change_mask_change_accuracy():
    def _change_accuracy(y_true, y_pred):
        y_pred = y_true * y_pred
        acc_all = K.cast(K.not_equal(y_true, K.round(y_pred)), 'float32')
        acc = 1.0 - K.clip(K.sum(acc_all)/(K.epsilon() + K.sum(y_true)), 0, 1.0)
        return acc
    return _change_accuracy

def count_accuracy():
    def _count_accuracy(y_true, y_pred):
        acc = K.mean(K.cast(K.equal(y_true, K.round(y_pred)), 'float32'), axis=-1)
        return acc
    return _count_accuracy
