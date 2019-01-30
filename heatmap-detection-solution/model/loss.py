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


# def change_mask_loss():
#     """loss for change_mask
#     """
#     def _myloss(y_true, y_pred):
#         loss = binary_cross_entropy(y_true, y_pred)
#         return 1000*loss

#     return _myloss

def binary_cross_entropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred))

def change_mask_loss():
    """loss for change_mask
    """
    def _myloss(y_true, y_pred):
        return 1000*K.mean(K.square(y_pred - y_true) * K.binary_crossentropy(y_true, y_pred))

    return _myloss


def focal(alpha=0.25):
    """ Create a functor for computing the focal loss.
    Args
        alpha: change object weight.
    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """
    def _focal(y_true, y_pred):
        # compute the focal weight
        alpha_factor = K.switch(y_true > 0.5, alpha, 1-alpha)
        focal_weight = alpha_factor * K.square(y_pred - y_true)
        # compute loss
        cls_loss = focal_weight * K.binary_crossentropy(y_true, y_pred)
        return 1000 * K.mean(cls_loss)

    return _focal

def focal_loss_3(alpha=1000):
    """loss for change_mask
    """
    def _myloss(y_true, y_pred):
        
        y_mask = K.switch(y_true > 0.9, K.pool2d(y_pred, (3, 3), strides=(1, 1), padding='same', pool_mode='max'), y_true)
        y_pred = K.switch(y_true > 0.5, y_mask, y_pred)
        loss_1 = K.switch(y_true < 0.5, K.binary_crossentropy(y_true, y_pred), alpha * K.binary_crossentropy(y_true, y_pred))
        return K.mean(loss_1)

    return _myloss
