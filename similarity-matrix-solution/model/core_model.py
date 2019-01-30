#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = 'core_model'
__author__ = 'fangwudi'
__time__ = '18-12-13 10:44'

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
from keras.models import *
from keras.layers import *
from keras import backend as K
from .switchnorm import SwitchNormalization
K.set_image_dim_ordering('tf')
from .my_inception_v3_sn import InceptionV3_sn
# from .switchnorm import SwitchNormalization
# from .DepthwiseConv2D import DepthwiseConv2D
def name_and_axis(name=None):
    if name is not None:
        bn_name = name + '_bn'
        ac_name = name + '_ac'
    else:
        bn_name = None
        ac_name = None
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    return bn_name, ac_name, channel_axis


def my_conv2d_transpose(x, filters, kernel_size, strides=(1, 1), padding='same', name=None):
    bn_name, ac_name, channel_axis = name_and_axis(name)
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding,
                        use_bias=False, name=name)(x)
    x = SwitchNormalization(axis=channel_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=ac_name)(x)
    return x


def my_conv2d(x, filters, kernel_size, strides=(1, 1), padding='same', name=None, use_bias=False):
    bn_name, ac_name, channel_axis = name_and_axis(name)
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, name=name)(x)
    x = SwitchNormalization(axis=channel_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=ac_name)(x)
    return x


def my_depthwise(x, kernel_size, strides=(1, 1), padding='same', dilation_rate=(1, 1),
                 depth_multiplier=1, name=None):
    bn_name, ac_name, channel_axis = name_and_axis(name)
    x = DepthwiseConv2D(kernel_size, strides=strides, padding=padding,
                        depth_multiplier=depth_multiplier, use_bias=False,
                        dilation_rate=dilation_rate, name=name)(x)
    x = SwitchNormalization(axis=channel_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=ac_name)(x)
    return x

def block_Unet(x, filters, name, updown_type=None, up_filters=None):
    for i in range(2):
        x = my_conv2d(x, filters, (3, 3), name=name + '_conv_' + str(i+1))
    y = None
    if updown_type == 'downsample':
        y = my_depthwise(x, (3, 3), strides=(2, 2), padding='valid', name=name + '_downsample')
    elif updown_type == 'upsample':
        y = my_conv2d_transpose(x, up_filters, (3, 3), strides=(2, 2), padding='valid', name=name + '_upsample')
    return x, y

def block_inner_sub(a, b, filters, name):
    out_a   = my_conv2d(a, filters, (1, 1), name = name + '_a_decrease')
    out_b   = my_conv2d(b, filters, (1, 1), name = name + '_b_decrease')
    a_inner = my_conv2d(a, filters, (1, 1), name = name + '_a_inner_decrease')
    b_inner = my_conv2d(b, filters, (1, 1), name = name + '_b_inner_decrease')
    out_sub = Subtract(name = name + '_sub')([a_inner, b_inner])
    return Concatenate(name = name + '_out')([out_a, out_b, out_sub])


def ab_similarity(b, a, name=None):
    # first use a and b to generate weight matrix,
    # then use that to arregate A as response of one position in b
    batchsize, h, w, c = K.int_shape(b)
    b = Reshape((-1, c), name = name + '_reshape_b')(b)
    a = Reshape((-1, c), name = name + '_reshape_a')(a)
    similarity_matrix = Dot(axes=-1, name=name+'_dot1', normalize=True)([b, a])
    similarity_matrix = Reshape((h, w, h, w, 1), name = name + '_reshape_out')(similarity_matrix)
    return similarity_matrix

def model_v6(image_size=(512, 512), mask_size=(63, 63)):
    # build basemodel
    input_img = Input((*image_size, 3))
    basemodel = InceptionV3_sn(input_tensor=input_img, include_top=False)
    low = basemodel.get_layer(name='mixed2').output     # 288
    middle = basemodel.get_layer(name='mixed7').output  # 768
    high = basemodel.get_layer(name='mixed8').output    # 1280
    vision_model = Model(input_img, [low, middle, high], name='vision_model')
    
    input_a_all = Input((*mask_size, 1))
    
    input_img_a = Input((*image_size, 3))
    input_img_b = Input((*image_size, 3))
    low_a, middle_a, high_a = vision_model(input_img_a)
    low_b, middle_b, high_b = vision_model(input_img_b)
    
    mp = my_conv2d(input_a_all, 16, (3, 3), name='mp_conv_1')
    mp = my_conv2d(mp, 64, (3, 3), name='mp_conv_2')

    x = Concatenate(name='x_low_concat')([mp, block_inner_sub(low_b, low_a, 64, name='low_inner_sub')])
    x_low, x = block_Unet(x, 128, 'x_low_block', updown_type='downsample', up_filters=None)

    x = Concatenate(name='x_middle_concat')([x, block_inner_sub(middle_b, middle_a, 128, name='middle_inner_sub')])
    x_middle, x = block_Unet(x, 256, 'x_middle_block', updown_type='downsample', up_filters=None)

    x = Concatenate(name='x_high_concat')([x, block_inner_sub(high_b, high_a, 128, name='high_inner_sub')])
    _, x = block_Unet(x, 256, 'x_high_block', updown_type='upsample', up_filters=256)

    x = Concatenate(name='x_middle_up_concat')([x, x_middle])
    _, x = block_Unet(x, 256, 'x_middle_up_block', updown_type='upsample', up_filters=128)

    x = Concatenate(name='x_low_up_concat')([x, x_low])
    x, _ = block_Unet(x, 128, 'x_low_up_block', updown_type=None, up_filters=None)

    x_heatmp = Conv2D(1, (1, 1), activation='sigmoid', name='heatmap')(x)

    x_count = GlobalAveragePooling2D(name='pooling')(x)
    x_count = Dense(1, name='count')(x_count)

    return Model([input_img_a, input_img_b, input_a_all], [x_heatmp, x_count], name='main_model')


def model_v1(image_size = (512, 512), mask_size=(31, 31, 31, 31)):
    # build basemodel
    input_img = Input((*image_size, 3))
    # here sn not use
    basemodel = InceptionV3_sn(input_tensor=input_img, weights='imagenet', include_top=False)
    middle = basemodel.get_layer(name='mixed7').output
   # middle = my_conv2d_transpose(middle, 768, (3, 3), strides=(2, 2), padding='valid', name = 'x_upsample')
    vision_model = Model(input_img, middle, name='vision_model')
    # input
    input_img_a = Input((*image_size, 3))
    input_img_b = Input((*image_size, 3))
    input_mask = Input((*mask_size, 1))
    a = vision_model(input_img_a)
    b = vision_model(input_img_b)
    
    s = ab_similarity(b, a, name='similarity')
    s_mask = Multiply(name="similarity_mask")([s, input_mask])
    return Model([input_img_a, input_img_b, input_mask], s_mask)
