#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = 'preposed_model'
__author__ = 'fangwudi'
__time__ = '18-11-21 11:03'

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
from .my_inception_v3_sn import InceptionV3_sn

K.set_image_dim_ordering('tf')
from .switchnorm import SwitchNormalization
from .DepthwiseConv2D import DepthwiseConv2D


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


def block_change_detect_v1(x, x_sub, filters, switch, name):
    x = my_conv2d(x, filters, (1, 1), name=name+'_x_decrease')
    x_sub = my_conv2d(x_sub, filters, (1, 1), name=name + '_sub_decrease')
    x = Multiply(name=name + "_multiply")([x, x_sub])
    for i in range(3):
        x = my_conv2d(x, filters, (3, 3), name=name + '_conv_' + str(i+1))
    if switch == 'low':
        x = my_conv2d(x, filters, (3, 3), strides=(2, 2), padding='valid', name=name + 'downsample')
    elif switch == 'high':
        x = my_conv2d_transpose(x, filters, (3, 3), strides=(2, 2), padding='valid', name=name + 'upsample')
    else:
        x = my_conv2d(x, filters, (3, 3), name=name + '_conv_middle')
    x = my_conv2d(x, filters, (3, 3), name=name + '_conv_end')
    x = Conv2D(1, (1, 1), name=name + '_out')(x)
    return x


def model_v0(image_size=(512, 512)):
    # build basemodel
    input_img = Input((*image_size, 3))
    basemodel = InceptionV3_sn(input_tensor=input_img, include_top=False)
    low = basemodel.get_layer(name='mixed2').output
    middle = basemodel.get_layer(name='mixed7').output
    high = basemodel.get_layer(name='mixed10').output
    vision_model = Model(input_img, [low, middle, high], name='vision_model')
    # mask input
    input_mask_a_all = Input((31, 31, 1))
    # img input
    input_img_a = Input((*image_size, 3))
    input_img_b = Input((*image_size, 3))
    low_a, middle_a, high_a = vision_model(input_img_a)
    low_b, middle_b, high_b = vision_model(input_img_b)
    low_sub = Lambda(lambda z: K.abs(z), name='low_sub')(Subtract()([low_a, low_b]))
    middle_sub = Lambda(lambda z: K.abs(z), name='middle_sub')(Subtract()([middle_a, middle_b]))
    high_sub = Lambda(lambda z: K.abs(z), name='high_sub')(Subtract()([high_a, high_b]))
    # call model for change detect
    low_c = block_change_detect_v1(low_a, low_sub, 128, switch='low', name='change_low')
    middle_c = block_change_detect_v1(middle_a, middle_sub, 256, switch='middle', name='change_middle')
    high_c = block_change_detect_v1(high_a, high_sub, 256, switch='high', name='change_high')
    out = Add(name='merge_change')([low_c, middle_c, high_c])
    a_change_mask = Activation('sigmoid')(out)
    # deal with input_mask_ab_all
    a_change_mask = Multiply(name="a_change_mask")([a_change_mask, input_mask_a_all])
    return Model([input_img_a, input_img_b, input_mask_a_all], a_change_mask, name='main_model')


def model_v1(image_size=(512, 512)):
    # build basemodel
    input_img = Input((*image_size, 3))
    basemodel = InceptionV3_sn(input_tensor=input_img, include_top=False)
    middle = basemodel.get_layer(name='mixed7').output
    vision_model = Model(input_img, middle, name='vision_model')
    # img input
    input_img_a = Input((*image_size, 3))
    input_img_b = Input((*image_size, 3))
    middle_a = vision_model(input_img_a)
    middle_b = vision_model(input_img_b)
    middle_sub = Lambda(lambda z: K.abs(z), name='middle_sub')(Subtract()([middle_a, middle_b]))
    # call model for change detect
    middle_c = block_change_detect_v1(middle_a, middle_sub, 256, switch='middle', name='change_middle')
    a_change_mask = Activation('sigmoid')(middle_c)
    # deal with input_mask_ab_all
    input_mask_a_all = Input((31, 31, 1))
    a_change_mask = Multiply(name="a_change_mask")([a_change_mask, input_mask_a_all])
    return Model([input_img_a, input_img_b, input_mask_a_all], a_change_mask, name='main_model')

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


def model_v5(image_size=(512, 512)):
    # build basemodel
    input_img = Input((*image_size, 3))
    basemodel = InceptionV3_sn(input_tensor=input_img, include_top=False)
    low = basemodel.get_layer(name='mixed2').output     # 288
    middle = basemodel.get_layer(name='mixed7').output  # 768
    high = basemodel.get_layer(name='mixed8').output    # 1280
    vision_model = Model(input_img, [low, middle, high], name='vision_model')
    
    mask_a_all = Input((63, 63, 1))
    input_a_all = Input((63, 63, 1))
    
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

    x = Conv2D(1, (1, 1), activation='sigmoid', name='out')(x)

    return Model([input_img_a, input_img_b, input_a_all], x, name='main_model')

def model_v6(image_size=(512, 512)):
    # build basemodel
    input_img = Input((*image_size, 3))
    basemodel = InceptionV3_sn(input_tensor=input_img, include_top=False)
    low = basemodel.get_layer(name='mixed2').output     # 288
    middle = basemodel.get_layer(name='mixed7').output  # 768
    high = basemodel.get_layer(name='mixed8').output    # 1280
    vision_model = Model(input_img, [low, middle, high], name='vision_model')
    
    mask_a_all = Input((63, 63, 1))
    input_a_all = Input((63, 63, 1))
    
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