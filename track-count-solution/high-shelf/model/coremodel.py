#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = 'coremodel'
__author__ = 'fangwudi'
__time__ = '18-1-9 16:11'

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
from keras.applications import *
from keras.applications.inception_v3 import conv2d_bn
from keras import backend as K
from .my_inception_v3 import myInceptionV3

K.set_image_dim_ordering('tf')
# modified keras version of DepthwiseConv2D using tensorflow
from .DepthwiseConv2D import DepthwiseConv2D
from .interleaved_group_conv import  interleaved_group_conv
#from .grouped_conv import grouped_conv
from model.inception_v3_shake import InceptionV3_shake



def my_block(input_tensor, filters_before, filters_after, stage, block_idx, block_type='A',
             kernel_size=(3, 3), depth_multiplier=1):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    if K.image_data_format() == 'channels_last':
        channel_axis = 3
    else:
        channel_axis = 1
    block_name = 'stage_' + str(stage) + '_idx_' + str(block_idx) + '_type_' + block_type

    x_1 = Conv2D(filters_before, (1, 1), name=block_name + '_1_before_conv')(input_tensor)
    x_1 = BatchNormalization(axis=channel_axis, name=block_name + '_1_before_bn')(x_1)
    x_1 = Activation('relu', name=block_name + '_1_before_ac')(x_1)

    x_1 = DepthwiseConv2D(kernel_size, padding='same', depth_multiplier=depth_multiplier, use_bias=False,
                          dilation_rate=(2, 2), name=block_name + '_1_dilation_conv')(x_1)
    x_1 = BatchNormalization(axis=channel_axis, name=block_name + '_1_dilation_bn')(x_1)
    x_1 = Activation('relu', name=block_name + '_1_dilation_ac')(x_1)

    x_2 = Conv2D(filters_before, (1, 1), name=block_name + '_2_before_conv')(input_tensor)
    x_2 = BatchNormalization(axis=channel_axis, name=block_name + '_2_before_bn')(x_2)
    x_2 = Activation('relu', name=block_name + '_2_before_ac')(x_2)

    x_2 = DepthwiseConv2D(kernel_size, padding='same', depth_multiplier=depth_multiplier, use_bias=False,
                          dilation_rate=(1, 3), name=block_name + '_2_dilation_conv')(x_2)
    x_2 = BatchNormalization(axis=channel_axis, name=block_name + '_2_dilation_bn')(x_2)
    x_2 = Activation('relu', name=block_name + '_2_dilation_ac')(x_2)

    x_3 = Conv2D(filters_before, (1, 1), name=block_name + '_3_before_conv')(input_tensor)
    x_3 = BatchNormalization(axis=channel_axis, name=block_name + '_3_before_bn')(x_3)
    x_3 = Activation('relu', name=block_name + '_3_before_ac')(x_3)

    x_3 = DepthwiseConv2D(kernel_size, padding='same', depth_multiplier=depth_multiplier, use_bias=False,
                          dilation_rate=(3, 1), name=block_name + '_3_dilation_conv')(x_3)
    x_3 = BatchNormalization(axis=channel_axis, name=block_name + '_3_dilation_bn')(x_3)
    x_3 = Activation('relu', name=block_name + '_3_dilation_ac')(x_3)

    x = concatenate([x_1, x_2, x_3], name=block_name + '_concat')
    x = Conv2D(filters_after, (1, 1), name=block_name + '_after_conv')(x)
    x = BatchNormalization(axis=channel_axis, name=block_name + '_after_bn')(x)

    if block_type == 'A':
        shortcut = input_tensor
    elif block_type == 'B':
        shortcut = Conv2D(filters_after, (1, 1), name=block_name + '_origin_conv')(input_tensor)
        shortcut = BatchNormalization(axis=channel_axis, name=block_name + '_origin_bn')(shortcut)
    else:
        print('block type not defined！')
        raise NameError

    x = Add(name=block_name + '_add')([x, shortcut])
    x = Activation('relu', name=block_name + '_after_ac')(x)
    return x

def my_conv2d(x, filters, kernel_size, strides=(1, 1), padding='same', name=None):
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
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=name)(x)
    x = BatchNormalization(axis=channel_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=ac_name)(x)
    return x


def my_depthwise(x, kernel_size, strides=(1, 1), padding='same', dilation_rate=(1, 1),
                 depth_multiplier=1, name=None):
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
    x = DepthwiseConv2D(kernel_size, strides=strides, padding=padding,
                        depth_multiplier=depth_multiplier, use_bias=False,
                        dilation_rate=dilation_rate, name=name)(x)
    x = BatchNormalization(axis=channel_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=ac_name)(x)
    return x


def model_v15lc(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'inception_v3':
        basemodel = myInceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    low = basemodel.get_layer(name='mixed2').output
    middle = basemodel.get_layer(name='mixed7').output
    # stage 5
    x = my_block(middle, 256, 768, stage=5, block_idx=1, block_type='B')
    for block_idx in range(2, 4):
        x = my_block(x, 256, 768, stage=5, block_idx=block_idx, block_type='A')
    # stage 6
    x = my_block(x, 256, 768, stage=6, block_idx=1, block_type='B')
    for block_idx in range(2, 4):
        x = my_block(x, 256, 768, stage=6, block_idx=block_idx, block_type='A')
    # connect with others
    p_middle = my_conv2d(middle, 288, (1, 1), name='p_middle_conv')
    # down multiply
    p_low = Multiply(name="low_multiply")([UpSampling2D(size=(2, 2), name="middle_upsampled")(p_middle), low])
    # add low to stage 6
    p_low = my_depthwise(p_low, (3, 3), strides=(2, 2), name='low_depth')
    p_low = my_conv2d(p_low, 768, (1, 1), name="low_conv")
    x = Add(name='low_add')([x, p_low])
    # stage 7
    x = my_block(x, 256, 768, stage=7, block_idx=1, block_type='B')
    for block_idx in range(2, 4):
        x = my_block(x, 256, 768, stage=7, block_idx=block_idx, block_type='A')
    # after flatten
    x = conv2d_bn(x, 128, 1, 1, name='x_conv_1')
    x = conv2d_bn(x, 10, 1, 1, name='x_conv_2')
    x = Flatten(name='x_flatten')(x)
    x = Dropout(0.1)(x)
    # deal with lc
    my_expand = Lambda(lambda y: K.expand_dims(y, axis=-1))
    lc = my_expand(input_lc)
    lc = Conv1D(16, 2, strides=2, activation='relu', name='lc_conv_1')(lc)
    lc_1 = Conv1D(8, 2, padding='same', activation='relu', name='lc_conv_2')(lc)
    lc_2 = Conv1D(8, 3, padding='same', activation='relu', name='lc_conv_3')(lc)
    lc_3 = Conv1D(8, 4, padding='same', activation='relu', name='lc_conv_4')(lc)
    lc = concatenate([lc_1, lc_2, lc_3], name='lc_concat')
    lc = Conv1D(48, 1, activation='relu', name='lc_conv_5')(lc)
    lc = Flatten(name='lc_flatten')(lc)
    # concat
    x_2 = Dense(48, activation='relu', name='x_2_dense')(x)
    concat_lc = concatenate([x_2, lc], name='concat_lc')
    concat_lc = Dense(128, activation='relu', name='dense_lc')(concat_lc)
    x = concatenate([x, concat_lc], name='concat_final')
    x = Dense(128, activation='relu', name='x_dense')(x)
    out = Dense(column_num, name='dense_final')(x)
    return Model([input_img, input_lc], out)


def model_v14lc(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'inception_v3':
        basemodel = myInceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    low = basemodel.get_layer(name='mixed2').output
    middle = basemodel.get_layer(name='mixed7').output
    # stage 5
    x = my_block(middle, 256, 768, stage=5, block_idx=1, block_type='B')
    for block_idx in range(2, 4):
        x = my_block(x, 256, 768, stage=5, block_idx=block_idx, block_type='A')
    # stage 6
    x = my_block(x, 256, 768, stage=6, block_idx=1, block_type='B')
    for block_idx in range(2, 4):
        x = my_block(x, 256, 768, stage=6, block_idx=block_idx, block_type='A')
    # connect with others
    p_middle = my_conv2d(middle, 288, (1, 1), name='p_middle_conv')
    # down multiply
    p_low = Multiply(name="low_multiply")([UpSampling2D(size=(2, 2), name="middle_upsampled")(p_middle), low])
    # add low to stage 6
    p_low = my_depthwise(p_low, (3, 3), strides=(2, 2), name='low_depth')
    p_low = my_conv2d(p_low, 768, (1, 1), name="low_conv")
    x = Add(name='low_add')([x, p_low])
    # stage 7
    x = my_block(x, 256, 768, stage=7, block_idx=1, block_type='B')
    for block_idx in range(2, 4):
        x = my_block(x, 256, 768, stage=7, block_idx=block_idx, block_type='A')
    # after flatten
    x = conv2d_bn(x, 128, 1, 1, name='x_conv_1')
    x = conv2d_bn(x, 10, 1, 1, name='x_conv_2')
    x = Flatten(name='x_flatten')(x)
    x = Dropout(0.1)(x)
    # deal with lc
    my_expand = Lambda(lambda y: K.expand_dims(y, axis=-1))
    lc = my_expand(input_lc)
    lc = Conv1D(16, 2, strides=2, activation='relu', name='lc_conv_1')(lc)
    lc_1 = Conv1D(8, 2, padding='same', activation='relu', name='lc_conv_2')(lc)
    lc_2 = Conv1D(8, 3, padding='same', activation='relu', name='lc_conv_3')(lc)
    lc_3 = Conv1D(8, 4, padding='same', activation='relu', name='lc_conv_4')(lc)
    lc = concatenate([lc_1, lc_2, lc_3], name='lc_concat')
    lc = Conv1D(48, 1, activation='relu', name='lc_conv_5')(lc)
    lc = Flatten(name='lc_flatten')(lc)
    # concat
    x_2 = Dense(48, activation='relu', name='x_2_dense')(x)
    concat_lc = concatenate([x_2, lc], name='concat_lc')
    concat_lc = Dense(128, activation='relu', name='dense_lc')(concat_lc)
    x = concatenate([x, concat_lc], name='concat_final')
    x = Dense(128, activation='relu', name='x_dense')(x)
    out = Dense(column_num, name='dense_final')(x)
    return Model([input_img, input_lc], out)


def model_v13lc(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'inception_v3':
        basemodel = myInceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    low = basemodel.get_layer(name='mixed2').output
    middle = basemodel.get_layer(name='mixed6').output
    # stage 5
    x = my_block(middle, 256, 768, stage=5, block_idx=1, block_type='B')
    for block_idx in range(2, 4):
        x = my_block(x, 256, 768, stage=5, block_idx=block_idx, block_type='A')
    # connect with others
    p_middle = my_conv2d(middle, 288, (1, 1), name='p_middle_conv')
    # down multiply
    p_low = Multiply(name="low_multiply")([UpSampling2D(size=(2, 2), name="middle_upsampled")(p_middle), low])
    # add low to stage 5
    p_low = my_depthwise(p_low, (3, 3), strides=(2, 2), name='low_depth')
    p_low = my_conv2d(p_low, 768, (1, 1), name="low_conv")
    x = Add(name='low_add')([x, p_low])
    # stage 6
    x = my_block(x, 256, 768, stage=6, block_idx=1, block_type='B')
    for block_idx in range(2, 4):
        x = my_block(x, 256, 768, stage=6, block_idx=block_idx, block_type='A')
    # after flatten
    x = conv2d_bn(x, 128, 1, 1, name='x_conv_1')
    x = conv2d_bn(x, 10, 1, 1, name='x_conv_2')
    x = Flatten(name='x_flatten')(x)
    x = Dropout(0.1)(x)
    # deal with lc
    my_expand = Lambda(lambda y: K.expand_dims(y, axis=-1))
    lc = my_expand(input_lc)
    lc = Conv1D(16, 2, strides=2, activation='relu', name='lc_conv_1')(lc)
    lc_1 = Conv1D(8, 2, padding='same', activation='relu', name='lc_conv_2')(lc)
    lc_2 = Conv1D(8, 3, padding='same', activation='relu', name='lc_conv_3')(lc)
    lc_3 = Conv1D(8, 4, padding='same', activation='relu', name='lc_conv_4')(lc)
    lc = concatenate([lc_1, lc_2, lc_3], name='lc_concat')
    lc = Conv1D(48, 1, activation='relu', name='lc_conv_5')(lc)
    lc = Flatten(name='lc_flatten')(lc)
    # concat
    x_2 = Dense(48, activation='relu', name='x_2_dense')(x)
    concat_lc = concatenate([x_2, lc], name='concat_lc')
    concat_lc = Dense(128, activation='relu', name='dense_lc')(concat_lc)
    x = concatenate([x, concat_lc], name='concat_final')
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu', name='x_dense')(x)
    out = Dense(column_num, name='dense_final')(x)
    return Model([input_img, input_lc], out)


def model_v12lc(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    low = basemodel.get_layer(name='mixed4').output
    middle = basemodel.get_layer(name='mixed7').output
    high = basemodel.get_layer(name='mixed10').output
    # low
    low_1 = my_conv2d(low, 12, (1, 1), name='low_1_conv')
    low_2 = my_depthwise(low, (3, 3), name='low_2_depth')
    low_2 = my_conv2d(low_2, 12, (1, 1), name='low_2_conv')
    low_3 = my_depthwise(low, (3, 3), dilation_rate=(2, 2), name='low_3_depth')
    low_3 = my_conv2d(low_3, 12, (1, 1), name='low_3_conv')
    low_4 = my_depthwise(low, (3, 3), dilation_rate=(4, 4), name='low_4_depth')
    low_4 = my_conv2d(low_4, 12, (1, 1), name='low_4_conv')
    low = Add(name='add_low')([low_1, low_2, low_3, low_4])
    low = Flatten(name='low_flatten')(low)
    low = Dense(128, activation='relu', name='low_before_dense')(low)
    low = Dense(column_num, name='low_dense')(low)
    # middle
    middle_1 = my_conv2d(middle, 12, (1, 1), name='middle_1_conv')
    middle_2 = my_depthwise(middle, (3, 3), name='middle_2_depth')
    middle_2 = my_conv2d(middle_2, 12, (1, 1), name='middle_2_conv')
    middle_3 = my_depthwise(middle, (3, 3), dilation_rate=(2, 2), name='middle_3_depth')
    middle_3 = my_conv2d(middle_3, 12, (1, 1), name='middle_3_conv')
    middle_4 = my_depthwise(middle, (3, 3), dilation_rate=(4, 4), name='middle_4_depth')
    middle_4 = my_conv2d(middle_4, 12, (1, 1), name='middle_4_conv')
    middle = Add(name='add_middle')([middle_1, middle_2, middle_3, middle_4])
    middle = Flatten(name='middle_flatten')(middle)
    middle = Dense(128, activation='relu', name='middle_before_dense')(middle)
    middle = Dense(column_num, name='middle_dense')(middle)
    # high
    high_1 = my_conv2d(high, 24, (1, 1), name='high_1_conv')
    high_2 = my_depthwise(high, (3, 3), name='high_2_depth')
    high_2 = my_conv2d(high_2, 24, (1, 1), name='high_2_conv')
    high_3 = my_depthwise(high, (3, 3), dilation_rate=(2, 2), name='high_3_depth')
    high_3 = my_conv2d(high_3, 24, (1, 1), name='high_3_conv')
    high_4 = my_depthwise(high, (3, 3), dilation_rate=(4, 4), name='high_4_depth')
    high_4 = my_conv2d(high_4, 24, (1, 1), name='high_4_conv')
    high = Add(name='add_high')([high_1, high_2, high_3, high_4])
    high = Flatten(name='high_flatten')(high)
    high = Dense(256, activation='relu', name='high_before_dense')(high)
    high = Dense(column_num, name='high_dense')(high)
    # deal with lc
    my_expand = Lambda(lambda y: K.expand_dims(y, axis=-1))
    lc = my_expand(input_lc)
    lc = Conv1D(16, 2, strides=2, activation='relu', name='lc_conv_1')(lc)
    lc_1 = Conv1D(8, 2, padding='same', activation='relu', name='lc_conv_2')(lc)
    lc_2 = Conv1D(8, 3, padding='same', activation='relu', name='lc_conv_3')(lc)
    lc_3 = Conv1D(8, 4, padding='same', activation='relu', name='lc_conv_4')(lc)
    lc = concatenate([lc_1, lc_2, lc_3], name='lc_concat')
    lc = Conv1D(48, 1, activation='relu', name='lc_conv_5')(lc)
    lc = Flatten(name='lc_flatten')(lc)
    # concat
    concat_lc = concatenate([low, middle, high, lc], name='concat_lc')
    concat_lc = Dense(column_num, name='lc_dense')(concat_lc)
    # add
    out = Add(name='add_final')([low, middle, high, concat_lc])
    return Model([input_img, input_lc], out)


def model_v11nolc(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = myInceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    low = basemodel.get_layer(name='mixed2').output
    middle = basemodel.get_layer(name='mixed7').output
    high = basemodel.get_layer(name='mixed10').output
    # low
    low_1 = my_conv2d(low, 64, (1, 1), name='low_1_conv')
    low_2 = my_depthwise(low, (3, 3), name='low_2_depth')
    low_2 = my_conv2d(low_2, 64, (1, 1), name='low_2_conv')
    low_3 = my_depthwise(low, (3, 3), dilation_rate=(2, 2), name='low_3_depth')
    low_3 = my_conv2d(low_3, 64, (1, 1), name='low_3_conv')
    low_4 = my_depthwise(low, (3, 3), dilation_rate=(4, 4), name='low_4_depth')
    low_4 = my_conv2d(low_4, 64, (1, 1), name='low_4_conv')
    low = Add(name='add_low')([low_1, low_2, low_3, low_4])
    # middle
    middle_1 = my_conv2d(middle, 128, (1, 1), name='middle_1_conv')
    middle_2 = my_depthwise(middle, (3, 3), name='middle_2_depth')
    middle_2 = my_conv2d(middle_2, 128, (1, 1), name='middle_2_conv')
    middle_3 = my_depthwise(middle, (3, 3), dilation_rate=(2, 2), name='middle_3_depth')
    middle_3 = my_conv2d(middle_3, 128, (1, 1), name='middle_3_conv')
    middle_4 = my_depthwise(middle, (3, 3), dilation_rate=(4, 4), name='middle_4_depth')
    middle_4 = my_conv2d(middle_4, 128, (1, 1), name='middle_4_conv')
    middle = Add(name='add_middle')([middle_1, middle_2, middle_3, middle_4])
    # high
    high_1 = my_conv2d(high, 256, (1, 1), name='high_1_conv')
    high_2 = my_depthwise(high, (3, 3), name='high_2_depth')
    high_2 = my_conv2d(high_2, 256, (1, 1), name='high_2_conv')
    high_3 = my_depthwise(high, (3, 3), dilation_rate=(2, 2), name='high_3_depth')
    high_3 = my_conv2d(high_3, 256, (1, 1), name='high_3_conv')
    high_4 = my_depthwise(high, (3, 3), dilation_rate=(4, 4), name='high_4_depth')
    high_4 = my_conv2d(high_4, 256, (1, 1), name='high_4_conv')
    high = Add(name='add_high')([high_1, high_2, high_3, high_4])
    # Bidirectional Pyramid Networks
    p_high = my_conv2d(high, 256, (1, 1), name='p_high')
    p_middle = Add(name="p_middle_add")([UpSampling2D(size=(2, 2), name="p_middle_upsampled")(p_high),
                                         my_conv2d(middle, 256, (1, 1), name='p_middle_conv')])
    p_low = Add(name="p_low_add")([UpSampling2D(size=(2, 2), name="p_low_upsampled")(p_middle),
                                   my_conv2d(low, 256, (1, 1), name='p_low_conv')])
    rp_low = p_low
    rp_low_2 = my_depthwise(rp_low, (3, 3), name='rp_low_2_depth')
    rp_low_3 = my_depthwise(p_middle, (3, 3), name='rp_low_3_depth')
    rp_middle = Add(name="rp_middle_add")([my_conv2d(rp_low_2, 256, (1, 1), strides=(2, 2), name='rp_middle_conv_in1'),
                                           my_conv2d(rp_low_3, 256, (1, 1), name='rp_middle_conv_in2')])
    rp_middle_2 = my_depthwise(rp_middle, (3, 3), name='rp_middle_2_depth')
    rp_middle_3 = my_depthwise(p_high, (3, 3), name='rp_middle_3_depth')
    rp_high = Add(name="rp_high_add")([my_conv2d(rp_middle_2, 256, (1, 1), strides=(2, 2), name='rp_high_conv_in1'),
                                       my_conv2d(rp_middle_3, 256, (1, 1), name='rp_high_conv_in2')])
    # count flatten and dense
    count_high = my_conv2d(rp_high, 16, (1, 1), name='high_before_conv')
    count_high = Flatten(name='high_flatten')(count_high)
    count_high = Dense(128, activation='relu', name='high_before_dense')(count_high)
    out = Dense(column_num, name='high_dense')(count_high)
    return Model(input_img, out)


def model_v9lc_tk(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    lower = basemodel.get_layer(name='mixed7').output

    l_x0 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                             depth_multiplier=1, use_bias=False,
                             dilation_rate=(1, 1), name='l_rate1_depthconv1')(lower)
    l_x0 = BatchNormalization(name='l_rate1_depthconv1_bn')(l_x0)
    l_x0 = Activation('relu', name='l_rate1_depthconv1_act')(l_x0)
    # rate 1 pointwise convolution
    l_x0 = conv2d_bn(l_x0, 128, 1, 1, name='l_x0_conv_1')
    l_x0 = conv2d_bn(l_x0, 10, 1, 1, name='l_x0_conv_2')
    l_x0 = Flatten(name='l_x0_flatten')(l_x0)

    
    
    # rate 1 depth convolution
    l_x1_1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(2, 1), name='l_rate1_depthconv1')(lower)
    l_x1 = BatchNormalization(name='l_rate1_depthconv1_bn')(l_x1_1)
    l_x1 = Activation('relu', name='l_rate1_depthconv1_act')(l_x1)
    # rate 1 pointwise convolution
    l_x1 = conv2d_bn(l_x1, 128, 1, 1, name='l_x1_conv_1')
    l_x1_4 = conv2d_bn(l_x1, 10, 1, 1, name='l_x1_conv_2')
    l_x1 = Flatten(name='l_x1_flatten')(l_x1)


    # rate 2 depth convolution
    l_x2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(4, 1), name='l_rate2_depthconv1')(lower)
    l_x2 = BatchNormalization(name='l_rate2_depthconv1_bn')(l_x2)
    l_x2 = Activation('relu', name='l_rate2_depthconv1_act')(l_x2)
    # rate 2 pointwise convolution
    l_x2 = conv2d_bn(l_x2, 128, 1, 1, name='l_x2_conv_1')
    l_x2 = conv2d_bn(l_x2, 10, 1, 1, name='l_x2_conv_2')
    l_x2 = Flatten(name='l_x2_flatten')(l_x2)



    # rate 1 depth convolution
    x1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                         depth_multiplier=1, use_bias=False,
                         dilation_rate=(2, 1), name='rate1_depthconv1')(basemodel.output)
    x1 = BatchNormalization(name='rate1_depthconv1_bn')(x1)
    x1 = Activation('relu', name='rate1_depthconv1_act')(x1)
    # rate 1 pointwise convolution
    x1 = conv2d_bn(x1, 128, 1, 1, name='x1_conv_1')
    x1 = conv2d_bn(x1, 10, 1, 1, name='x1_conv_2')
    x1 = Flatten(name='x1_flatten')(x1)
    x1 = Dense(128, activation='relu', name='x1_before_dense')(x1)
    x1 = Dense(column_num, name='x1_dense')(x1)
    # rate 2 depth convolution
    x2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                         depth_multiplier=1, use_bias=False,
                         dilation_rate=(3, 1), name='rate2_depthconv1')(basemodel.output)
    x2 = BatchNormalization(name='rate2_depthconv1_bn')(x2)
    x2 = Activation('relu', name='rate2_depthconv1_act')(x2)
    # rate 2 pointwise convolution
    x2 = conv2d_bn(x2, 128, 1, 1, name='x2_conv_1')
    x2 = conv2d_bn(x2, 10, 1, 1, name='x2_conv_2')
    x2 = Flatten(name='x2_flatten')(x2)
    x2 = Dense(128, activation='relu', name='x2_before_dense')(x2)
    x2 = Dense(column_num, name='x2_dense')(x2)
    # deal with lc
    my_expand = Lambda(lambda y: K.expand_dims(y, axis=-1))
    lc = my_expand(input_lc)

    lc = Conv1D(16, 2, strides=2, activation='relu', name='lc_conv_1')(lc)
    lc_1 = Conv1D(8, 2, padding='same', activation='relu', name='lc_conv_2')(lc)
    lc_2 = Conv1D(8, 3, padding='same', activation='relu', name='lc_conv_3')(lc)
    lc_3 = Conv1D(8, 4, padding='same', activation='relu', name='lc_conv_4')(lc)
    lc = concatenate([lc_1, lc_2, lc_3], name='lc_concat')
    lc = Conv1D(48, 1, activation='relu', name='lc_conv_5')(lc)
    lc = Flatten(name='lc_flatten')(lc)

    # concat
    concat_lc = concatenate([l_x1, l_x2, l_x0, x1, x2, lc], name='concat_lc')
    concat_lc = Dense(column_num, name='lc_dense')(concat_lc)
    # add
    out = Add(name='add_final')([l_x1, l_x2, l_x0, x1, x2, concat_lc])
    return Model([input_img, input_lc], out)

def model_v10lc(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    low = basemodel.get_layer(name='mixed2').output
    middle = basemodel.get_layer(name='mixed7').output
    high = basemodel.get_layer(name='mixed10').output
    # low
    low_1 = my_conv2d(low, 6, (1, 1), name='low_1_conv')
    low_2 = my_depthwise(low, (3, 3), name='low_2_depth')
    low_2 = my_conv2d(low_2, 6, (1, 1), name='low_2_conv')
    low_3 = my_depthwise(low, (3, 3), dilation_rate=(2, 2), name='low_3_depth')
    low_3 = my_conv2d(low_3, 6, (1, 1), name='low_3_conv')
    low_4 = my_depthwise(low, (3, 3), dilation_rate=(4, 4), name='low_4_depth')
    low_4 = my_conv2d(low_4, 6, (1, 1), name='low_4_conv')
    low = Add(name='add_low')([low_1, low_2, low_3, low_4])
    low = Flatten(name='low_flatten')(low)
    low = Dense(64, activation='relu', name='low_before_dense')(low)
    low = Dense(column_num, name='low_dense')(low)
    # middle
    middle_1 = my_conv2d(middle, 12, (1, 1), name='middle_1_conv')
    middle_2 = my_depthwise(middle, (3, 3), name='middle_2_depth')
    middle_2 = my_conv2d(middle_2, 12, (1, 1), name='middle_2_conv')
    middle_3 = my_depthwise(middle, (3, 3), dilation_rate=(2, 2), name='middle_3_depth')
    middle_3 = my_conv2d(middle_3, 12, (1, 1), name='middle_3_conv')
    middle_4 = my_depthwise(middle, (3, 3), dilation_rate=(4, 4), name='middle_4_depth')
    middle_4 = my_conv2d(middle_4, 12, (1, 1), name='middle_4_conv')
    middle = Add(name='add_middle')([middle_1, middle_2, middle_3, middle_4])
    middle = Flatten(name='middle_flatten')(middle)
    middle = Dense(128, activation='relu', name='middle_before_dense')(middle)
    middle = Dense(column_num, name='middle_dense')(middle)
    # high
    high_1 = my_conv2d(high, 24, (1, 1), name='high_1_conv')
    high_2 = my_depthwise(high, (3, 3), name='high_2_depth')
    high_2 = my_conv2d(high_2, 24, (1, 1), name='high_2_conv')
    high_3 = my_depthwise(high, (3, 3), dilation_rate=(2, 2), name='high_3_depth')
    high_3 = my_conv2d(high_3, 24, (1, 1), name='high_3_conv')
    high_4 = my_depthwise(high, (3, 3), dilation_rate=(4, 4), name='high_4_depth')
    high_4 = my_conv2d(high_4, 24, (1, 1), name='high_4_conv')
    high = Add(name='add_high')([high_1, high_2, high_3, high_4])
    high = Flatten(name='high_flatten')(high)
    high = Dense(256, activation='relu', name='high_before_dense')(high)
    high = Dense(column_num, name='high_dense')(high)
    # deal with lc
    my_expand = Lambda(lambda y: K.expand_dims(y, axis=-1))
    lc = my_expand(input_lc)
    lc = Conv1D(16, 2, strides=2, activation='relu', name='lc_conv_1')(lc)
    lc_1 = Conv1D(8, 2, padding='same', activation='relu', name='lc_conv_2')(lc)
    lc_2 = Conv1D(8, 3, padding='same', activation='relu', name='lc_conv_3')(lc)
    lc_3 = Conv1D(8, 4, padding='same', activation='relu', name='lc_conv_4')(lc)
    lc = concatenate([lc_1, lc_2, lc_3], name='lc_concat')
    lc = Conv1D(48, 1, activation='relu', name='lc_conv_5')(lc)
    lc = Flatten(name='lc_flatten')(lc)
    # concat
    concat_lc = concatenate([low, middle, high, lc], name='concat_lc')
    concat_lc = Dense(column_num, name='lc_dense')(concat_lc)
    # add
    out = Add(name='add_final')([low, middle, high, concat_lc])
    return Model([input_img, input_lc], out)


def model_v9lc_interleaved(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    lower = basemodel.get_layer(name='mixed7').output
    #import pdb;pdb.set_trace()
    # rate 1 depth convolution
    # l_x1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
    #                        depth_multiplier=1, use_bias=False,
    #                        dilation_rate=(2, 1), name='l_rate1_depthconv1')(lower)
    l_x1 = interleaved_group_conv(lower, 2,384 , 1)
    l_x1 = BatchNormalization(name='l_rate1_depthconv1_bn')(l_x1)
    l_x1 = Activation('relu', name='l_rate1_depthconv1_act')(l_x1)
    # rate 1 pointwise convolution
    l_x1 = conv2d_bn(l_x1, 128, 1, 1, name='l_x1_conv_1')
    l_x1 = conv2d_bn(l_x1, 10, 1, 1, name='l_x1_conv_2')
    l_x1 = Flatten(name='l_x1_flatten')(l_x1)
    l_x1 = Dense(128, activation='relu', name='l_x1_before_dense')(l_x1)
    l_x1 = Dense(column_num, name='l_x1_dense')(l_x1)
    # rate 2 depth convolution
    l_x2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(3, 1), name='l_rate2_depthconv1')(lower)


    l_x2 = BatchNormalization(name='l_rate2_depthconv1_bn')(l_x2)
    l_x2 = Activation('relu', name='l_rate2_depthconv1_act')(l_x2)
    # rate 2 pointwise convolution
    l_x2 = conv2d_bn(l_x2, 128, 1, 1, name='l_x2_conv_1')
    l_x2 = conv2d_bn(l_x2, 10, 1, 1, name='l_x2_conv_2')
    l_x2 = Flatten(name='l_x2_flatten')(l_x2)
    l_x2 = Dense(128, activation='relu', name='l_x2_before_dense')(l_x2)
    l_x2 = Dense(column_num, name='l_x2_dense')(l_x2)
    # rate 3 depth convolution
    l_x3 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(4, 1), name='l_rate3_depthconv1')(lower)
    l_x3 = BatchNormalization(name='l_rate3_depthconv1_bn')(l_x3)
    l_x3 = Activation('relu', name='l_rate3_depthconv1_act')(l_x3)
    # rate 3 pointwise convolution
    l_x3 = conv2d_bn(l_x3, 128, 1, 1, name='l_x3_conv_1')
    l_x3 = conv2d_bn(l_x3, 10, 1, 1, name='l_x3_conv_2')
    l_x3 = Flatten(name='l_x3_flatten')(l_x3)
    l_x3 = Dense(128, activation='relu', name='l_x3_before_dense')(l_x3)
    l_x3 = Dense(column_num, name='l_x3_dense')(l_x3)
    # rate 1 depth convolution
    x1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                         depth_multiplier=1, use_bias=False,
                         dilation_rate=(2, 1), name='rate1_depthconv1')(basemodel.output)
    x1 = BatchNormalization(name='rate1_depthconv1_bn')(x1)
    x1 = Activation('relu', name='rate1_depthconv1_act')(x1)
    # rate 1 pointwise convolution
    x1 = conv2d_bn(x1, 128, 1, 1, name='x1_conv_1')
    x1 = conv2d_bn(x1, 10, 1, 1, name='x1_conv_2')
    x1 = Flatten(name='x1_flatten')(x1)
    x1 = Dense(128, activation='relu', name='x1_before_dense')(x1)
    x1 = Dense(column_num, name='x1_dense')(x1)
    # rate 2 depth convolution
    x2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                         depth_multiplier=1, use_bias=False,
                         dilation_rate=(3, 1), name='rate2_depthconv1')(basemodel.output)
    x2 = BatchNormalization(name='rate2_depthconv1_bn')(x2)
    x2 = Activation('relu', name='rate2_depthconv1_act')(x2)
    # rate 2 pointwise convolution
    x2 = conv2d_bn(x2, 128, 1, 1, name='x2_conv_1')
    x2 = conv2d_bn(x2, 10, 1, 1, name='x2_conv_2')
    x2 = Flatten(name='x2_flatten')(x2)
    x2 = Dense(128, activation='relu', name='x2_before_dense')(x2)
    x2 = Dense(column_num, name='x2_dense')(x2)
    # deal with lc
    my_expand = Lambda(lambda y: K.expand_dims(y, axis=-1))
    lc = my_expand(input_lc)

    lc = Conv1D(16, 2, strides=2, activation='relu', name='lc_conv_1')(lc)
    lc_1 = Conv1D(8, 2, padding='same', activation='relu', name='lc_conv_2')(lc)
    lc_2 = Conv1D(8, 3, padding='same', activation='relu', name='lc_conv_3')(lc)
    lc_3 = Conv1D(8, 4, padding='same', activation='relu', name='lc_conv_4')(lc)
    lc = concatenate([lc_1, lc_2, lc_3], name='lc_concat')
    lc = Conv1D(48, 1, activation='relu', name='lc_conv_5')(lc)
    lc = Flatten(name='lc_flatten')(lc)

    # concat
    concat_lc = concatenate([l_x1, l_x2, l_x3, x1, x2, lc], name='concat_lc')
    concat_lc = Dense(column_num, name='lc_dense')(concat_lc)
    # add
    out = Add(name='add_final')([l_x1, l_x2, l_x3, x1, x2, concat_lc])
    return Model([input_img, input_lc], out)


def model_v9lc_large(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    lower = basemodel.get_layer(name='mixed7').output
    lower2=basemodel.get_layer(name='mixed2').output

    #l_x0 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
    #                       depth_multiplier=1, use_bias=False,
    #                       dilation_rate=(2, 1), name='l_rate0_depthconv1')(lower2)
    #l_x0 = BatchNormalization(name='l_rate0_depthconv1_bn')(l_x0)
    #l_x0 = Activation('relu', name='l_rate0_depthconv1_act')(l_x0)
    # rate 1 pointwise convolution
    l_x0 = conv2d_bn(lower2, 128, 1, 1, name='l_x0_conv_1')
    l_x0 = conv2d_bn(l_x0, 10, 1, 1, name='l_x0_conv_2')
    l_x0 = Flatten(name='l_x0_flatten')(l_x0)
    l_x0 = Dense(128, activation='relu', name='l_x0_before_dense')(l_x0)
    l_x0 = Dense(column_num, name='l_x0_dense')(l_x0)
    
    # rate 1 depth convolution
    l_x1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(2, 1), name='l_rate1_depthconv1')(lower)
    l_x1 = BatchNormalization(name='l_rate1_depthconv1_bn')(l_x1)
    l_x1 = Activation('relu', name='l_rate1_depthconv1_act')(l_x1)
    # rate 1 pointwise convolution
    l_x1 = conv2d_bn(l_x1, 128, 1, 1, name='l_x1_conv_1')
    l_x1 = conv2d_bn(l_x1, 10, 1, 1, name='l_x1_conv_2')
    l_x1 = Flatten(name='l_x1_flatten')(l_x1)
    l_x1 = Dense(128, activation='relu', name='l_x1_before_dense')(l_x1)
    l_x1 = Dense(column_num, name='l_x1_dense')(l_x1)
    # rate 2 depth convolution
    l_x2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(3, 1), name='l_rate2_depthconv1')(lower)
    l_x2 = BatchNormalization(name='l_rate2_depthconv1_bn')(l_x2)
    l_x2 = Activation('relu', name='l_rate2_depthconv1_act')(l_x2)
    # rate 2 pointwise convolution
    l_x2 = conv2d_bn(l_x2, 128, 1, 1, name='l_x2_conv_1')
    l_x2 = conv2d_bn(l_x2, 10, 1, 1, name='l_x2_conv_2')
    l_x2 = Flatten(name='l_x2_flatten')(l_x2)
    l_x2 = Dense(128, activation='relu', name='l_x2_before_dense')(l_x2)
    l_x2 = Dense(column_num, name='l_x2_dense')(l_x2)
    # rate 3 depth convolution
    l_x3 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(4, 1), name='l_rate3_depthconv1')(lower)
    l_x3 = BatchNormalization(name='l_rate3_depthconv1_bn')(l_x3)
    l_x3 = Activation('relu', name='l_rate3_depthconv1_act')(l_x3)
    # rate 3 pointwise convolution
    l_x3 = conv2d_bn(l_x3, 128, 1, 1, name='l_x3_conv_1')
    l_x3 = conv2d_bn(l_x3, 10, 1, 1, name='l_x3_conv_2')
    l_x3 = Flatten(name='l_x3_flatten')(l_x3)
    l_x3 = Dense(128, activation='relu', name='l_x3_before_dense')(l_x3)
    l_x3 = Dense(column_num, name='l_x3_dense')(l_x3)

    # rate 4 pointwise convolution
    l_x4 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(6, 1), name='l_rate4_depthconv1')(lower)
    l_x4 = BatchNormalization(name='l_rate4_depthconv1_bn')(l_x4)
    l_x4 = Activation('relu', name='l_rate4_depthconv1_act')(l_x4)
    # rate 3 pointwise convolution
    l_x4 = conv2d_bn(l_x4, 128, 1, 1, name='l_x4_conv_1')
    l_x4 = conv2d_bn(l_x4, 10, 1, 1, name='l_x4_conv_2')
    l_x4 = Flatten(name='l_x4_flatten')(l_x4)
    l_x4 = Dense(128, activation='relu', name='l_x4_before_dense')(l_x4)
    l_x4 = Dense(column_num, name='l_x4_dense')(l_x4)

    # rate 4 pointwise convolution
     
    # rate 1 depth convolution
    x1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                         depth_multiplier=1, use_bias=False,
                         dilation_rate=(2, 1), name='rate1_depthconv1')(basemodel.output)
    x1 = BatchNormalization(name='rate1_depthconv1_bn')(x1)
    x1 = Activation('relu', name='rate1_depthconv1_act')(x1)
    # rate 1 pointwise convolution
    x1 = conv2d_bn(x1, 128, 1, 1, name='x1_conv_1')
    x1 = conv2d_bn(x1, 10, 1, 1, name='x1_conv_2')
    x1 = Flatten(name='x1_flatten')(x1)
    x1 = Dense(128, activation='relu', name='x1_before_dense')(x1)
    x1 = Dense(column_num, name='x1_dense')(x1)
    # rate 2 depth convolution
    x2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                         depth_multiplier=1, use_bias=False,
                         dilation_rate=(3, 1), name='rate2_depthconv1')(basemodel.output)
    x2 = BatchNormalization(name='rate2_depthconv1_bn')(x2)
    x2 = Activation('relu', name='rate2_depthconv1_act')(x2)
    # rate 2 pointwise convolution
    x2 = conv2d_bn(x2, 128, 1, 1, name='x2_conv_1')
    x2 = conv2d_bn(x2, 10, 1, 1, name='x2_conv_2')
    x2 = Flatten(name='x2_flatten')(x2)
    x2 = Dense(128, activation='relu', name='x2_before_dense')(x2)
    x2 = Dense(column_num, name='x2_dense')(x2)
    # deal with lc
    my_expand = Lambda(lambda y: K.expand_dims(y, axis=-1))
    lc = my_expand(input_lc)

    lc = Conv1D(16, 2, strides=2, activation='relu', name='lc_conv_1')(lc)
    lc_1 = Conv1D(8, 2, padding='same', activation='relu', name='lc_conv_2')(lc)
    lc_2 = Conv1D(8, 3, padding='same', activation='relu', name='lc_conv_3')(lc)
    lc_3 = Conv1D(8, 4, padding='same', activation='relu', name='lc_conv_4')(lc)
    lc = concatenate([lc_1, lc_2, lc_3], name='lc_concat')
    lc = Conv1D(48, 1, activation='relu', name='lc_conv_5')(lc)
    lc = Flatten(name='lc_flatten')(lc)

    # concat
    concat_lc = concatenate([l_x0,l_x1, l_x2,l_x3,  x1, x2, lc], name='concat_lc')
    concat_lc = Dense(column_num, name='lc_dense')(concat_lc)
    # add
    out = Add(name='add_final')([l_x0,l_x1, l_x2,l_x3, x1, x2, concat_lc])
    return Model([input_img, input_lc], out)


def model_v9lc_tk(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    lower = basemodel.get_layer(name='mixed7').output

    l_x0 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                             depth_multiplier=1, use_bias=False,
                             dilation_rate=(1, 1), name='l_rate1_depthconv0')(lower)
    l_x0 = BatchNormalization(name='l_rate1_depthconv0_bn')(l_x0)
    l_x0 = Activation('relu', name='l_rate1_depthconv0_act')(l_x0)
    # rate 1 pointwise convolution
    l_x0 = conv2d_bn(l_x0, 128, 1, 1, name='l_x0_conv_1')
    l_x0 = conv2d_bn(l_x0, 10, 1, 1, name='l_x0_conv_2')
    l_x0 = Flatten(name='l_x0_flatten')(l_x0)
    l_x0 = Dense(128, activation='relu', name='l_x0_before_dense')(l_x0)
    #l_x0 = Flatten(name='l_x0_flatten')(l_x0)

    
    # rate 1 depth convolution
    l_x1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(2, 1), name='l_rate1_depthconv1')(lower)
    l_x1 = BatchNormalization(name='l_rate1_depthconv1_bn')(l_x1)
    l_x1 = Activation('relu', name='l_rate1_depthconv1_act')(l_x1)
    # rate 1 pointwise convolution
    l_x1 = conv2d_bn(l_x1, 128, 1, 1, name='l_x1_conv_1')
    l_x1 = conv2d_bn(l_x1, 10, 1, 1, name='l_x1_conv_2')
    l_x1 = Flatten(name='l_x1_flatten')(l_x1)
    l_x1 = Dense(128, activation='relu', name='l_x1_before_dense')(l_x1)
    #l_x1 = Flatten(name='l_x1_flatten')(l_x1)


    # rate 2 depth convolution
    l_x2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(4, 1), name='l_rate2_depthconv1')(lower)
    l_x2 = BatchNormalization(name='l_rate2_depthconv1_bn')(l_x2)
    l_x2 = Activation('relu', name='l_rate2_depthconv1_act')(l_x2)
    # rate 2 pointwise convolution
    l_x2 = conv2d_bn(l_x2, 128, 1, 1, name='l_x2_conv_1')
    l_x2 = conv2d_bn(l_x2, 10, 1, 1, name='l_x2_conv_2')
    l_x2 = Flatten(name='l_x2_flatten')(l_x2)
    l_x2 = Dense(128, activation='relu', name='l_x2_before_dense')(l_x2)
    #l_x2 = Flatten(name='l_x2_flatten')(l_x2)


    # rate 1 depth convolution
    x0=basemodel.output
    x0 = Flatten(name='x0_flatten')(x0)
    x0 = Dense(128, activation='relu', name='x0_before_dense')(x0)

    x1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                         depth_multiplier=1, use_bias=False,
                         dilation_rate=(2, 1), name='rate1_depthconv1')(basemodel.output)
    x1 = BatchNormalization(name='rate1_depthconv1_bn')(x1)
    x1 = Activation('relu', name='rate1_depthconv1_act')(x1)
    # rate 1 pointwise convolution
    x1 = conv2d_bn(x1, 128, 1, 1, name='x1_conv_1')
    x1 = conv2d_bn(x1, 10, 1, 1, name='x1_conv_2')
    x1 = Flatten(name='x1_flatten')(x1)
    x1 = Dense(128, activation='relu', name='x1_before_dense')(x1)
    # rate 2 depth convolution
    x2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                         depth_multiplier=1, use_bias=False,
                         dilation_rate=(3, 1), name='rate2_depthconv1')(basemodel.output)
    x2 = BatchNormalization(name='rate2_depthconv1_bn')(x2)
    x2 = Activation('relu', name='rate2_depthconv1_act')(x2)
    # rate 2 pointwise convolution
    x2 = conv2d_bn(x2, 128, 1, 1, name='x2_conv_1')
    x2 = conv2d_bn(x2, 10, 1, 1, name='x2_conv_2')
    x2 = Flatten(name='x2_flatten')(x2)
    x2 = Dense(128, activation='relu', name='x2_before_dense')(x2)

    # deal with lc
    my_expand = Lambda(lambda y: K.expand_dims(y, axis=-1))
    lc = my_expand(input_lc)

    lc = Conv1D(16, 2, strides=2, activation='relu', name='lc_conv_1')(lc)
    lc_1 = Conv1D(8, 2, padding='same', activation='relu', name='lc_conv_2')(lc)
    lc_2 = Conv1D(8, 3, padding='same', activation='relu', name='lc_conv_3')(lc)
    lc_3 = Conv1D(8, 4, padding='same', activation='relu', name='lc_conv_4')(lc)
    lc = concatenate([lc_1, lc_2, lc_3], name='lc_concat')
    lc_1 = Conv1D(48, 1, activation='relu', name='lc_conv_5')(lc)
    lc = Flatten(name='lc_flatten')(lc_1)

    # concat
    concat_lc = concatenate([l_x1, l_x2, l_x0, x1, x2,x0, lc], name='concat_lc')
    concat_lc = concatenate([l_x1, l_x2, l_x0, x1, x2,x0, concat_lc], name='concat_lc1')

    out = Dense(column_num, name='lc_dense')(concat_lc)
    # add

    return Model([input_img, input_lc], out)


def model_v9lc(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    lower = basemodel.get_layer(name='mixed7').output
    # rate 1 depth convolution
    l_x1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(2, 1), name='l_rate1_depthconv1')(lower)
    l_x1 = BatchNormalization(name='l_rate1_depthconv1_bn')(l_x1)
    l_x1 = Activation('relu', name='l_rate1_depthconv1_act')(l_x1)
    # rate 1 pointwise convolution
    l_x1 = conv2d_bn(l_x1, 128, 1, 1, name='l_x1_conv_1')
    l_x1 = conv2d_bn(l_x1, 10, 1, 1, name='l_x1_conv_2')
    l_x1 = Flatten(name='l_x1_flatten')(l_x1)
    l_x1 = Dense(128, activation='relu', name='l_x1_before_dense')(l_x1)
    l_x1 = Dense(column_num, name='l_x1_dense')(l_x1)
    # rate 2 depth convolution
    l_x2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(3, 1), name='l_rate2_depthconv1')(lower)
    l_x2 = BatchNormalization(name='l_rate2_depthconv1_bn')(l_x2)
    l_x2 = Activation('relu', name='l_rate2_depthconv1_act')(l_x2)
    # rate 2 pointwise convolution
    l_x2 = conv2d_bn(l_x2, 128, 1, 1, name='l_x2_conv_1')
    l_x2 = conv2d_bn(l_x2, 10, 1, 1, name='l_x2_conv_2')
    l_x2 = Flatten(name='l_x2_flatten')(l_x2)
    l_x2 = Dense(128, activation='relu', name='l_x2_before_dense')(l_x2)
    l_x2 = Dense(column_num, name='l_x2_dense')(l_x2)
    # rate 3 depth convolution
    l_x3 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(4, 1), name='l_rate3_depthconv1')(lower)
    l_x3 = BatchNormalization(name='l_rate3_depthconv1_bn')(l_x3)
    l_x3 = Activation('relu', name='l_rate3_depthconv1_act')(l_x3)
    # rate 3 pointwise convolution
    l_x3 = conv2d_bn(l_x3, 128, 1, 1, name='l_x3_conv_1')
    l_x3 = conv2d_bn(l_x3, 10, 1, 1, name='l_x3_conv_2')
    l_x3 = Flatten(name='l_x3_flatten')(l_x3)
    l_x3 = Dense(128, activation='relu', name='l_x3_before_dense')(l_x3)
    l_x3 = Dense(column_num, name='l_x3_dense')(l_x3)
    # rate 1 depth convolution
    x1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                         depth_multiplier=1, use_bias=False,
                         dilation_rate=(2, 1), name='rate1_depthconv1')(basemodel.output)
    x1 = BatchNormalization(name='rate1_depthconv1_bn')(x1)
    x1 = Activation('relu', name='rate1_depthconv1_act')(x1)
    # rate 1 pointwise convolution
    x1 = conv2d_bn(x1, 128, 1, 1, name='x1_conv_1')
    x1 = conv2d_bn(x1, 10, 1, 1, name='x1_conv_2')
    x1 = Flatten(name='x1_flatten')(x1)
    x1 = Dense(128, activation='relu', name='x1_before_dense')(x1)
    x1 = Dense(column_num, name='x1_dense')(x1)
    # rate 2 depth convolution
    x2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                         depth_multiplier=1, use_bias=False,
                         dilation_rate=(3, 1), name='rate2_depthconv1')(basemodel.output)
    x2 = BatchNormalization(name='rate2_depthconv1_bn')(x2)
    x2 = Activation('relu', name='rate2_depthconv1_act')(x2)
    # rate 2 pointwise convolution
    x2 = conv2d_bn(x2, 128, 1, 1, name='x2_conv_1')
    x2 = conv2d_bn(x2, 10, 1, 1, name='x2_conv_2')
    x2 = Flatten(name='x2_flatten')(x2)
    x2 = Dense(128, activation='relu', name='x2_before_dense')(x2)
    x2 = Dense(column_num, name='x2_dense')(x2)
    # deal with lc
    my_expand = Lambda(lambda y: K.expand_dims(y, axis=-1))
    lc = my_expand(input_lc)
    
    lc = Conv1D(16, 2, strides=2, activation='relu', name='lc_conv_1')(lc)
    lc_1 = Conv1D(8, 2, padding='same', activation='relu', name='lc_conv_2')(lc)
    lc_2 = Conv1D(8, 3, padding='same', activation='relu', name='lc_conv_3')(lc)
    lc_3 = Conv1D(8, 4, padding='same', activation='relu', name='lc_conv_4')(lc)
    lc = concatenate([lc_1, lc_2, lc_3], name='lc_concat')
    lc = Conv1D(48, 1, activation='relu', name='lc_conv_5')(lc)
    lc = Flatten(name='lc_flatten')(lc)
    
    # concat
    concat_lc = concatenate([l_x1, l_x2, l_x3, x1, x2, lc], name='concat_lc')
    concat_lc = Dense(column_num, name='lc_dense')(concat_lc)
    # add
    out = Add(name='add_final')([l_x1, l_x2, l_x3, x1, x2, concat_lc])
    return Model([input_img, input_lc], out)

def model_v9lc_dropout(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    lower = basemodel.get_layer(name='mixed7').output
    # rate 1 depth convolution
    l_x1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(2, 1), name='l_rate1_depthconv1')(lower)
    l_x1 = BatchNormalization(name='l_rate1_depthconv1_bn')(l_x1)
    l_x1 = Activation('relu', name='l_rate1_depthconv1_act')(l_x1)
    # rate 1 pointwise convolution
    l_x1 = conv2d_bn(l_x1, 128, 1, 1, name='l_x1_conv_1')
    l_x1 = conv2d_bn(l_x1, 10, 1, 1, name='l_x1_conv_2')
    l_x1 = Flatten(name='l_x1_flatten')(l_x1)
    l_x1 = Dropout(0.2)(l_x1)
    l_x1 = Dense(128, activation='relu', name='l_x1_before_dense')(l_x1)
    l_x1 = Dense(column_num, name='l_x1_dense')(l_x1)
    # rate 2 depth convolution
    l_x2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(3, 1), name='l_rate2_depthconv1')(lower)
    l_x2 = BatchNormalization(name='l_rate2_depthconv1_bn')(l_x2)
    l_x2 = Activation('relu', name='l_rate2_depthconv1_act')(l_x2)
    # rate 2 pointwise convolution
    l_x2 = conv2d_bn(l_x2, 128, 1, 1, name='l_x2_conv_1')
    l_x2 = conv2d_bn(l_x2, 10, 1, 1, name='l_x2_conv_2')
    l_x2 = Flatten(name='l_x2_flatten')(l_x2)
    l_x2 = Dropout(0.2)(l_x2)
    l_x2 = Dense(128, activation='relu', name='l_x2_before_dense')(l_x2)
    l_x2 = Dense(column_num, name='l_x2_dense')(l_x2)
    # rate 3 depth convolution
    l_x3 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(4, 1), name='l_rate3_depthconv1')(lower)
    l_x3 = BatchNormalization(name='l_rate3_depthconv1_bn')(l_x3)
    l_x3 = Activation('relu', name='l_rate3_depthconv1_act')(l_x3)
    # rate 3 pointwise convolution
    l_x3 = conv2d_bn(l_x3, 128, 1, 1, name='l_x3_conv_1')
    l_x3 = conv2d_bn(l_x3, 10, 1, 1, name='l_x3_conv_2')
    l_x3 = Flatten(name='l_x3_flatten')(l_x3)
    l_x3 = Dropout(0.2)(l_x3)
    l_x3 = Dense(128, activation='relu', name='l_x3_before_dense')(l_x3)
    l_x3 = Dense(column_num, name='l_x3_dense')(l_x3)
    # rate 1 depth convolution
    x1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                         depth_multiplier=1, use_bias=False,
                         dilation_rate=(2, 1), name='rate1_depthconv1')(basemodel.output)
    x1 = BatchNormalization(name='rate1_depthconv1_bn')(x1)
    x1 = Activation('relu', name='rate1_depthconv1_act')(x1)
    # rate 1 pointwise convolution
    x1 = conv2d_bn(x1, 128, 1, 1, name='x1_conv_1')
    x1 = conv2d_bn(x1, 10, 1, 1, name='x1_conv_2')
    x1 = Flatten(name='x1_flatten')(x1)
    x1 = Dropout(0.2)(x1)
    x1 = Dense(128, activation='relu', name='x1_before_dense')(x1)
    x1 = Dense(column_num, name='x1_dense')(x1)
    # rate 2 depth convolution
    x2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                         depth_multiplier=1, use_bias=False,
                         dilation_rate=(3, 1), name='rate2_depthconv1')(basemodel.output)
    x2 = BatchNormalization(name='rate2_depthconv1_bn')(x2)
    x2 = Activation('relu', name='rate2_depthconv1_act')(x2)
    # rate 2 pointwise convolution
    x2 = conv2d_bn(x2, 128, 1, 1, name='x2_conv_1')
    x2 = conv2d_bn(x2, 10, 1, 1, name='x2_conv_2')
    x2 = Flatten(name='x2_flatten')(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Dense(128, activation='relu', name='x2_before_dense')(x2)
    x2 = Dense(column_num, name='x2_dense')(x2)
    # deal with lc
    my_expand = Lambda(lambda y: K.expand_dims(y, axis=-1))
    lc = my_expand(input_lc)
    
    lc = Conv1D(16, 2, strides=2, activation='relu', name='lc_conv_1')(lc)
    lc_1 = Conv1D(8, 2, padding='same', activation='relu', name='lc_conv_2')(lc)
    lc_2 = Conv1D(8, 3, padding='same', activation='relu', name='lc_conv_3')(lc)
    lc_3 = Conv1D(8, 4, padding='same', activation='relu', name='lc_conv_4')(lc)
    lc = concatenate([lc_1, lc_2, lc_3], name='lc_concat')
    lc = Conv1D(48, 1, activation='relu', name='lc_conv_5')(lc)
    lc = Flatten(name='lc_flatten')(lc)
    
    # concat
    concat_lc = concatenate([l_x1, l_x2, l_x3, x1, x2, lc], name='concat_lc')
    concat_lc = Dense(column_num, name='lc_dense')(concat_lc)
    # add
    out = Add(name='add_final')([l_x1, l_x2, l_x3, x1, x2, concat_lc])
    return Model([input_img, input_lc], out)

def model_v9lc_shake(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3_shake(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    lower = basemodel.get_layer(name='mixed7').output
    # rate 1 depth convolution
    l_x1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(2, 1), name='l_rate1_depthconv1')(lower)
    l_x1 = BatchNormalization(name='l_rate1_depthconv1_bn')(l_x1)
    l_x1 = Activation('relu', name='l_rate1_depthconv1_act')(l_x1)
    # rate 1 pointwise convolution
    l_x1 = conv2d_bn(l_x1, 128, 1, 1, name='l_x1_conv_1')
    l_x1 = conv2d_bn(l_x1, 10, 1, 1, name='l_x1_conv_2')
    l_x1 = Flatten(name='l_x1_flatten')(l_x1)
    l_x1 = Dense(128, activation='relu', name='l_x1_before_dense')(l_x1)
    l_x1 = Dense(column_num, name='l_x1_dense')(l_x1)
    # rate 2 depth convolution
    l_x2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(3, 1), name='l_rate2_depthconv1')(lower)
    l_x2 = BatchNormalization(name='l_rate2_depthconv1_bn')(l_x2)
    l_x2 = Activation('relu', name='l_rate2_depthconv1_act')(l_x2)
    # rate 2 pointwise convolution
    l_x2 = conv2d_bn(l_x2, 128, 1, 1, name='l_x2_conv_1')
    l_x2 = conv2d_bn(l_x2, 10, 1, 1, name='l_x2_conv_2')
    l_x2 = Flatten(name='l_x2_flatten')(l_x2)
    l_x2 = Dense(128, activation='relu', name='l_x2_before_dense')(l_x2)
    l_x2 = Dense(column_num, name='l_x2_dense')(l_x2)
    # rate 3 depth convolution
    l_x3 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                           depth_multiplier=1, use_bias=False,
                           dilation_rate=(4, 1), name='l_rate3_depthconv1')(lower)
    l_x3 = BatchNormalization(name='l_rate3_depthconv1_bn')(l_x3)
    l_x3 = Activation('relu', name='l_rate3_depthconv1_act')(l_x3)
    # rate 3 pointwise convolution
    l_x3 = conv2d_bn(l_x3, 128, 1, 1, name='l_x3_conv_1')
    l_x3 = conv2d_bn(l_x3, 10, 1, 1, name='l_x3_conv_2')
    l_x3 = Flatten(name='l_x3_flatten')(l_x3)
    l_x3 = Dense(128, activation='relu', name='l_x3_before_dense')(l_x3)
    l_x3 = Dense(column_num, name='l_x3_dense')(l_x3)
    # rate 1 depth convolution
    x1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                         depth_multiplier=1, use_bias=False,
                         dilation_rate=(2, 1), name='rate1_depthconv1')(basemodel.output)
    x1 = BatchNormalization(name='rate1_depthconv1_bn')(x1)
    x1 = Activation('relu', name='rate1_depthconv1_act')(x1)
    # rate 1 pointwise convolution
    x1 = conv2d_bn(x1, 128, 1, 1, name='x1_conv_1')
    x1 = conv2d_bn(x1, 10, 1, 1, name='x1_conv_2')
    x1 = Flatten(name='x1_flatten')(x1)
    x1 = Dense(128, activation='relu', name='x1_before_dense')(x1)
    x1 = Dense(column_num, name='x1_dense')(x1)
    # rate 2 depth convolution
    x2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid',
                         depth_multiplier=1, use_bias=False,
                         dilation_rate=(3, 1), name='rate2_depthconv1')(basemodel.output)
    x2 = BatchNormalization(name='rate2_depthconv1_bn')(x2)
    x2 = Activation('relu', name='rate2_depthconv1_act')(x2)
    # rate 2 pointwise convolution
    x2 = conv2d_bn(x2, 128, 1, 1, name='x2_conv_1')
    x2 = conv2d_bn(x2, 10, 1, 1, name='x2_conv_2')
    x2 = Flatten(name='x2_flatten')(x2)
    x2 = Dense(128, activation='relu', name='x2_before_dense')(x2)
    x2 = Dense(column_num, name='x2_dense')(x2)
    # deal with lc
    my_expand = Lambda(lambda y: K.expand_dims(y, axis=-1))
    lc = my_expand(input_lc)

    lc = Conv1D(16, 2, strides=2, activation='relu', name='lc_conv_1')(lc)
    lc_1 = Conv1D(8, 2, padding='same', activation='relu', name='lc_conv_2')(lc)
    lc_2 = Conv1D(8, 3, padding='same', activation='relu', name='lc_conv_3')(lc)
    lc_3 = Conv1D(8, 4, padding='same', activation='relu', name='lc_conv_4')(lc)
    lc = concatenate([lc_1, lc_2, lc_3], name='lc_concat')
    lc = Conv1D(48, 1, activation='relu', name='lc_conv_5')(lc)
    lc = Flatten(name='lc_flatten')(lc)

    # concat
    concat_lc = concatenate([l_x1, l_x2, l_x3, x1, x2, lc], name='concat_lc')
    concat_lc = Dense(column_num, name='lc_dense')(concat_lc)
    # add
    out = Add(name='add_final')([l_x1, l_x2, l_x3, x1, x2, concat_lc])
    return Model([input_img, input_lc], out)


def model_lyt_v9lc(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    # begin
    print(basemodel.output)
    x1 = conv2d_bn(basemodel.output, 10, 3, 3, name='x1_conv_1')
    x1 = Flatten(name='x1_flatten')(x1)
    x1 = Dropout(.5)(x1)
    x1 = Dense(7, activation='relu')(x1)
    x2 = conv2d_bn(basemodel.output, 10, 1, 1, name='x2_conv_1')
    x2 = Flatten(name='x2_flatten')(x2)
    x2 = Dropout(.5)(x2)
    x2 = Dense(7, activation='relu')(x2)
    # deal with lc
    my_expand = Lambda(lambda y: K.expand_dims(y, axis=-1))
    lc = my_expand(input_lc)
    lc = Conv1D(16, 2, strides=2, activation='relu', name='lc_conv_1')(lc)
    lc_1 = Conv1D(8, 2, padding='same', activation='relu', name='lc_conv_2')(lc)
    lc_2 = Conv1D(8, 3, padding='same', activation='relu', name='lc_conv_3')(lc)
    lc_3 = Conv1D(8, 4, padding='same', activation='relu', name='lc_conv_4')(lc)
    lc = concatenate([lc_1, lc_2, lc_3], name='lc_concat')
    lc = Conv1D(48, 1, activation='relu', name='lc_conv_5')(lc)
    lc = Flatten(name='lc_flatten')(lc)
    lc = Dense(7, activation='relu')(lc)
    # concat
    x3 = concatenate([lc, x1])
    x3 = Dense(7)(x3)
    out = Add()([x1, x2, x3])
    return Model([input_img, input_lc], out)


def model_v7lc(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    # begin
    x1 = conv2d_bn(basemodel.output, 10, 3, 3, name='x1_conv_1')
    x1 = Flatten(name='x1_flatten')(x1)
    x2 = conv2d_bn(basemodel.output, 128, 3, 3, name='x2_conv_1')
    x2 = conv2d_bn(x2, 10, 3, 3, name='x2_conv_2')
    x2 = Flatten(name='x2_flatten')(x2)
    # deal with lc
    my_expand = Lambda(lambda y: K.expand_dims(y, axis=-1))
    lc = my_expand(input_lc)
    lc = Conv1D(16, 2, strides=2, activation='relu', name='lc_conv_1')(lc)
    lc_1 = Conv1D(8, 2, padding='same', activation='relu', name='lc_conv_2')(lc)
    lc_2 = Conv1D(8, 3, padding='same', activation='relu', name='lc_conv_3')(lc)
    lc_3 = Conv1D(8, 4, padding='same', activation='relu', name='lc_conv_4')(lc)
    lc = concatenate([lc_1, lc_2, lc_3], name='lc_concat')
    lc = Conv1D(48, 1, activation='relu', name='lc_conv_5')(lc)
    lc = Flatten(name='lc_flatten')(lc)
    # concatenate
    x = concatenate([x1, x2, lc], name='after_concat1')
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', name='after_dense1')(x)
    # output
    out = Dense(column_num, name='count')(x)
    return Model([input_img, input_lc], out)


def model_v6nolc(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    # begin
    x1 = conv2d_bn(basemodel.output, 10, 3, 3, name='x1_conv_1')
    x1 = Flatten(name='x1_flatten')(x1)
    x2 = conv2d_bn(basemodel.output, 128, 3, 3, name='x2_conv_1')
    x2 = conv2d_bn(x2, 10, 3, 3, name='x2_conv_2')
    x2 = Flatten(name='x2_flatten')(x2)
    x = concatenate([x1, x2], name='after_concat1')
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', name='after_dense1')(x)
    # output
    out = Dense(column_num, name='count')(x)
    return Model([input_img, input_lc], out)


def model_v4nolc(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    # begin
    x1 = conv2d_bn(basemodel.output, 10, 3, 3, name='x1_conv_1')
    x1 = Flatten(name='x1_flatten')(x1)
    x2 = conv2d_bn(basemodel.output, 128, 3, 3, name='x2_conv_1')
    x2 = conv2d_bn(x2, 10, 3, 3, name='x2_conv_2')
    x2 = Flatten(name='x2_flatten')(x2)
    x = concatenate([x1, x2], name='after_concat1')
    x = Dense(128, activation='relu', name='after_dense1')(x)
    # output
    out = Dense(column_num, name='count')(x)
    return Model([input_img, input_lc], out)


def model_v5lc(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    # begin
    x = basemodel.output
    # branch1x7
    x1 = conv2d_bn(x, 192, 1, 1, name='x1_conv_1')
    x1 = conv2d_bn(x1, 192, 1, 7, name='x1_conv_2')
    # branch7x1
    x2 = conv2d_bn(x, 192, 1, 1, name='x2_conv_1')
    x2 = conv2d_bn(x2, 192, 7, 1, name='x2_conv_2')
    # branch_combine
    x3 = concatenate([x1, x2], name='x1_x2_combine')
    x3 = conv2d_bn(x3, 192, 3, 3, name='x3_conv_1')
    # branch pool
    x4 = conv2d_bn(x3, 192, 3, 3, strides=(2, 2), name='x4_conv_1')
    # x1 flatten
    x1 = conv2d_bn(x1, 10, 3, 3, padding='valid', name='x1_conv_3')
    x1 = Flatten(name='x1_flatten')(x1)
    # x2 flatten
    x2 = conv2d_bn(x2, 10, 3, 3, padding='valid', name='x2_conv_3')
    x2 = Flatten(name='x2_flatten')(x2)
    # x3 flatten
    x3 = conv2d_bn(x3, 10, 3, 3, padding='valid', name='x3_conv_2')
    x3 = Flatten(name='x3_flatten')(x3)
    # x4 flatten
    x4 = conv2d_bn(x4, 10, 3, 3, padding='valid', name='x4_conv_2')
    x4 = Flatten(name='x4_flatten')(x4)
    # deal with lc
    my_expand = Lambda(lambda y: K.expand_dims(y, axis=-1))
    lc = my_expand(input_lc)
    lc = Conv1D(16, 2, strides=2, activation='relu', name='lc_conv_1')(lc)
    lc_1 = Conv1D(8, 2, padding='same', activation='relu', name='lc_conv_2')(lc)
    lc_2 = Conv1D(8, 3, padding='same', activation='relu', name='lc_conv_3')(lc)
    lc_3 = Conv1D(8, 4, padding='same', activation='relu', name='lc_conv_4')(lc)
    lc = concatenate([lc_1, lc_2, lc_3], name='lc_concat')
    lc = Conv1D(24, 1, activation='relu', name='lc_conv_5')(lc)
    lc = Flatten(name='lc_flatten')(lc)
    # concat all
    x = concatenate([x1, x2, x3, x4, lc], name='after_concat1')
    x = Dropout(0.5)(x)
    # after dense
    x = Dense(128, activation='relu', name='after_dense1')(x)
    # output
    out = Dense(column_num, name='count')(x)
    return Model([input_img, input_lc], out)


def model_v5(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    # begin
    x = basemodel.output
    # branch1x7
    x1 = conv2d_bn(x, 192, 1, 1, name='x1_conv_1')
    x1 = conv2d_bn(x1, 192, 1, 7, name='x1_conv_2')
    # branch7x1
    x2 = conv2d_bn(x, 192, 1, 1, name='x2_conv_1')
    x2 = conv2d_bn(x2, 192, 7, 1, name='x2_conv_2')
    # branch_combine
    x3 = concatenate([x1, x2], name='x1_x2_combine')
    x3 = conv2d_bn(x3, 192, 3, 3, name='x3_conv_1')
    # branch pool
    x4 = conv2d_bn(x3, 192, 3, 3, strides=(2, 2), name='x4_conv_1')
    # x1 flatten
    x1 = conv2d_bn(x1, 10, 3, 3, padding='valid', name='x1_conv_3')
    x1 = Flatten(name='x1_flatten')(x1)
    # x2 flatten
    x2 = conv2d_bn(x2, 10, 3, 3, padding='valid', name='x2_conv_3')
    x2 = Flatten(name='x2_flatten')(x2)
    # x3 flatten
    x3 = conv2d_bn(x3, 10, 3, 3, padding='valid', name='x3_conv_2')
    x3 = Flatten(name='x3_flatten')(x3)
    # x4 flatten
    x4 = conv2d_bn(x4, 10, 3, 3, padding='valid', name='x4_conv_2')
    x4 = Flatten(name='x4_flatten')(x4)
    x = concatenate([x1, x2, x3, x4], name='after_concat1')
    x = Dropout(0.5)(x)
    # output
    out = Dense(column_num, name='count')(x)
    return Model([input_img, input_lc], out)


def model_v4b2(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    # begin
    x1 = conv2d_bn(basemodel.output, 10, 3, 3, name='x1_conv_1')
    x1 = Flatten(name='x1_flatten')(x1)
    x2 = conv2d_bn(basemodel.output, 128, 3, 3, name='x2_conv_1')
    x2 = conv2d_bn(x2, 10, 3, 3, name='x2_conv_2')
    x2 = Flatten(name='x2_flatten')(x2)
    # deal with lc
    my_expand = Lambda(lambda y: K.expand_dims(y, axis=-1))
    lc = my_expand(input_lc)
    lc = Conv1D(16, 2, strides=2, activation='relu', name='lc_conv_1')(lc)
    lc_1 = Conv1D(8, 2, padding='same', activation='relu', name='lc_conv_2')(lc)
    lc_2 = Conv1D(8, 3, padding='same', activation='relu', name='lc_conv_3')(lc)
    lc_3 = Conv1D(8, 4, padding='same', activation='relu', name='lc_conv_4')(lc)
    lc = concatenate([lc_1, lc_2, lc_3], name='lc_concat')
    lc = Conv1D(24, 1, activation='relu', name='lc_conv_5')(lc)
    lc = Flatten(name='lc_flatten')(lc)
    x = concatenate([x1, x2, lc], name='after_concat1')
    x_2 = Dense(128, activation='relu', name='after_dense1')(x)
    x = concatenate([x, x_2], name='after_concat2')
    # output
    out = Dense(column_num, name='count')(x)
    return Model([input_img, input_lc], out)


def model_v4b(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    # begin
    x1 = conv2d_bn(basemodel.output, 10, 3, 3, name='x1_conv_1')
    x1 = Flatten(name='x1_flatten')(x1)
    x2 = conv2d_bn(basemodel.output, 128, 3, 3, name='x2_conv_1')
    x2 = conv2d_bn(x2, 10, 3, 3, name='x2_conv_2')
    x2 = Flatten(name='x2_flatten')(x2)
    # deal with lc
    my_expand = Lambda(lambda y: K.expand_dims(y, axis=-1))
    lc = my_expand(input_lc)
    lc = Conv1D(16, 2, strides=2, activation='relu', name='lc_conv_1')(lc)
    lc_1 = Conv1D(8, 2, padding='same', activation='relu', name='lc_conv_2')(lc)
    lc_2 = Conv1D(8, 3, padding='same', activation='relu', name='lc_conv_3')(lc)
    lc_3 = Conv1D(8, 4, padding='same', activation='relu', name='lc_conv_4')(lc)
    lc = concatenate([lc_1, lc_2, lc_3], name='lc_concat')
    lc = Conv1D(24, 1, activation='relu', name='lc_conv_5')(lc)
    lc = Flatten(name='lc_flatten')(lc)
    x = concatenate([x1, x2, lc], name='after_concat1')
    x = Dense(128, activation='relu', name='after_dense1')(x)
    # output
    out = Dense(column_num, name='count')(x)
    return Model([input_img, input_lc], out)


def model_v3b(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    # begin
    x = conv2d_bn(basemodel.output, 10, 3, 3, name='x_conv_1')
    x = Flatten(name='x_flatten')(x)
    # output
    out = Dense(column_num, name='count')(x)
    return Model([input_img, input_lc], out)


def model_v2(basemodel_name, model_image_size=(378, 504), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        # basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
        print('basemodel_name not defined！')
        raise NameError
    elif basemodel_name == 'xception':
        # basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
        print('basemodel_name not defined！')
        raise NameError
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
        # get serveral output layers
        # mixed 2: 35 x 35 x 256
        p1 = basemodel.get_layer(name='mixed2').output
        # mixed 7: 17 x 17 x 768
        p2 = basemodel.get_layer(name='mixed7').output
        # mixed 10: 8 x 8 x 2048
        p3 = basemodel.get_layer(name='mixed10').output
    elif basemodel_name == 'inception_resnet_v2':
        # basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
        print('basemodel_name not defined！')
        raise NameError
    else:
        print('basemodel_name not defined！')
        raise NameError
    # begin
    p1_pool = MaxPooling2D((3, 3), strides=(2, 2))(p1)
    # b1 17 x 17 use pool of 35 and 17
    b1 = concatenate([p1_pool, p2])
    b1 = conv2d_bn(b1, 128, 1, 1, name='b1_conv_1')
    b1 = conv2d_bn(b1, 10, 1, 1, name='b1_conv_2')
    b1 = Flatten(name='b1_flatten')(b1)
    b1 = Dense(128, activation='relu', name='b1_dense_1')(b1)
    # b2 8 x 8
    b2 = conv2d_bn(p3, 128, 1, 1, name='b2_conv_1')
    b2 = conv2d_bn(b2, 10, 1, 1, name='b2_conv_2')
    b2 = Flatten(name='b2_flatten')(b2)
    b2 = Dense(128, activation='relu', name='b2_dense_1')(b2)
    # merge b1 and b2 and input_lc
    x = concatenate([b1, b2, input_lc])
    x = Dense(128, activation='relu', name='x_dense_1')(x)
    # output
    out = Dense(column_num, name='count')(x)
    return Model([input_img, input_lc], out)


def model_v1(basemodel_name, model_image_size=(378, 504), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 720*540?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        # basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
        print('basemodel_name not defined！')
        raise NameError
    elif basemodel_name == 'xception':
        # basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
        print('basemodel_name not defined！')
        raise NameError
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
        # get serveral output layers
        # mixed 2: 35 x 35 x 256
        p1 = basemodel.get_layer(name='mixed2').output
        # mixed 6: 17 x 17 x 768
        p2 = basemodel.get_layer(name='mixed6').output
        # mixed 10: 8 x 8 x 2048
        p3 = basemodel.get_layer(name='mixed10').output
    elif basemodel_name == 'inception_resnet_v2':
        # basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
        print('basemodel_name not defined！')
        raise NameError
    else:
        print('basemodel_name not defined！')
        raise NameError
    # begin
    p1_pool = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(p1)
    # b1 17 x 17 use pool of 35 and 17
    b1 = concatenate([p1_pool, p2])
    b1 = conv2d_bn(b1, 128, 3, 3, name='b1_conv_1')
    b1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(b1)
    # b2 8 x 8
    b2 = concatenate([b1, p3])
    b2 = conv2d_bn(b2, 128, 2, 2, name='b2_conv_1')
    b2 = conv2d_bn(b2, 10, 4, 4, name='b2_conv_2')
    b2 = Flatten(name='b2_flatten')(b2)
    # merge b2 and input_lc
    lc = Dense(128, activation='relu', name='lc_dense_1')(input_lc)
    x = concatenate([b2, lc])
    x = Dense(128, activation='relu', name='x_dense_2')(x)
    # output
    out = Dense(column_num, name='count')(x)
    return Model([input_img, input_lc], out)


def model_s0(basemodel_name, model_image_size=(378, 504), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 1280*960?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    # begin
    x = GlobalAveragePooling2D()(basemodel.output)
    # output
    out = Dense(column_num, name='count')(x)
    return Model([input_img, input_lc], out)


def model_s0b(basemodel_name, model_image_size=(444, 592), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 1280*960?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    # begin
    x = GlobalAveragePooling2D()(basemodel.output)
    # output
    out = Dense(column_num, name='count')(x)
    return Model([input_img, input_lc], out)
