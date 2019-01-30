#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = 'model_count'
__author__ = 'fangwudi'
__time__ = '18-12-3 19:46'

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
from keras.applications.inception_v3 import InceptionV3


K.set_image_dim_ordering('tf')
from .my_inception_v3_sn import InceptionV3_sn
from .switchnorm import SwitchNormalization
from .DepthwiseConv2D import DepthwiseConv2D

def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis,
                                      scale=False,
                                      name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x



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


def my_conv2d(x, filters, kernel_size, strides=(1, 1), dilation_rate=(1, 1), padding='same', name=None):
    bn_name, ac_name, channel_axis = name_and_axis(name)
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, dilation_rate = dilation_rate,
               use_bias=False, name=name)(x)
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

def my_count_block_1(x, dilation_rate, strides=(1, 1), name=None):
    x = my_depthwise(x, (3, 3), strides=strides, padding='valid', dilation_rate=dilation_rate, name=name+'_depth')
    x = my_conv2d(x, 128, (1, 1), name=name+'_conv1')
    x = my_conv2d(x, 10, (1, 1), name=name+'_conv2')
    x = Flatten(name=name+'_flatten', data_format='channels_last')(x)
    x = Dense(128, activation='relu', name=name+'_dense')(x)
    x = Dense(1, name=name+'_block_count')(x)
    return x


def model_v1():
    image_size = (512, 512)
    # build basemodel
    input_img = Input((*image_size, 3))
    basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    middle = basemodel.get_layer(name='mixed7').output
    high = basemodel.get_layer(name='mixed10').output
    m1 = my_count_block_1(middle, dilation_rate=(2, 2), name='m1')
    m2 = my_count_block_1(middle, dilation_rate=(3, 3), name='m2')
    m3 = my_count_block_1(middle, dilation_rate=(4, 4), name='m3')
    h1 = my_count_block_1(high, dilation_rate=(2, 2), name='h1')
    h2 = my_count_block_1(high, dilation_rate=(3, 3), name='h2')
    # add
    out = Add(name='add_final')([m1, m2, m3, h1, h2])
    return Model(input_img, out)


def model_v2(image_size):
    #image_size = (512, 512)
    # img input
    input_img = Input((*image_size, 3))
    basemodel = InceptionV3_sn(input_tensor=input_img, weights='imagenet', include_top=False)
    m = basemodel.get_layer('mixed8').output

    b1 = my_conv2d(m, 256, (1, 1), name='before_conv_r1')
    R1 = my_conv2d(b1, 256, (3, 3), dilation_rate=(1, 1), name='conv_r1')

    b2 = my_conv2d(m, 256, (1, 1), name='before_conv_r2')
    R2 = my_conv2d(b2, 256, (3, 3), dilation_rate=(2, 2), name='conv_r2')

    b3 = my_conv2d(m, 256, (1, 1), name='before_conv_r3')
    R3 = my_conv2d(b3, 256, (3, 3), dilation_rate=(3, 3), name='conv_r3')

    b4 = my_conv2d(m, 256, (1, 1), name='before_conv_r4')
    R4 = my_conv2d(b4, 256, (3, 3), dilation_rate=(4, 4), name='conv_r4')

    x = concatenate([R1, R2, R3, R4], axis=-1, name='Rn_concat')
    x = my_conv2d(x, 1080, (1, 1), name='before_pooling')
    x = GlobalAveragePooling2D(name='pooling')(x)
    x = Dense(1, name='output')(x)
    return Model(input_img, x)


def model_v3(image_size):
    #image_size = (512, 512)
    # img input
    input_img = Input((*image_size, 3))
    basemodel = InceptionV3_sn(input_tensor=input_img, weights='imagenet', include_top=False)
    m = basemodel.get_layer('mixed8').output

    b1 = my_conv2d(m, 256, (1, 1), name='before_conv_r1')
    R1 = my_conv2d(b1, 256, (3, 3), dilation_rate=(1, 1), name='conv_r1')

    b2 = my_conv2d(m, 256, (1, 1), name='before_conv_r2')
    R2 = my_conv2d(b2, 256, (3, 3), dilation_rate=(2, 2), name='conv_r2')

    b3 = my_conv2d(m, 256, (1, 1), name='before_conv_r3')
    R3 = my_conv2d(b3, 256, (3, 3), dilation_rate=(3, 3), name='conv_r3')

    b4 = my_conv2d(m, 256, (1, 1), name='before_conv_r4')
    R4 = my_conv2d(b4, 256, (3, 3), dilation_rate=(4, 4), name='conv_r4')

    x = concatenate([R1, R2, R3, R4], axis=-1, name='Rn_concat')
    
    for i in range(4):
        x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depth_multiplier=1, use_bias=False,
                            dilation_rate=(2, 1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='valid', depth_multiplier=1, use_bias=False,
                            dilation_rate=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #import pdb;pdb.set_trace()
    x = GlobalAveragePooling2D(name='pooling')(x)
    
    #x=Dropout(0.2)(x)
    x = Dense(1, name='output')(x)
    return Model(input_img, x)

def model_v4(image_size):
    #image_size = (512, 512)
    # img input
    input_img = Input((*image_size, 3))
    basemodel = InceptionV3_sn(input_tensor=input_img, weights='imagenet', include_top=False)
    m = basemodel.get_layer('mixed8').output

    b1 = my_conv2d(m, 256, (1, 1), name='before_conv_r1')
    R1 = my_conv2d(b1, 256, (3, 3), dilation_rate=(1, 1), name='conv_r1')

    b2 = my_conv2d(m, 256, (1, 1), name='before_conv_r2')
    R2 = my_conv2d(b2, 256, (3, 3), dilation_rate=(2, 2), name='conv_r2')

    b3 = my_conv2d(m, 256, (1, 1), name='before_conv_r3')
    R3 = my_conv2d(b3, 256, (3, 3), dilation_rate=(3, 3), name='conv_r3')

    b4 = my_conv2d(m, 256, (1, 1), name='before_conv_r4')
    R4 = my_conv2d(b4, 256, (3, 3), dilation_rate=(4, 4), name='conv_r4')

    x = concatenate([R1, R2, R3, R4], axis=-1, name='Rn_concat')
    
    for i in range(4):
        x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depth_multiplier=1, use_bias=False,
                            dilation_rate=(2, 1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='valid', depth_multiplier=1, use_bias=False,
                            dilation_rate=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #import pdb;pdb.set_trace()
    x = GlobalMaxPooling2D(name='pooling')(x)
    
    #x=Dropout(0.2)(x)
    x = Dense(1, name='output')(x)
    return Model(input_img, x)

def model_v5(image_size):
    #image_size = (512, 512)
    # img input
    input_img = Input((*image_size, 3))
    basemodel = InceptionV3_sn(input_tensor=input_img, weights='imagenet', include_top=False)
    m = basemodel.get_layer('mixed8').output

    b1 = my_conv2d(m, 256, (1, 1), name='before_conv_r1')
    R1 = my_conv2d(b1, 256, (3, 3), dilation_rate=(1, 1), name='conv_r1')

    b2 = my_conv2d(m, 256, (1, 1), name='before_conv_r2')
    R2 = my_conv2d(b2, 256, (3, 3), dilation_rate=(2, 2), name='conv_r2')

    b3 = my_conv2d(m, 256, (1, 1), name='before_conv_r3')
    R3 = my_conv2d(b3, 256, (3, 3), dilation_rate=(3, 3), name='conv_r3')

    b4 = my_conv2d(m, 256, (1, 1), name='before_conv_r4')
    R4 = my_conv2d(b4, 256, (3, 3), dilation_rate=(4, 4), name='conv_r4')
    x = concatenate([R1, R2, R3, R4], axis=-1, name='Rn_concat')
    
    for i in range(3):
        x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depth_multiplier=1, use_bias=False,
                            dilation_rate=(2, 1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depth_multiplier=1, use_bias=False,
                            dilation_rate=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    
    x=MaxPooling2D(pool_size=(2, 2))(x) 
    #x = GlobalMaxPooling2D(name='pooling')(x)
    x = Flatten(name='after_flatten')(x) 
    x = Dropout(0.1)(x)
    x = Dense(1, name='output')(x)
    return Model(input_img, x)

def model_v6(image_size):
    #image_size = (512, 512)
    # img input
    input_img = Input((*image_size, 3))
    basemodel = InceptionV3_sn(input_tensor=input_img, weights='imagenet', include_top=False)
    m = basemodel.get_layer('mixed8').output

    b1 = my_conv2d(m, 256, (1, 1), name='before_conv_r1')
    R1 = my_conv2d(b1, 256, (3, 3), dilation_rate=(1, 1), name='conv_r1')

    b2 = my_conv2d(m, 256, (1, 1), name='before_conv_r2')
    R2 = my_conv2d(b2, 256, (3, 3), dilation_rate=(2, 2), name='conv_r2')

    b3 = my_conv2d(m, 256, (1, 1), name='before_conv_r3')
    R3 = my_conv2d(b3, 256, (3, 3), dilation_rate=(3, 3), name='conv_r3')

    b4 = my_conv2d(m, 256, (1, 1), name='before_conv_r4')
    R4 = my_conv2d(b4, 256, (3, 3), dilation_rate=(4, 4), name='conv_r4')
    x = concatenate([R1, R2, R3, R4], axis=-1, name='Rn_concat')
    
    for i in range(3):
        x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depth_multiplier=1, use_bias=False,
                            dilation_rate=(2, 1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depth_multiplier=1, use_bias=False,
                            dilation_rate=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='valid', depth_multiplier=1, use_bias=False,
                            dilation_rate=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #import pdb;pdb.set_trace()

    x = Flatten(name='after_flatten')(x) 
    x = Dropout(0.1)(x)
    x = Dense(1, name='output')(x)
    return Model(input_img, x)
def model_v7(image_size):
    #image_size = (512, 512)
    # img input
    input_img = Input((*image_size, 3))
    basemodel = InceptionV3_sn(input_tensor=input_img, weights='imagenet', include_top=False)
    m = basemodel.get_layer('mixed8').output

    b1 = my_conv2d(m, 256, (1, 1), name='before_conv_r1')
    R1 = my_conv2d(b1, 256, (3, 3), dilation_rate=(1, 1), name='conv_r1')

    b2 = my_conv2d(m, 256, (1, 1), name='before_conv_r2')
    R2 = my_conv2d(b2, 256, (3, 3), dilation_rate=(2, 2), name='conv_r2')

    b3 = my_conv2d(m, 256, (1, 1), name='before_conv_r3')
    R3 = my_conv2d(b3, 256, (3, 3), dilation_rate=(3, 3), name='conv_r3')

    b4 = my_conv2d(m, 256, (1, 1), name='before_conv_r4')
    R4 = my_conv2d(b4, 256, (3, 3), dilation_rate=(4, 4), name='conv_r4')
    x = concatenate([R1, R2, R3, R4], axis=-1, name='Rn_concat')
    
    for i in range(3):
        x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depth_multiplier=1, use_bias=False,
                            dilation_rate=(2, 1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depth_multiplier=1, use_bias=False,
                            dilation_rate=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='valid', depth_multiplier=1, use_bias=False,
                            dilation_rate=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #import pdb;pdb.set_trace()

    x = Flatten(name='after_flatten')(x) 
    #x = Dropout(0.1)(x)
    x = Dense(1, name='output')(x)
    return Model(input_img, x)



def model_v8(image_size):
    # image_size = (512, 512)
    # img input
    input_img = Input((*image_size, 3))
    basemodel = InceptionV3_sn(input_tensor=input_img, weights='imagenet', include_top=False)
    x = basemodel.get_layer('mixed8').output
    for i in range(3):
        branch_0 = conv2d_bn(x, 192,1)
        branch_1 = conv2d_bn(x, 192,1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        mixed = concatenate([branch_0, branch_1],axis=-1)
        x = conv2d_bn(mixed, K.int_shape(x)[-1],1)

     
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid')
    branches = [branch_0, branch_1]
    x = concatenate(branches,axis=-1, name='mixed')
    x = conv2d_bn(x, 768, 1, name='last_conv')
    #import pdb;pdb.set_trace()
        
    x = GlobalMaxPooling2D(name='pooling')(x)

    #x = Flatten(name='after_flatten')(x)
    x = Dense(1, name='output')(x)
    return Model(input_img, x)


def model_v9(image_size):
    # image_size = (512, 512)
    # img input
    input_img = Input((*image_size, 3))
    basemodel = InceptionV3_sn(input_tensor=input_img, weights='imagenet', include_top=False)
    x = basemodel.get_layer('mixed8').output
    for i in range(3):
        branch_0 = conv2d_bn(x, 192,1)
        branch_1 = conv2d_bn(x, 192,1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        mixed = concatenate([branch_0, branch_1],axis=-1)
        x = conv2d_bn(mixed, K.int_shape(x)[-1],1)


    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='valid', depth_multiplier=1, use_bias=False,
                            dilation_rate=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = conv2d_bn(x, 768, 1, name='last_conv')    #可能是非常有问题的点
    #import pdb;pdb.set_trace()      
    x = GlobalMaxPooling2D(name='pooling')(x)

    #x = Flatten(name='after_flatten')(x)
    x = Dense(1, name='output')(x)
    return Model(input_img, x)

def model_v10(image_size):
    # image_size = (512, 512)
    # img input
    input_img = Input((*image_size, 3))
    basemodel = InceptionV3_sn(input_tensor=input_img, weights='imagenet', include_top=False)
    x = basemodel.get_layer('mixed8').output
    import pdb;pdb.set_trace()

    branch_0 = my_conv2d(x, 192, [1, 1])
    branch_0 = my_conv2d(branch_0, 192, [3, 3], strides=2,
                                       padding='valid')
    branch_1 = my_conv2d(x, 256, [1, 1])
    branch_1 = my_conv2d(branch_1, 256, [1, 5])
    branch_1 = my_conv2d(branch_1, 320, [5, 1])
    branch_1 = my_conv2d(branch_1, 320, [3, 3], strides=2,
                                       padding='valid')
    branch_2 = my_conv2d(x, 256, [1, 1])
    branch_2 = MaxPooling2D(3, strides=2, padding='valid')(branch_2)
    x=concatenate([branch_0, branch_1, branch_2],axis=-1)
    
    for i in range(3):
        branch_0 = conv2d_bn(x, 192,1)
        branch_1 = conv2d_bn(x, 192,1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        mixed = concatenate([branch_0, branch_1],axis=-1)
        x = conv2d_bn(mixed, K.int_shape(x)[-1],1)

    x = GlobalAveragePooling2D(name='pooling')(x)
    #import pdb;pdb.set_trace()
    x = Dense(1, name='output')(x)
    return Model(input_img, x)


def model_v11(image_size):
    # image_size = (512, 512)
    # img input
    input_img = Input((*image_size, 3))
    basemodel = InceptionV3_sn(input_tensor=input_img, weights='imagenet', include_top=False)
    x = basemodel.get_layer('mixed8').output

    for i in range(3):
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 256, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        mixed = concatenate([branch_0, branch_1], axis=-1)
        x = conv2d_bn(mixed, K.int_shape(x)[-1], 1)



    branch_0 = my_conv2d(x, 192, [1, 1])
    branch_0 = my_conv2d(branch_0, 192, [3, 3], strides=2,
                         padding='valid')
    branch_1 = my_conv2d(x, 256, [1, 1])
    branch_1 = my_conv2d(branch_1, 256, [1, 5])
    branch_1 = my_conv2d(branch_1, 320, [5, 1])
    branch_1 = my_conv2d(branch_1, 320, [3, 3], strides=2,
                         padding='valid')
    branch_2 = my_conv2d(x, 256, [1, 1])
    branch_2 = MaxPooling2D(3, strides=2, padding='valid')(branch_2)
    x = concatenate([branch_0, branch_1, branch_2], axis=-1)

    x = GlobalAveragePooling2D(name='pooling')(x)
    # import pdb;pdb.set_trace()
    # x = Flatten(name='after_flatten')(x)
    x = Dense(1, name='output')(x)
    return Model(input_img, x)

def model_v12(image_size):
    # img input
    input_img = Input((*image_size, 3))
    basemodel = InceptionV3_sn(input_tensor=input_img, weights='imagenet', include_top=False)
    x = basemodel.get_layer('mixed9').output
    
    branch_0 = my_conv2d(x, 192, [1, 1])
    branch_0 = my_conv2d(branch_0, 192, [3, 3], strides=2, padding='valid')
    branch_1 = my_conv2d(x, 256, [1, 1])
    branch_1 = my_conv2d(branch_1, 256, [1, 5])
    branch_1 = my_conv2d(branch_1, 320, [5, 1])
    branch_1 = my_conv2d(branch_1, 320, [3, 3], strides=2, padding='valid')
    branch_2 = my_conv2d(x, 256, [1, 1])
    branch_2 = MaxPooling2D(3, strides=2, padding='valid')(branch_2)
    x=concatenate([branch_0, branch_1, branch_2],axis=-1)
    
    for i in range(2):
        x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depth_multiplier=1, use_bias=False,
                            dilation_rate=(1, 1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    
    x = GlobalAveragePooling2D(name='pooling')(x)
    x = Dense(1, name='output')(x)
    return Model(input_img, x)