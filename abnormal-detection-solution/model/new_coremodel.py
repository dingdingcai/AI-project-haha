#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = 'new_coremodel'
__author__ = 'dingding'
__time__ = '18-5-21 11:11'

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
from keras.regularizers import l2
from keras.applications import *
#from keras.applications.inception_v3 import conv2d_bn
from keras.layers.merge import concatenate as concat
from keras import backend as K
from PIL import Image
import sys
from keras.utils.data_utils import get_file
from .switchnorm import SwitchNormalization
from keras.applications import imagenet_utils
#from keras.applications.imagenet_utils import _obtain_input_shape, decode_predictions
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras import layers



sys.path.append('../')

K.set_image_dim_ordering('tf')

def conv2d_sn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    # x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = SwitchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x



def conv2d_sn_dilation(x,
              filters,
              num_row,
              num_col,
              padding='same',
              dilation_rate=(1,1),
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        dilation_rate = dilation_rate,
        use_bias=False,
        name=conv_name)(x)
    # x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = SwitchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

def conv2d_group(tensor, groups=1, name=None):
    wid = int(tensor.shape[1])
    hei = int(tensor.shape[2])
    channel = int(tensor.shape[3])
    if channel % groups == 0:
        batch_per_group = int(channel / groups)
    else:
        print('input channels must be divided by groups.')
        raise('input channels must be divided by groups.')
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Reshape((wid, hei, groups, batch_per_group), name=name+'_reshape')(tensor)
    x = Permute((1, 2, 4, 3))(x)
    x1 = Lambda(lambda z: K.expand_dims(z[:, :, :, :, 0], axis=-1))(x)
    x1 = Conv3D(1, kernel_size=(3, 3, batch_per_group), strides=(1, 1, batch_per_group), padding='same', name=conv_name+'_0')(x1)
    for i in range(1, groups):
        x2 = Lambda(lambda z: K.expand_dims(z[:, :, :, :, i], axis=-1))(x)
        x2 = Conv3D(1, kernel_size=(3, 3, batch_per_group), strides=(1, 1, batch_per_group), padding='same', name=conv_name+'_' +str(i))(x2)
        x1 = concatenate([x1, x2], axis=-1)
    y = Lambda(lambda z: K.squeeze(z, axis=3))(x1)

    y = BatchNormalization(axis=3, scale=False, name=bn_name)(y)
    y = Activation('relu', name=name)(y)

    return y

def Slice(x, index):  
    return x[:, :, :, index] 

def Split(x, index):  
    return x[:, :, index*5 - 5:index*5, :] 


def model_mixed8_v7_sn_3(basemodel_name, model_image_size=(444, 592), column_num=1):

    input_img = Input((*model_image_size, 3))
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

    m8 = basemodel.get_layer(name='mixed8').output
    
    b1 = conv2d_sn(m8, 128, 1, 1, name='before_conv_r1')
    R1 = conv2d_sn_dilation(b1, 128, 3, 3, dilation_rate=(1, 1), name='conv_r1')
    
    b2 = conv2d_sn(m8, 128, 1, 1, name='before_conv_r2')
    R2 = conv2d_sn_dilation(b2, 128, 3, 3, dilation_rate=(1, 2), name='conv_r2')
    
    b3 = conv2d_sn(m8, 128, 1, 1, name='before_conv_r3')
    R3 = conv2d_sn_dilation(b3, 128, 3, 3, dilation_rate=(1, 3), name='conv_r3')
    
    b4 = conv2d_sn(m8, 128, 1, 1, name='before_conv_r4')
    R4 = conv2d_sn_dilation(b4, 128, 3, 3, dilation_rate=(1, 4), name='conv_r4')
     
    m8 = concat([R1, R2, R3, R4], axis=-1, name='Rn_concat')
        
    m81 = conv2d_sn(m8, 128, 3, 3, name='alpha_conv')
    alpha = Lambda(lambda x: K.random_uniform([K.shape(x)[0], 1, 1, 128], 0, 1))(m81) 
    alpha_e = Lambda(lambda x: K.repeat_elements(x, 12, axis=1))(alpha) 
    alpha_e = Lambda(lambda x: K.repeat_elements(x, 17, axis=2))(alpha_e) 
    m81 = Multiply(name='alpha')([alpha_e, m81])
    
    m82 = conv2d_sn(m8, 128, 3, 3, name='beta_conv')
    ones  = Lambda(lambda x: K.ones([K.shape(x)[0], 1, 1, 128]))(m82)
    beta = Subtract()([ones, alpha])
    beta_e = Lambda(lambda x: K.repeat_elements(x, 12, axis=1))(beta) 
    beta_e = Lambda(lambda x: K.repeat_elements(x, 17, axis=2))(beta_e)
    m82 = Multiply(name='beta')([beta_e, m82])
    
    x = Add()([m81, m82])
    
    x1 = conv2d_sn(x, 2, 3, 3, name='x_conv1')
    x1 = Flatten(name='x1_flatten')(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(1, activation='sigmoid', name='x1')(x1)

    x2 = conv2d_sn(x, 7, 3, 3, name='x_conv2')
    x2 = Flatten(name='x2_flatten')(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(7, activation='sigmoid', name='x2_lc')(x2)

    return Model(input_img, [x1, x2])



def model_mixed8_v7_sn_3_test(basemodel_name, model_image_size=(444, 592), column_num=1):

    input_img = Input((*model_image_size, 3))
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

    m8 = basemodel.get_layer(name='mixed8').output
    
    b1 = conv2d_sn(m8, 128, 1, 1, name='before_conv_r1')
    R1 = conv2d_sn_dilation(b1, 128, 3, 3, dilation_rate=(1, 1), name='conv_r1')
    
    b2 = conv2d_sn(m8, 128, 1, 1, name='before_conv_r2')
    R2 = conv2d_sn_dilation(b2, 128, 3, 3, dilation_rate=(1, 2), name='conv_r2')
    
    b3 = conv2d_sn(m8, 128, 1, 1, name='before_conv_r3')
    R3 = conv2d_sn_dilation(b3, 128, 3, 3, dilation_rate=(1, 3), name='conv_r3')
    
    b4 = conv2d_sn(m8, 128, 1, 1, name='before_conv_r4')
    R4 = conv2d_sn_dilation(b4, 128, 3, 3, dilation_rate=(1, 4), name='conv_r4')
     
    m8 = concat([R1, R2, R3, R4], axis=-1, name='Rn_concat')
        
    m81 = conv2d_sn(m8, 128, 3, 3, name='alpha_conv')
    alpha = Lambda(lambda x: 0.5 * K.ones(K.shape(x)))(m81) 
    m81 = Multiply(name='alpha')([alpha, m81])
    
    m82 = conv2d_sn(m8, 128, 3, 3, name='beta_conv')
    beta  = Lambda(lambda x: 0.5 * K.ones(K.shape(x)))(m82)
    m82 = Multiply(name='beta')([beta, m82])
    
    x = Add()([m81, m82])
    
    x1 = conv2d_sn(x, 2, 3, 3, name='x_conv1')
    x1 = Flatten(name='x1_flatten')(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(1, activation='sigmoid', name='x1')(x1)

    x2 = conv2d_sn(x, 7, 3, 3, name='x_conv2')
    x2 = Flatten(name='x2_flatten')(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(7, activation='sigmoid', name='x2_lc')(x2)

    return Model(input_img, [x1, x2])




