#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = 'coremodel'
__author__ = 'fangwudi'
__time__ = '18-4-8 10:53'

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
K.set_image_dim_ordering('tf')
# modified keras version of DepthwiseConv2D using tensorflow
from .DepthwiseConv2D import DepthwiseConv2D
from .my_inception_v3 import myInceptionV3


def my_conv2d(x, filters, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        ac_name = name + '_ac'
    else:
        bn_name = None
        ac_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
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
        bn_axis = 1
    else:
        bn_axis = 3
    x = DepthwiseConv2D(kernel_size, strides=strides, padding=padding,
                        depth_multiplier=depth_multiplier, use_bias=False,
                        dilation_rate=dilation_rate, name=name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=ac_name)(x)
    return x

def model_v10(basemodel_name, model_image_size=(384, 512)):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: original size is 540(height)*720(width)
    :return: keras model
    """
    # First, define the vision modules
    # img input
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
    # use serveral layers as output
    out_2_35 = basemodel.get_layer(name='mixed2').output   #block1_conv1
    out_5_17 = basemodel.get_layer(name='mixed4').output
    vision_model = Model(input_img, [out_2_35, out_5_17],
                         name='vision_model')
    # get output for a and b
    input_a = Input((*model_image_size, 3), name='input_a')
    input_b = Input((*model_image_size, 3), name='input_b')
    a_out_2_35, a_out_5_17 = vision_model(input_a)
    b_out_2_35, b_out_5_17 = vision_model(input_b)





    # define abs layers
    #my_abs = Lambda(lambda z: K.abs(z))
    # sub
    #p_low = concatenate([a_out_2_35, b_out_2_35])
    #sub_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(Subtract(name='sub2')([a_out_2_35, b_out_2_35]))

    sub_4 =(Subtract(name='sub4')([a_out_5_17, b_out_5_17]))

    x1 = sub_4

    x = conv2d_bn(x1, 10, 1, 1, name='after_conv_2')
    # flatten
    y = Flatten(name='after_flatten')(x)
    # add some dense layers
    y = Dense(128, activation='relu', name='after_dense_1')(y)
    out = Dense(1, activation='sigmoid')(y)
    return Model([input_a, input_b], out)



# model-v9 : use shared base model and then concat feature pyramid to predict
def model_v9(basemodel_name, model_image_size=(384, 512)):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: original size is 540(height)*720(width)
    :return: keras model
    """
    # First, define the vision modules
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
    # use serveral layers as output
    out_2_35 = basemodel.get_layer(name='mixed2').output
    out_5_17 = basemodel.get_layer(name='mixed5').output
    vision_model = Model(input_img, [out_2_35, out_5_17], name='vision_model')
    # get output for a and b
    input_a = Input((*model_image_size, 3), name='input_a')
    input_b = Input((*model_image_size, 3), name='input_b')
    a_out_2_35, a_out_5_17 = vision_model(input_a)
    b_out_2_35, b_out_5_17 = vision_model(input_b)
    # define abs layers
    my_abs = Lambda(lambda z: K.abs(z))
    # sub
    p_low = my_abs(Subtract(name='sub2')([a_out_2_35, b_out_2_35]))
    import tensorflow as tf
    tf.summary.image(name="sub_image",tensor=p_low, max_outputs=3,)
    # maximum
    max_5_17 = Maximum(name="maximum_high")([a_out_5_17, b_out_5_17])
    p_middle = my_conv2d(max_5_17, 288, (1, 1), name='p_middle_conv')
    # down multiply
    x = Multiply(name="x_multiply")([UpSampling2D(size=(2, 2), name="x_upsampled")(p_middle), p_low])
    x_2 = GlobalMaxPooling2D(name="global_max")(x)
    # max poool
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='x_s_1')(x)
    p_middle_2 = my_conv2d(max_5_17, 128, (1, 1), name='p_middle_2_conv')
    x = concatenate([x, p_middle_2])
    x = my_conv2d(x, 128, 1, 1, name='x_conv_1')
    # max poool
    x = MaxPooling2D((5, 5), strides=(3, 3), padding='same', name='x_s_2')(x)
    x = my_conv2d(x, 10, 1, 1, name='x_conv_2')
    # flatten
    y = Flatten(name='after_flatten')(x)
    y = concatenate([y, x_2])
    # add some dense layers
    y = Dense(128, activation='relu', name='after_dense_1')(y)
    out = Dense(1, activation='sigmoid')(y)
    #self.merged = tf.summary.merge_all()
    return Model([input_a, input_b], out)


# model-v8 : use shared base model and then concat feature pyramid to predict
def model_v8(basemodel_name, model_image_size=(384, 512)):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: original size is 540(height)*720(width)
    :return: keras model
    """
    # First, define the vision modules
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
    # use serveral layers as output
    out_2_35 = basemodel.get_layer(name='mixed2').output
    out_5_17 = basemodel.get_layer(name='mixed5').output
    vision_model = Model(input_img, [out_2_35, out_5_17],
                         name='vision_model')
    # get output for a and b
    input_a = Input((*model_image_size, 3), name='input_a')
    input_b = Input((*model_image_size, 3), name='input_b')
    a_out_2_35, a_out_5_17 = vision_model(input_a)
    b_out_2_35, b_out_5_17 = vision_model(input_b)
    # define abs layers
    my_abs = Lambda(lambda z: K.abs(z))
    # sub
    sub_2_35 = my_abs(Subtract(name='sub2')([a_out_2_35, b_out_2_35]))
    p_low = my_conv2d(sub_2_35, 256, (1, 1), name='p_low_conv')
    # maximum
    max_5_17 = Maximum(name="maximum_high")([a_out_5_17, b_out_5_17])
    p_middle = my_conv2d(max_5_17, 256, (1, 1), name='p_middle_conv')
    # down multiply
    x = Multiply(name="x_multiply")([UpSampling2D(size=(2, 2), name="x_upsampled")(p_middle), p_low])
    # max poool
    x = my_depthwise(x, (3, 3), strides=(2, 2), name='x_s_1')
    p_middle_2 = my_conv2d(max_5_17, 256, (1, 1), name='p_middle_2_conv')
    x = concatenate([x, p_middle_2])
    x = my_conv2d(x, 128, 1, 1, name='x_conv_1')
    # max poool
    x = my_depthwise(x, (3, 3), strides=(2, 2), name='x_s_2')
    x = my_conv2d(x, 10, 1, 1, name='x_conv_2')
    # flatten
    y = Flatten(name='after_flatten')(x)
    # add some dense layers
    y = Dense(128, activation='relu', name='after_dense_1')(y)
    out = Dense(1, activation='sigmoid')(y)
    return Model([input_a, input_b], out)


# model-v7 : use shared base model and then concat feature pyramid to predict
def model_v7(basemodel_name, model_image_size=(360, 480)):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: original size is 540(height)*720(width)
    :return: keras model
    """
    # First, define the vision modules
    # img input
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
    # use serveral layers as output
    out_2_35 = basemodel.get_layer(name='mixed2').output
    out_5_17 = basemodel.get_layer(name='mixed5').output
    vision_model = Model(input_img, [out_2_35, out_5_17],
                         name='vision_model')
    # get output for a and b
    input_a = Input((*model_image_size, 3), name='input_a')
    input_b = Input((*model_image_size, 3), name='input_b')
    a_out_2_35, a_out_5_17 = vision_model(input_a)
    b_out_2_35, b_out_5_17 = vision_model(input_b)
    # define abs layers
    my_abs = Lambda(lambda z: K.abs(z))
    # sub
    sub_2_35 = my_abs(Subtract(name='sub2')([a_out_2_35, b_out_2_35]))
    sub_5_17 = my_abs(Subtract(name='sub5')([a_out_5_17, b_out_5_17]))
    # define max layers
    my_maximum = Lambda(lambda z: K.maximum(z[0], z[1]))
    # maximum
    max_2_35 = my_maximum([a_out_2_35, b_out_2_35])
    max_5_17 = my_maximum([a_out_5_17, b_out_5_17])
    # max poool
    l_35 = concatenate([sub_2_35, max_2_35])
    l_35 = my_conv2d(l_35, 256, (1, 1), name='l_35_conv_1')
    l_35 = my_depthwise(l_35, (3, 3), strides=(2, 2), padding='valid', name='l_35_s_1')
    l_17 = concatenate([sub_5_17, max_5_17])
    l_17 = my_conv2d(l_17, 512, (1, 1), name='l_17_conv_1')
    x = concatenate([l_35, l_17])
    x = my_conv2d(x, 256, (1, 1), name='x_conv_1')
    x = my_depthwise(x, (3, 3), strides=(2, 2), name='x_s_1')
    x = my_conv2d(x, 10, 1, 1, name='x_conv_2')
    # flatten
    y = Flatten(name='after_flatten')(x)
    # add some dense layers
    y = Dense(128, activation='relu', name='after_dense_1')(y)
    out = Dense(1, activation='sigmoid')(y)
    return Model([input_a, input_b], out)


# model-v6 : use shared base model and then concat feature pyramid to predict
def model_v6(basemodel_name, model_image_size=(360, 480)):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: original size is 540(height)*720(width)
    :return: keras model
    """
    # First, define the vision modules
    # img input
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
    # use serveral layers as output
    out_0_35 = basemodel.get_layer(name='mixed0').output
    out_1_35 = basemodel.get_layer(name='mixed1').output
    out_2_35 = basemodel.get_layer(name='mixed2').output
    out_3_17 = basemodel.get_layer(name='mixed3').output
    out_4_17 = basemodel.get_layer(name='mixed4').output
    out_5_17 = basemodel.get_layer(name='mixed5').output
    vision_model = Model(input_img, [out_0_35, out_1_35, out_2_35, out_3_17, out_4_17, out_5_17],
                         name='vision_model')
    # get output for a and b
    input_a = Input((*model_image_size, 3), name='input_a')
    input_b = Input((*model_image_size, 3), name='input_b')
    a_out_0_35, a_out_1_35, a_out_2_35, a_out_3_17, a_out_4_17, a_out_5_17 = vision_model(input_a)
    b_out_0_35, b_out_1_35, b_out_2_35, b_out_3_17, b_out_4_17, b_out_5_17 = vision_model(input_b)
    # define abs layers
    my_abs = Lambda(lambda z: K.abs(z))
    # sub
    sub_0_35 = my_abs(Subtract(name='sub0')([a_out_0_35, b_out_0_35]))
    sub_1_35 = my_abs(Subtract(name='sub1')([a_out_1_35, b_out_1_35]))
    sub_2_35 = my_abs(Subtract(name='sub2')([a_out_2_35, b_out_2_35]))
    sub_3_17 = my_abs(Subtract(name='sub3')([a_out_3_17, b_out_3_17]))
    sub_4_17 = my_abs(Subtract(name='sub4')([a_out_4_17, b_out_4_17]))
    sub_5_17 = my_abs(Subtract(name='sub5')([a_out_5_17, b_out_5_17]))
    # define max layers
    my_maximum = Lambda(lambda z: K.maximum(z[0], z[1]))
    # maximum
    max_0_35 = my_maximum([a_out_0_35, b_out_0_35])
    max_1_35 = my_maximum([a_out_1_35, b_out_1_35])
    max_2_35 = my_maximum([a_out_2_35, b_out_2_35])
    max_3_17 = my_maximum([a_out_3_17, b_out_3_17])
    max_4_17 = my_maximum([a_out_4_17, b_out_4_17])
    max_5_17 = my_maximum([a_out_5_17, b_out_5_17])
    # max poool
    l_35 = concatenate([sub_0_35, sub_1_35, sub_2_35, max_0_35, max_1_35, max_2_35])
    l_35 = my_conv2d(l_35, 128, (1, 1), name='l_35_conv_1')
    l_35 = my_depthwise(l_35, (3, 3), strides=(2, 2), padding='valid', name='l_35_s_1')
    l_17 = concatenate([sub_3_17, sub_4_17, sub_5_17, max_3_17, max_4_17, max_5_17])
    l_17 = my_conv2d(l_17, 256, (1, 1), name='l_17_conv_1')
    x = concatenate([l_35, l_17])
    x = my_conv2d(x, 128, (1, 1), name='x_conv_1')
    x = my_depthwise(x, (3, 3), strides=(2, 2), name='x_s_1')
    x = my_conv2d(x, 64, 1, 1, name='x_conv_2')
    x = my_depthwise(x, (3, 3), strides=(2, 2), name='x_s_2')
    x = my_conv2d(x, 32, 1, 1, name='x_conv_3')
    # flatten
    y = Flatten(name='after_flatten')(x)
    # add some dense layers
    y = Dense(128, activation='relu', name='after_dense_1')(y)
    out = Dense(1, activation='sigmoid')(y)
    return Model([input_a, input_b], out)


# model-v5 : use shared base model and then concat feature pyramid to predict
def model_v5(basemodel_name, model_image_size=(360, 480)):
    """
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: original size is 540(height)*720(width)
    :return: keras model
    """
    # First, define the vision modules
    # img input
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
    # use serveral layers as output
    out_0_35 = basemodel.get_layer(name='mixed0').output
    out_1_35 = basemodel.get_layer(name='mixed1').output
    out_2_35 = basemodel.get_layer(name='mixed2').output
    out_3_17 = basemodel.get_layer(name='mixed3').output
    out_4_17 = basemodel.get_layer(name='mixed4').output
    out_5_17 = basemodel.get_layer(name='mixed5').output
    vision_model = Model(input_img, [out_0_35, out_1_35, out_2_35, out_3_17, out_4_17, out_5_17], name='vision_model')
    # get output for a and b
    input_a = Input((*model_image_size, 3), name='input_a')
    input_b = Input((*model_image_size, 3), name='input_b')
    a_out_0_35, a_out_1_35, a_out_2_35, a_out_3_17, a_out_4_17, a_out_5_17 = vision_model(input_a)
    b_out_0_35, b_out_1_35, b_out_2_35, b_out_3_17, b_out_4_17, b_out_5_17 = vision_model(input_b)
    # Then, define the tell-img-apart model
    # define abs layers
    my_abs = Lambda(lambda z: K.abs(z))
    # sub
    sub_0_35 = my_abs(Subtract(name='sub0')([a_out_0_35, b_out_0_35]))
    sub_1_35 = my_abs(Subtract(name='sub1')([a_out_1_35, b_out_1_35]))
    sub_2_35 = my_abs(Subtract(name='sub2')([a_out_2_35, b_out_2_35]))
    sub_3_17 = my_abs(Subtract(name='sub3')([a_out_3_17, b_out_3_17]))
    sub_4_17 = my_abs(Subtract(name='sub4')([a_out_4_17, b_out_4_17]))
    sub_5_17 = my_abs(Subtract(name='sub5')([a_out_5_17, b_out_5_17]))
    # max poool
    sub_35 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(concatenate([sub_0_35, sub_1_35, sub_2_35]))
    x_sub = concatenate([sub_35, sub_3_17, sub_4_17, sub_5_17])
    # original
    ori_35 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(concatenate([a_out_2_35, b_out_2_35]))
    x_ori = concatenate([ori_35, a_out_5_17, b_out_5_17])
    # add some conv layers
    x_sub = conv2d_bn(x_sub, 256, 1, 1, name='after_conv_sub')
    x_ori = conv2d_bn(x_ori, 256, 1, 1, name='after_conv_ori')
    x = concatenate([x_sub, x_ori])
    x_1 = conv2d_bn(x, 128, 1, 1, name='after_conv_1')
    x_2 = conv2d_bn(x, 128, 1, 1, name='after_conv_2')
    x_2 = conv2d_bn(x_2, 128, 3, 3, name='after_conv_3')
    x = concatenate([x_1, x_2])
    x = conv2d_bn(x, 10, 1, 1, name='after_conv_4')
    # flatten
    y = Flatten(name='after_flatten')(x)
    # add some dense layers
    y = Dense(128, activation='relu', name='after_dense_1')(y)
    out = Dense(1, activation='sigmoid', name='output_dense')(y)
    return Model([input_a, input_b], out)


# model-v4 : use shared base model and then concat feature pyramid to predict
def model_v4(basemodel_name, model_image_size=(360, 480)):
    """
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: original size is 540(height)*720(width)
    :return: keras model
    """
    # First, define the vision modules
    # img input
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
    # use serveral layers as output
    out_0_35 = basemodel.get_layer(name='mixed0').output
    out_1_35 = basemodel.get_layer(name='mixed1').output
    out_2_35 = basemodel.get_layer(name='mixed2').output
    out_3_17 = basemodel.get_layer(name='mixed3').output
    out_4_17 = basemodel.get_layer(name='mixed4').output
    out_5_17 = basemodel.get_layer(name='mixed5').output
    vision_model = Model(input_img, [out_0_35, out_1_35, out_2_35, out_3_17, out_4_17, out_5_17], name='vision_model')
    # get output for a and b
    input_a = Input((*model_image_size, 3), name='input_a')
    input_b = Input((*model_image_size, 3), name='input_b')
    a_out_0_35, a_out_1_35, a_out_2_35, a_out_3_17, a_out_4_17, a_out_5_17 = vision_model(input_a)
    b_out_0_35, b_out_1_35, b_out_2_35, b_out_3_17, b_out_4_17, b_out_5_17 = vision_model(input_b)
    # Then, define the tell-img-apart model
    # define abs layers
    my_abs = Lambda(lambda z: K.abs(z))
    # sub
    sub_0_35 = my_abs(Subtract(name='sub0')([a_out_0_35, b_out_0_35]))
    sub_1_35 = my_abs(Subtract(name='sub1')([a_out_1_35, b_out_1_35]))
    sub_2_35 = my_abs(Subtract(name='sub2')([a_out_2_35, b_out_2_35]))
    sub_3_17 = my_abs(Subtract(name='sub3')([a_out_3_17, b_out_3_17]))
    sub_4_17 = my_abs(Subtract(name='sub4')([a_out_4_17, b_out_4_17]))
    sub_5_17 = my_abs(Subtract(name='sub5')([a_out_5_17, b_out_5_17]))
    # max poool
    sub_35 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(concatenate([sub_0_35, sub_1_35, sub_2_35]))
    x_sub = concatenate([sub_35, sub_3_17, sub_4_17, sub_5_17])
    # original
    ori_35 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(concatenate([a_out_2_35, b_out_2_35]))
    x_ori = concatenate([ori_35, a_out_5_17, b_out_5_17])
    # add some conv layers
    x_sub = conv2d_bn(x_sub, 128, 1, 1, name='after_conv_sub')
    x_ori = conv2d_bn(x_ori, 128, 1, 1, name='after_conv_ori')
    x = concatenate([x_sub, x_ori])
    x = conv2d_bn(x, 10, 1, 1, name='after_conv_1')
    # flatten
    y = Flatten(name='after_flatten')(x)
    # add some dense layers
    y = Dense(128, activation='relu', name='after_dense_1')(y)
    out = Dense(1, activation='sigmoid', name='output_dense')(y)
    return Model([input_a, input_b], out)


# model-v3 : use shared base model and then concat feature pyramid to predict
def model_v3(basemodel_name, model_image_size=(360, 480)):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: original size is 540(height)*720(width)
    :return: keras model
    """
    # First, define the vision modules
    # img input
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
    # get serveral output layers
    out_0_35 = basemodel.get_layer(name='mixed0').output
    out_1_35 = basemodel.get_layer(name='mixed1').output
    out_2_35 = basemodel.get_layer(name='mixed2').output
    out_3_17 = basemodel.get_layer(name='mixed3').output
    out_4_17 = basemodel.get_layer(name='mixed4').output
    vision_model = Model(input_img, [out_0_35, out_1_35, out_2_35, out_3_17, out_4_17], name='vision_model')

    # Then define the tell-img-apart model
    input_a = Input((*model_image_size, 3), name='input_a')
    input_b = Input((*model_image_size, 3), name='input_b')
    # The vision model will be shared, weights and all
    a_out_0_35, a_out_1_35, a_out_2_35, a_out_3_17, a_out_4_17 = vision_model(input_a)
    b_out_0_35, b_out_1_35, b_out_2_35, b_out_3_17, b_out_4_17 = vision_model(input_b)
    sub_0_35 = Subtract(name='sub0')([a_out_0_35, b_out_0_35])
    sub_1_35 = Subtract(name='sub1')([a_out_1_35, b_out_1_35])
    sub_2_35 = Subtract(name='sub2')([a_out_2_35, b_out_2_35])
    sub_3_17 = Subtract(name='sub3')([a_out_3_17, b_out_3_17])
    sub_4_17 = Subtract(name='sub4')([a_out_4_17, b_out_4_17])
    l0 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(sub_0_35)
    l1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(sub_1_35)
    l2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(sub_2_35)
    l_35 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(concatenate([a_out_2_35, b_out_2_35]))
    l_17 = concatenate([a_out_4_17, b_out_4_17])
    x1 = concatenate([l0, l1, l2, sub_3_17, sub_4_17])
    x2 = concatenate([l_35, l_17])
    # add some conv layers
    x1 = conv2d_bn(x1, 128, 1, 1, name='after_conv_x1')
    x2 = conv2d_bn(x2, 128, 1, 1, name='after_conv_x2')
    x = concatenate([x1, x2])
    x = conv2d_bn(x, 10, 1, 1, name='after_conv_2')
    # flatten
    y = Flatten(name='after_flatten')(x)
    # add some dense layers
    y = Dense(128, activation='relu', name='after_dense_1')(y)
    out = Dense(1, activation='sigmoid')(y)
    return Model([input_a, input_b], out)
