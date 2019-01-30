from keras.models import Model
from keras.layers.core import Dense, Lambda
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D
from keras.layers import Input
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
from .grouped_conv import grouped_convolution_block
import numpy as np

def interleaved_group_conv(input, grouped_channels, cardinality, strides, weight_decay=5e-4):
    """

    :param input:
    :param grouped_channels:
    :param cardinality:
    :param strides:
    :param weight_decay:
    :return:
    """
    x = grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay)

    ### get permutation
    permutation = []
    permutation_tensor = []
    for i in range(cardinality):
        for j in range(grouped_channels):
            permutation.append(int(j*cardinality + i))
    #permutation = np.array(permutation).astype(np.int32)
    for i in range(grouped_channels * cardinality):
        permutation_tensor.append(Lambda(lambda  z:K.expand_dims(z[:,:,:,permutation[i]], axis=-1))(x))

    permutation_merge = concatenate(permutation_tensor, axis=-1)
    x = grouped_convolution_block(permutation_merge, cardinality, grouped_channels, strides, weight_decay)

    return x
