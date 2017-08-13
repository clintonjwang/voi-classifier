# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
'''

from __future__ import print_function
from __future__ import absolute_import

from functools import partial
from keras.layers import Input, Add, Dense, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, TimeDistributed
from keras import backend as K
from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization

def get_weight_path():
    if K.image_dim_ordering() == 'th': 
        return 'resnet50_weights_th_dim_ordering_th_kernels_notop.h5'
    else:
        return 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

def get_img_output_length(width, height):
    def get_output_length(input_length):
        # zero_pad
        input_length += 6
        # apply 4 strided convolutions
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length

    return get_output_length(width), get_output_length(height) 

def add_block(type, input_tensor, filters, block_name, strides=(2, 2), trainable=True, input_shape=None):
    conv_name_base = 'res' + block_name + '_'
    bn_name_base = 'bn' + block_name + '_'

    n_layers = len(filters)
    if n_layers == 3:
        kernels = [1,3,1]
    elif n_layers == 2:
        kernels = [3,3]
    else:
        raise ValueError("Unhandled layer size %d in add_block!" % n_layers)

    bn_layer = partial(FixedBatchNormalization,
        axis = 3 if K.image_dim_ordering() == 'tf' else 1)

    if 'td' in type:
        k_init = 'glorot_uniform'
    else:
        k_init = 'normal'

    for layer in range(n_layers):
        conv_name = conv_name_base + str(layer)
        bn_name = bn_name_base + str(layer)

        conv_layer = partial(Conv2D, filters[layer],
            (kernels[layer], kernels[layer]),
            kernel_initializer = k_init,
            padding='same', trainable=trainable)

        if type == "identity":
            x = conv_layer(name = conv_name)(x if layer != 0 else input_tensor)
            x = bn_layer(name = bn_name)(x)
        
        elif type == "id-td":
            x = TimeDistributed(conv_layer(),
                name = conv_name)(x if layer != 0 else input_tensor)
            x = TimeDistributed(bn_layer(), name = bn_name)(x)
        
        elif type == "conv":
            x = conv_layer(strides = strides if layer == 0 else (1, 1),
                name = conv_name)(x if layer != 0 else input_tensor)
            x = bn_layer(name = bn_name)(x)
        
        elif type == "conv-td":
            if layer == 0 and input_shape is not None:
                x = TimeDistributed(conv_layer(
                    strides = strides if layer == 0 else (1, 1),
                    input_shape = input_shape),
                    name = conv_name)(input_tensor)
            else:
                x = TimeDistributed(conv_layer(
                    strides = strides if layer == 0 else (1, 1)),
                    name = conv_name)(x if layer != 0 else input_tensor)
            
            x = TimeDistributed(bn_layer(), name = bn_name)(x)

        else:
            raise ValueError("Unrecognized type %s in add_block" % type)

        if layer != n_layers - 1:
            x = Activation('relu')(x)

    if "id" in type:
        x = Add()([x, input_tensor])
        x = Activation('relu')(x)

    elif type == "conv":
        shortcut = Conv2D(filters[-1], (1, 1), strides=strides,
                    name=conv_name_base + 'shortcut', trainable=trainable)(input_tensor)
        shortcut = bn_layer(name=bn_name_base + 'shortcut')(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)

    elif type == "conv-td":
        shortcut = TimeDistributed(Conv2D(filters[-1], (1, 1), strides=strides,
                    trainable=trainable, kernel_initializer='normal'), 
                    name=conv_name_base + 'shortcut')(input_tensor)
        shortcut = TimeDistributed(bn_layer(),
                    name=bn_name_base + 'shortcut')(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)

    else:
        raise ValueError("Bad type %s in add_block()" % type)

    return x

def nn_base(input_tensor=None, trainable=False, total_layers=50):
    """Shared layers of the RPN and classifier. Uses ResNet50 architecture."""
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    elif not K.is_keras_tensor(input_tensor):
        img_input = Input(tensor=input_tensor, shape=input_shape)
    else:
        img_input = input_tensor

    x = ZeroPadding2D((3, 3))(img_input)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=3 if K.image_dim_ordering() == 'tf' else 1,
        name='bn1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    def add_blocks(x, block_types, nb_filters, block_num, trainable):
        for i, block_type in enumerate(block_types):
            x = add_block(block_type, x, nb_filters, block_name=str(block_num)+chr(ord('a')+i), trainable=trainable)
        return x

    if total_layers == 50:
        x = add_blocks(x, ('conv ' + 'identity '*2).split(),
            nb_filters=[64, 64, 256], block_num=2, trainable=trainable)
        x = add_blocks(x, ('conv ' + 'identity '*3).split(),
            nb_filters=[128, 128, 512], block_num=3, trainable=trainable)
        x = add_blocks(x, ('conv ' + 'identity '*5).split(),
            nb_filters=[256, 256, 1024], block_num=4, trainable=trainable)

    elif total_layers == 18:
        pass
        # TODO, [64, 64], [128, 128], [256, 256]
    else:
        raise ValueError("Number of layers %d not supported in nn_base()" % total_layers)

    return x

def rpn(base_layers, num_anchors):
    """Region proposal network"""

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]

def classifier(base_layers, input_rois, num_rois, nb_classes, trainable=False):
    """ROI classifier. Takes rois as input."""

    def classifier_layers(x, input_shape, trainable=False):
        # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
        # (hence a smaller stride in the region that follows the ROI pool)
        nb_filters = [512, 512, 2048]

        if K.backend() == 'tensorflow':
            x = add_block('conv-td', x, nb_filters, block_name='5a', input_shape=input_shape, strides=(2, 2), trainable=trainable)
        elif K.backend() == 'theano':
            x = add_block('conv-td', x, nb_filters, block_name='5a', input_shape=input_shape, strides=(1, 1), trainable=trainable)

        x = add_block('id-td', x, nb_filters, block_name='5b', trainable=trainable)
        x = add_block('id-td', x, nb_filters, block_name='5c', trainable=trainable)
        x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

        return x

    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax',
        kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear',
        kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]

