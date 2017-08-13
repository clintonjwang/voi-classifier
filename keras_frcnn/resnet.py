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

def add_block(type, input_tensor, filters, block_name, strides=(2, 2), trainable=True):
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + block_name + '_branch'
    bn_name_base = 'bn' + block_name + '_branch'

    n_layers = len(filters)
    if n_layers == 3:
        kernels = [1,3,1]
    elif n_layers == 2:
        kernels = [3,3]
    else:
        print("Unhandled layer size in add_block!")
        return None

    # TODO: convert to partials
    if type == "identity":
        for layer in range(n_layers):
            #conv_layer = partial(Conv2D, filters[layer], (kernels[layer], kernels[layer]), padding='same', trainable=trainable)
            x = Conv2D(filters[layer], (kernels[layer], kernels[layer]),
                padding='same', trainable=trainable,
                name = conv_name_base + str(layer))(x if layer == 0 else input_tensor)

            x = FixedBatchNormalization(axis=bn_axis,
                name = bn_name_base + str(layer))(x)

            if layer != n_layers - 1:
                x = Activation('relu')(x)

        x = Add()([x, input_tensor]) # residual
        x = Activation('relu')(x)

    elif type == "id-td":
        for layer in range(n_layers):
            x = TimeDistributed(Conv2D(filters[layer], (kernels[layer], kernels[layer]),
                padding='same', trainable=trainable, kernel_initializer='normal'), 
                name = conv_name_base + str(layer))(x if layer == 0 else input_tensor)

            x = TimeDistributed(FixedBatchNormalization(axis=bn_axis),
                name = bn_name_base + str(layer))(x)

            if layer != n_layers - 1:
                x = Activation('relu')(x)

        x = Add()([x, input_tensor])
        x = Activation('relu')(x)

    elif type == "conv":
        for layer in range(n_layers):
            x = Conv2D(filters[layer], (kernels[layer], kernels[layer]),
                strides = strides if layer == 0 else (1, 1)
                padding='same', trainable=trainable,
                name = conv_name_base + str(layer))(x if layer == 0 else input_tensor)
            x = FixedBatchNormalization(axis=bn_axis,
                name = bn_name_base + str(layer))(x)
            if layer != n_layers:
                x = Activation('relu')(x)

        shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                    name=conv_name_base + '1', trainable=trainable)(input_tensor)
        shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)

    elif type == "conv-td":
        for layer in range(n_layers):
            x = TimeDistributed(Conv2D(filters[layer], (kernels[layer], kernels[layer]),
                strides = strides if layer == 0 else (1, 1)
                padding='same', trainable=trainable, kernel_initializer='normal'),
                name = conv_name_base + str(layer))(x if layer == 0 else input_tensor)
            x = TimeDistributed(FixedBatchNormalization(axis=bn_axis,
                name = bn_name_base + str(layer))(x)
            if layer != n_layers:
                x = Activation('relu')(x)

        shortcut = TimeDistributed(Conv2D(nb_filter3, (1, 1), strides=strides,
                    trainable=trainable, kernel_initializer='normal'), 
                    name=conv_name_base + '1')(input_tensor)
        shortcut = TimeDistributed(FixedBatchNormalization(axis=bn_axis,
                        name=bn_name_base + '1'))(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)

    else:
        print("Unrecognized type %s" % type)

    return x

def nn_base(input_tensor=None, trainable=False, total_layers=50):
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

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    if total_layers == 50:
        nb_filters = [64, 64, 256]
        x = add_block('conv', x, nb_filters, block_name='2a', strides=(1, 1), trainable=trainable)
        x = add_block('identity', x, nb_filters, block_name='2b', trainable=trainable)
        x = add_block('identity', x, nb_filters, block_name='2c', trainable=trainable)

        nb_filters = [128, 128, 512]
        x = add_block('conv', x, nb_filters, block_name='3a', trainable=trainable)
        x = add_block('identity', x, nb_filters, block_name='3b', trainable=trainable)
        x = add_block('identity', x, nb_filters, block_name='3c', trainable=trainable)
        x = add_block('identity', x, nb_filters, block_name='3d', trainable=trainable)

        nb_filters = [256, 256, 1024]
        x = add_block('conv', x, nb_filters, block_name='4a', trainable=trainable)
        x = add_block('identity', x, nb_filters, block_name='4b', trainable=trainable)
        x = add_block('identity', x, nb_filters, block_name='4c', trainable=trainable)
        x = add_block('identity', x, nb_filters, block_name='4d', trainable=trainable)
        x = add_block('identity', x, nb_filters, block_name='4e', trainable=trainable)
        x = add_block('identity', x, nb_filters, block_name='4f', trainable=trainable)

    elif total_layers == 18:
        # TODO
        nb_filters = [64, 64]
        x = add_block('conv', x, nb_filters, block_name='2a', strides=(1, 1), trainable=trainable)
        x = add_block('identity', x, nb_filters, block_name='2b', trainable=trainable)

        nb_filters = [128, 128]
        x = add_block('conv', x, nb_filters, block_name='3a', trainable=trainable)
        x = add_block('identity', x, nb_filters, block_name='3b', trainable=trainable)

        nb_filters = [256, 256]
        x = add_block('conv', x, nb_filters, block_name='4a', trainable=trainable)
        x = add_block('identity', x, nb_filters, block_name='4b', trainable=trainable)

    return x


def classifier_layers(x, input_shape, trainable=False):
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    # (hence a smaller stride in the region that follows the ROI pool)
    if K.backend() == 'tensorflow':
        x = add_block('conv-td', x, [512, 512, 2048], block_name='5a', input_shape=input_shape, strides=(2, 2), trainable=trainable)
    elif K.backend() == 'theano':
        x = add_block('conv-td', x, [512, 512, 2048], block_name='5a', input_shape=input_shape, strides=(1, 1), trainable=trainable)

    x = add_block('id-td', x, [512, 512, 2048], block_name='5b', trainable=trainable)
    x = add_block('id-td', x, [512, 512, 2048], block_name='5c', trainable=trainable)
    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

    return x


def rpn(base_layers,num_anchors):

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]

def classifier(base_layers, input_rois, num_rois, nb_classes = 21, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]

