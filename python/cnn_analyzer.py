from keras.models import Model
import keras.models
import keras.layers as layers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras_contrib.layers.normalization import InstanceNormalization
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from keras import backend as K

import cnn_builder as cbuild
import config
import csv
import niftiutils.helper_fxns as hf
import importlib
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import pandas as pd
import random


def get_distribution(feature, population_activations):
    pass

"""Original code by the Keras Team at https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py"""

def visualize_layer(model, layer_name, save_path, num_f=None):
    """Visualize the model inputs that would maximally activate a layer."""

    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    input_img = model.input

    if num_f is None:
        num_f = layer_dict[layer_name].output.shape[-1]

    for filter_index in range(num_f):
        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        loss = K.mean(layer_output[:, :, :, :, filter_index])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        input_img_data = np.random.random((1, C.dims[0], C.dims[1], C.dims[2], 3)) * 2.
        # run gradient ascent for 20 steps
        step = 1.
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

        img = input_img_data[0]
        img = deprocess_image(img)
        hf.plot_section_auto(img, save_path=os.path.join(save_path, "%s_filter_%d.png" % (layer_name, filter_index)))

def build_dual_cnn(optimizer='adam', dilation_rate=(1,1,1), padding=['same', 'valid'], pool_sizes = [(2,2,2), (2,2,2)],
    dropout=[0.1,0.1], activation_type='relu', f=[64,128,128], dense_units=100, kernel_size=(3,3,2), stride=(1,1,1)):
    """Main class for setting up a CNN. Returns the compiled model."""

    C = config.Config()
    ActivationLayer = Activation
    activation_args = 'relu'

    nb_classes = len(C.classes_to_include)

    context_img = layers.Input(shape=(C.context_dims[0], C.context_dims[1], C.context_dims[2], 3))
    cx = context_img

    cx = InstanceNormalization(axis=4)(cx)
    cx = layers.Reshape((C.context_dims[0], C.context_dims[1], C.context_dims[2], 3, 1))(context_img)
    cx = layers.Permute((4,1,2,3,5))(cx)
    cx = layers.TimeDistributed(layers.Conv3D(filters=256, kernel_size=(8,8,3), padding='valid'))(cx)
    cx = layers.TimeDistributed(layers.MaxPooling3D((2,2,2)))(cx)
    cx = layers.TimeDistributed(layers.Conv3D(filters=256, kernel_size=(6,6,3), padding='valid'))(cx)
    cx = layers.TimeDistributed(layers.Flatten())(cx)

    img = layers.Input(shape=(C.dims[0], C.dims[1], C.dims[2], 3))

    x = img

    x = layers.Reshape((C.dims[0], C.dims[1], C.dims[2], 3, 1))(x)
    x = layers.Permute((4,1,2,3,5))(x)

    for layer_num in range(len(f)):
        if layer_num == 1:
            x = layers.TimeDistributed(layers.Conv3D(filters=f[layer_num], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[1]))(x) #, kernel_regularizer=l2(.01)
        #elif layer_num == 0:
        #   x = TimeDistributed(Conv3D(filters=f[layer_num], kernel_size=kernel_size, strides=stride, padding=padding[1]))(x) #, kernel_regularizer=l2(.01)
        else:
            x = layers.TimeDistributed(layers.Conv3D(filters=f[layer_num], kernel_size=kernel_size, padding=padding[1 * (layer_num > 1)]))(x) #, kernel_regularizer=l2(.01)
        x = layers.TimeDistributed(Dropout(dropout[0]))(x)
        x = layers.ActivationLayer(activation_args)(x)
        x = layers.TimeDistributed(BatchNormalization(axis=5))(x)
        if layer_num == 0:
            x = layers.TimeDistributed(layers.MaxPooling3D(pool_sizes[0]))(x)

    x = layers.TimeDistributed(layers.MaxPooling3D(pool_sizes[1]))(x)

    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.Concatenate(axis=1)([x, cx])

    #x = SimpleRNN(128, return_sequences=True)(x)
    x = layers.SimpleRNN(dense_units)(x)
    x = layers.ActivationLayer(activation_args)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout[1])(x)

    pred_class = layers.Dense(nb_classes, activation='softmax')(x)

    model = Model([img, context_img], pred_class)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

###########################
### FOR OUTPUTTING IMAGES AFTER TRAINING
###########################

def save_output(Z, y_pred, y_true, C=None, save_dir=None):
    """Saves large and small cropped images of all lesions in Z.
    Uses y_true and y_pred to separate correct and incorrect predictions.
    Requires C.classes_to_include, C.output_img_dir, C.crops_dir, C.orig_dir"""

    if C is None:
        C = config.Config()
    if save_dir is None:
        save_dir = C.output_img_dir

    cls_mapping = C.classes_to_include

    for cls in cls_mapping:
        if not os.path.exists(save_dir + "\\correct\\" + cls):
            os.makedirs(save_dir + "\\correct\\" + cls)
        if not os.path.exists(save_dir + "\\incorrect\\" + cls):
            os.makedirs(save_dir + "\\incorrect\\" + cls)

    for i in range(len(Z)):
        if y_pred[i] != y_true[i]:
            vm.save_img_with_bbox(cls=y_true[i], lesion_nums=[Z[i]],
                fn_suffix = " (bad_pred %s).png" % cls_mapping[y_pred[i]],
                save_dir=save_dir + "\\incorrect\\" + cls_mapping[y_true[i]])
        else:
            vm.save_img_with_bbox(cls=y_true[i], lesion_nums=[Z[i]],
                fn_suffix = " (good_pred %s).png" % cls_mapping[y_pred[i]],
                save_dir=save_dir + "\\correct\\" + cls_mapping[y_true[i]])

def merge_classes(y_true, y_pred, cls_mapping=None):
    """From lists y_true and y_pred with class numbers, """
    C = config.Config()

    if cls_mapping is None:
        cls_mapping = C.classes_to_include
    
    y_true_simp = np.array([C.simplify_map[cls_mapping[y]] for y in y_true])
    y_pred_simp = np.array([C.simplify_map[cls_mapping[y]] for y in y_pred])
    
    return y_true_simp, y_pred_simp, ['LR5', 'LR1', 'LRM']

#####################################
### Subroutines
#####################################

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #x = x.transpose((1, 2, 3, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    
    return x[:,:,x.shape[2]//2,:]