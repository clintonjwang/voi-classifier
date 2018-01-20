from keras.models import Model
import keras.models
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
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