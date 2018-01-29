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
import math


def obtain_params(A):
	"""Returns mu and var for the normal distribution p(A|f) based on the annotated set
	- A (100*10) is the list of annotated image activations for a given feature
	- F () is the list of feature labels
	"""
	return np.linalg.lstsq(A, F)

def obtain_params(A):
	"""Returns mu and var for the normal distribution p(A|f) based on the annotated set
	- A (100*10) is the list of annotated image activations for a given feature
	"""
	return np.mean(A, axis=0), np.std(A, axis=0)

def fit_ls(A, Theta):
	"""
	- A (100) is the list of activations
	- Theta (100*15) is the matrix linking F to A
	- Returns feature labels (15)
	"""
	return np.linalg.lstsq(Theta, np.expand_dims(A, axis=1))

def neg_log_like(A, f, mu, var):
	"""Returns negative log likelihood, -log( p(A|f;mu,var) )
	- A (100) is the list of activations
	- f (15) is the list of feature labels, either 1 or 0
	- mu and var (15*100) are the params of the normal dist p(A|f)
	"""
	prob_A = np.sum([f[i] * np.prod([norm.pdf(A[a], mu[i,a], var[i,a]) for a in range(len(A))]) for i in range(len(f))]) / np.sum(f)
	return -math.log( prob_A )

def get_distribution(feature, population_activations):
	"""Returns the set of feature labels f that minimizes -log( p(A|f;mu,var) )
	"""
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