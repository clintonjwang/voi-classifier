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
import niftiutils.transforms as tr
import importlib
from math import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import operator
import os
import pandas as pd
import random
import math
from sklearn.manifold import TSNE



###########################
### Output graphs
###########################

def tsne(filter_results):
	C = config.Config()

	X = []
	z = [0]
	for i,cls in enumerate(C.classes_to_include):
		X.append(filter_results[cls])
		z.append(len(filter_results[cls]) + z[-1])
	z.append(len(X))
	X = np.concatenate(X, axis=0)

	X_emb = TSNE(n_components=2, init='pca').fit_transform(X)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	for i, cls in enumerate(C.classes_to_include):
		ax.scatter(X_emb[z[i]:z[i+1], 0], X_emb[z[i]:z[i+1], 1], color=plt.cm.Set1(i/6.), marker='.', alpha=.8)

	ax.legend(C.short_cls_names, framealpha=0.5)
	ax.set_title("t-SNE")
	ax.xaxis.set_major_formatter(NullFormatter())
	ax.yaxis.set_major_formatter(NullFormatter())
	ax.axis('tight')

	return fig


###########################
### Analyze annotations
###########################

def collect_features():
	C = config.Config()
	feature_sheet = pd.read_excel(C.xls_name, "Descriptions")

	features_by_cls = {}
	feat_count = {}
	for cls in C.classes_to_include:
		features_by_cls[cls] = list(feature_sheet["evidence1"+cls].dropna().values)
		features_by_cls[cls] = features_by_cls[cls] + list(feature_sheet["evidence2"+cls].dropna().values)

	feat_count = dict(zip(*np.unique([f for cls in features_by_cls for f in features_by_cls[cls]], return_counts=True)))
	for cls in C.classes_to_include:
		features_by_cls[cls] = list(set(features_by_cls[cls]))

	return features_by_cls, feat_count

def get_annotated_files(features_by_cls):
	C = config.Config()
	feature_sheet = pd.read_excel(C.xls_name, "Descriptions")

	Z_features_by_cls = {cls: {} for cls in features_by_cls}
	Z_features = {}
	for cls in C.classes_to_include:
		for f in features_by_cls[cls]:
			if f not in Z_features:
				Z_features[f] = []
				
			Z_features_by_cls[cls][f] = [x for x in feature_sheet[feature_sheet["evidence1"+cls] == f][cls].values]
			Z_features[f] += [x for x in feature_sheet[feature_sheet["evidence1"+cls] == f][cls].values]
			if feature_sheet["evidence2"+cls].dropna().size > 0:
				Z_features_by_cls[cls][f] = Z_features_by_cls[cls][f] + [x+".npy" for x in feature_sheet[feature_sheet["evidence2"+cls] == f][cls].values]
				Z_features[f] += [x for x in feature_sheet[feature_sheet["evidence2"+cls] == f][cls].values]

	return Z_features

def get_evidence_strength(feature_filters, pred_filters):
	"""A good pred_filter has high values for all the key (non-zero) features of feature_filter.
	These values should be unscaled.
	Returns average percentage of the mean value of the key filters (capped at 100%)"""
	
	strength = 0
	num_key_filters = sum(feature_filters > 0)
	
	for i in range(len(pred_filters)):
		t = feature_filters[i]
		p = pred_filters[i]
		
		if t == 0:
			continue
			
		strength += min(p/t, 1.1)#t*p / filter_avgs[i]**.7
	return (strength / num_key_filters / 1.1)**.3

###########################
### Bayesian Modeling
###########################

def obtain_params(A):
	"""Returns mu and var for the normal distribution p(A|f) based on the annotated set
	- A (100*10) is the list of annotated image activations for a given feature
	- F () is the list of feature labels
	"""
	return np.linalg.lstsq(A, F)

def obtain_params_dnu(A):
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

def visualize_layer_weighted(model, layer_name, save_path, channel_weights=None, init_img=None):
	"""Visualize the model inputs that would maximally activate a layer.
	channel_ixs is the set of channels to optimize over; keep as None to use the whole layer
	Original code by the Keras Team at
	https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py"""
	C = config.Config()

	layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

	input_img = model.input

	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered
	layer_output = layer_dict[layer_name].output
	layer_output = K.mean(layer_output, (0,1,2,3))
	loss = K.dot(K.expand_dims(layer_output,0), K.expand_dims(channel_weights))

	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(loss, input_img)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_img], [loss, grads])

	if init_img is None:
		input_img_data = np.random.random((1, C.dims[0], C.dims[1], C.dims[2], 3))
	else:
		input_img_data = np.expand_dims(init_img, 0)

	# run gradient ascent for 20 steps
	step = 5.
	for i in range(250):
		loss_value, grads_value = iterate([input_img_data])
		input_img_data += grads_value * step
		if i % 2 == 0:
			step *= .98
		if i % 5 == 0:
			#random rotations for transformation robustness, see https://distill.pub/2017/feature-visualization/#enemy-of-feature-vis
			input_img_data = np.pad(input_img_data[0], ((5,5),(5,5),(0,0),(0,0)), 'constant')
			input_img_data = tr.rotate(input_img_data, random.uniform(-5,5)*pi/180)
			input_img_data = np.expand_dims(input_img_data[5:-5, 5:-5, :, :], 0)

	img = input_img_data[0]
	img = deprocess_image(img)
	hf.draw_slices(img, save_path=os.path.join(save_path, "%s_filter.png" % layer_name))

def visualize_layer(model, layer_name, save_path, channel_ixs=None, init_img=None):
	"""Visualize the model inputs that would maximally activate a layer.
	channel_ixs is the set of channels to optimize over; keep as None to use the whole layer
	Original code by the Keras Team at
	https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py"""
	from keras import backend as K
	K.set_learning_phase(0)

	C = config.Config()

	layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

	input_img = model.input

	if channel_ixs is None:
		channel_ixs = list(range(layer_dict[layer_name].output.shape[-1]))

	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered
	layer_output = layer_dict[layer_name].output
	layer_output = K.permute_dimensions(layer_output, (4,0,1,2,3))
	layer_output = K.gather(layer_output, channel_ixs)
	loss = K.mean(layer_output)#[:, :, :, :, channel_ixs])

	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(loss, input_img)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_img], [loss, grads])

	if init_img is None:
		input_img_data = np.random.random((1, C.dims[0], C.dims[1], C.dims[2], 3))
	else:
		input_img_data = np.expand_dims(init_img, 0)

	# run gradient ascent for 20 steps
	step = 1.
	for i in range(250):
		loss_value, grads_value = iterate([input_img_data])
		input_img_data += grads_value * step
		if i % 2 == 0:
			step *= .99
		if i % 5 == 0:
			#random rotations for transformation robustness, see https://distill.pub/2017/feature-visualization/#enemy-of-feature-vis
			input_img_data = np.pad(input_img_data[0], ((5,5),(5,5),(0,0),(0,0)), 'constant')
			input_img_data = tr.rotate(input_img_data, random.uniform(-5,5)*pi/180)
			input_img_data = np.expand_dims(input_img_data[5:-5, 5:-5, :, :], 0)

	img = input_img_data[0]
	img = deprocess_image(img)
	hf.draw_slices(img, save_path=os.path.join(save_path, "%s_filter.png" % layer_name))

def visualize_channel(model, layer_name, save_path, num_ch=None):
	"""Visualize the model inputs that would maximally activate a layer.
	num_ch is the number of channels to optimize over; keep as None to use the whole layer
	Original code by the Keras Team at
	https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py"""
	C = config.Config()

	layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

	input_img = model.input

	if num_ch is None:
		num_ch = layer_dict[layer_name].output.shape[-1]

	for filter_index in range(num_ch):
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
			input_img_data = np.pad(input_img_data[0], ((5,5),(5,5),(0,0),(0,0)), 'constant')
			input_img_data = tr.rotate(input_img_data, random.uniform(-5,5)*pi/180)
			input_img_data = np.expand_dims(input_img_data[5:-5, 5:-5, :, :], 0)
			#random rotations for transformation robustness, see https://distill.pub/2017/feature-visualization/#enemy-of-feature-vis

		img = input_img_data[0]
		img = deprocess_image(img)
		hf.save_slices(img, save_path=os.path.join(save_path, "%s_filter_%d.png" % (layer_name, filter_index)))

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