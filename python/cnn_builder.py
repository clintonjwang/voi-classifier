"""
Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

import copy
import glob
import importlib
import math
import operator
import os
from os.path import *
import random
import time

import keras
import keras.backend as K
import keras.layers as layers
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import (ELU, Activation, Concatenate, Dense, Dropout,
						  Flatten, Input, Lambda, Permute,
						  Reshape, SimpleRNN, TimeDistributed)
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
from keras_contrib.layers.normalization import InstanceNormalization
from scipy.misc import imsave
from skimage.transform import rescale
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import config
import dr_methods as drm
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import niftiutils.deep_learning.cnn_components as cnnc
import niftiutils.deep_learning.dcgan as dcgan
import niftiutils.deep_learning.densenet as densenet
import niftiutils.deep_learning.uncertainty as uncert
import niftiutils.deep_learning.common as common
import voi_methods as vm

importlib.reload(config)
importlib.reload(uncert)
importlib.reload(drm)
importlib.reload(cnnc)
importlib.reload(common)
C = config.Config()

####################################
### Build CNNs
####################################

def build_cnn_hyperparams(T=None):
	if T is None:
		T = config.Hyperparams()

	if T.cnn_type == 'inception':
		M = build_inception(T.optimizer)

	elif T.cnn_type == 'dense':
		M = densenet.DenseNet((*C.dims, C.nb_channels), C.nb_classes,
			optimizer=T.optimizer, depth=T.depth, growth_rate=T.f[1]//4, nb_filter=T.f[0]//2,
        	pool_type='max', dropout_rate=T.dropout)

	elif C.dual_img_inputs:
		M = build_dual_cnn(optimizer=T.optimizer,
			padding=T.padding, pool_sizes=T.pool_sizes, dropout=T.dropout,
			f=T.f, dense_units=T.dense_units, kernel_size=T.kernel_size)

	else:
		M = build_cnn(optimizer=T.optimizer, skip_con=T.skip_con,
			padding=T.padding, pool_sizes=T.pool_sizes, dropout=T.dropout,
			f=T.f, dense_units=T.dense_units, kernel_size=T.kernel_size,
			dual_inputs=C.clinical_inputs, mc_sampling=T.mc_sampling)

	if C.aleatoric:
		pred_model, train_model = uncert.add_aleatoric_var(M, C.nb_classes, focal_loss=C.focal_loss)
		return pred_model, train_model
	else:
		return M

def build_cnn(optimizer='adam', padding=['same','same'], pool_sizes=[2,(2,2,1)],
	dropout=.1, f=[64,128,128], dense_units=100, kernel_size=(3,3,2),
	dual_inputs=False, skip_con=False, trained_model=None, mc_sampling=False):
	"""Main class for setting up a CNN. Returns the compiled model."""
	"""if trained_model is not None:
		inputs = Input(shape=(C.dims[0]//2, C.dims[1]//2, C.dims[2]//2, 128))
		x = ActivationLayer(activation_args)(inputs)
		x = layers.MaxPooling3D(pool_sizes[1])(x)
		x = Flatten()(x)
		x = Dense(dense_units)(x)
		x = cnnc.bn_relu_etc(x, dropout, mc_sampling)
		pred_class = Dense(nb_classes, activation='softmax')(x)
		model = Model(inputs, pred_class)

		num_l = len(model.layers)
		dl = len(trained_model.layers)-num_l
		for ix in range(num_l):
			model.layers[-ix].set_weights(trained_model.layers[-ix].get_weights())

		return model"""

	img = Input(shape=(*C.dims, C.nb_channels))

	art_x = Lambda(lambda x: K.expand_dims(x[...,0], axis=4))(img)
	ven_x = Lambda(lambda x: K.expand_dims(x[...,1], axis=4))(img)
	eq_x = Lambda(lambda x: K.expand_dims(x[...,2], axis=4))(img)
	#art_x = cnnc.bn_relu_etc(art_x, dropout, mc_sampling, cv_u=f[0], cv_k=kernel_size)
	art_x = layers.Conv3D(f[0], kernel_size, kernel_initializer="he_uniform", padding=padding[0])(art_x)
	ven_x = layers.Conv3D(f[0], kernel_size, kernel_initializer="he_uniform", padding=padding[0])(ven_x)
	eq_x = layers.Conv3D(f[0], kernel_size, kernel_initializer="he_uniform", padding=padding[0])(eq_x)

	x = Concatenate(axis=-1)([art_x, ven_x, eq_x])
	if mc_sampling:
		x = cnnc.bn_relu_etc(x, dropout, mc_sampling)
	else:
		x = cnnc.bn_relu_etc(x)
		x = layers.SpatialDropout3D(dropout)(x)
	x = layers.MaxPooling3D(pool_sizes[0])(x)

	for layer_num in range(1,len(f)):
		#x = layers.Conv3D(filters=f[layer_num], kernel_size=kernel_size, kernel_initializer="he_uniform", padding=padding[1])(x)

		if skip_con and layer_num==1:
			skip_layer = x
		elif skip_con and layer_num==5:
			x = layers.Add()([x, skip_layer])

		x = cnnc.bn_relu_etc(x, dropout, mc_sampling, cv_u=f[layer_num], cv_k=kernel_size, cv_pad=padding[1])

	x = layers.MaxPooling3D(pool_sizes[1])(x)
	x = Flatten()(x)
	x = cnnc.bn_relu_etc(x, dropout, mc_sampling, fc_u=dense_units)

	if C.clinical_inputs > 0:
		clinical_inputs = Input(shape=(C.clinical_inputs,))
		#y = layers.PReLU(alpha_constraint=keras.constraints.non_neg())(clinical_inputs)
		y = layers.Masking(mask_value=0.)(clinical_inputs)
		y = cnnc._expand_dims(y)
		y = layers.LocallyConnected1D(1, 1, activation='tanh')(y)
		y = layers.Flatten()(y)
		#y = cnnc.bn_relu_etc(y, drop=.3)
		y = layers.Dropout(.2)(y)
		x = Concatenate(axis=1)([x, y])
		
	if C.aleatoric:
		pred_class = Dense(C.nb_classes+1)(x)
		loss = 'categorical_crossentropy'
		metrics = None
	elif C.focal_loss:
		pred_class = Dense(C.nb_classes, activation='softmax')(x)
		loss = common.focal_loss(gamma=C.focal_loss)
		metrics = ['accuracy']
	else:
		pred_class = Dense(C.nb_classes, activation='softmax')(x)
		loss = 'categorical_crossentropy'
		metrics = ['accuracy']

	if C.clinical_inputs == 0:
		model = Model(img, pred_class)
	else:
		model = Model([img, clinical_inputs], pred_class)

	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

	return model

def build_inception(optimizer='adam'):
	"""Main class for setting up a CNN. Returns the compiled model."""

	img = Input(shape=(*C.dims, 3))

	x = cnnc.bn_relu_etc(img, drop=.1, cv_u=64, pool=2)
	x = cnnc.Inception3D(x)
	x = cnnc.bn_relu_etc(x, pool=2)
	x = cnnc.bn_relu_etc(x, drop=.1, cv_u=64, pool=3)
	x = Flatten()(x)
	x = cnnc.bn_relu_etc(x, fc_u=64)

	if C.clinical_inputs > 0:
		clinical_inputs = Input(shape=(C.clinical_inputs,))
		y = layers.Masking(mask_value=0.)(clinical_inputs)
		y = cnnc._expand_dims(y)
		y = layers.LocallyConnected1D(1, 1, activation='tanh')(y)
		y = layers.Flatten()(y)
		x = Concatenate(axis=1)([x, y])
		
	x = layers.Dropout(.2)(x)
	if C.aleatoric:
		pred_class = Dense(C.nb_classes+1)(x)
		loss = 'categorical_crossentropy'
		metrics = None
	elif C.focal_loss:
		pred_class = Dense(C.nb_classes, activation='softmax')(x)
		loss = common.focal_loss(gamma=C.focal_loss)
		metrics = ['accuracy']
	else:
		pred_class = Dense(C.nb_classes, activation='softmax')(x)
		loss = 'categorical_crossentropy'
		metrics = ['accuracy']

	if C.clinical_inputs == 0:
		model = Model(img, pred_class)
	else:
		model = Model([img, clinical_inputs], pred_class)

	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

	return model

def build_rcnn(optimizer='adam', padding=['same','same'], pool_sizes=[2,2],
	dropout=.1, activation_type='relu', f=[64,64,64,64,64], dense_units=100, kernel_size=(3,3,2),
	dual_inputs=False, skip_con=False, trained_model=None, first_layer=0, last_layer=0,
	add_activ=False, debug=False, mc_sampling=False):
	"""Main class for setting up a CNN. Returns the compiled model."""

	if activation_type == 'elu':
		ActivationLayer = ELU
		activation_args = 1
	elif activation_type == 'relu':
		ActivationLayer = Activation
		activation_args = 'relu'

	if first_layer == 0:
		inputs = Input(shape=(*C.dims, C.nb_channels))
		x = Reshape((*C.dims, C.nb_channels, 1))(inputs)
		x = Permute((4,1,2,3,5))(x)

		for layer_num in range(len(f)):
			x = layers.TimeDistributed(layers.Conv3D(f[layer_num], kernel_size=kernel_size, padding='same'))(x)
			if layer_num == len(f)+last_layer+2:
				break
			x = cnnc.bn_relu_etc(x, dropout, mc_sampling)
			if layer_num == 0:
				x = TimeDistributed(layers.MaxPooling3D(pool_sizes[0]))(x)
	else:
		inputs = Input(shape=(C.nb_channels, C.dims[0]//2, C.dims[1]//2, C.dims[2]//2, f[-1]))
		x = inputs
		for layer_num in range(first_layer,0):
			x = cnnc.bn_relu_etc(x, dropout, mc_sampling)
			if layer_num!=-1:
				x = layers.TimeDistributed(layers.Conv3D(f[layer_num], kernel_size=kernel_size, padding='same'))(x)
		#x = ActivationLayer(activation_args)(x)
		#x = layers.TimeDistributed(layers.BatchNormalization(axis=4))(x)

	if last_layer == 0:
		x = layers.TimeDistributed(layers.MaxPooling3D(pool_sizes[1]))(x)
		x = layers.TimeDistributed(Flatten())(x)

		#x = SimpleRNN(128, return_sequences=True)(x)
		x = layers.SimpleRNN(dense_units)(x)
		x = cnnc.bn_relu_etc(x, dropout, mc_sampling)
		x = Dense(dense_units)(x)
		x = cnnc.bn_relu_etc(x, dropout, mc_sampling)

		if dual_inputs:
			clinical_inputs = Input(shape=(C.clinical_inputs,))
			y = Dense(20)(clinical_inputs)
			y = BatchNormalization()(y)
			#y = DropoutLayer(y)
			#y = ActivationLayer(activation_args)(y)
			x = Concatenate(axis=1)([x, y])
			model = Model([inputs, clinical_inputs], pred_class)
			
		else:
			pred_class = Dense(C.nb_classes, activation='softmax')(x)
			model = Model(inputs, pred_class)

		if first_layer == 0:
			model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		elif not debug:
			num_l = len(model.layers)
			dl = len(trained_model.layers)-num_l

			for l in range(num_l-1, 0, -1):
				model.layers[l].set_weights(trained_model.layers[l+dl].get_weights())

	else:
		if add_activ:
			x = ActivationLayer(activation_args)(x)
		model = Model(inputs, x)

		if not debug:
			for l in range(1,len(model.layers)):
				model.layers[l].set_weights(trained_model.layers[l].get_weights())

	return model

def build_dual_cnn(optimizer='adam', padding=['same','same'], pool_sizes=[2,2],
	dropout=[0.1,0.1], activation_type='relu', f=[64,128,128], dense_units=100, kernel_size=(3,3,2)):
	"""Main class for setting up a CNN. Returns the compiled model."""

	ActivationLayer = Activation
	activation_args = 'relu'

	nb_classes = len(C.cls_names)

	context_img = layers.Input(shape=(C.context_dims[0], C.context_dims[1], C.context_dims[2], 3))
	cx = context_img

	cx = layers.Reshape((*C.context_dims, 3, 1))(context_img)
	cx = layers.Permute((4,1,2,3,5))(cx)
	cx = layers.TimeDistributed(layers.Conv3D(filters=64, kernel_size=(4,4,1), padding='valid', activation='relu'))(cx)
	cx = layers.TimeDistributed(layers.MaxPooling3D(2))(cx)
	cx = layers.TimeDistributed(layers.BatchNormalization(axis=4))(cx)
	cx = layers.TimeDistributed(layers.Conv3D(filters=64, kernel_size=(3,3,1), padding='valid', activation='relu'))(cx)
	cx = layers.TimeDistributed(layers.BatchNormalization(axis=4))(cx)
	cx = layers.TimeDistributed(layers.Conv3D(filters=64, kernel_size=(3,3,1), padding='valid', activation='relu'))(cx)
	cx = layers.TimeDistributed(layers.BatchNormalization(axis=4))(cx)
	cx = layers.TimeDistributed(layers.Conv3D(filters=64, kernel_size=(3,3,1), padding='same', activation='relu'))(cx)
	cx = layers.TimeDistributed(layers.BatchNormalization(axis=4))(cx)

	img = layers.Input(shape=(*C.dims, C.nb_channels))
	x = img
	x = layers.Reshape((*C.dims, C.nb_channels, 1))(x)
	x = layers.Permute((4,1,2,3,5))(x)

	for layer_num in range(len(f)):
		if layer_num == 3:
			x = layers.Concatenate(axis=5)([x, cx])
		#...
	#...
	pred_class = layers.Dense(nb_classes, activation='softmax')(x)

	model = Model([img, context_img], pred_class)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	return model

def pretrain_cnn(trained_model, padding=['same','same'], pool_sizes=[2, (2,2,1)],
	activation_type='relu', f=[64,128,128], kernel_size=(3,3,2), dense_units=100, skip_con=False,
	last_layer=-2, add_activ=False, training=True, debug=False):
	"""Sets up CNN with pretrained weights"""

	ActivationLayer = Activation
	activation_args = 'relu'

	nb_classes = len(C.cls_names)

	img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 3))

	art_x = Lambda(lambda x : K.expand_dims(x[...,0], axis=4))(img)
	art_x = layers.Conv3D(filters=f[0], kernel_size=kernel_size, padding=padding[0], trainable=False)(art_x)
	ven_x = Lambda(lambda x : K.expand_dims(x[...,1], axis=4))(img)
	ven_x = layers.Conv3D(filters=f[0], kernel_size=kernel_size, padding=padding[0], trainable=False)(ven_x)
	eq_x = Lambda(lambda x : K.expand_dims(x[...,2], axis=4))(img)
	eq_x = layers.Conv3D(filters=f[0], kernel_size=kernel_size, padding=padding[0], trainable=False)(eq_x)

	#if padding != ['same', 'same']:
	#   art_x = BatchNormalization(trainable=False)(art_x, training=training)
	#   ven_x = BatchNormalization(trainable=False)(ven_x, training=training)
	#   eq_x = BatchNormalization(trainable=False)(eq_x, training=training)

	if last_layer >= -5:
		#if padding != ['same', 'same']:
		#   art_x = ActivationLayer(activation_args)(art_x)
		#   ven_x = ActivationLayer(activation_args)(ven_x)
		#   eq_x = ActivationLayer(activation_args)(eq_x)

		x = Concatenate(axis=4)([art_x, ven_x, eq_x])
		#if padding == ['same', 'same']:
		x = ActivationLayer(activation_args)(x)
		x = Dropout(0)(x)
		x = layers.MaxPooling3D(pool_sizes[0])(x)
		x = BatchNormalization(axis=4, trainable=False)(x)
		#else:
		#x = layers.MaxPooling3D(pool_sizes[0])(x)

		for layer_num in range(1, len(f)+last_layer+2):
			x = layers.Conv3D(filters=f[layer_num], kernel_size=kernel_size, padding=padding[1], trainable=False)(x)

			if skip_con and layer_num==1:
				skip_layer = x
			elif skip_con and layer_num==5:
				x = layers.Add()([x, skip_layer])


			x = BatchNormalization(trainable=False)(x)
			#if layer_num == len(f)+last_layer+1:
			#   break
			x = ActivationLayer(activation_args)(x)
			x = Dropout(0)(x)

		if last_layer >= -2:
			x = layers.MaxPooling3D(pool_sizes[1])(x)

			x = Flatten()(x)
			x = Dense(dense_units, trainable=False)(x)
			x = BatchNormalization(trainable=False)(x)
			if last_layer >= -1:
				x = ActivationLayer(activation_args)(x)
				x = Dropout(0)(x)
				x = Dense(6, trainable=False)(x)
				x = BatchNormalization(trainable=False)(x)
	else:
		x = Concatenate(axis=4)([art_x, ven_x, eq_x])

	if add_activ:
		x = ActivationLayer(activation_args)(x)

	model_pretrain = Model(img, x)
	model_pretrain.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	if not debug:
		for l in range(1,len(model_pretrain.layers)-add_activ):
			model_pretrain.layers[l].set_weights(trained_model.layers[l].get_weights())

	return model_pretrain

def pretrain_model_back(trained_model, padding=['same', 'same'], pool_sizes = [2, (2,2,1)],
	activation_type='relu', f=[64,128,128], kernel_size=(3,3,2), dense_units=100, first_layer=-3):
	"""Sets up CNN with pretrained weights"""
	ActivationLayer = Activation
	activation_args = 'relu'

	nb_classes = len(C.cls_names)

	if first_layer <= -5:
		img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], f[0]*3))
		x = ActivationLayer(activation_args)(img)
		x = Dropout(0)(x)
		x = layers.MaxPooling3D(pool_sizes[0])(x)
		x = BatchNormalization(axis=4, trainable=False)(x)

	else:
		img = Input(shape=(C.dims[0]//2, C.dims[1]//2, C.dims[2]//2, f[1]))
		x = ActivationLayer(activation_args)(img)
		x = Dropout(0)(x)

	for layer_num in range(-3-first_layer):
		x = layers.Conv3D(filters=f[len(f)-layer_num-1], kernel_size=kernel_size, padding=padding[1], trainable=False)(x)
		x = BatchNormalization(trainable=False)(x)
		x = ActivationLayer(activation_args)(x)
		x = Dropout(0)(x)

	x = layers.MaxPooling3D(pool_sizes[1])(x)
	x = Flatten()(x)
	x = Dense(dense_units, trainable=False)(x)
	x = BatchNormalization(trainable=False)(x)
	x = Dropout(0)(x)
	x = ActivationLayer(activation_args)(x)
	x = Dense(nb_classes, trainable=False)(x)

	model_pretrain = Model(img, x)
	model_pretrain.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	num_l = len(model_pretrain.layers)
	dl = len(trained_model.layers)-num_l

	for l in range(num_l-1, 0, -1):
		model_pretrain.layers[l].set_weights(trained_model.layers[l+dl].get_weights())

	return model_pretrain

####################################
### Load Data
####################################

def get_cnn_data(n=4, use_vois=True, Z_test_fixed=None, verbose=False):
	"""Subroutine to run CNN
	n is number of real samples
	Z_test is filenames"""
	orig_data_dict, num_samples = _collect_unaug_data(use_vois)

	train_ids = {} #filenames of training set originals
	test_ids = {} #filenames of test set
	X_test = []
	X2_test = []
	Y_test = []
	Z_test = []
	X_train_orig = []
	X2_train_orig = []
	Y_train_orig = []
	Z_train_orig = []

	train_samples = {}

	if Z_test_fixed is not None:
		orders = {cls: np.where(np.isin(orig_data_dict[cls][1], Z_test_fixed)) for cls in orig_data_dict}
		for cls in C.cls_names:
			orders[cls] = list(set(range(num_samples[cls])).difference(list(orders[cls][0]))) + list(orders[cls][0])

	for cls in orig_data_dict:
		cls_num = C.cls_names.index(cls)

		train_samples[cls] = num_samples[cls] - C.test_num
		
		if Z_test_fixed is None:
			order = np.random.permutation(list(range(num_samples[cls])))
		else:
			order = orders[cls]

		train_ids[cls] = list(orig_data_dict[cls][-1][order[:train_samples[cls]]])
		test_ids[cls] = list(orig_data_dict[cls][-1][order[train_samples[cls]:]])
		
		X_test = X_test + list(orig_data_dict[cls][0][order[train_samples[cls]:]])
		X2_test = X2_test + list(orig_data_dict[cls][1][order[train_samples[cls]:]])
		Y_test = Y_test + [cls_num] * (num_samples[cls] - train_samples[cls])
		Z_test = Z_test + test_ids[cls]
		
		X_train_orig = X_train_orig + list(orig_data_dict[cls][0][order[:train_samples[cls]]])
		X2_train_orig = X2_train_orig + list(orig_data_dict[cls][1][order[:train_samples[cls]]])
		Y_train_orig = Y_train_orig + [cls_num] * (train_samples[cls])
		Z_train_orig = Z_train_orig + train_ids[cls]
		
		if verbose:
			print("%s has %d samples for training (%d after augmentation) and %d for testing" %
				  (cls, train_samples[cls], train_samples[cls] * C.aug_factor, num_samples[cls] - train_samples[cls]))


	Y_test = np_utils.to_categorical(Y_test, C.nb_classes)
	Y_train_orig = np_utils.to_categorical(Y_train_orig, C.nb_classes)
	if C.dual_img_inputs or C.clinical_inputs:
		X_test = [np.array(X_test), np.array(X2_test)]
		X_train_orig = [np.array(X_train_orig), np.array(X2_train_orig)]
	else:
		X_test = np.array(X_test)
		X_train_orig = np.array(X_train_orig)

	Y_test = np.array(Y_test)
	Y_train_orig = np.array(Y_train_orig)

	Z_test = np.array(Z_test)
	Z_train_orig = np.array(Z_train_orig)

	"""if use_vois is None:
		accnums = {}
		for cls in C.cls_names:
			src_data_df = drm.get_coords_df(cls)
			accnums[cls] = src_data_df["acc #"].values"""

	train_generator = _train_gen_classifier(test_ids, n=n)

	return X_test, Y_test, train_generator, num_samples, [X_train_orig, Y_train_orig], [Z_test, Z_train_orig]

def train_gen_ensemble(n=4, Z_exc=None):
	orig_data_dict, num_samples = _collect_unaug_data()
	test_ids = {cls: [z for z in orig_data_dict[cls][1] if z in Z_exc] for cls in orig_data_dict}

	return _train_gen_classifier(test_ids, n=n)

def load_data_capsnet(n=2, Z_test_fixed=None):
	orig_data_dict, num_samples = _collect_unaug_data()

	test_ids = {} #filenames of test set
	X_test = []
	Y_test = []
	Z_test = []
	train_samples = {}

	if Z_test_fixed is not None:
		orders = {cls: np.where(np.isin(orig_data_dict[cls][1], Z_test_fixed)) for cls in orig_data_dict}
		for cls in C.cls_names:
			orders[cls] = list(set(range(num_samples[cls])).difference(list(orders[cls][0]))) + list(orders[cls][0])

	for cls in orig_data_dict:
		cls_num = C.cls_names.index(cls)
		train_samples[cls] = num_samples[cls] - C.test_num
		
		if Z_test_fixed is None:
			order = np.random.permutation(list(range(num_samples[cls])))
		else:
			order = orders[cls]

		X_test = X_test + list(orig_data_dict[cls][0][order[train_samples[cls]:]])
		test_ids[cls] = list(orig_data_dict[cls][-1][order[train_samples[cls]:]])
		Y_test += [cls_num] * (num_samples[cls] - train_samples[cls])
		Z_test = Z_test + test_ids[cls]

	Y_test = np_utils.to_categorical(Y_test, C.nb_classes)
	X_test = np.array(X_test)
	Z_test = np.array(Z_test)

	return _train_gen_capsnet(test_ids, n=n), (X_test, Y_test), Z_test

####################################
### Training Submodules
####################################

"""def _train_gen_capsnet(test_ids, n=4):
	while True:
		x1 = np.empty((n*C.nb_classes, *C.dims, C.nb_channels))
		y = np.zeros((n*C.nb_classes, C.nb_classes))

		train_cnt = 0
		for cls in C.cls_names:
			img_fns = os.listdir(C.aug_dir)
			while n > 0:
				img_fn = random.choice(img_fns)
				lesion_id = img_fn[:img_fn.rfind('_')]
				if lesion_id not in test_ids[cls]:
					x1[train_cnt] = np.load(join(C.aug_dir, img_fn))
					
					y[train_cnt][C.cls_names.index(cls)] = 1
					
					train_cnt += 1
					if train_cnt % n == 0:
						break

		x_batch = np.array(x1)
		y_batch = np.array(y)
		yield ([x_batch, y_batch], [y_batch, x_batch])"""

"""def _train_gen_unet(test_accnums=[]):
	raise ValueError("Not usable")
	lesion_df_art = drm.get_lesion_df()[0]
	lesion_df_art.accnum = lesion_df_art.accnum.astype(str)
	img_fns = [fn for fn in glob.glob(join(C.full_img_dir, "*.npy")) if not fn.endswith("_seg.npy") \
				and basename(fn)[:-4] not in test_accnums]

	while True:
		img_fn = random.choice(img_fns)
		img = np.load(img_fn)
		img = tr.rescale_img(img, C.dims)
		seg = np.load(img_fn[:-4]+"_seg.npy")
		seg = tr.rescale_img(seg, C.dims) > .5
		seg = np_utils.to_categorical(seg, 2).astype(int)
		cls = np_utils.to_categorical(C.cls_names.index(lesion_df_art.loc[lesion_df_art["accnum"] \
			== basename(img_fn[:-4]), "cls"].values[0]), len(C.cls_names)).astype(int)

		yield [np.expand_dims(img, 0), np.expand_dims(seg, 0), np.expand_dims(cls, 0)], None"""

def pretrain_gen(self, side_len, n=8, test_accnums=[]):
	dims_df = pd.read_csv(C.dims_df_path, index_col=0)
	dims_df.index = dims_df.index.map(str)

	accnums = {}
	for cls in C.cls_names:
		src_data_df = drm.get_coords_df(cls)
		accnums[cls] = src_data_df["acc #"].values

	img_fns = [fn for fn in glob.glob(join(C.full_img_dir, "*.npy")) if not fn.endswith("seg.npy") \
				and basename(fn)[:-4] not in test_accnums]

	cropI = np.empty(n, *C.dims, C.nb_channels)
	crop_true_seg = np.empty(n, *C.dims, C.num_segs)
	true_cls = np.empty(n, C.nb_classes)
	train_ix = 0

	while True:
		img_fn = random.choice(img_fns)
		img = np.load(img_fn)
		img = tr.rescale_img(img, C.dims)
		seg = np.load(img_fn[:-4]+"_liverseg.npy")
		seg = tr.rescale_img(seg, C.dims) > .5
		seg = np_utils.to_categorical(seg, 2).astype(int)
		cls = np_utils.to_categorical(C.cls_names.index(lesion_df_art.loc[lesion_df_art["accnum"] \
			== basename(img_fn[:-4]), "cls"].values[0]), len(C.cls_names)).astype(int)


		cropIs, crop_segs = tr.split_img(img, seg, L=side_len*10)
		D = dims_df.loc[accnum].values
		
		for crop_ix in range(cropIs.shape[0]):
			if train_ix == n:
				yield [cropI, crop_true_seg, true_cls], [None]*n
				train_ix = 0

			cropI[train_ix] = cropIs[crop_ix]
			crop_true_seg[train_ix, ..., 1:] = crop_segs[crop_ix]
			crop_true_seg[train_ix, ..., 0] = 1 - crop_true_seg[train_ix, ..., 1:].sum(-1)
			true_cls[train_ix] = cls
			train_ix += 1

def _train_gen_ddpg(test_accnums=[]):
	"""X is the whole abdominal MR (20s only), ; Y is the set of true bboxes"""

	lesion_df = drm.get_lesion_df()
	lesion_df.accnum = lesion_df_art.accnum.astype(str)
	img_fns = [fn for fn in glob.glob(join(C.full_img_dir, "*.npy")) if not fn.endswith("seg.npy") \
				and basename(fn)[:-4] not in test_accnums]

	while True:
		img_fn = random.choice(img_fns)
		accnum = img_fn[:-4]

		if not exists(accnum+"_tumorseg.npy") or not exists(accnum+"_liverseg.npy"):
			continue

		img = np.load(img_fn)
		img = tr.rescale_img(img, C.context_dims)
		img = tr.normalize_intensity(img, 1, -1)
		tumorM = np.load(accnum+"_tumorseg.npy")
		liverM = np.load(accnum+"_liverseg.npy")
		try:
			liverM[tumorM > 0] = 0
		except:
			print(basename(accnum), end="','")
			continue
		seg = np.zeros((*C.context_dims, C.num_segs))
		seg[...,-1] = tr.rescale_img(tumorM, C.context_dims)
		seg[...,1] = tr.rescale_img(liverM, C.context_dims)
		seg[...,0] = np.clip(1 - seg[...,1] - seg[...,-1], 0, 1)

		if basename(accnum) in lesion_df_art["accnum"].values:
			cls_num = C.cls_names.index(lesion_df_art.loc[lesion_df_art["accnum"] \
				== basename(accnum), "cls"].values[0])
		else:
			cls_num = 0
		cls = np_utils.to_categorical(cls_num, len(C.cls_names)).astype(int)

		yield (img, seg, cls)

def _train_gen_classifier(test_ids, accnums=None, n=12):
	"""n is the number of samples from each class"""

	if accnums is None:
		lesion_df = drm.get_lesion_df()
		accnums = {}
		for cls in C.cls_names:
			accnums[cls] = lesion_df.loc[(lesion_df["cls"]==cls) & \
				(lesion_df["run_num"] <= C.run_num), "accnum"].values

	if C.clinical_inputs > 0:
		train_path="E:\\LIRADS\\excel\\clinical_data_test.xlsx" #clinical_data_train
		clinical_df = pd.read_excel(train_path, index_col=0)
		clinical_df.index = clinical_df.index.astype(str)

	if type(accnums) == dict:
		img_fns = {cls:[fn for fn in os.listdir(C.aug_dir) if fn[:fn.find('_')] in accnums[cls]] for cls in C.cls_names}

	while True:
		x1 = np.empty((n*C.nb_classes, *C.dims, C.nb_channels))
		y = np.zeros((n*C.nb_classes, C.nb_classes))

		if C.dual_img_inputs:
			x2 = np.empty((n*C.nb_classes, *C.context_dims, C.nb_channels))
		elif C.clinical_inputs:
			x2 = np.empty((n*C.nb_classes, C.clinical_inputs))

		train_cnt = 0
		for cls in C.cls_names:
			while n > 0:
				img_fn = random.choice(img_fns[cls])
				lesion_id = img_fn[:img_fn.rfind('_')]
				if lesion_id not in test_ids[cls]:
					x1[train_cnt] = np.load(join(C.aug_dir, img_fn))
					try:
						if C.post_scale > 0:
							x1[train_cnt] = tr.normalize_intensity(x1[train_cnt], 1., -1., C.post_scale)
					except:
						print(lesion_id)
						vm.reset_accnum(lesion_id[:lesion_id.find('_')])

					if C.dual_img_inputs:
						tmp = np.load(join(C.crops_dir, lesion_id+".npy"))
						x2[train_cnt] = tr.rescale_img(tmp, C.context_dims)[0]
					elif C.clinical_inputs > 0:
						x2[train_cnt] = clinical_df.loc[lesion_id[:lesion_id.find('_')]].values[:C.clinical_inputs]
					
					y[train_cnt][C.cls_names.index(cls)] = 1
					
					train_cnt += 1
					if train_cnt % n == 0:
						break

		if C.dual_img_inputs or C.clinical_inputs>0:
			yield [np.array(x1), np.array(x2)], np.array(y)
		elif C.aleatoric:
			yield [np.array(x1), np.array(y)], None
		else:
			yield np.array(x1), np.array(y)

def _separate_phases(X, clinical_inputs=False):
	"""Assumes X[0] contains imaging and X[1] contains dimension data.
	Reformats such that X[0:2] has 3 phases and X[3] contains dimension data.
	Image data still is 5D (nb_samples, 3D, 1 channel).
	Handles both 2D and 3D images"""
	
	if clinical_inputs:
		dim_data = copy.deepcopy(X[1])
		img_data = X[0]
		
		axis = len(X[0].shape) - 1
		X[1] = np.expand_dims(X[0][...,1], axis=axis)
		X += [np.expand_dims(X[0][...,2], axis=axis)]
		X += [dim_data]
		X[0] = np.expand_dims(X[0][...,0], axis=axis)
	
	else:
		X = np.array(X)
		axis = len(X[0].shape) - 1
		X = [np.expand_dims(X[...,ix], axis=axis) for ix in range(3)]

	return X

def _collect_unaug_data(use_vois=True):
	"""Return dictionary pointing to X (img data) and Z (filenames) and dictionary storing number of samples of each class."""
	orig_data_dict = {}
	num_samples = {}
	if use_vois:
		lesion_df = drm.get_lesion_df()
		lesion_df = lesion_df[lesion_df["run_num"] <= C.test_run_num]

	if C.clinical_inputs > 0:
		test_path="E:\\LIRADS\\excel\\clinical_data_test.xlsx"
		clinical_df = pd.read_excel(test_path, index_col=0)
		clinical_df.index = clinical_df.index.astype(str)

	for cls in C.cls_names:
		x = np.empty((10000, *C.dims, C.nb_channels))
		z = []

		if C.dual_img_inputs:
			x2 = np.empty((10000, *C.context_dims, C.nb_channels))
		elif C.clinical_inputs > 0:
			x2 = np.empty((10000, C.clinical_inputs))

		if use_vois:
			lesion_ids = [x for x in lesion_df[lesion_df["cls"] == cls].index if exists(join(C.unaug_dir, x+".npy"))]
		else:
			src_data_df = drm.get_coords_df(cls)
			accnums = src_data_df["acc #"].values
			lesion_ids = [x[:-4] for x in os.listdir(C.crops_dir) if x[:x.find('_')] in accnums]

		index = 0
		for index, lesion_id in enumerate(lesion_ids):
			img_path = join(C.unaug_dir, lesion_id+".npy")
			try:
				x[index] = np.load(img_path)
				if C.post_scale > 0:
					x[index] = tr.normalize_intensity(x[index], 1., -1., C.post_scale)
			except:
				raise ValueError(img_path + " not found")
			z.append(lesion_id)
			
			if C.dual_img_inputs:
				tmp = np.load(join(C.crops_dir, lesion_id+".npy"))
				x2[index] = tr.rescale_img(tmp, C.context_dims)[0]

			elif C.clinical_inputs > 0:
				x2[index] = clinical_df.loc[lesion_id[:lesion_id.find('_')]].values[:C.clinical_inputs]

		x.resize((index+1, *C.dims, C.nb_channels)) #shrink first dimension to fit
		if C.dual_img_inputs or C.clinical_inputs>0:
			x2.resize((index+1, *x2.shape[1:]))
			orig_data_dict[cls] = [x, x2, np.array(z)]
		else:
			orig_data_dict[cls] = [x, np.array(z)]

		num_samples[cls] = index + 1
		
	return orig_data_dict, num_samples

####################################
### Misc
####################################

def aleatoric_xentropy(y_true, y_pred):
	eps = 1e-8
	rv = K.random_normal((1000,1), mean=0.0, stddev=1.0)
	#rv = np.random.laplace(loc=0., scale=1.0, size=(1000,1))
	y_noisy = K.mean( 1/(1+K.exp(-(y_pred[:,0] + y_pred[:,1]*rv))) , 0 )
	loss = - y_true[:,0] * K.log(y_noisy + eps) - (1-y_true[:,0]) * K.log(1-y_noisy + eps)
	
	return loss

def acc_logit(y_true, y_pred):
	acc = K.abs(y_true[:,0] - K.cast(y_pred[:,0] < 0, 'float32'))
	return acc
