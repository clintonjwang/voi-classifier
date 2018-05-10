"""
Usage:
	python cnn_builder.py

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

import keras.backend as K
import keras.layers as layers
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Concatenate, Flatten, Dropout, Lambda
from keras.layers import SimpleRNN, Conv2D, MaxPooling2D, ZeroPadding3D, Activation, ELU, TimeDistributed, Permute, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils

import argparse
import cnn_analyzer as cnna
import copy
import config
import niftiutils.cnn_components as cnnc
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import math
from math import log, ceil
import numpy as np
import operator
import os
import pandas as pd
import random
from scipy.misc import imsave
from skimage.transform import rescale
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import time
import importlib

import dr_methods as drm
import voi_methods as vm

def aleatoric_xentropy(y_true, y_pred):
    eps = K.random_normal((1000,2), mean=0.0, stddev=1.0)
    y_noisy = y_pred[0] + y_pred[1]*eps
    #loss = K.binary_crossentropy(y_true, y_noisy)
    lnDen = K.log(K.exp(y_noisy) + K.exp(1 - y_noisy))
    loss = y_true * K.mean(1/(1+K.exp(-y_noisy))) + (1-y_true) * K.mean(K.exp(1 - y_noisy - lnDen))
    
    return loss

def build_cnn_hyperparams(hyperparams):
	C = config.Config()
	if C.aleatoric:
		return build_prob_cnn(optimizer=hyperparams.optimizer,
			padding=hyperparams.padding, pool_sizes=hyperparams.pool_sizes, dropout=hyperparams.dropout,
			activation_type=hyperparams.activation_type, f=hyperparams.f, dense_units=hyperparams.dense_units,
			kernel_size=hyperparams.kernel_size)
	elif C.dual_img_inputs:
		return build_dual_cnn(optimizer=hyperparams.optimizer,
			padding=hyperparams.padding, pool_sizes=hyperparams.pool_sizes, dropout=hyperparams.dropout,
			activation_type=hyperparams.activation_type, f=hyperparams.f, dense_units=hyperparams.dense_units,
			kernel_size=hyperparams.kernel_size)
	elif hyperparams.rcnn:
		return build_rcnn(optimizer=hyperparams.optimizer,
			padding=hyperparams.padding, pool_sizes=hyperparams.pool_sizes, dropout=hyperparams.dropout,
			activation_type=hyperparams.activation_type, f=hyperparams.f, dense_units=hyperparams.dense_units,
			kernel_size=hyperparams.kernel_size,
			dual_inputs=C.non_imaging_inputs,
			skip_con=hyperparams.skip_con)
	else:
		return build_cnn(optimizer=hyperparams.optimizer,
			padding=hyperparams.padding, pool_sizes=hyperparams.pool_sizes, dropout=hyperparams.dropout,
			activation_type=hyperparams.activation_type, f=hyperparams.f, dense_units=hyperparams.dense_units,
			kernel_size=hyperparams.kernel_size,
			dual_inputs=C.non_imaging_inputs, run_2d=hyperparams.run_2d,
			skip_con=hyperparams.skip_con)

def build_cnn(optimizer='adam', dilation_rate=(1,1,1), padding=['same','same'], pool_sizes = [(2,2,2),(2,2,1)],
	dropout=[0.1,0.1], activation_type='relu', f=[64,128,128], dense_units=100, kernel_size=(3,3,2),
	dual_inputs=False, run_2d=False, stride=(1,1,1), skip_con=False, trained_model=None):
	"""Main class for setting up a CNN. Returns the compiled model."""

	importlib.reload(config)
	C = config.Config()

	if activation_type == 'elu':
		ActivationLayer = ELU
		activation_args = 1
	elif activation_type == 'relu':
		ActivationLayer = Activation
		activation_args = 'relu'

	nb_classes = len(C.classes_to_include)

	if trained_model is not None:
		inputs = Input(shape=(C.dims[0]//2, C.dims[1]//2, C.dims[2]//2, 128))
		x = ActivationLayer(activation_args)(inputs)
		#x = layers.Conv3D(filters=f[-1], kernel_size=kernel_size, padding=padding[1])(x)
		#x = BatchNormalization()(x)
		#x = ActivationLayer(activation_args)(x)
		#x = Dropout(dropout[0])(x)
		x = layers.MaxPooling3D(pool_sizes[1])(x)
		x = Flatten()(x)
		x = Dense(dense_units)(x)
		x = BatchNormalization()(x)
		x = Dropout(dropout[1])(x)
		x = ActivationLayer(activation_args)(x)
		pred_class = Dense(nb_classes, activation='softmax')(x)
		model = Model(inputs, pred_class)

		num_l = len(model.layers)
		dl = len(trained_model.layers)-num_l
		for ix in range(num_l):
			model.layers[-ix].set_weights(trained_model.layers[-ix].get_weights())

		return model


	if not run_2d:
		img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 3))
	else:
		img = Input(shape=(C.dims[0], C.dims[1], 3))

	art_x = Lambda(lambda x : K.expand_dims(x[:,:,:,:,0], axis=4))(img)
	art_x = layers.Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0])(art_x)
	ven_x = Lambda(lambda x : K.expand_dims(x[:,:,:,:,1], axis=4))(img)
	ven_x = layers.Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0])(ven_x)
	eq_x = Lambda(lambda x : K.expand_dims(x[:,:,:,:,2], axis=4))(img)
	eq_x = layers.Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0])(eq_x)

	x = Concatenate(axis=4)([art_x, ven_x, eq_x])
	x = ActivationLayer(activation_args)(x)
	x = Dropout(dropout[0])(x)
	x = layers.MaxPooling3D(pool_sizes[0])(x)
	x = BatchNormalization(axis=4)(x)

	for layer_num in range(1,len(f)):
		x = layers.Conv3D(filters=f[layer_num], kernel_size=kernel_size, padding=padding[1])(x)

		if skip_con and layer_num==1:
			skip_layer = x
		elif skip_con and layer_num==5:
			x = layers.Add()([x, skip_layer])

		x = BatchNormalization()(x)
		x = ActivationLayer(activation_args)(x)
		x = Dropout(dropout[0])(x)
	x = layers.MaxPooling3D(pool_sizes[1])(x)
	x = Flatten()(x)

	x = Dense(dense_units)(x)
	x = BatchNormalization()(x)
	x = Dropout(dropout[1])(x)
	x = ActivationLayer(activation_args)(x)

	if dual_inputs:
		non_img_inputs = Input(shape=(C.num_non_image_inputs,))
		#y = Dense(20)(non_img_inputs)
		#y = BatchNormalization()(y)
		#y = Dropout(dropout[1])(y)
		#y = ActivationLayer(activation_args)(y)
		x = Concatenate(axis=1)([x, non_img_inputs])
		
	else:
		pred_class = Dense(nb_classes, activation='softmax')(x)

		if not dual_inputs:
			model = Model(img, pred_class)
		else:
			model = Model([img, non_img_inputs], pred_class)

		#optim = Adam(lr=0.01)#5, decay=0.001)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	return model

def build_prob_cnn(optimizer='adam', dilation_rate=(1,1,1), padding=['same','same'], pool_sizes = [(2,2,2),(2,2,1)],
	dropout=[0.1,0.1], activation_type='relu', f=[64,128,128], dense_units=100, kernel_size=(3,3,2),
	dual_inputs=False, run_2d=False, stride=(1,1,1), skip_con=False, trained_model=None):
	"""Main class for setting up a CNN. Returns the compiled model."""

	importlib.reload(config)
	C = config.Config()

	if activation_type == 'elu':
		ActivationLayer = ELU
		activation_args = 1
	elif activation_type == 'relu':
		ActivationLayer = Activation
		activation_args = 'relu'

	if trained_model is not None:
		inputs = Input(shape=(C.dims[0]//2, C.dims[1]//2, C.dims[2]//2, 128))
		x = ActivationLayer(activation_args)(inputs)
		x = layers.MaxPooling3D(pool_sizes[1])(x)
		x = Flatten()(x)
		x = Dense(dense_units)(x)
		x = BatchNormalization()(x)
		x = Dropout(dropout[1])(x)
		x = ActivationLayer(activation_args)(x)
		pred_class = Dense(2)(x)
		model = Model(inputs, pred_class)

		num_l = len(model.layers)
		dl = len(trained_model.layers)-num_l
		for ix in range(num_l):
			model.layers[-ix].set_weights(trained_model.layers[-ix].get_weights())

		return model


	img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 3))

	art_x = Lambda(lambda x : K.expand_dims(x[:,:,:,:,0], axis=4))(img)
	art_x = layers.Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0])(art_x)
	ven_x = Lambda(lambda x : K.expand_dims(x[:,:,:,:,1], axis=4))(img)
	ven_x = layers.Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0])(ven_x)
	eq_x = Lambda(lambda x : K.expand_dims(x[:,:,:,:,2], axis=4))(img)
	eq_x = layers.Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0])(eq_x)

	x = Concatenate(axis=4)([art_x, ven_x, eq_x])
	x = ActivationLayer(activation_args)(x)
	x = Dropout(dropout[0])(x)
	x = layers.MaxPooling3D(pool_sizes[0])(x)
	x = BatchNormalization(axis=4)(x)

	for layer_num in range(1,len(f)):
		x = layers.Conv3D(filters=f[layer_num], kernel_size=kernel_size, padding=padding[1])(x)

		if skip_con and layer_num==1:
			skip_layer = x
		elif skip_con and layer_num==5:
			x = layers.Add()([x, skip_layer])

		x = BatchNormalization()(x)
		x = ActivationLayer(activation_args)(x)
		x = Dropout(dropout[0])(x)
	x = layers.MaxPooling3D(pool_sizes[1])(x)
	x = Flatten()(x)

	x = Dense(dense_units)(x)
	x = BatchNormalization()(x)
	x = Dropout(dropout[1])(x)
	x = ActivationLayer(activation_args)(x)
	pred_class = Dense(2)(x)

	model = Model(img, pred_class)

	#optim = Adam(lr=0.01)#5, decay=0.001)
	model.compile(optimizer=optimizer, loss=aleatoric_xentropy)

	return model

def build_rcnn(optimizer='adam', padding=['same','same'], pool_sizes = [(2,2,2), (2,2,2)],
	dropout=[0.1,0.1], activation_type='relu', f=[64,64,64,64,64], dense_units=100, kernel_size=(3,3,2),
	dual_inputs=False, skip_con=False, trained_model=None, first_layer=0, last_layer=0, add_activ=False, debug=False):
	"""Main class for setting up a CNN. Returns the compiled model."""

	C = config.Config()

	if activation_type == 'elu':
		ActivationLayer = ELU
		activation_args = 1
	elif activation_type == 'relu':
		ActivationLayer = Activation
		activation_args = 'relu'

	nb_classes = len(C.classes_to_include)

	if first_layer == 0:
		inputs = Input(shape=(C.dims[0], C.dims[1], C.dims[2], C.nb_channels))
		x = Reshape((C.dims[0], C.dims[1], C.dims[2], C.nb_channels, 1))(inputs)
		x = Permute((4,1,2,3,5))(x)

		for layer_num in range(len(f)):
			x = layers.TimeDistributed(layers.Conv3D(f[layer_num], kernel_size=kernel_size, padding='same'))(x)
			if layer_num == len(f)+last_layer+2:
				break
			x = layers.TimeDistributed(layers.Dropout(dropout[0]))(x)
			x = ActivationLayer(activation_args)(x)
			x = layers.TimeDistributed(layers.BatchNormalization(axis=4))(x)
			if layer_num == 0:
				x = TimeDistributed(layers.MaxPooling3D(pool_sizes[0]))(x)
	else:
		inputs = Input(shape=(C.nb_channels, C.dims[0]//2, C.dims[1]//2, C.dims[2]//2, f[-1]))
		x = inputs
		for layer_num in range(first_layer,0):
			x = layers.TimeDistributed(layers.Dropout(dropout[0]))(x)
			x = ActivationLayer(activation_args)(x)
			x = layers.TimeDistributed(layers.BatchNormalization(axis=4))(x)
			if layer_num!=-1:
				x = layers.TimeDistributed(layers.Conv3D(f[layer_num], kernel_size=kernel_size, padding='same'))(x)
		#x = ActivationLayer(activation_args)(x)
		#x = layers.TimeDistributed(layers.BatchNormalization(axis=4))(x)

	if last_layer == 0:
		x = layers.TimeDistributed(layers.MaxPooling3D(pool_sizes[1]))(x)
		x = layers.TimeDistributed(Flatten())(x)

		#x = SimpleRNN(128, return_sequences=True)(x)
		x = layers.SimpleRNN(dense_units)(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(dropout[1])(x)
		x = Dense(dense_units)(x)
		x = BatchNormalization()(x)
		x = Dropout(dropout[1])(x)
		x = ActivationLayer(activation_args)(x)

		if dual_inputs:
			non_img_inputs = Input(shape=(C.num_non_image_inputs,))
			#y = Dense(20)(non_img_inputs)
			#y = BatchNormalization()(y)
			#y = Dropout(dropout[1])(y)
			#y = ActivationLayer(activation_args)(y)
			x = Concatenate(axis=1)([x, non_img_inputs])
			model = Model([inputs, non_img_inputs], pred_class)
			
		else:
			pred_class = Dense(nb_classes, activation='softmax')(x)
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

def build_dual_cnn(optimizer='adam', dilation_rate=(1,1,1), padding=['same','same'], pool_sizes = [(2,2,2), (2,2,2)],
	dropout=[0.1,0.1], activation_type='relu', f=[64,128,128], dense_units=100, kernel_size=(3,3,2), stride=(1,1,1)):
	"""Main class for setting up a CNN. Returns the compiled model."""

	C = config.Config()
	ActivationLayer = Activation
	activation_args = 'relu'

	nb_classes = len(C.classes_to_include)

	context_img = layers.Input(shape=(C.context_dims[0], C.context_dims[1], C.context_dims[2], 3))
	cx = context_img

	#cx = InstanceNormalization(axis=4)(cx)
	cx = layers.Reshape((*C.context_dims, 3, 1))(context_img)
	cx = layers.Permute((4,1,2,3,5))(cx)
	cx = layers.TimeDistributed(layers.Conv3D(filters=64, kernel_size=(4,4,1), padding='valid', activation='relu'))(cx)
	cx = layers.TimeDistributed(layers.MaxPooling3D((2,2,2)))(cx)
	cx = layers.TimeDistributed(layers.BatchNormalization(axis=4))(cx)
	cx = layers.TimeDistributed(layers.Conv3D(filters=64, kernel_size=(3,3,1), padding='valid', activation='relu'))(cx)
	cx = layers.TimeDistributed(layers.BatchNormalization(axis=4))(cx)
	cx = layers.TimeDistributed(layers.Conv3D(filters=64, kernel_size=(3,3,1), padding='valid', activation='relu'))(cx)
	cx = layers.TimeDistributed(layers.BatchNormalization(axis=4))(cx)
	cx = layers.TimeDistributed(layers.Conv3D(filters=64, kernel_size=(3,3,1), padding='same', activation='relu'))(cx)
	cx = layers.TimeDistributed(layers.BatchNormalization(axis=4))(cx)


	img = layers.Input(shape=(C.dims[0], C.dims[1], C.dims[2], 3))
	x = img
	x = layers.Reshape((C.dims[0], C.dims[1], C.dims[2], 3, 1))(x)
	x = layers.Permute((4,1,2,3,5))(x)

	for layer_num in range(len(f)):
		if layer_num == 3:
			x = layers.Concatenate(axis=5)([x, cx])
		x = layers.TimeDistributed(layers.Conv3D(filters=f[layer_num], kernel_size=kernel_size,
			padding='same', activation='relu'))(x)
		x = layers.TimeDistributed(Dropout(dropout[0]))(x)
		x = layers.TimeDistributed(BatchNormalization(axis=4))(x)
		if layer_num == 0:
			x = layers.TimeDistributed(layers.MaxPooling3D(pool_sizes[0]))(x)

	x = layers.TimeDistributed(layers.MaxPooling3D(pool_sizes[1]))(x)
	x = layers.TimeDistributed(layers.Conv3D(filters=128, kernel_size=(1,1,1), padding='same', activation='relu'))(x)
	x = layers.TimeDistributed(layers.BatchNormalization(axis=4))(x)
	x = layers.TimeDistributed(layers.Conv3D(filters=64, kernel_size=(3,3,2), padding='same', activation='relu'))(x)
	x = layers.TimeDistributed(layers.BatchNormalization(axis=4))(x)
	x = layers.TimeDistributed(layers.Flatten())(x)

	x = SimpleRNN(128)(x)
	x = layers.Dense(dense_units, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(dropout[1])(x)

	pred_class = layers.Dense(nb_classes, activation='softmax')(x)

	model = Model([img, context_img], pred_class)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	return model

def pretrain_cnn(trained_model, padding=['same','same'], pool_sizes=[(2,2,2), (2,2,1)],
	activation_type='relu', f=[64,128,128], kernel_size=(3,3,2), dense_units=100, skip_con=False,
	last_layer=-2, add_activ=False, training=True, debug=False):
	"""Sets up CNN with pretrained weights"""

	C = config.Config()
	dilation_rate=(1,1,1)

	ActivationLayer = Activation
	activation_args = 'relu'

	nb_classes = len(C.classes_to_include)

	img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 3))

	art_x = Lambda(lambda x : K.expand_dims(x[:,:,:,:,0], axis=4))(img)
	art_x = layers.Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0], trainable=False)(art_x)

	ven_x = Lambda(lambda x : K.expand_dims(x[:,:,:,:,1], axis=4))(img)
	ven_x = layers.Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0], trainable=False)(ven_x)

	eq_x = Lambda(lambda x : K.expand_dims(x[:,:,:,:,2], axis=4))(img)
	eq_x = layers.Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0], trainable=False)(eq_x)

	#if padding != ['same', 'same']:
	#	art_x = BatchNormalization(trainable=False)(art_x, training=training)
	#	ven_x = BatchNormalization(trainable=False)(ven_x, training=training)
	#	eq_x = BatchNormalization(trainable=False)(eq_x, training=training)

	if last_layer >= -5:
		#if padding != ['same', 'same']:
		#	art_x = ActivationLayer(activation_args)(art_x)
		#	ven_x = ActivationLayer(activation_args)(ven_x)
		#	eq_x = ActivationLayer(activation_args)(eq_x)

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
			#	break
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

def pretrain_model_back(trained_model, dilation_rate=(1,1,1), padding=['same', 'same'], pool_sizes = [(2,2,2), (2,2,1)],
	activation_type='relu', f=[64,128,128], kernel_size=(3,3,2), dense_units=100, first_layer=-3):
	"""Sets up CNN with pretrained weights"""

	C = config.Config()

	ActivationLayer = Activation
	activation_args = 'relu'

	nb_classes = len(C.classes_to_include)

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

def build_bbox_cnn(dilation_rate=(1,1,1), 
	dropout=[0.1,0.1], activation_type='relu', f=[64,128,128], dense_units=100, kernel_size=(3,3,2),
	dual_inputs=False, run_2d=False, stride=(1,1,1), skip_con=False, trained_model=None):
	"""Main class for setting up a CNN. Returns the compiled model."""

	C = config.Config()
	ActivationLayer = Activation
	activation_args = 'relu'

	nb_classes = len(C.classes_to_include)

	img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 3))

	for layer_depth in range(depth):
		layer1 = cnnc.conv_block(current_layer, base_f*2**layer_depth, strides=1)
		layer1 = layers.Dropout(.1)(layer1)
		layer2 = cnnc.conv_block(layer1, base_f*2**(layer_depth+1))

		if layer_depth < depth - 1:
			current_layer = layers.MaxPooling3D((2,2,2))(layer2)
			levels.append([layer1, layer2, current_layer])
		else:
			current_layer = layer2
			levels.append([layer1, layer2])
			
		model.compile(optimizer='adam', loss='mse')

	return model

def build_autoencoder(pool_sizes=[(2,2,2), (2,2,1)], f=[64,128,128], kernel_size=(3,3,2), dense_units=100,
				lr=.001, trained_model=None, first_layer=0, last_layer=0):
	C = config.Config()

	ActivationLayer = Activation
	activation_args = 'relu'

	nb_classes = len(C.classes_to_include)

	img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 3))
	x = Reshape((C.dims[0], C.dims[1], C.dims[2], 3, 1))(img)
	x = Permute((4,1,2,3,5))(x)

	for layer_num in range(len(f)):
		x = cnnc.td_conv_block(x, f[layer_num])
		if layer_num == 0:
			x = TimeDistributed(layers.MaxPooling3D(pool_sizes[0]))(x)

	if last_layer == 0:
		for layer_num in range(len(f)):
			x = layers.Conv3DTranspose(filters=f[-1-layer_num], kernel_size=[3,3,2], strides=1)(x)
			#x = layers.BatchNormalization(axis=4)(x)
			#x = layers.BatchNormalization(axis=4)(x)
		x = layers.Conv3DTranspose(filters=f[-1-layer_num], kernel_size=[3,3,2], strides=2)(x)
		x = layers.Dense(3, activation='sigmoid')(x)
		model = Model(inputs, x)

		if first_layer == 0:
			model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
		else:
			num_l = len(model.layers)
			dl = len(trained_model.layers)-num_l

			for l in range(num_l-1, 0, -1):
				model.layers[l].set_weights(trained_model.layers[l+dl].get_weights())

	else:
		model = Model(img, filter_weights)
		model.compile(optimizer=Adam(lr=lr), loss='mse', metrics=['accuracy'])

		for l in range(1,len(model.layers)):
			model.layers[l].set_weights(trained_model.layers[l].get_weights())

	return model

def build_model_dropout(trained_model, dropout, last_layer="final", dilation_rate=(1,1,1), padding=['same', 'valid'], pool_sizes = [(2,2,2), (2,2,1)],
	activation_type='relu', f=[64,128,128], kernel_size=(3,3,2), dense_units=100):
	"""Sets up CNN with pretrained weights"""

	C = config.Config()

	ActivationLayer = Activation
	activation_args = 'relu'

	nb_classes = len(C.classes_to_include)

	img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 3))

	art_x = Lambda(lambda x : K.expand_dims(x[:,:,:,:,0], axis=4))(img)
	art_x = layers.Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0], trainable=False)(art_x)
	art_x = BatchNormalization(trainable=False)(art_x)
	art_x = ActivationLayer(activation_args)(art_x)

	ven_x = Lambda(lambda x : K.expand_dims(x[:,:,:,:,1], axis=4))(img)
	ven_x = layers.Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0], trainable=False)(ven_x)
	ven_x = BatchNormalization(trainable=False)(ven_x)
	ven_x = ActivationLayer(activation_args)(ven_x)

	eq_x = Lambda(lambda x : K.expand_dims(x[:,:,:,:,2], axis=4))(img)
	eq_x = layers.Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0], trainable=False)(eq_x)
	eq_x = BatchNormalization(trainable=False)(eq_x)
	eq_x = ActivationLayer(activation_args)(eq_x)

	x = Concatenate(axis=4)([art_x, ven_x, eq_x])
	#x = Lambda(lambda x: K.dropout(x, level=dropout))(x)
	x = layers.MaxPooling3D(pool_sizes[0])(x)

	for layer_num in range(1,len(f)):
		x = layers.Conv3D(filters=f[layer_num], kernel_size=kernel_size, padding=padding[1], trainable=False)(x)
		x = BatchNormalization(trainable=False)(x)
		x = ActivationLayer(activation_args)(x)
		x = Lambda(lambda x: K.dropout(x, level=dropout))(x)

	x = layers.MaxPooling3D(pool_sizes[1])(x)
	x = Flatten()(x)
	x = Dense(dense_units, trainable=False)(x)
	x = BatchNormalization(trainable=False)(x)
	x = ActivationLayer(activation_args)(x)

	if last_layer == "final":
		x = Lambda(lambda x: K.dropout(x, level=dropout))(x)
		x = Dense(nb_classes, activation='softmax')(x)

	model_pretrain = Model(img, x)
	model_pretrain.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	for l in range(1,len(model_pretrain.layers)):
		if type(model_pretrain.layers[l]) == type(trained_model.layers[l]):
			model_pretrain.layers[l].set_weights(trained_model.layers[l].get_weights())
		else:
			model_pretrain.layers[l].set_weights(trained_model.layers[l-1].get_weights())

	return model_pretrain

####################################
### Load Data
####################################

def get_cnn_data(n=4, n_art=0, run_2d=False, Z_test_fixed=None, verbose=False):
	"""Subroutine to run CNN
	n is number of real samples, n_art is number of artificial samples
	Z_test is filenames"""
	importlib.reload(config)
	C = config.Config()

	nb_classes = len(C.classes_to_include)
	orig_data_dict, num_samples = _collect_unaug_data()

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
		for cls in C.classes_to_include:
			orders[cls] = list(set(range(num_samples[cls])).difference(list(orders[cls][0]))) + list(orders[cls][0])

	for cls in orig_data_dict:
		cls_num = C.classes_to_include.index(cls)

		if C.train_frac is None:
			train_samples[cls] = num_samples[cls] - C.test_num
		else:
			train_samples[cls] = round(num_samples[cls]*C.train_frac)
		
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


	Y_test = np_utils.to_categorical(Y_test, nb_classes)
	Y_train_orig = np_utils.to_categorical(Y_train_orig, nb_classes)
	if C.dual_img_inputs or C.non_imaging_inputs:
		X_test = [np.array(X_test), np.array(X2_test)]
		X_train_orig = [np.array(X_train_orig), np.array(X2_train_orig)]
	else:
		X_test = np.array(X_test)
		X_train_orig = np.array(X_train_orig)

	Y_test = np.array(Y_test)
	Y_train_orig = np.array(Y_train_orig)

	Z_test = np.array(Z_test)
	Z_train_orig = np.array(Z_train_orig)

	if run_2d:
		train_generator = _train_generator_func_2d(test_ids, n=n, n_art=n_art)
	else:
		train_generator = _train_generator_func(test_ids, n=n, n_art=n_art)

	return X_test, Y_test, train_generator, num_samples, [X_train_orig, Y_train_orig], [Z_test, Z_train_orig]

def load_data_capsnet(n=2, Z_test_fixed=None):
	C = config.Config()

	nb_classes = len(C.classes_to_include)
	orig_data_dict, num_samples = _collect_unaug_data()

	test_ids = {} #filenames of test set
	X_test = []
	Y_test = []
	Z_test = []

	train_samples = {}


	if Z_test_fixed is not None:
		orders = {cls: np.where(np.isin(orig_data_dict[cls][1], Z_test_fixed)) for cls in orig_data_dict}
		for cls in C.classes_to_include:
			orders[cls] = list(set(range(num_samples[cls])).difference(list(orders[cls][0]))) + list(orders[cls][0])

	for cls in orig_data_dict:
		cls_num = C.classes_to_include.index(cls)

		if C.train_frac is None:
			train_samples[cls] = num_samples[cls] - C.test_num
		else:
			train_samples[cls] = round(num_samples[cls]*C.train_frac)
		
		if Z_test_fixed is None:
			order = np.random.permutation(list(range(num_samples[cls])))
		else:
			order = orders[cls]

		X_test = X_test + list(orig_data_dict[cls][0][order[train_samples[cls]:]])
		test_ids[cls] = list(orig_data_dict[cls][-1][order[train_samples[cls]:]])
		Y_test += [cls_num] * (num_samples[cls] - train_samples[cls])
		Z_test = Z_test + test_ids[cls]

	Y_test = np_utils.to_categorical(Y_test, nb_classes)
	X_test = np.array(X_test)
	Z_test = np.array(Z_test)

	return _train_gen_capsnet(test_ids, n=n), (X_test, Y_test), Z_test

####################################
### Predict demographics
####################################

def build_cnn_demogr(optimizer='adam', dropout=[0.1,0.1], pool_sizes=[(2,2,2), (2,2,2)],
			f=[32,32,64,64,64,64], dense_units=100, kernel_size=(3,3,2), trained_model=None):
	C = config.Config()
	ActivationLayer = Activation
	activation_args = 'relu'

	img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 3))
	x = Reshape((C.dims[0], C.dims[1], C.dims[2], 3, 1))(img)
	x = Permute((4,1,2,3,5))(x)

	for layer_num in range(len(f)):
		x = cnnc.td_conv_block(x, f[layer_num])
		if layer_num == 1:
			x = layers.TimeDistributed(layers.MaxPooling3D(pool_sizes[0]))(x)

	if trained_model is None:
		x = layers.Dropout(dropout[0])(x)
		x = layers.TimeDistributed(layers.MaxPooling3D(pool_sizes[1]))(x)
		x = layers.TimeDistributed(Flatten())(x)

		x = layers.SimpleRNN(dense_units)(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(dropout[1])(x)
		x = layers.Dense(dense_units, activation='relu')(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(dropout[1])(x)

		pred_sex = Dense(2, activation='softmax')(x)
		pred_age = Dense(1, activation='relu')(x)
		model = Model(img, [pred_sex, pred_age])

		model.compile(optimizer=optimizer, loss=['binary_crossentropy', 'mse'],
						loss_weights=[1., 2.], metrics=['accuracy'])
	else:
		model = Model(img, x)

		for l in range(1,len(model.layers)):
			model.layers[l].set_weights(trained_model.layers[l].get_weights())

	return model

def get_cnn_demogr(n=4):
	C = config.Config()
	orig_data_dict, num_samples = _collect_unaug_demogr()

	test_ids = {} #filenames of test set
	X_test = []
	X2_test = []
	Y_test = []
	X_train_orig = []
	X2_train_orig = []
	Y_train_orig = []

	train_samples = {}
	for cls in orig_data_dict:
		cls_num = C.classes_to_include.index(cls)

		if C.train_frac is None:
			train_samples[cls] = num_samples[cls] - C.test_num
		else:
			train_samples[cls] = round(num_samples[cls]*C.train_frac)
		
		order = np.random.permutation(list(range(num_samples[cls])))
		
		test_ids[cls] = list(orig_data_dict[cls][-1][order[train_samples[cls]:]])

		X_test += list(orig_data_dict[cls][0][order[train_samples[cls]:]])
		Y_test += list(orig_data_dict[cls][1][order[train_samples[cls]:]])
		
		X_train_orig += list(orig_data_dict[cls][0][order[:train_samples[cls]]])
		Y_train_orig += list(orig_data_dict[cls][1][order[:train_samples[cls]]])

	X_test = np.array(X_test)
	X_train_orig = np.array(X_train_orig)

	Y_test = np.array(Y_test)
	Y_test = [np_utils.to_categorical(Y_test[:,0], 2), Y_test[:,1]]
	Y_train_orig = np.array(Y_train_orig)
	Y_train_orig = [np_utils.to_categorical(Y_train_orig[:,0], 2), Y_train_orig[:,1]]

	train_generator = _train_gen_demogr(test_ids, n=n)

	return X_test, Y_test, train_generator, num_samples, [X_train_orig, Y_train_orig]

def _collect_unaug_demogr():
	"""Return dictionary pointing to X (img data) and Z (filenames) and dictionary storing number of samples of each class."""

	C = config.Config()
	orig_data_dict = {}
	voi_df = drm.get_voi_dfs()[0]
	voi_df = voi_df[voi_df["run_num"] <= C.test_run_num]
	patient_info_df = pd.read_csv(C.patient_info_path)
	patient_info_df["AccNum"] = patient_info_df["AccNum"].astype(str)
	num_samples = {}

	for cls in C.classes_to_include:
		x = np.empty((10000, C.dims[0], C.dims[1], C.dims[2], C.nb_channels))
		x2 = np.empty((10000, 2))
		z = []

		for index, lesion_id in enumerate(voi_df[voi_df["cls"] == cls].index):
			img_path = os.path.join(C.orig_dir, cls, lesion_id+".npy")
			try:
				x[index] = np.load(img_path)
				if C.post_scale > 0:
					x[index] = tr.normalize_intensity(x[index], 1., -1., C.post_scale)
			except:
				raise ValueError(img_path + " not found")
			z.append(lesion_id)

			voi_row = voi_df.loc[lesion_id]
			patient_row = patient_info_df[patient_info_df["AccNum"] == voi_row["acc_num"]]
			try:
				age = float(patient_row["AgeAtImaging"].values[0])
				sex = 0 if patient_row["Sex"].values[0]=="M" else 1
			except:
				raise ValueError(str(lesion_id))
			x2[index] = [sex, age]

		x.resize((index+1, C.dims[0], C.dims[1], C.dims[2], C.nb_channels)) #shrink first dimension to fit
		x2.resize((index+1, *x2.shape[1:]))
		orig_data_dict[cls] = [x, x2, np.array(z)]

		num_samples[cls] = index + 1

	return orig_data_dict, num_samples

def _train_gen_demogr(test_ids, n):
	"""n is the number of samples from each class, n_art is the number of artificial samples"""

	C = config.Config()
	voi_df = drm.get_voi_dfs()[0]
	voi_df = voi_df[voi_df["run_num"] <= C.test_run_num]
	patient_info_df = pd.read_csv(C.patient_info_path)
	patient_info_df["AccNum"] = patient_info_df["AccNum"].astype(str)

	while True:
		x1 = np.empty((n*C.nb_classes, C.dims[0], C.dims[1], C.dims[2], C.nb_channels))
		y = np.zeros((n*C.nb_classes, 2))

		train_cnt = 0
		for cls in C.classes_to_include:
			img_fns = os.listdir(C.aug_dir+cls)
			while n > train_cnt:
				img_fn = random.choice(img_fns)
				lesion_id = img_fn[:img_fn.rfind('_')]
				if lesion_id not in test_ids[cls] and lesion_id in voi_df.index:
					x1[train_cnt] = np.load(C.aug_dir+cls+"\\"+img_fn)

					voi_row = voi_df.loc[lesion_id]
					patient_row = patient_info_df[patient_info_df["AccNum"] == voi_row["acc_num"]]
					try:
						age = float(patient_row["AgeAtImaging"].values[0])
						sex = 0 if patient_row["Sex"].values[0]=="M" else 1
					except:
						raise ValueError(str(lesion_id))
					y[train_cnt] = [sex, age]
					
					train_cnt += 1
					if train_cnt % n == 0:
						break

		y = [np_utils.to_categorical(y[:,0], 2), y[:,1]]
		yield np.array(x1), y

####################################
### Training Submodules
####################################

def _train_gen_capsnet(test_ids, n=4):
	"""n is the number of samples from each class, n_art is the number of artificial samples"""
	n_art=0

	C = config.Config()

	num_classes = len(C.classes_to_include)
	while True:
		x1 = np.empty(((n+n_art)*num_classes, C.dims[0], C.dims[1], C.dims[2], C.nb_channels))
		y = np.zeros(((n+n_art)*num_classes, num_classes))

		train_cnt = 0
		for cls in C.classes_to_include:
			img_fns = os.listdir(C.aug_dir+cls)
			while n > 0:
				img_fn = random.choice(img_fns)
				lesion_id = img_fn[:img_fn.rfind('_')]
				if lesion_id not in test_ids[cls]:
					x1[train_cnt] = np.load(C.aug_dir+cls+"\\"+img_fn)
					
					y[train_cnt][C.classes_to_include.index(cls)] = 1
					
					train_cnt += 1
					if train_cnt % (n+n_art) == 0:
						break

		x_batch = np.array(x1)
		y_batch = np.array(y)
		yield ([x_batch, y_batch], [y_batch, x_batch])

def _train_generator_func(test_ids, n=12, n_art=0):
	"""n is the number of samples from each class, n_art is the number of artificial samples"""

	C = config.Config()

	voi_df = drm.get_voi_dfs()[0]

	#avg_X2 = {}
	#for cls in orig_data_dict:
	#	avg_X2[cls] = np.mean(orig_data_dict[cls][1], axis=0)
	
	if C.non_imaging_inputs:
		patient_info_df = pd.read_csv(C.patient_info_path)
		patient_info_df["AccNum"] = patient_info_df["AccNum"].astype(str)

	num_classes = len(C.classes_to_include)
	while True:
		x1 = np.empty(((n+n_art)*num_classes, C.dims[0], C.dims[1], C.dims[2], C.nb_channels))
		y = np.zeros(((n+n_art)*num_classes, num_classes))

		if C.dual_img_inputs:
			x2 = np.empty(((n+n_art)*num_classes, *C.context_dims, C.nb_channels))
		elif C.non_imaging_inputs:
			x2 = np.empty(((n+n_art)*num_classes, C.num_non_image_inputs))

		train_cnt = 0
		for cls in C.classes_to_include:
			if n_art > 0:
				img_fns = os.listdir(C.artif_dir+cls)
				for _ in range(n_art):
					img_fn = random.choice(img_fns)
					x1[train_cnt] = np.load(C.artif_dir + cls + "\\" + img_fn)
					#x2[train_cnt] = avg_X2[cls]
					y[train_cnt][C.classes_to_include.index(cls)] = 1

					train_cnt += 1

			img_fns = os.listdir(C.aug_dir+cls)
			while n > 0:
				img_fn = random.choice(img_fns)
				lesion_id = img_fn[:img_fn.rfind('_')]
				if lesion_id not in test_ids[cls]:
					x1[train_cnt] = np.load(C.aug_dir+cls+"\\"+img_fn)
					try:
						if C.post_scale > 0:
							x1[train_cnt] = tr.normalize_intensity(x1[train_cnt], 1., -1., C.post_scale)
					except:
						vm.reset_accnum(lesion_id[:lesion_id.find('_')])

					if C.dual_img_inputs:
						tmp = np.load(os.path.join(C.crops_dir, cls, lesion_id+".npy"))
						x2[train_cnt] = tr.rescale_img(tmp, C.context_dims)[0]

					elif C.non_imaging_inputs:
						voi_row = voi_df.loc[lesion_id]
						patient_row = patient_info_df[patient_info_df["AccNum"] == voi_row["acc_num"]]
						x2[train_cnt] = get_non_img_inputs(voi_row, patient_row)
					
					y[train_cnt][C.classes_to_include.index(cls)] = 1
					
					train_cnt += 1
					if train_cnt % (n+n_art) == 0:
						break

		if C.dual_img_inputs or C.non_imaging_inputs:
			yield [np.array(x1), np.array(x2)], np.array(y) #[np.array(x1), np.array(x2)], np.array(y) #
		else:
			yield np.array(x1), np.array(y) #[np.array(x1), np.array(x2)], np.array(y) #

def _train_generator_func_2d(train_ids, voi_df, avg_X2, n=12, n_art=0, C=None):
	"""n is the number of samples from each class, n_art is the number of artificial samples"""

	classes_to_include = C.classes_to_include
	if C is None:
		C = config.Config()
	
	num_classes = len(classes_to_include)
	while True:
		x1 = np.empty(((n+n_art)*num_classes, C.dims[0], C.dims[1], C.nb_channels))
		x2 = np.empty(((n+n_art)*num_classes, C.num_non_image_inputs))
		y = np.zeros(((n+n_art)*num_classes, num_classes))

		train_cnt = 0
		for cls in classes_to_include:
			if n_art>0:
				img_fns = os.listdir(C.artif_dir+cls)
				for _ in range(n_art):
					img_fn = random.choice(img_fns)
					temp = np.load(C.artif_dir + cls + "\\" + img_fn)
					x1[train_cnt] = temp[:,:,temp.shape[2]//2,:]
					x2[train_cnt] = avg_X2[cls]
					y[train_cnt][C.classes_to_include.index(cls)] = 1

					train_cnt += 1

			img_fns = os.listdir(C.aug_dir+cls)
			while n>0:
				img_fn = random.choice(img_fns)
				if img_fn[:img_fn.rfind('_')] + ".npy" in train_ids[cls]:
					temp = np.load(C.aug_dir+cls+"\\"+img_fn)
					x1[train_cnt] = temp[:,:,temp.shape[2]//2,:]

					row = voi_df[(voi_df["Filename"] == img_fn[:img_fn.find('_')] + ".npy") &
								 (voi_df["lesion_num"] == int(img_fn[img_fn.find('_')+1:img_fn.rfind('_')]))]
					x2[train_cnt] = [(float(row["real_dx"]) * float(row["real_dy"]) * float(row["real_dz"])) ** (1/3) / 50,
										max(float(row["real_dx"]), float(row["real_dy"])) / float(row["real_dz"])]
					
					y[train_cnt][C.classes_to_include.index(cls)] = 1
					
					train_cnt += 1
					if train_cnt % (n+n_art) == 0:
						break
			
		
		yield _separate_phases([np.array(x1), np.array(x2)]), np.array(y) #[np.array(x1), np.array(x2)], np.array(y) #

def _separate_phases(X, non_imaging_inputs=False):
	"""Assumes X[0] contains imaging and X[1] contains dimension data.
	Reformats such that X[0:2] has 3 phases and X[3] contains dimension data.
	Image data still is 5D (nb_samples, 3D, 1 channel).
	Handles both 2D and 3D images"""
	
	if non_imaging_inputs:
		dim_data = copy.deepcopy(X[1])
		img_data = X[0]
		
		if len(X[0].shape)==5:
			X[1] = np.expand_dims(X[0][:,:,:,:,1], axis=4)
			X += [np.expand_dims(X[0][:,:,:,:,2], axis=4)]
			X += [dim_data]
			X[0] = np.expand_dims(X[0][:,:,:,:,0], axis=4)
		
		else:
			X[1] = np.expand_dims(X[0][:,:,:,1], axis=3)
			X += [np.expand_dims(X[0][:,:,:,2], axis=3)]
			X += [dim_data]
			X[0] = np.expand_dims(X[0][:,:,:,0], axis=3)
	
	else:
		X = np.array(X)
		if len(X.shape)==5:
			X = [np.expand_dims(X[:,:,:,:,0], axis=4), np.expand_dims(X[:,:,:,:,1], axis=4), np.expand_dims(X[:,:,:,:,2], axis=4)]
		else:
			X = [np.expand_dims(X[:,:,:,0], axis=3), np.expand_dims(X[:,:,:,1], axis=3), np.expand_dims(X[:,:,:,2], axis=3)]

	return X

def _collect_unaug_data():
	"""Return dictionary pointing to X (img data) and Z (filenames) and dictionary storing number of samples of each class."""

	C = config.Config()
	orig_data_dict = {}
	num_samples = {}
	voi_df = drm.get_voi_dfs()[0]
	voi_df = voi_df[voi_df["run_num"] <= C.test_run_num]

	if C.non_imaging_inputs:
		patient_info_df = pd.read_csv(C.patient_info_path)
		patient_info_df["AccNum"] = patient_info_df["AccNum"].astype(str)

	for cls in C.classes_to_include:
		x = np.empty((10000, C.dims[0], C.dims[1], C.dims[2], C.nb_channels))
		z = []

		if C.dual_img_inputs:
			x2 = np.empty((10000, *C.context_dims, C.nb_channels))
		elif C.non_imaging_inputs:
			x2 = np.empty((10000, C.num_non_image_inputs))

		for index, lesion_id in enumerate(voi_df[voi_df["cls"] == cls].index):
			img_path = os.path.join(C.orig_dir, cls, lesion_id+".npy")
			try:
				x[index] = np.load(img_path)
				if C.post_scale > 0:
					x[index] = tr.normalize_intensity(x[index], 1., -1., C.post_scale)
			except:
				raise ValueError(img_path + " not found")
			z.append(lesion_id)
			
			if C.dual_img_inputs:
				tmp = np.load(os.path.join(C.crops_dir, cls, lesion_id+".npy"))
				x2[index] = tr.rescale_img(tmp, C.context_dims)[0]

			elif C.non_imaging_inputs:
				voi_row = voi_df.loc[lesion_id]
				patient_row = patient_info_df[patient_info_df["AccNum"] == voi_row["acc_num"]]
				x2[index] = get_non_img_inputs(voi_row, patient_row)

		x.resize((index+1, C.dims[0], C.dims[1], C.dims[2], C.nb_channels)) #shrink first dimension to fit
		if C.dual_img_inputs or C.non_imaging_inputs:
			x2.resize((index+1, *x2.shape[1:]))
			orig_data_dict[cls] = [x, x2, np.array(z)]
		else:
			orig_data_dict[cls] = [x, np.array(z)]

		num_samples[cls] = index + 1
		
	return orig_data_dict, num_samples

def get_non_img_inputs(voi_info, patient_info):
	side_length = ((float(voi_info["real_dx"]) * float(voi_info["real_dy"]) * float(voi_info["real_dz"])) ** (1/3) - 26.98) / 14.78
	aspect_ratio = max(float(voi_info["real_dx"]), float(voi_info["real_dy"])) / float(voi_info["real_dz"])
	age = (float(patient_info["AgeAtImaging"].values[0]) - 56.58) / 13.24
	sex = 0 if patient_info["Sex"].values[0]=="M" else 1

	return [side_length, age, sex]

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Convert DICOMs to npy files and transfer voi coordinates from excel to csv.')
	parser.add_argument('-m', '--max_runs', type=int, help='max number of runs to allow')
	#parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite')
	args = parser.parse_args()

	run_fixed_hyperparams(max_runs=args.max_runs)