"""
Converts a nifti file to a numpy array.
Accepts either a single nifti file or a folder of niftis as the input argument.

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

import dr_methods as drm
import voi_methods as vm


def build_cnn_hyperparams(hyperparams):
	C = config.Config()
	if C.dual_img_inputs:
		return build_dual_cnn(optimizer=hyperparams.optimizer, dilation_rate=hyperparams.dilation_rate,
			padding=hyperparams.padding, pool_sizes=hyperparams.pool_sizes, dropout=hyperparams.dropout,
			activation_type=hyperparams.activation_type, f=hyperparams.f, dense_units=hyperparams.dense_units,
			kernel_size=hyperparams.kernel_size)
	else:
		return build_cnn(optimizer=hyperparams.optimizer, dilation_rate=hyperparams.dilation_rate,
			padding=hyperparams.padding, pool_sizes=hyperparams.pool_sizes, dropout=hyperparams.dropout,
			activation_type=hyperparams.activation_type, f=hyperparams.f, dense_units=hyperparams.dense_units,
			kernel_size=hyperparams.kernel_size, merge_layer=hyperparams.merge_layer,
			dual_inputs=C.non_imaging_inputs, run_2d=hyperparams.run_2d, time_dist=hyperparams.time_dist)

def build_cnn(optimizer='adam', dilation_rate=(1,1,1), padding=['same', 'valid'], pool_sizes = [(2,2,2), (2,2,2)],
	dropout=[0.1,0.1], activation_type='relu', f=[64,128,128], dense_units=100, kernel_size=(3,3,2), merge_layer=1,
	dual_inputs=False, run_2d=False, time_dist=True, stride=(1,1,1)):
	"""Main class for setting up a CNN. Returns the compiled model."""

	C = config.Config()

	if activation_type == 'elu':
		ActivationLayer = ELU
		activation_args = 1
	elif activation_type == 'relu':
		ActivationLayer = Activation
		activation_args = 'relu'

	nb_classes = len(C.classes_to_include)

	if not run_2d:
		img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 3))
	else:
		img = Input(shape=(C.dims[0], C.dims[1], 3))

	if merge_layer == 1:
		art_x = Lambda(lambda x : K.expand_dims(x[:,:,:,:,0], axis=4))(img)
		art_x = layers.Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0])(art_x)
		art_x = ActivationLayer(activation_args)(art_x)

		ven_x = Lambda(lambda x : K.expand_dims(x[:,:,:,:,1], axis=4))(img)
		ven_x = layers.Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0])(ven_x)
		ven_x = ActivationLayer(activation_args)(ven_x)

		eq_x = Lambda(lambda x : K.expand_dims(x[:,:,:,:,2], axis=4))(img)
		eq_x = layers.Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0])(eq_x)
		eq_x = ActivationLayer(activation_args)(eq_x)

		x = Concatenate(axis=4)([art_x, ven_x, eq_x])
		x = layers.MaxPooling3D(pool_sizes[0])(x)
		x = BatchNormalization(axis=4)(x)
		#x = Dropout(dropout[0])(x)

		for layer_num in range(1,len(f)):
			x = layers.Conv3D(filters=f[layer_num], kernel_size=kernel_size, padding=padding[1])(x)
			x = BatchNormalization()(x)
			x = ActivationLayer(activation_args)(x)
			x = Dropout(dropout[0])(x)

	elif merge_layer == 0:
		x = img

		if time_dist:
			x = Reshape((C.dims[0], C.dims[1], C.dims[2], 3, 1))(x)
			x = Permute((4,1,2,3,5))(x)

			for layer_num in range(len(f)):
				if layer_num == 1:
					x = layers.TimeDistributed(layers.Conv3D(filters=f[layer_num], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[1]))(x) #, kernel_regularizer=l2(.01)
				#elif layer_num == 0:
				#	x = TimeDistributed(layers.Conv3D(filters=f[layer_num], kernel_size=kernel_size, strides=stride, padding=padding[1]))(x) #, kernel_regularizer=l2(.01)
				else:
					x = layers.TimeDistributed(layers.Conv3D(filters=f[layer_num], kernel_size=kernel_size, padding=padding[1 * (layer_num > 1)]))(x) #, kernel_regularizer=l2(.01)
				x = layers.TimeDistributed(layers.Dropout(dropout[0]))(x)
				x = ActivationLayer(activation_args)(x)
				x = layers.TimeDistributed(layers.BatchNormalization(axis=4))(x)
				if layer_num == 0:
					x = TimeDistributed(layers.MaxPooling3D(pool_sizes[0]))(x)
			
		else:
			for layer_num in range(len(f)):
				x = layers.Conv3D(filters=f[layer_num], kernel_size=kernel_size, padding=padding[1])(x)
				x = BatchNormalization()(x)
				x = ActivationLayer(activation_args)(x)
				x = Dropout(dropout[0])(x)

	if time_dist:
		x = layers.TimeDistributed(layers.MaxPooling3D(pool_sizes[1]))(x)
		x = layers.TimeDistributed(Flatten())(x)

		#x = SimpleRNN(128, return_sequences=True)(x)
		x = layers.SimpleRNN(dense_units)(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(dropout[1])(x)
	else:
		x = layers.MaxPooling3D(pool_sizes[1])(x)
		x = Flatten()(x)

		x = Dense(dense_units)(x)#, kernel_initializer='normal', kernel_regularizer=l2(.01), kernel_constraint=max_norm(3.))(x)
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

	pred_class = Dense(nb_classes, activation='softmax')(x)

	if not dual_inputs:
		model = Model(img, pred_class)
	else:
		model = Model([img, non_img_inputs], pred_class)
	
	#optim = Adam(lr=0.01)#5, decay=0.001)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	return model

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
	cx = layers.Reshape((*C.context_dims, 3, 1))(context_img)
	cx = layers.Permute((4,1,2,3,5))(cx)
	cx = layers.TimeDistributed(layers.Conv3D(filters=128, kernel_size=(8,8,3), dilation_rate=(2,2,2), padding='valid', activation='relu'))(cx)
	cx = layers.TimeDistributed(layers.MaxPooling3D((2,2,2)))(cx)
	cx = layers.TimeDistributed(BatchNormalization(axis=4))(cx)
	cx = layers.TimeDistributed(layers.Conv3D(filters=128, kernel_size=(6,6,3), padding='valid', activation='relu'))(cx)
	cx = layers.TimeDistributed(BatchNormalization(axis=4))(cx)
	cx = layers.TimeDistributed(Dropout(dropout[0]))(cx)
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
		x = ActivationLayer(activation_args)(x)
		x = layers.TimeDistributed(BatchNormalization(axis=4))(x)
		if layer_num == 0:
			x = layers.TimeDistributed(layers.MaxPooling3D(pool_sizes[0]))(x)

	x = layers.TimeDistributed(layers.MaxPooling3D(pool_sizes[1]))(x)

	x = layers.TimeDistributed(layers.Flatten())(x)
	x = layers.Concatenate(axis=2)([x, cx])

	#x = SimpleRNN(128, return_sequences=True)(x)
	x = layers.SimpleRNN(dense_units)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(dropout[1])(x)

	pred_class = layers.Dense(nb_classes, activation='softmax')(x)

	model = Model([img, context_img], pred_class)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	return model

def build_pretrain_model(trained_model, dilation_rate=(1,1,1), padding=['same', 'valid'], pool_sizes = [(2,2,2), (2,2,1)],
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
	x = layers.MaxPooling3D(pool_sizes[0])(x)

	for layer_num in range(1,len(f)):
		x = layers.Conv3D(filters=f[layer_num], kernel_size=kernel_size, padding=padding[1], trainable=False)(x)
		x = BatchNormalization(trainable=False)(x)
		x = ActivationLayer(activation_args)(x)
		x = Dropout(0)(x)

	x = layers.MaxPooling3D(pool_sizes[1])(x)
	#x = layers.AveragePooling3D((4,4,4))(x)
	#filter_weights = Flatten()(x)

	x = Flatten()(x)

	x = Dense(dense_units, trainable=False)(x)
	x = BatchNormalization(trainable=False)(x)
	filter_weights = ActivationLayer(activation_args)(x)

	model_pretrain = Model(img, filter_weights)
	model_pretrain.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	for l in range(1,len(model_pretrain.layers)):
		model_pretrain.layers[l].set_weights(trained_model.layers[l].get_weights())

	return model_pretrain

def build_model_forced_dropout(trained_model, dropout, dilation_rate=(1,1,1), padding=['same', 'valid'], pool_sizes = [(2,2,2), (2,2,1)],
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
	x = Lambda(lambda x: K.dropout(x, level=dropout))(x)

	pred_class = Dense(nb_classes, activation='softmax')(x)

	model_pretrain = Model(img, pred_class)
	model_pretrain.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	for l in range(1,len(model_pretrain.layers)):
		model_pretrain.layers[l].set_weights(trained_model.layers[l].get_weights())

	return model_pretrain

def get_cnn_data(n=4, n_art=0, run_2d=False, Z_test_fixed=None, verbose=False):
	"""Subroutine to run CNN
	n is number of real samples, n_art is number of artificial samples
	Z_test is filenames"""

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
					if C.hard_scale:
						x1[train_cnt] = vm.scale_intensity(x1[train_cnt], 1, max_int=2, keep_min=False)
					
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
					if C.hard_scale:
						x1[train_cnt] = vm.scale_intensity(x1[train_cnt], 1, max_int=2, keep_min=False)

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
	#voi_df = voi_df[voi_df["run_num"] <= C.test_run_num]
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
				if C.hard_scale:
					x[index] = vm.scale_intensity(x[index], 1, max_int=2)#, keep_min=True)
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