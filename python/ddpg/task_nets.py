"""
Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

import copy
import importlib
import math
import os
import random
import time
from collections import deque
from os.path import *

import keras.backend as K
import keras.layers as layers
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten, Input, Lambda, merge, Lambda, Layer
from keras.layers.normalization import BatchNormalization
from keras.models import Model, model_from_json, Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.initializers import Constant
from keras.models import Model
from tensorflow.contrib import distributions

import config
import dr_methods as drm
import feature_interpretation as cnna
import niftiutils.cnn_components as cnnc
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import spatial_transformer as st
import voi_methods as vm

def get_models(lr):
	C = config.Config()

	prediction_model = unet_cls()
	inp = Input(prediction_model.input_shape[1:], name='inp')
	y1_pred, y2_pred = prediction_model(inp)
	y1_true = Input((*prediction_model.input_shape[1:-1], 2), name='y1_true') #prediction_model.output_shape[0][1:]
	y2_true = Input((len(C.cls_names),), name='y2_true')
	out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, y1_pred, y2_pred])
	trainable_model = Model([inp, y1_true, y2_true], out)
	#prediction_model.compile(optimizer=None, loss=None)
	trainable_model.compile(optimizer=Adam(lr), loss=None)
	return prediction_model, trainable_model

def unet_cls(nb_segs=2, optimizer='adam', depth=3, base_f=32, dropout=.1, lr=.001):
	importlib.reload(cnnc)
	C = config.Config()
	levels = []

	img = Input(shape=(*C.dims, 3))

	# initial weights
	b = np.zeros((3, 4), dtype='float32')
	b[0, 0] = 1
	b[1, 1] = 1
	b[2, 2] = 1
	W = np.zeros((64, 12), dtype='float32')
	weights = [W, b.flatten()]

	locnet = Sequential()
	locnet.add(layers.MaxPooling3D(2))
	locnet.add(layers.Conv3D(32, 3, activation='relu')) #input_shape=(C.dims[0], C.dims[1], C.dims[2], 1)
	locnet.add(layers.Conv3D(32, 3, activation='relu'))
	locnet.add(Flatten())
	locnet.add(Dense(64, activation='relu'))
	locnet.add(Dense(12, weights=weights))

	v_Tx = Lambda(lambda x: K.expand_dims(x[...,1],-1))(img)
	e_Tx = Lambda(lambda x: K.expand_dims(x[...,2],-1))(img)
	v_Tx = st.SpatialTransformer(localization_net=locnet, downsample_factor=1)(v_Tx)
	e_Tx = st.SpatialTransformer(localization_net=locnet, downsample_factor=1)(e_Tx)

	reg_img = Lambda(lambda x: K.stack([x[0][...,0], x[1][...,0], x[2][...,0]], -1))([img, v_Tx, e_Tx])
	reg_img = Lambda(lambda x: K.stack([x[...,0], x[...,1]-x[...,0], x[...,2]-x[...,0]], -1))(reg_img)
	current_layer = reg_img
	
	for layer_depth in range(depth):
		layer1 = cnnc.conv_block(current_layer, base_f*2**layer_depth, strides=1)
		dropl = layers.Dropout(dropout)(layer1)
		layer2 = cnnc.conv_block(dropl, base_f*2**(layer_depth+1))

		if layer_depth < depth - 1:
			current_layer = layers.MaxPooling3D(2)(layer2)
			levels.append([layer1, layer2, current_layer])
		else:
			current_layer = layer2
			levels.append([layer1, layer2])

	cls_layer = layers.Flatten()(current_layer)
	cls_layer = layers.Dense(128, activation='relu')(cls_layer)
	cls_layer = layers.Dense(len(C.cls_names)+1)(cls_layer)
			
	for layer_depth in range(depth-2, -1, -1):
		up_convolution = cnnc.up_conv_block(pool_size=2, deconvolution=False,
											f=current_layer._keras_shape[1])(current_layer)
		concat = layers.Concatenate(axis=-1)([up_convolution, levels[layer_depth][1]])
		current_layer = cnnc.conv_block(concat, levels[layer_depth][1]._keras_shape[-1]//2)
		current_layer = layers.Dropout(dropout)(current_layer)
		current_layer = cnnc.conv_block(current_layer, levels[layer_depth][1]._keras_shape[-1]//2)

	segs = layers.Conv3D(nb_segs+1, (1,1,1))(current_layer)

	model = Model(img, [segs, cls_layer]) #logits + logvar

	return model

def hetero_cls_loss(true, pred_var, T=500, num_classes=3):
	# Bayesian categorical cross entropy.
	# https://github.com/kyle-dorman/bayesian-neural-network-blogpost
	# N data points, C classes, T monte carlo simulations
	# true - true values. Shape: (N, C)
	# pred_var - predicted logit values and log variance. Shape: (N, C + 1)
	# returns - loss (N,)

	true = K.reshape(true, [-1, num_classes])
	pred_var = K.reshape(pred_var, [-1, num_classes+1])
	std = K.sqrt(K.exp(pred_var[:, num_classes:])) # shape: (N,)
	pred = pred_var[:, :num_classes] # shape: (N, C)

	dist = distributions.Normal(loc=K.zeros_like(std), scale=std) # shape: (T,)
	std_samples = K.transpose(dist.sample(num_classes))
	distorted_loss = K.categorical_crossentropy(true, pred + std_samples, from_logits=True)

	return K.mean(distorted_loss, axis=-1)

	# shape: (N,)
	variance = K.exp(pred_var[:, num_classes])
	variance_depressor = K.exp(variance) - K.ones_like(variance)
	# shape: (N,)
	undistorted_loss = K.categorical_crossentropy(true, pred, from_logits=True)

	iterable = K.variable(np.ones(T))
	monte_carlo_results = K.map_fn(gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss, num_classes), iterable, name='monte_carlo_results')

	variance_loss = K.mean(monte_carlo_results, axis=0) * undistorted_loss

	return variance_loss + undistorted_loss# + variance_depressor

def gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss, num_classes):
	# for a single monte carlo simulation, 
	#   calculate categorical_crossentropy of 
	#   predicted logit values plus gaussian 
	#   noise vs true values.
	# true - true values. Shape: (N, C)
	# pred - predicted logit values. Shape: (N, C)
	# dist - normal distribution to sample from. Shape: (N, C)
	# undistorted_loss - the crossentropy loss without variance distortion. Shape: (N,)
	# num_classes - the number of classes. C
	# returns - total differences for all classes (N,)
	def map_fn(i):
		std_samples = K.transpose(dist.sample(num_classes))
		distorted_loss = K.categorical_crossentropy(true, pred + std_samples, from_logits=True)
		diff = undistorted_loss - distorted_loss
		return -K.elu(diff)
	return map_fn

class CustomMultiLossLayer(Layer):
	# Custom loss layer
	# https://github.com/yaringal/multi-task-learning-example
	def __init__(self, nb_outputs=2, **kwargs):
		self.nb_outputs = nb_outputs
		super(CustomMultiLossLayer, self).__init__(**kwargs)
		
	def build(self, input_shape=None):
		# initialise log_vars
		self.log_vars = []
		for i in range(self.nb_outputs):
			self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
											  initializer=Constant(0.), trainable=True)]
		super(CustomMultiLossLayer, self).build(input_shape)

	def multi_loss(self, ys_true, ys_pred):
		assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
		loss = 0
		clss = [2,3]
		for y_true, y_pred, log_var, num_classes in zip(ys_true, ys_pred, self.log_vars, clss):
			precision = K.exp(-log_var[0])
			loss += K.sum(precision * hetero_cls_loss(y_true, y_pred,
					num_classes=num_classes) + log_var[0], -1)
		return K.mean(loss)

	def call(self, inputs):
		ys_true = inputs[:self.nb_outputs]
		ys_pred = inputs[self.nb_outputs:]
		loss = self.multi_loss(ys_true, ys_pred)
		self.add_loss(loss, inputs=inputs)
		# We won't actually use the output.
		return inputs#K.concatenate(inputs, -1)
