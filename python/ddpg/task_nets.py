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
from keras.layers import Dense, Flatten, Input, Lambda, merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model, model_from_json, Sequential
from keras.optimizers import Adam
from keras.utils import np_utils

import config
import dr_methods as drm
import feature_interpretation as cnna
import niftiutils.cnn_components as cnnc
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import spatial_transformer as st
import voi_methods as vm

class TaskNets():
	def __init__(self):
		# Input shape
		C = config.Config()
		optimizer = Adam(0.001)#Adam(0.0002, 0.5)
		return

		# Build and compile the discriminator
		self.reg_net = self.build_reg_network()
		self.compile(optimizer=optimizer, loss='mse') #local difference and l1 terms

		# Build the generator
		self.unet = self.build_unet_network()
		self.unet.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

		self.classifier = self.build_unet_network()
		self.classifier.compile(loss='categorical_crossentropy', optimizer=optimizer)

		# The generator takes projections as input and generates 3D imgs
		img = Input(C.dims)
		reg_img = self.reg_net(img)
		self.reg_img.trainable = False
		segs = self.unet(reg_img)
		self.unet_reg = Model(img, segs)
		self.unet_reg.compile(loss='binary_crossentropy', optimizer=optimizer)

	def build_unet_network(state_size,action_dim):
		reg_img, x = common_network()
		x, segs = unet()
		model = Model(reg_img, segs)

		return model, model.trainable_weights, S

	def build_cls_network(state_size,action_dim):
		reg_img, x = common_network()
		x = layers.Dense(128, activation='relu')(x)
		segs = Dense(len(C.cls_names), activation='softmax')(x)
		model = Model(input=S,output=V)

		return model, model.trainable_weights, S


	def build_reg_network(f=64):
		importlib.reload(st)
		C = config.Config()
		img = Input(shape=(*C.dims, 3))

		# initial weights
		b = np.zeros((3, 4), dtype='float32')
		b[0, 0] = 1
		b[1, 1] = 1
		b[2, 2] = 1
		W = np.zeros((64, 12), dtype='float32')
		weights = [W, b.flatten()]

		locnet = Sequential()
		locnet.add(layers.Conv3D(32, 3, activation='relu')) #input_shape=(C.dims[0], C.dims[1], C.dims[2], 1)
		locnet.add(layers.Conv3D(32, 3, activation='relu'))
		locnet.add(Flatten())
		locnet.add(Dense(64, activation='relu'))
		locnet.add(Dense(12, weights=weights))

		v_Tx = Lambda(lambda x: K.expand_dims(x[...,1],-1))(img)
		v_Tx = st.SpatialTransformer(localization_net=locnet, downsample_factor=1)(v_Tx)

		e_Tx = Lambda(lambda x: K.expand_dims(x[...,2],-1))(img)
		e_Tx = st.SpatialTransformer(localization_net=locnet, downsample_factor=1)(e_Tx)

		reg_img = Lambda(lambda x: K.stack([x[0][...,0], x[1][...,0], x[2][...,0]], -1))([img, v_Tx, e_Tx])
		x = layers.Conv3D(64, 3, activation='relu')(reg_img)
		x = layers.Conv3D(64, 3, activation='relu')(x)
		x = layers.Conv3D(64, 3, activation='relu')(x)

		model = Model(img, x)

		return model

def get_trainable_model():
	prediction_model = unet_cls()
	inp = Input(prediction_model.input_shape[1:], name='inp')
	y1_pred, y2_pred = prediction_model(inp)
	y1_true = Input(prediction_model.output_shape[0], name='y1_true')
	y2_true = Input(prediction_model.output_shape[1], name='y2_true')
	out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, y1_pred, y2_pred])
	return Model([inp, y1_true, y2_true], out)

def unet_cls(nb_segs=3, optimizer='adam', depth=3, base_f=32, dropout=.1, lr=.001):
	importlib.reload(cnnc)
	C = config.Config()
	levels = []

	img = Input(shape=(*C.dims, 3))
	current_layer = img
	
	for layer_depth in range(depth):
		layer1 = cnnc.conv_block(current_layer, base_f*2**layer_depth, strides=1)
		dropl = layers.Dropout(dropout)(layer1)
		layer2 = cnnc.conv_block(dropl, base_f*2**(layer_depth+1))

		if layer_depth < depth - 1:
			current_layer = layers.MaxPooling3D((2,2,2))(layer2)
			levels.append([layer1, layer2, current_layer])
		else:
			current_layer = layer2
			levels.append([layer1, layer2])

	cls_layer = layers.Dense(128, activation='relu')(current_layer)
	cls_logit = layers.Dense(len(C.cls_names), activation='softmax')(cls_layer)
	cls_var = layers.Dense(1)(cls_layer)
	cls_layers = layers.Concatenate(axis=-1)([cls_logit, cls_var])
			
	for layer_depth in range(depth-2, -1, -1):
		up_convolution = cnnc.up_conv_block(pool_size=2, deconvolution=False,
											f=current_layer._keras_shape[1])(current_layer)
		concat = layers.Concatenate(axis=-1)([up_convolution, levels[layer_depth][1]])
		current_layer = cnnc.conv_block(concat, levels[layer_depth][1]._keras_shape[-1]//2)
		current_layer = layers.Dropout(dropout)(current_layer)
		current_layer = cnnc.conv_block(current_layer, levels[layer_depth][1]._keras_shape[-1]//2)

	segs = layers.Conv3D(nb_segs, (1,1,1), activation='softmax')(current_layer)
	seg_var = layers.Conv3D(1, (1,1,1))(current_layer)
	segs = layers.Concatenate(axis=-1)([segs, seg_var])

	model = Model(img, [segs, cls_layers])

	return model


from keras.layers import Input, Dense, Lambda, Layer
from keras.initializers import Constant
from keras.models import Model
from keras import backend as K
from tensorflow.contrib import distributions

# Bayesian categorical cross entropy.
# https://github.com/kyle-dorman/bayesian-neural-network-blogpost
# N data points, C classes, T monte carlo simulations
# true - true values. Shape: (N, C)
# pred_var - predicted logit values and variance. Shape: (N, C + 1)
# returns - loss (N,)
def hetero_cls_loss(true, pred_var, T=1000, num_classes=3):
	# shape: (N,)
	std = K.sqrt(pred_var[:, num_classes:])
	# shape: (N,)
	variance = pred_var[:, num_classes]
	variance_depressor = K.exp(variance) - K.ones_like(variance)
	# shape: (N, C)
	pred = pred_var[:, 0:num_classes]
	# shape: (N,)
	undistorted_loss = K.categorical_crossentropy(pred, true, from_logits=True)
	# shape: (T,)
	iterable = K.variable(np.ones(T))
	dist = distributions.Normal(loc=K.zeros_like(std), scale=std)
	monte_carlo_results = K.map_fn(gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss, num_classes), iterable, name='monte_carlo_results')

	variance_loss = K.mean(monte_carlo_results, axis=0) * undistorted_loss

	return variance_loss + undistorted_loss + variance_depressor

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
def gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss, num_classes):
	def map_fn(i):
		std_samples = K.transpose(dist.sample(num_classes))
		distorted_loss = K.categorical_crossentropy(pred + std_samples, true, from_logits=True)
		diff = undistorted_loss - distorted_loss
		return -K.elu(diff)
	return map_fn

# Custom loss layer
# https://github.com/yaringal/multi-task-learning-example
class CustomMultiLossLayer(Layer):
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

	def hetero_cls_loss(y_true, y_pred):
		rv = K.random_normal((1000,1), mean=0.0, stddev=1.0)
		#rv = np.random.laplace(loc=0., scale=1.0, size=(1000,1))
		y_hetero = K.stack([y_pred[:,0] + y_pred[:,-1]*rv,
							y_pred[:,1] + y_pred[:,-1]*rv,
							y_pred[:,2] + y_pred[:,-1]*rv], -1)
		return K.categorical_crossentropy(y_true, y_hetero, from_logits=True)

	def multi_loss(self, ys_true, ys_pred):
		assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
		loss = 0
		for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
			precision = K.exp(-log_var[0])
			loss += K.sum(precision * hetero_cls_loss(y_true, y_pred) + log_var[0], -1)
		return K.mean(loss)

	def call(self, inputs):
		ys_true = inputs[:self.nb_outputs]
		ys_pred = inputs[self.nb_outputs:]
		loss = self.multi_loss(ys_true, ys_pred)
		self.add_loss(loss, inputs=inputs)
		# We won't actually use the output.
		return K.concatenate(inputs, -1)
