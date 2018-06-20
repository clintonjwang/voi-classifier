# https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/DenseNet/densenet.py

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

from keras.models import Model
import keras.layers as layers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling3D
from keras.layers import Input, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
import config

importlib.reload(config)
C = config.Config()

def conv_factory(x, nb_filter, dropout_rate=None, w_decay=1E-4):
	"""Apply BatchNorm, Relu 3x3Conv2D, optional dropout
	:param x: Input keras network
	:param concat_axis: int -- index of contatenate axis
	:param nb_filter: int -- number of filters
	:param dropout_rate: int -- dropout rate
	:param w_decay: int -- weight decay factor
	:returns: keras network with b_norm, relu and Conv2D added
	:rtype: keras network
	"""

	x = BatchNormalization(gamma_regularizer=l2(w_decay),
												 beta_regularizer=l2(w_decay))(x)
	x = Activation('relu')(x)
	x = layers.Conv3D(nb_filter, 3,
						 kernel_initializer="he_uniform",
						 padding="same",
						 use_bias=False,
						 kernel_regularizer=l2(w_decay))(x)
	if dropout_rate:
		x = Dropout(dropout_rate)(x)

	return x

def transition(x, nb_filter, dropout_rate=None, w_decay=1E-4, pool=(2,2,2)):
	"""Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D
	:param x: keras model
	:param concat_axis: int -- index of contatenate axis
	:param nb_filter: int -- number of filters
	:param dropout_rate: int -- dropout rate
	:param w_decay: int -- weight decay factor
	:returns: model
	:rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool
	"""

	x = BatchNormalization(gamma_regularizer=l2(w_decay), beta_regularizer=l2(w_decay))(x)
	x = Activation('relu')(x)
	x = layers.Conv3D(nb_filter, 1,
						 kernel_initializer="he_uniform",
						 padding="same",
						 use_bias=False,
						 kernel_regularizer=l2(w_decay))(x)
	if dropout_rate:
		x = Dropout(dropout_rate)(x)
	x = layers.AveragePooling3D(pool, strides=pool)(x)

	return x

def denseblock(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, w_decay=1E-4):
	"""Build a denseblock where the output of each
		 conv_factory is fed to subsequent ones
	:param x: keras model
	:returns: keras model with nb_layers of conv_factory appended
	:rtype: keras model
	"""

	list_feat = [x]

	for i in range(nb_layers):
		x = conv_factory(x, growth_rate, dropout_rate, w_decay)
		list_feat.append(x)
		x = Concatenate()(list_feat)
		nb_filter += growth_rate

	return x, nb_filter

"""def denseblock_altern(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, w_decay=1E-4):
	for i in range(nb_layers):
		merge_tensor = conv_factory(x, growth_rate,
																dropout_rate, w_decay)
		x = Concatenate()([merge_tensor, x])
		nb_filter += growth_rate

	return x, nb_filter"""

def DenseNet(lr=.001, depth=22, nb_dense_block=3, growth_rate=16, nb_filter=32, dropout_rate=None, w_decay=1E-6):
	"""DCCN"""
	model_input = Input(shape=(*C.dims, C.nb_channels))

	assert (depth - 4) % 3 == 0, "Depth must be 3N+4"

	# layers in each dense block
	nb_layers = int((depth - 4) / 3)

	# Initial convolution
	x = layers.Conv3D(nb_filter, 3,
						 kernel_initializer="he_uniform",
						 padding="same",
						 name="initial_conv2D",
						 use_bias=False,
						 kernel_regularizer=l2(w_decay))(model_input)

	# Add dense blocks
	for block_idx in range(nb_dense_block - 1):
		if block_idx == 0:
			x, nb_filter = denseblock(x, 2, nb_filter, growth_rate, 
							dropout_rate=dropout_rate, w_decay=w_decay)
		else:
			x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate, 
							dropout_rate=dropout_rate, w_decay=w_decay)
		# add transition
		if block_idx == 1:
			x = transition(x, nb_filter, dropout_rate=dropout_rate,
							w_decay=w_decay, pool=(2,2,1))
		else:
			x = transition(x, nb_filter, dropout_rate=dropout_rate,
							w_decay=w_decay, pool=(2,2,2))

	# The last denseblock does not have a transition
	x, nb_filter = denseblock(x, nb_layers,
							nb_filter, growth_rate, 
							dropout_rate=dropout_rate,
							w_decay=w_decay)

	x = BatchNormalization(gamma_regularizer=l2(w_decay),
												 beta_regularizer=l2(w_decay))(x)
	x = Activation('relu')(x)
	x = GlobalAveragePooling3D(data_format=K.image_data_format())(x)

	if C.non_img_inputs > 0:
		non_img_inputs = Input(shape=(C.non_img_inputs,))
		y = layers.Reshape((C.non_img_inputs, 1))(non_img_inputs)
		y = layers.LocallyConnected1D(1, 1, bias_regularizer=l2(.1), activation='tanh')(y)
		y = layers.Flatten()(y)
		y = Dense(32, activation='relu')(y)
		y = BatchNormalization()(y)
		#y = Dropout(dropout[1])(y)
		#y = ActivationLayer(activation_args)(y)
		x = Concatenate(axis=1)([x, y])

	#x = Dense(C.nb_classes+1)(x)
	x = Dense(C.nb_classes,
			activation='softmax',
			kernel_regularizer=l2(w_decay),
			bias_regularizer=l2(w_decay))(x)

	if C.non_img_inputs > 0:
		densenet = Model(inputs=[model_input, non_img_inputs], outputs=[x], name="DenseNet")
	else:
		densenet = Model(inputs=[model_input], outputs=[x], name="DenseNet")
	densenet.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

	return densenet

def get_models(sess, lr=.001):
	pred_model = DenseNet()
	inp = Input(pred_model.input_shape[1:], name='inp')
	y_pred = pred_model(inp)
	y_true = Input((C.nb_classes,), name='y_true')
	out = CustomMultiLossLayer(sess, num_classes=[C.nb_classes])([y_true, y_pred])
	train_model = Model([inp, y_true], out)
	train_model.compile(optimizer=Adam(lr), loss=None)

	return pred_model, train_model


class CustomMultiLossLayer(Layer):
	# https://github.com/yaringal/multi-task-learning-example
	def __init__(self, sess, nb_outputs=1, num_classes=3, **kwargs):
		self.nb_outputs = nb_outputs
		self.clss = list(num_classes)
		self.sess = sess

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
		for y_true, y_pred, log_var, num_classes in zip(ys_true, ys_pred, self.log_vars, self.clss):
			precision = K.exp(-log_var[0])
			loss += precision * hetero_cls_loss(y_true, y_pred,
					num_classes=num_classes) + log_var[0]
		return K.mean(loss)

	def call(self, inputs):
		ys_true = inputs[:self.nb_outputs]
		ys_pred = inputs[self.nb_outputs:]
		loss = self.multi_loss(ys_true, ys_pred)
		self.add_loss(loss, inputs=inputs)
		# We won't actually use the output.
		return inputs#K.concatenate(inputs, -1)

def hetero_cls_loss(true, pred, num_classes=3, weights=[1,1,1], T=500):
	# Bayesian categorical cross entropy.
	# https://github.com/kyle-dorman/bayesian-neural-network-blogpost
	# N data points, C classes, T monte carlo simulations
	# true - true values. Shape: (N, C)
	# pred - predicted logit values and log variance. Shape: (N, C + 1)
	# returns - loss (N,)

	true = K.reshape(true, [-1, num_classes])
	pred = K.reshape(pred, [-1, num_classes+1])
	weights = K.cast(weights, tf.float32)
	pred_scale = K.sqrt(K.exp(pred[:, num_classes:])) # shape: (N,1)
	pred = pred[:, :num_classes] # shape: (N, C)

	dist = distributions.Normal(loc=K.zeros_like(pred_scale), scale=pred_scale)
	#std_samples = K.transpose(dist.sample(num_classes))
	#distorted_loss = K.categorical_crossentropy(true, pred + std_samples, from_logits=True) * weights
	
	iterable = K.variable(np.ones(T))
	mc_loss = K.mean(K.map_fn(gaussian_categorical_crossentropy(true, pred, dist, num_classes), iterable, name='monte_carlo'), 0)

	return K.mean(mc_loss * weights)

def gaussian_categorical_crossentropy(true, pred, dist, num_classes):
	def map_fn(i):
		std_samples = K.transpose(dist.sample(num_classes))
		distorted_loss = K.categorical_crossentropy(true, pred + std_samples, from_logits=True)
		return tf.cast(K.mean(distorted_loss, -1), tf.float32)
	return map_fn

