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

importlib.reload(config)
importlib.reload(cnnc)
C = config.Config()

def get_models(sess, lr=.001):
	pred_model = unet_cls()
	inp = Input(pred_model.input_shape[1:], name='inp')
	y1_pred, y2_pred = pred_model(inp)
	y1_true = Input((*pred_model.input_shape[1:-1], C.num_segs), name='y1_true')
	y2_true = Input((len(C.cls_names),), name='y2_true')
	out = CustomMultiLossLayer(sess, num_classes=[3,3], weights=C.loss_weights)([y1_true, y2_true, y1_pred, y2_pred])
	train_model = Model([inp, y1_true, y2_true], out)
	train_model.compile(optimizer=Adam(lr), loss=None)

	return pred_model, train_model

def unet_cls(optimizer='adam'): #, depth=3, base_f=32, dropout=.1, lr=.001
	img = Input(shape=(*C.dims, 3))
	bottom_layer, end_layer = cnnc.UNet(img)

	cls_layer = layers.Conv3D(32, 1)(bottom_layer)
	cls_layer = layers.AveragePooling3D((2,2,1))(cls_layer)
	cls_layer = layers.BatchNormalization()(cls_layer)
	cls_layer = layers.Flatten()(cls_layer)
	cls_layer = layers.Dense(128, activation='relu')(cls_layer)
	cls_layer = layers.BatchNormalization()(cls_layer)
	cls_layer = layers.Dense(len(C.cls_names)+1, activation='elu')(cls_layer)
	
	segs = layers.Conv3D(C.num_segs+1, (1,1,1), activation='elu')(end_layer)

	model = Model(img, [segs, cls_layer]) #logits + logvar

	return model

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
	# for a single monte carlo simulation, calculate xentropy
	#   of predicted logit values plus gaussian 
	#   noise vs true values.
	# true - true values. Shape: (N, C)
	# pred - predicted logit values. Shape: (N, C)
	# dist - normal distribution to sample from. Shape: (N, C)
	# num_classes - the number of classes. C
	# returns - mean differences for each class (N,)
	def map_fn(i):
		std_samples = K.transpose(dist.sample(num_classes))
		distorted_loss = K.categorical_crossentropy(true, pred + std_samples, from_logits=True)
		return tf.cast(K.mean(distorted_loss, -1), tf.float32)
	return map_fn

class CustomMultiLossLayer(Layer):
	# https://github.com/yaringal/multi-task-learning-example
	def __init__(self, sess, nb_outputs=2, num_classes=[3,3], weights=[[1,1,1], [1,1,1]], **kwargs):
		self.nb_outputs = nb_outputs
		self.clss = num_classes
		self.W = weights
		self.sess = sess

		super(CustomMultiLossLayer, self).__init__(**kwargs)
		
	def build(self, input_shape=None):
		# initialise log_vars
		self.log_vars = []
		for i in range(self.nb_outputs):
			self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
											  initializer=Constant(0.), trainable=True)]

		self.tf_true_img = tf.placeholder(tf.float32, (None,*C.context_dims,C.num_segs))
		self.tf_pred_img = tf.placeholder(tf.float32, (None,*C.context_dims,C.num_segs+1))
		self.tf_true_cls = tf.placeholder(tf.float32, (None, len(C.cls_names)))
		self.tf_pred_cls = tf.placeholder(tf.float32, (None, len(C.cls_names)+1))
		self.tf_weights = [tf.placeholder(tf.float32, [3]), tf.placeholder(tf.float32, [3])]
		self.loss_graph = tf.exp(-self.log_vars[0]) * hetero_cls_loss(self.tf_true_img,
							self.tf_pred_img, C.num_segs, weights=self.tf_weights[0]) + \
							tf.exp(-self.log_vars[1]) * hetero_cls_loss(self.tf_true_cls,
							self.tf_pred_cls, len(C.cls_names), weights=self.tf_weights[1]) + \
							self.log_vars[0] + self.log_vars[1]

		super(CustomMultiLossLayer, self).build(input_shape)

	def multi_loss(self, ys_true, ys_pred):
		assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
		loss = 0
		for y_true, y_pred, log_var, num_classes, w in zip(ys_true, ys_pred, self.log_vars, self.clss, self.W):
			precision = K.exp(-log_var[0])
			loss += precision * hetero_cls_loss(y_true, y_pred,
					num_classes=num_classes, weights=w) + log_var[0]
		return K.mean(loss)

	def get_loss(self, true_img, pred_img, true_cls, pred_cls, w):
		return self.sess.run(self.loss_graph, feed_dict={
			self.tf_true_img: np.expand_dims(true_img,0),
			self.tf_pred_img: np.expand_dims(pred_img,0),
			self.tf_true_cls: np.expand_dims(true_cls,0),
			self.tf_pred_cls: np.expand_dims(pred_cls,0),
			self.tf_weights[0]: w[0],
			self.tf_weights[1]: w[1]
		})

	def call(self, inputs):
		ys_true = inputs[:self.nb_outputs]
		ys_pred = inputs[self.nb_outputs:]
		loss = self.multi_loss(ys_true, ys_pred)
		self.add_loss(loss, inputs=inputs)
		# We won't actually use the output.
		return inputs#K.concatenate(inputs, -1)
