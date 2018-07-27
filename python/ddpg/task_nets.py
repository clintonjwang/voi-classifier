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
import niftiutils.deep_learning.cnn_components as cnnc
import niftiutils.deep_learning.uncertainty as uncert
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
	out = AleatoricLossLayer(sess, num_classes=[3,3], weights=C.loss_weights)([y1_true, y2_true, y1_pred, y2_pred])
	train_model = Model([inp, y1_true, y2_true], out)
	train_model.compile(optimizer=Adam(lr), loss=None)

	return pred_model, train_model

def unet_cls(optimizer='adam'): #, depth=3, base_f=32, dropout=.1, lr=.001
	img = Input(shape=(*C.dims, 3))
	bottom_layer, end_layer = cnnc.UNet(img)

	cls_layer = layers.bn_relu_etc(cv_u=32, cv_k=1, pool=(3,3,3))(bottom_layer)
	cls_layer = layers.GlobalAveragePooling3D()(cls_layer)
	cls_layer = layers.Dense(len(C.cls_names)+1)(cls_layer)
	
	segs = layers.Conv3D(C.num_segs+1, (1,1,1))(end_layer)

	model = Model(img, [segs, cls_layer]) #logits + logvar

	return model

class AleatoricLossLayer(Layer):
	# https://github.com/yaringal/multi-task-learning-example
	def __init__(self, sess, nb_outputs=2, num_classes=[3,3], weights=[[1,1,1], [1,1,1]], **kwargs):
		self.nb_outputs = nb_outputs
		self.clss = num_classes
		self.W = weights
		self.sess = sess

		super(AleatoricLossLayer, self).__init__(**kwargs)
		
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
		self.loss_graph = tf.exp(-self.log_vars[0]) * uncert.hetero_cls_loss(self.tf_true_img,
							self.tf_pred_img, C.num_segs, weights=self.tf_weights[0]) + \
							tf.exp(-self.log_vars[1]) * uncert.hetero_cls_loss(self.tf_true_cls,
							self.tf_pred_cls, len(C.cls_names), weights=self.tf_weights[1]) + \
							self.log_vars[0] + self.log_vars[1]

		super(AleatoricLossLayer, self).build(input_shape)

	def multi_loss(self, ys_true, ys_pred):
		assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
		loss = 0
		for y_true, y_pred, log_var, num_classes, w in zip(ys_true, ys_pred, self.log_vars, self.clss, self.W):
			precision = K.exp(-log_var[0])
			loss += precision * uncert.hetero_cls_loss(y_true, y_pred,
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
