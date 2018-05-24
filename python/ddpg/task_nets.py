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
		self.unet.compile(loss='categorical_crossentropy', optimizer=optimizer)

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
		x = layers.Dense(128, activation='relu')(x)
		segs = Dense(3, activation='softmax')(x)
		model = Model(reg_img, segs)

		return model, model.trainable_weights, S

	def build_cls_network(state_size,action_dim):
		S,x = common_network()
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
		#input_shape=(C.dims[0], C.dims[1], C.dims[2], 1)
		locnet.add(layers.Conv3D(32, 3, activation='relu'))
		#locnet.add(layers.MaxPooling3D(2))
		locnet.add(layers.Conv3D(32, 3, activation='relu'))
		locnet.add(Flatten())
		locnet.add(Dense(64, activation='relu'))
		locnet.add(Dense(12, weights=weights))

		#locnet = Model()


		dx = layers.Lambda(lambda x: K.expand_dims(K.stack([x[...,1]-x[...,0], x[...,2]-x[...,1]], 1), -1))(img)
		#dx = layers.TimeDistributed(layers.Conv3D(f, (5,5,3), strides=2, padding='same', activation='relu'))(dx)
		d_va = layers.Lambda(lambda x: x[:,0])(dx)
		#d_va = layers.MaxPooling3D((2,2,2))(d_va)
		#d_va = layers.Conv3D(f, 3, padding='same', activation='relu')(d_va)
		#d_va = layers.Conv3DTranspose(f, (5,5,3), strides=2, padding='same', activation='relu')(d_va)
		d_va = st.SpatialTransformer(localization_net=locnet,
                             downsample_factor=1, input_shape=d_va.shape)(d_va)

		d_ev = Lambda(lambda x: x[:,1])(dx)
		#d_ev = layers.MaxPooling3D((2,2,2))(d_ev)
		#d_ev = layers.Conv3D(f, 3, padding='same', activation='relu')(d_ev)
		#d_ev = layers.Conv3DTranspose(f, (5,5,3), strides=2, padding='same', activation='relu')(d_ev)
		d_ev = st.SpatialTransformer(localization_net=locnet,
                             downsample_factor=1, input_shape=d_ev.shape)(d_ev)

		model = Model(img, [d_va, d_ev])

		return model
