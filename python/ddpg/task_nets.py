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
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.utils import np_utils

import config
import dr_methods as drm
import feature_interpretation as cnna
import niftiutils.cnn_components as cnnc
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import voi_methods as vm

def build_unet_network(state_size,action_dim):
	S,x = common_network()
	x = layers.Dense(128, activation='relu')(x)
	V = Dense(6, activation='sigmoid')(x)
	model = Model(input=S,output=V)

	return model, model.trainable_weights, S

def build_cls_network(state_size,action_dim):
	S,x = common_network()
	x = layers.Dense(128, activation='relu')(x)
	V = Dense(6, activation='sigmoid')(x)
	model = Model(input=S,output=V)

	return model, model.trainable_weights, S

def build_reg_network(f=64):
	C = config.Config()
	img = Input(shape=(*C.dims, 3))

	dx = layers.Lambda(lambda x: K.expand_dims(K.stack([x[...,1]-x[...,0], x[...,2]-x[...,1]], 1), -1))(img)
	dx = layers.TimeDistributed(layers.Conv3D(f, (5,5,3), strides=2, padding='same', activation='relu'))(dx)
	d_va = Lambda(lambda x: x[:,0])(dx)
	d_va = layers.Conv3D(f, 3, padding='same', activation='relu')(d_va)
	d_va = layers.Conv3DTranspose(f, (5,5,3), strides=2, padding='same', activation='relu')(d_va)

	d_ev = Lambda(lambda x: x[:,1])(dx)
	d_ev = layers.Conv3D(f, 3, padding='same', activation='relu')(d_ev)
	d_ev = layers.Conv3DTranspose(f, (5,5,3), strides=2, padding='same', activation='relu')(d_ev)




	model = Model(img, [d_va, d_ev])
	model.compile(optimizer=Adam(lr=0.01), loss='mse') #local difference and l1 terms

	return model
