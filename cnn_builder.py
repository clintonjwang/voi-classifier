import keras.backend as K
from keras.layers import Input, Dense, Concatenate, Flatten, Dropout, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.models import Model
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from keras.constraints import max_norm
from keras.layers.noise import GaussianNoise
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

import cnn_methods as cfunc
import config
import csv
import helper_fxns as hf
import importlib
import numpy as np
import operator
import os
import pandas as pd
import random

def build_cnn(C, inputs=4):
	dims = C.dims
	nb_classes = len(C.classes_to_include)

	if inputs == 2:
		voi_img = Input(shape=(dims[0], dims[1], dims[2], C.nb_channels))
		x = voi_img
		#x = GaussianNoise(1)(x)
		#x = ZeroPadding3D(padding=(3,3,2))(voi_img)
		x = Conv3D(filters=128, kernel_size=(3,3,2), activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Conv3D(filters=128, kernel_size=(3,3,2), activation='relu')(x)
		x = MaxPooling3D((2, 2, 2))(x)
		x = Dropout(0.5)(x)
		#x = Conv3D(filters=64, kernel_size=(3,3,2), strides=(2, 2, 2), activation='relu', kernel_constraint=max_norm(4.))(x)
		#x = Dropout(0.5)(x)
		x = Conv3D(filters=64, kernel_size=(3,3,2), activation='relu')(x)
		x = MaxPooling3D((2, 2, 1))(x)
		x = Dropout(0.5)(x)
		x = Flatten()(x)

		img_traits = Input(shape=(2,)) #bounding volume and aspect ratio of lesion

		intermed = Concatenate(axis=1)([x, img_traits])
		x = Dense(64, activation='relu')(intermed)#, kernel_initializer='normal', kernel_regularizer=l1(.01), kernel_constraint=max_norm(3.))(x)
		x = Dropout(0.5)(x)
		pred_class = Dense(nb_classes, activation='softmax')(x)#Dense(nb_classes, activation='softmax')(x)

		model = Model([voi_img, img_traits], pred_class)

	elif inputs == 4:
		art_img = Input(shape=(dims[0], dims[1], dims[2], 1))
		art_x = art_img
		art_x = Conv3D(filters=64, kernel_size=(3,3,2), activation='relu')(art_x)
		art_x = MaxPooling3D((2, 2, 2))(art_x)
		art_x = Dropout(0.5)(art_x)

		ven_img = Input(shape=(dims[0], dims[1], dims[2], 1))
		ven_x = ven_img
		ven_x = Conv3D(filters=64, kernel_size=(3,3,2), activation='relu')(ven_x)
		ven_x = MaxPooling3D((2, 2, 2))(ven_x)
		ven_x = Dropout(0.5)(ven_x)

		eq_img = Input(shape=(dims[0], dims[1], dims[2], 1))
		eq_x = eq_img
		eq_x = Conv3D(filters=64, kernel_size=(3,3,2), activation='relu')(eq_x)
		eq_x = MaxPooling3D((2, 2, 2))(eq_x)
		eq_x = Dropout(0.5)(eq_x)

		intermed = Concatenate(axis=4)([art_x, ven_x, eq_x])
		x = Conv3D(filters=100, kernel_size=(3,3,2), activation='relu')(intermed)
		x = MaxPooling3D((2, 2, 1))(x)
		x = Dropout(0.5)(x)
		x = Flatten()(x)

		img_traits = Input(shape=(2,)) #bounding volume and aspect ratio of lesion

		intermed = Concatenate(axis=1)([x, img_traits])
		x = Dense(100, activation='relu')(intermed)#, kernel_initializer='normal', kernel_regularizer=l1(.01), kernel_constraint=max_norm(3.))(x)
		x = Dropout(0.5)(x)
		pred_class = Dense(nb_classes, activation='softmax')(x)#Dense(nb_classes, activation='softmax')(x)

		model = Model([art_img, ven_img, eq_img, img_traits], pred_class)
	
	#optim = Adam(lr=0.01)#5, decay=0.001)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	return model


def build_cnn(C):
	dims = C.dims
	nb_classes = len(C.classes_to_include)

	return model

def build_pretrain_model(C):
	dims = C.dims
	nb_classes = len(C.classes_to_include)

	voi_img = Input(shape=(dims[0], dims[1], dims[2], C.nb_channels))
	x = voi_img
	x = Conv3D(filters=128, kernel_size=(3,3,2), activation='relu', trainable=False)(x)
	x = Dropout(0.5)(x)
	x = Conv3D(filters=128, kernel_size=(3,3,2), activation='relu', trainable=False)(x)
	x = MaxPooling3D((2, 2, 2))(x)
	x = Dropout(0.5)(x)
	x = Conv3D(filters=64, kernel_size=(3,3,2), activation='relu', trainable=False)(x)
	x = MaxPooling3D((2, 2, 1))(x)
	x = Dropout(0.5)(x)
	x = Flatten()(x)

	img_traits = Input(shape=(2,)) #bounding volume and aspect ratio of lesion

	intermed = Concatenate(axis=1)([x, img_traits])
	x = Dense(64, activation='relu')(intermed)#, kernel_initializer='normal', kernel_regularizer=l1(.01), kernel_constraint=max_norm(3.))(x)
	x = Dropout(0.5)(x)
	pred_class = Dense(nb_classes, activation='softmax')(x)

	#optim = Adam(lr=0.01)#5, decay=0.001)

	model_pretrain = Model([voi_img, img_traits], pred_class)
	model_pretrain.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	return model_pretrain