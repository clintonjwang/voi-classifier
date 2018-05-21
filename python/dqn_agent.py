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
import feature_interpretation as cnna
import copy
import dqn_env
from collections import deque
import config
import niftiutils.cnn_components as cnnc
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import math
from math import log, ceil
import numpy as np
import operator
import os
from os.path import *
import pandas as pd
import random
from scipy.misc import imsave
from skimage.transform import rescale
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import time
import importlib
import tensorflow as tf

import dr_methods as drm
import voi_methods as vm

#def jagged_activ(x):
#	a = K.switch(x>1, x-1, x-x+0)
#	a = K.switch(a<-1, a+1, a-a+0) #tf.where
#	return a

class DQNAgent:
	def __init__(self, state_size):
		self.state_size = state_size
		self.action_size = 9
		self.memory = deque(maxlen=10000)
		self.gamma = 0.95    # discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.dqn_model, self.cls_model = self._build_model()
		self.max_t = 25
		#self.target_model = self._build_model()
		#self.update_target_model()

	def _huber_loss(self, target, prediction):
		error = prediction - target
		return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

	#get_custom_objects().update({'custom_activation': Activation(custom_activation)})

	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		img = layers.Input(self.state_size)
		x = layers.Reshape((*self.state_size,1))(img)
		x = layers.Conv3D(64, 3, activation='relu')(x)
		x = layers.MaxPooling3D((2,2,2))(x)
		x = layers.Conv3D(64, 3, activation='relu')(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Conv3D(64, 3, activation='relu')(x)
		x = layers.MaxPooling3D((2,2,1))(x)
		x = layers.Flatten()(x)
		qn = layers.Dense(128, activation='relu')(x)
		qn = layers.BatchNormalization()(qn)
		qn = layers.Dense(self.action_size, activation='linear')(qn)

		#run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
		dqn_model = Model(img, qn)
		dqn_model.compile(loss=self._huber_loss,
					  optimizer=Adam(lr=self.learning_rate))#, options=run_opts)

		classifier = layers.Dense(64, activation='relu')(x)
		classifier = layers.BatchNormalization()(classifier)
		classifier = layers.Dense(self.action_size, activation='relu')(classifier)

		cls_model = Model(img, classifier)
		cls_model.compile(loss='categorical_crossentropy',
					  optimizer=Adam(lr=self.learning_rate))

		return dqn_model, cls_model

	#def update_target_model(self):
	#	self.target_model.set_weights(self.dqn_model.get_weights())

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return np_utils.to_categorical(random.randint(0,self.action_size-2), self.action_size)
		act_values = self.dqn_model.predict(state)
		return np_utils.to_categorical(np.argmax(act_values[0]), self.action_size)

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			if done:
				Q = reward
			else:
				Q = reward + self.gamma * \
					   np.amax(self.dqn_model.predict(next_state)[0])
			Q = np.expand_dims(Q, 0)
			self.dqn_model.fit(state, Q, epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

def train_dqn(img_generator):
	importlib.reload(dqn_env)
	C = config.Config()

	episodes = 250
	minibatch = 32
	state_size = C.context_dims

	env = dqn_env.DQNEnv(state_size)
	agent = DQNAgent(state_size)

	for e in range(episodes):
		img, true_bbox, cls, accnum = next(img_generator)
		save_path = join(C.orig_dir, cls, accnum)
		state = env.set_img(img, true_bbox, save_path)

		for time_t in range(agent.max_t):
			action = agent.act(state)
			next_state, reward, done = env.step(action)
			agent.remember(state, action, reward, next_state, done)
			state = next_state
			if done:
				break

		print(env.best_dice)
		if e % 5 == 0:
			dice = dqn_env.get_DICE(env.true_bbox, env.pred_bbox)
			print("episode: %d/%d, dice: %.2f" % (e, episodes, dice))

		try:
			agent.replay(minibatch)
		except Exception as e:
			print(e)
			continue

	return agent

def run_dqn(agent, img):
	env = dqn_env.DQNEnv(state_size)
	state = env.set_img(img)
	self.epsilon = 0
	# time_t represents each frame of the game
	for time_t in range(agent.max_t):
		# Decide action
		action = agent.act(state)
		state, done = env.step(action)
		if done:
			break
	return env.pred_bbox
