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
import config
import niftiutils.cnn_components as cnnc
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
import importlib
import tensorflow as tf

import dr_methods as drm
import voi_methods as vm

class DQNAgent:
	def __init__(self, state_size):
		self.state_size = state_size
		self.action_size = 9 #6 translations, 2 scaling, 1 trigger
		self.memory = deque(maxlen=2000)
		self.gamma = 0.95    # discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.model = self._build_model()
		self.target_model = self._build_model()
		self.update_target_model()

	def _huber_loss(self, target, prediction):
		# sqrt(1+error^2)-1
		error = prediction - target
		return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		img = layers.Input(self.state_size)
		x = Dense(24, activation='relu')(img)
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='relu'))
		model.compile(loss=self._huber_loss,
					  optimizer=Adam(lr=self.learning_rate))
		return model

	def update_target_model(self):
		# copy weights from model to target_model
		self.target_model.set_weights(self.model.get_weights())

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])  # returns action

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = reward + self.gamma * \
					   np.amax(self.model.predict(next_state)[0])
			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

def build_dqn():
	episodes = 500
	state_size = 9
	env = gym.make('CartPole-v0')
	agent = DQNAgent((100,100,50))
	# Iterate the game
	for e in range(episodes):
		# reset state in the beginning of each game
		state = env.reset()
		state = np.reshape(state, [1, state_size])
		# time_t represents each frame of the game
		for time_t in range(500):
			# Decide action
			action = agent.act(state)
			next_state, reward, done, _ = env.step(action)
			next_state = np.reshape(next_state, [1, state_size])
			agent.remember(state, action, reward, next_state, done)
			state = next_state
			if done:
				print("episode: {}/{}, score: {}"
					  .format(e, episodes, time_t))
				break
		# train the agent with the experience of the episode
		agent.replay(32)