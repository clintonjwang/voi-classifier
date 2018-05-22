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

class DQNAgent:
	def __init__(self, state_size):
		self.state_size = state_size
		self.action_size = 9
		self.memory = deque(maxlen=5000)
		self.gamma = 0.95    # discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.max_t = 25
		self.dqn_model, self.cls_model = self._build_model()
		self.target_model = self._build_model()[0]
		self.update_target_model()

	def _huber_loss(self, target, prediction):
		error = prediction - target
		return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

	def _build_model(self):
		importlib.reload(cnnc)
		# Neural Net for Deep-Q learning Model
		img = layers.Input(self.state_size)
		x = layers.Reshape((*self.state_size,1))(img)
		x = layers.Conv3D(64, 5, strides=2, activation='relu')(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Conv3D(64, 3, activation='relu')(x)
		x = cnnc.SeparableConv3D(x, 128, 3, activation='relu')
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Conv3D(64, 3, activation='relu')(x)
		x = layers.MaxPooling3D((2,2,1))(x)
		x = layers.Flatten()(x)
		Q = layers.Dense(128, activation='relu')(x)
		Q = layers.BatchNormalization()(Q)
		Q = layers.Dense(self.action_size, activation='linear')(Q)

		dqn_model = Model(img, Q)
		dqn_model.compile(loss=self._huber_loss,
					  optimizer=Adam(lr=self.learning_rate))

		classifier = layers.Dense(64, activation='relu')(x)
		classifier = layers.BatchNormalization()(classifier)
		classifier = layers.Dense(self.action_size, activation='relu')(classifier)

		cls_model = Model(img, classifier)
		cls_model.compile(loss='categorical_crossentropy',
					  optimizer=Adam(lr=self.learning_rate))

		return dqn_model, cls_model

	def update_target_model(self):
		self.target_model.set_weights(self.dqn_model.get_weights())

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return np_utils.to_categorical(random.randint(0,self.action_size-1), self.action_size)
		act_values = self.dqn_model.predict(state)
		return np_utils.to_categorical(np.argmax(act_values[0]), self.action_size)

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			if done:
				Q = reward
			else:
				Q = reward + self.gamma * \
					   np.amax(self.target_model.predict(next_state)[0])
			Q = np.expand_dims(Q, 0)
			self.dqn_model.fit(state, Q, epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay



	def create_actor_network(self):
		S = Input(shape=[self.state_size])
		h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
		h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
		Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   
		Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   
		Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   
		V = merge([Steering,Acceleration,Brake],mode='concat')          
		model = Model(input=S,output=V)

		return model, model.trainable_weights, S

	def create_critic_network(self):
		S = Input(self.state_size)
		A = Input(self.action_dim)
		w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
		a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
		h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
		h2 = merge([h1,a1],mode='sum')    
		h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
		V = Dense(action_dim,activation='linear')(h3)  
		model = Model(input=[S,A],output=V)
		model.compile(loss='mse', optimizer=Adam(lr=self.LEARNING_RATE))

		return model, A, S 

	def target_train(self):
		actor_weights = self.model.get_weights()
		actor_target_weights = self.target_model.get_weights()
		for i in range(len(actor_weights)):
			actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
		self.target_model.set_weights(actor_target_weights)




def train_dqn(dqn_generator, agent=None):
	importlib.reload(dqn_env)
	C = config.Config()

	episodes = 250
	minibatch = 32
	state_size = C.context_dims

	env = dqn_env.DQNEnv(state_size)
	if agent is None:
		agent = DQNAgent(state_size)
	else:
		agent.epsilon = .5

	for e in range(episodes):
		img, true_bbox, cls, accnum = next(dqn_generator)
		save_path = join(C.orig_dir, cls, accnum)
		state = env.set_img(img, true_bbox, save_path)

		for time_t in range(agent.max_t):
			action = agent.act(state)
			next_state, reward, done = env.step(action)
			agent.remember(state, action, reward, next_state, done)
			state = next_state
			if done:
				agent.update_target_model()
				break

		#print("%.3f" % env.best_dice)
		if e % 5 == 0:
			dice = dqn_env.get_DICE(env.true_bbox, env.pred_bbox)
			print("episode: %d/%d, dice: %.2f, eps: %.2f" % (e, episodes, dice, agent.epsilon))

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
