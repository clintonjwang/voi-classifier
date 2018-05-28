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

class OU(object):
	def function(self, x, mu, theta, sigma):
		return theta * (mu - x) + sigma * np.random.randn(1)

def huber_loss(target, pred):
	return K.mean(K.sqrt(1+K.square(pred - target))-1, axis=-1)

class ActorNetwork(object):
	def __init__(self, sess, state_size, action_size, BATCH_SIZE=32, TAU=.001, LEARNING_RATE=.0001):
		self.sess = sess
		self.BATCH_SIZE = BATCH_SIZE
		self.TAU = TAU
		self.LEARNING_RATE = LEARNING_RATE

		K.set_session(sess)

		#Now create the model
		self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)   
		self.target_model, _, _ = self.create_actor_network(state_size, action_size) 
		self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
		self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
		grads = zip(self.params_grad, self.weights)
		self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
		self.sess.run(tf.global_variables_initializer())

	def train(self, states, action_grads):
		self.sess.run(self.optimize, feed_dict={
			self.state: states,
			self.action_gradient: action_grads
		})

	def target_train(self):
		actor_weights = self.model.get_weights()
		actor_target_weights = self.target_model.get_weights()
		for i in range(len(actor_weights)):
			actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
		self.target_model.set_weights(actor_target_weights)

	def create_actor_network(self, state_size, action_dim):
		S,x = common_network(state_size)
		x = layers.Dense(128, activation='relu')(x)
		x = layers.BatchNormalization()(x)
		V = Dense(action_dim, activation='sigmoid')(x)
		model = Model(S,V)

		return model, model.trainable_weights, S

class CriticNetwork(object):
	def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
		self.sess = sess
		self.BATCH_SIZE = BATCH_SIZE
		self.TAU = TAU
		self.LEARNING_RATE = LEARNING_RATE
		self.action_size = action_size
		
		K.set_session(sess)

		self.model, self.action, self.state = self.create_critic_network(state_size, action_size)  
		self.target_model, _, _ = self.create_critic_network(state_size, action_size)  
		self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
		self.sess.run(tf.global_variables_initializer())

	def gradients(self, states, actions):
		return self.sess.run(self.action_grads, feed_dict={
			self.state: states,
			self.action: actions
		})[0]

	def target_train(self):
		critic_weights = self.model.get_weights()
		critic_target_weights = self.target_model.get_weights()
		for i in range(len(critic_weights)):
			critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
		self.target_model.set_weights(critic_target_weights)

	def create_critic_network(self, state_size, action_dim):
		#A = Input([C.context_dims+2])
		#a = layers.Lambda(lambda x: K.reshape(x[:,:-2]))(A)
		S,x = common_network(state_size)
		x = layers.Dense(64, activation='relu')(x)
		A = Input([action_dim])
		a = Dense(64, activation='relu')(A)
		a = layers.BatchNormalization()(a)
		x = layers.Add()([x,a])
		x = Dense(64, activation='relu')(x)
		x = layers.BatchNormalization()(x)
		V = Dense(action_dim)(x)
		model = Model([S,A], V)
		model.compile(loss=huber_loss, optimizer=Adam(lr=self.LEARNING_RATE))

		return model, A, S

def common_network(state_size):
	S = layers.Input(state_size)
	x = layers.Conv3D(64, 5, strides=(2,2,1), activation='relu')(S)
	x = layers.BatchNormalization()(x)
	x = cnnc.SeparableConv3D(x, 128, 3, strides=1, activation='relu')
	x = layers.BatchNormalization()(x)
	x = layers.Conv3D(128, 3, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPooling3D((2,2,2))(x)
	x = layers.Flatten()(x)
	return S,x

class UniformReplay(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def sample(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def store(self, *args):
        experience = tuple(args)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0
