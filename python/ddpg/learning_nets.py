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

class ActorNetwork(object):
	def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
		self.sess = sess
		self.BATCH_SIZE = BATCH_SIZE
		self.TAU = TAU
		self.LEARNING_RATE = LEARNING_RATE

		K.set_session(sess)

		#Now create the model
		self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
		self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
		self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
		self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
		grads = zip(self.params_grad, self.weights)
		self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
		self.sess.run(tf.initialize_all_variables())

	def train(self, states, action_grads):
		self.sess.run(self.optimize, feed_dict={
			self.state: states,
			self.action_gradient: action_grads
		})

	def target_train(self):
		actor_weights = self.model.get_weights()
		actor_target_weights = self.target_model.get_weights()
		for i in xrange(len(actor_weights)):
			actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
		self.target_model.set_weights(actor_target_weights)

	def create_actor_network(self, state_size,action_dim):
		S,x = common_network()
		x = layers.Dense(128, activation='relu')(x)
		V = Dense(6, activation='sigmoid')(x)
		model = Model(input=S,output=V)
		return model, model.trainable_weights, S

class CriticNetwork(object):
	def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
		self.sess = sess
		self.BATCH_SIZE = BATCH_SIZE
		self.TAU = TAU
		self.LEARNING_RATE = LEARNING_RATE
		self.action_size = action_size
		
		K.set_session(sess)

		#Now create the model
		self.model, self.action, self.state = self.create_critic_network(state_size, action_size)  
		self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)  
		self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
		self.sess.run(tf.initialize_all_variables())

	def gradients(self, states, actions):
		return self.sess.run(self.action_grads, feed_dict={
			self.state: states,
			self.action: actions
		})[0]

	def target_train(self):
		critic_weights = self.model.get_weights()
		critic_target_weights = self.target_model.get_weights()
		for i in xrange(len(critic_weights)):
			critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
		self.target_model.set_weights(critic_target_weights)

	def create_critic_network(self, state_size,action_dim):
		S = Input(shape=[state_size])  
		A = Input(shape=[action_dim],name='action2')   
		w1 = Dense(64, activation='relu')(S)
		a1 = Dense(64, activation='linear')(A) 
		h1 = Dense(64, activation='linear')(w1)
		h2 = merge([h1,a1],mode='sum')    
		h3 = Dense(64, activation='relu')(h2)
		V = Dense(action_dim,activation='linear')(h3)   
		model = Model(input=[S,A],output=V)
		adam = Adam(lr=self.LEARNING_RATE)
		model.compile(loss='mse', optimizer=adam)

		return model, A, S

def common_network():
	S = layers.Input(self.state_size)
	x = layers.Reshape((*self.state_size,1))(img)
	x = layers.Conv3D(64, 5, strides=2, activation='relu')(x)
	x = layers.BatchNormalization(axis=-1)(x)
	x = layers.Conv3D(64, 3, activation='relu')(x)
	x = cnnc.SeparableConv3D(x, 128, 3, activation='relu')
	x = layers.BatchNormalization(axis=-1)(x)
	x = layers.Conv3D(64, 3, activation='relu')(x)
	x = layers.MaxPooling3D((2,2,1))(x)
	x = layers.Flatten()(x)
	return S,x

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
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
