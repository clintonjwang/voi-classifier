"""
Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

import keras.backend as K
import keras.layers as layers
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Concatenate, Flatten, Dropout, Lambda
from keras.layers import SimpleRNN, Conv2D, MaxPooling2D, ZeroPadding3D, Activation, ELU, TimeDistributed, Permute, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
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

import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from keras.engine.training import collect_trainable_weights
import json

import replay
from actor import ActorNetwork
from critic import CriticNetwork
import timeit

class OU(object):
	def function(self, x, mu, theta, sigma):
		return theta * (mu - x) + sigma * np.random.randn(1)


def train_dqn(dqn_generator, agent=None):
	importlib.reload(dqn_env)
	C = config.Config()

	episodes = 250
	minibatch = 32

	env = dqn_env.DQNEnv()
	if agent is None:
		agent = DQNAgent()
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
	env = dqn_env.DQNEnv()
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


def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
	OU = OU()       #Ornstein-Uhlenbeck Process
	BUFFER_SIZE = 100000
	BATCH_SIZE = 32
	GAMMA = 0.99
	TAU = 0.001     #Target Network HyperParameters
	LRA = 0.0001    #Learning rate for Actor
	LRC = 0.001     #Lerning rate for Critic

	action_dim = 8  #6 bbox coordinates, cls weight, trigger
	state_dim = C.context_dims  #of sensors input

	np.random.seed(1337)

	EXPLORE = 10000.
	episode_count = 2000
	max_steps = 10000

	#Tensorflow GPU optimization
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	K.set_session(sess)

	actor = actor.ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
	critic = critic.CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
	buff = replay.ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

	env = Env()

	try:
		actor.model.load_weights("actormodel.h5")
		critic.model.load_weights("criticmodel.h5")
		actor.target_model.load_weights("actormodel.h5")
		critic.target_model.load_weights("criticmodel.h5")
	except:
		pass

	for i in range(episode_count):
		print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

		cur_seg = np.zeros((*C.context_dims, 2))
		seg_var = np.zeros(C.context_dims)

		cur_cls = np.zeros(len(C.cls_names))
		cls_var = 0

		img, true_seg, true_cls = next(dqn_generator)
		if np.mod(i, 3) == 0:
			ob = env.reset(img, true_seg, true_cls, relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
		else:
			ob = env.reset(img, true_seg, true_cls)

		s_t = env.train_unet_cls()

		epsilon = 1
		total_reward = 0.
		for j in range(max_steps):
			loss = 0 
			epsilon -= 1.0 / EXPLORE
			a_t = np.zeros([1,action_dim])
			noise_t = np.zeros([1,action_dim])
			
			a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
			for i in range(3):
				noise_t[0][i] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][i], .0, .60, .30)

			#The following code do the stochastic brake
			#if random.random() <= 0.1:
			#    print("********Now we apply the brake***********")
			#    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

			for i in range(action_dim):
				a_t[0][i] = a_t_original[0][i] + noise_t[0][i]

			ob, r_t, done, info = env.step(a_t[0])

			s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

			buff.add(s_t, a_t[0], r_t, s_t1, done, true_seg, true_cls)      #Add to replay buffer
			
			#Do the batch update
			batch = buff.getBatch(BATCH_SIZE)
			states = np.asarray([e[0] for e in batch])
			actions = np.asarray([e[1] for e in batch])
			rewards = np.asarray([e[2] for e in batch])
			new_states = np.asarray([e[3] for e in batch])
			dones = np.asarray([e[4] for e in batch])
			true_segs = np.asarray([e[-2] for e in batch])
			true_cls = np.asarray([e[-1] for e in batch])
			y_t = np.empty(states.shape)

			target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

			for k in range(len(batch)):
				if dones[k]:
					y_t[k] = rewards[k]
				else:
					y_t[k] = rewards[k] + GAMMA*target_q_values[k]

			if (train_indicator):
				unet_loss += unet.train_on_batch(imgs, true_segs)
				cls_loss += classifier.train_on_batch(imgs, true_cls)

				loss += critic.model.train_on_batch([states,actions], y_t) 
				a_for_grad = actor.model.predict(states)
				grads = critic.gradients(states, a_for_grad)
				actor.train(states, grads)
				actor.target_train()
				critic.target_train()

			total_reward += r_t
			s_t = s_t1
		
			print("Episode", i, "Step", j, "Action", a_t, "Reward", r_t, "Loss", loss)
		
			if done:
				break

		print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
		print("Total Step: " + str(step))
