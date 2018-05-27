"""
Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

import copy
import importlib
import math
import operator
import os
import random
import time
from os.path import *
import psutil

import keras.backend as K
import keras.layers as layers
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.misc import imsave
from skimage.transform import rescale
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import config
import ddpg.env as denv
import ddpg.learning_nets as ln
import ddpg.task_nets as tn
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr

C = config.Config()
importlib.reload(denv)
importlib.reload(ln)

def run_dqn(agent, img):
	env = denv.DQNEnv()
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

def memory():
    pid = os.getpid()
    py = psutil.Process(pid)
    print('Memory use:', py.memory_info()[0]/2.**30)

def play_game(dqn_generator, train_indicator=1):    #1 means Train, 0 means simply Run
	def load_weights():
		actor.model.load_weights(join(C.model_dir, "actor_model.h5"))
		critic.model.load_weights(join(C.model_dir, "critic_model.h5"))
		actor.target_model.load_weights(join(C.model_dir, "actor_model.h5"))
		critic.target_model.load_weights(join(C.model_dir, "critic_model.h5"))
		env.train_model.load_weights(join(C.model_dir, "train_model.h5"))
		env.pred_model.load_weights(join(C.model_dir, "pred_model.h5"))
		
	OU = ln.OU()       #Ornstein-Uhlenbeck Process
	BUFFER_SIZE = 100000
	BATCH_SIZE = 32
	GAMMA = 0.99
	TAU = 0.001     #Target Network HyperParameters
	LRA = .0001    #Learning rate for Actor
	LRC = .0002    #Learning rate for Critic
	LRU = .0002    #Learning rate for Unet

	action_dim = 8  #6 bbox coordinates, cls weight, trigger
	state_dim = (*C.dims, 3)

	EXPLORE = 10000.
	episode_count = 2000
	max_steps = 100

	#Tensorflow GPU optimization
	conf = tf.ConfigProto()
	conf.gpu_options.allow_growth = True
	sess = tf.Session(config=conf)
	K.set_session(sess)

	actor = ln.ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
	critic = ln.CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
	buff = ln.ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
	env = denv.Env(LRU)

	try:
		load_weights()
	except:
		pass

	for i in range(episode_count):
		print("Episode: " + str(i) + " Replay Buffer " + str(buff.count()))
		memory()

		if i % 3 == 2:
			actor.model.save_weights(join(C.model_dir, "actor_model.h5"))
			critic.model.save_weights(join(C.model_dir, "critic_model.h5"))
			env.train_model.save_weights(join(C.model_dir, "train_model.h5"))
			env.pred_model.save_weights(join(C.model_dir, "pred_model.h5"))

			del actor, critic, conf, sess
			K.clear_session()

			conf = tf.ConfigProto()
			conf.gpu_options.allow_growth = True
			sess = tf.Session(config=conf)
			K.set_session(sess)

			actor = ln.ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
			critic = ln.CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
			env = denv.Env(LRU)
			load_weights()

		img, true_seg, true_cls = next(dqn_generator)
		env.reset(img, true_seg, true_cls)

		s_t = env.get_state()

		epsilon = 1
		total_reward = 0.
		steps = 0
		for j in range(max_steps):
			loss = 0 
			if epsilon > .05:
				epsilon -= 1.0 / EXPLORE
			a_t = np.zeros(action_dim)
			noise_t = np.zeros(action_dim)
			
			a_t_original = actor.model.predict(np.expand_dims(s_t,0))[0]
			for ix in range(action_dim):
				noise_t[ix] = epsilon * OU.function(a_t_original[ix],.5,.3,.2) #shift, scale, gaussian

			a_t = np.clip(a_t_original + noise_t, 0, 1)
			s_t1, r_t, done, cropI, crop_true_seg = env.step(a_t, get_crops=True)

			buff.add(s_t, a_t, r_t, s_t1, done, cropI, crop_true_seg, env.true_cls)  #Add to replay buffer
			
			#Do the batch update
			batch = buff.getBatch(BATCH_SIZE)
			states = np.asarray([e[0] for e in batch])
			actions = np.asarray([e[1] for e in batch])
			rewards = np.asarray([e[2] for e in batch])
			new_states = np.asarray([e[3] for e in batch])
			dones = np.asarray([e[4] for e in batch])
			cropI = np.asarray([e[-3] for e in batch])
			crop_true_seg = np.asarray([e[-2] for e in batch])
			true_cls = np.asarray([e[-1] for e in batch])
			y_t = np.zeros(actions.shape)

			env.train_model.train_on_batch([cropI, crop_true_seg, true_cls], None)

			target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

			for k in range(len(batch)):
				if dones[k]:
					y_t[k] = rewards[k]
				else:
					y_t[k] = rewards[k] + GAMMA*target_q_values[k]

			for v in [crop_true_seg, cropI, new_states, y_t, target_q_values, states, actions]:
				if v.max() > 1e4:
					print(v)
					raise ValueError()

			loss += critic.model.train_on_batch([states,actions], y_t) 
			a_for_grad = actor.model.predict(states)
			grads = critic.gradients(states, a_for_grad)
			actor.train(states, grads)
			actor.target_train()
			critic.target_train()

			total_reward += r_t
			s_t = s_t1
		
			print("Step", j, "Action", a_t, "Reward", r_t, "Loss", loss)
		
			if done:
				steps = j
				break

		print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
		print("Total Step: " + str(steps))

	actor.model.save_weights(join(C.model_dir, "actor_model.h5"))
	critic.model.save_weights(join(C.model_dir, "critic_model.h5"))
	env.train_model.save_weights(join(C.model_dir, "train_model.h5"))
	env.pred_model.save_weights(join(C.model_dir, "pred_model.h5"))
