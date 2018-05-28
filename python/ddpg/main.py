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
import ddpg.rank_replay as rrep
import ddpg.task_nets as tn
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr

C = config.Config()
importlib.reload(denv)
importlib.reload(ln)

def memory():
    pid = os.getpid()
    py = psutil.Process(pid)
    print('Memory use:', py.memory_info()[0]/2.**30)

class CRSNet(object): #ClsRegSegNet
	def __init__(self, eps=1):
		self.epsilon = eps
		self.action_dim = 10  #6 bbox coordinates, 2 thresholds, cls weight, trigger

	def save_weights(self):
		try:
			self.actor.model.save_weights(join(C.model_dir, "actor.h5"))
			self.critic.model.save_weights(join(C.model_dir, "critic.h5"))
			self.actor.target_model.save_weights(join(C.model_dir, "actor_T.h5"))
			self.critic.target_model.save_weights(join(C.model_dir, "critic_T.h5"))
			self.env.train_model.save_weights(join(C.model_dir, "train.h5"))
			self.env.pred_model.save_weights(join(C.model_dir, "pred.h5"))
		except:
			print("Unable to save weights")


	def load_models(self, replay_conf):
		state_dim = (*C.dims, 3)
		TAU = .01     #Target Network update weight
		LRA = .0001    #Learning rate for Actor
		LRC = .00015    #Learning rate for Critic
		LRU = .00015    #Learning rate for Unet

		BATCH_SIZE = replay_conf["batch_size"]

		conf = tf.ConfigProto()
		conf.gpu_options.allow_growth = True
		sess = tf.Session(config=conf)
		K.set_session(sess)

		self.actor = ln.ActorNetwork(sess, state_dim, self.action_dim, BATCH_SIZE, TAU, LRA)
		self.critic = ln.CriticNetwork(sess, state_dim, self.action_dim, BATCH_SIZE, TAU, LRC)
		self.env = denv.Env(LRU)
		unet_buff = ln.UniformReplay(1000)
		try:
			self.load_weights()
			buff = hf.pickle_load(join(C.model_dir, "replay_buffer.bin"))
		except:
			buff = rrep.Experience(replay_conf)    #Create replay buffer

	def load_weights(self):
		try:
			self.actor.model.load_weights(join(C.model_dir, "actor.h5"))
			self.actor.target_model.load_weights(join(C.model_dir, "actor_T.h5"))
			if self.critic is not None:
				self.critic.model.load_weights(join(C.model_dir, "critic.h5"))
				self.critic.target_model.load_weights(join(C.model_dir, "critic_T.h5"))
			self.env.train_model.load_weights(join(C.model_dir, "train.h5"))
			self.env.pred_model.load_weights(join(C.model_dir, "pred.h5"))
		except:
			print("Models not initialized or weight files not found")

	def run(self, img):
		state_dim = (*C.dims, 3)
		max_steps = 100

		conf = tf.ConfigProto()
		conf.gpu_options.allow_growth = True
		sess = tf.Session(config=conf)
		K.set_session(sess)
		self.actor = ln.ActorNetwork(sess, state_dim, self.action_dim)
		self.critic = None
		self.env = denv.Env()
		self.load_weights()

		self.env.reset(img)
		s_t = self.env.get_state()

		step = 0
		for step in range(max_steps):
			print(".", end="")
			a_t = self.actor.model.predict(np.expand_dims(s_t,0))[0]
			s_t, done = self.env.step(a_t)
			if done:
				break

		print(step+1, "steps")

		seg = self.env.pred_seg
		seg = np.exp(seg[...,1]) / (np.exp(seg[...,0])+np.exp(seg[...,1]))
		cls = np.argmax(self.env.pred_cls)

		return seg, self.env.pred_seg_var, cls

	def transform_action(self, action):
		a_Tx = np.zeros(C.context_dims)
		bbox = env.get_bbox(action)
		sl = [slice(bbox[i], bbox[i+1]) for i in [0,2,4]]
		a_Tx[sl] = 1

		return np.concatenate([a_Tx.flatten(), action[6:]])

	def replay_Q(self, buff, g_step, GAMMA=.99):
		batch, _, ixs = buff.sample(g_step)
		states = np.asarray([e[0] for e in batch])
		actions = np.asarray([e[1] for e in batch])
		rewards = np.asarray([e[2] for e in batch])
		new_states = np.asarray([e[3] for e in batch])
		dones = np.asarray([e[4] for e in batch])
		y_t = np.zeros(actions.shape)

		target_a = self.actor.target_model.predict(new_states)
		#self.transform_action(self.actor.target_model.predict(new_states))
		target_q_values = self.critic.target_model.predict([new_states, target_a])

		#temporal difference error for prioritized exp replay
		TD = self.critic.model.predict([states, actions])
		for k in range(len(batch)):
			if dones[k]:
				y_t[k] = rewards[k]
			else:
				y_t[k] = rewards[k] + GAMMA*target_q_values[k]
		TD = np.mean(np.abs(TD - y_t), -1)
		buff.update_priority(ixs, TD)

		closs = self.critic.model.train_on_batch([states,actions], y_t) 
		a_for_grad = self.actor.model.predict(states)
		grads = self.critic.gradients(states, a_for_grad)
		self.actor.train(states, grads)
		self.actor.target_train()
		self.critic.target_train()

	def train(self, dqn_generator, verbose=False):    #1 means Train, 0 means simply Run
		state_dim = (*C.dims, 3)
		max_steps = 50
		OU = ln.OU()       #Ornstein-Uhlenbeck Process
		EXPLORE = 10000.
		episode_count = 2000

		replay_conf = {'size': 10000,
				'learn_start': 100,
				'partition_num': 100,
				'total_step': 10000,
				'batch_size': 4}
		BATCH_SIZE = replay_conf["batch_size"]

		self.load_models(replay_conf)

		unet_buff = ln.UniformReplay(1000)
		try:
			buff = hf.pickle_load(join(C.model_dir, "replay_buffer.bin"))
		except:
			buff = rrep.Experience(replay_conf)    #Create replay buffer

		g_step = 1
		for i in range(episode_count):
			print("Episode %d" % i, "Replay Buffer %d" % buff.record_size)

			img, true_seg, true_cls = next(dqn_generator)
			self.env.reset(img, true_seg, true_cls)

			s_t = self.env.get_state()

			total_reward = 0.
			steps = max_steps
			for j in range(max_steps):
				if self.epsilon > 0:
					self.epsilon -= 1.0 / EXPLORE
				noise_t = np.zeros(self.action_dim)
				
				a_t = self.actor.model.predict(np.expand_dims(s_t,0))[0]
				a_log = -np.log(1/np.clip(a_t,.01,.99) - 1)
				for ix in range(self.action_dim):
					noise_t[ix] = self.epsilon * OU.function(a_log[ix], 0,.3,3.) #shift, scale, gaussian
				a_t = 1/(1+np.exp(-a_log-noise_t))

				s_t1, r_t, done, cropI, crop_true_seg = self.env.step(a_t, get_crops=True)

				unet_buff.store(cropI, crop_true_seg, self.env.true_cls)
				buff.store(s_t, a_t, r_t, s_t1, done)

				#"Replay" for UNet
				if j % 2 == 0:
					batch = unet_buff.sample(BATCH_SIZE)
					cropI = np.asarray([e[0] for e in batch])
					crop_true_seg = np.asarray([e[1] for e in batch])
					true_cls = np.asarray([e[2] for e in batch])

					#cropI, crop_true_seg, true_cls = augment_crops(cropI, crop_true_seg, true_cls)
					uloss = self.env.train_model.train_on_batch([cropI, crop_true_seg, true_cls], None)

				#Replay for Q learning
				if buff.record_size > buff.learn_start:
					self.replay_Q(buff, g_step)

				if verbose:
					print("\tAction", a_t, "Reward %.1f" % r_t)
			
				total_reward += r_t
				s_t = s_t1
				g_step += 1
				if done:
					steps = j+1
					break

			if buff.record_size > buff.learn_start:
				buff.rebalance()

				if i % 3 == 2:
					self.save_weights()
					hf.pickle_dump(buff, join(C.model_dir, "replay_buffer.bin"))

					del self.actor, self.critic
					K.clear_session()

					self.load_models(replay_conf)
					memory()
					
			print("TOTAL REWARD: %.1f" % total_reward, "(%d steps)\n" % steps)

				
		self.save_weights()
		K.clear_session()
