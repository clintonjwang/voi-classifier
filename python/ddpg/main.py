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
	gigs = py.memory_info()[0]/2.**30
	print('Memory use:', gigs)
	return gigs

def augment_crops(cropI, crop_true_seg, true_cls):
	batch_size = cropI.shape[0]
	I = np.empty((batch_size*4, *cropI.shape[1:]))
	S = np.empty((batch_size*4, *crop_true_seg.shape[1:]))
	C = np.empty((batch_size*4, *true_cls.shape[1:]))

	I[:batch_size] = cropI
	S[:batch_size] = crop_true_seg
	C[:] = true_cls[0]

	ix = batch_size
	for k in range(1,4):
		for img, seg in zip(cropI, crop_true_seg):
			axes = (1,0)#random.choice([(1,0),(2,0),(2,1)]) if fully symmetrical
			sl = [slice(None, None, (-1)**(random.random() < .5)) for _ in range(3)]

			I[ix] = np.rot90(img[sl], k, axes) + np.random.normal(size=cropI.shape[1:]) * random.random()/5
			S[ix] = np.rot90(seg[sl], k, axes)

			ix += 1

	return I, S, C

class CRSNet(object): #ClsRegSegNet
	def __init__(self, eps=1):
		self.epsilon = eps
		self.action_dim = 10  #6 bbox coordinates, 2 thresholds, cls weight, trigger

	def save_models(self):
		try:
			hf.pickle_dump(self.q_buff, join(C.model_dir, "replay_buffer.bin"))
			hf.pickle_dump(self.u_buff, join(C.model_dir, "unet_buffer.bin"))
			self.actor.model.save_weights(join(C.model_dir, "actor.h5"))
			self.critic.model.save_weights(join(C.model_dir, "critic.h5"))
			self.actor.target_model.save_weights(join(C.model_dir, "actor_T.h5"))
			self.critic.target_model.save_weights(join(C.model_dir, "critic_T.h5"))
			self.env.train_model.save_weights(join(C.model_dir, "train.h5"))
			self.env.pred_model.save_weights(join(C.model_dir, "pred.h5"))
		except:
			print("Unable to save weights")

	def load_models(self, replay_conf):
		TAU = .01     #Target Network update weight
		LRA = .0001    #Learning rate for Actor
		LRC = .00015    #Learning rate for Critic
		LRU = .00015    #Learning rate for Unet

		BATCH_SIZE = replay_conf["batch_size"]

		conf = tf.ConfigProto()
		conf.gpu_options.allow_growth = True
		sess = tf.Session(config=conf)
		K.set_session(sess)

		self.actor = ln.ActorNetwork(sess, C.state_dim, self.action_dim, BATCH_SIZE, TAU, LRA)
		self.critic = ln.CriticNetwork(sess, C.state_dim, self.action_dim, BATCH_SIZE, TAU, LRC)
		self.env = denv.Env(sess, LRU)
		self.u_buff = ln.UniformReplay(3000)

		try:
			self.load_weights()
		except:
			pass

		sess.run(tf.global_variables_initializer())

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
		max_steps = 100

		conf = tf.ConfigProto()
		conf.gpu_options.allow_growth = True
		sess = tf.Session(config=conf)
		K.set_session(sess)
		self.actor = ln.ActorNetwork(sess, C.state_dim, self.action_dim)
		self.critic = None
		self.env = denv.Env(sess)
		self.load_weights()
		sess.run(tf.global_variables_initializer())

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
		den = np.sum(np.exp(seg),-1)
		liver_seg = np.exp(seg[...,1]) / den
		tumor_seg = np.exp(seg[...,-1]) / den
		cls = np.argmax(self.env.pred_cls)

		return liver_seg, tumor_seg, self.env.pred_seg_var, cls

	def transform_action(self, action):
		a_Tx = np.zeros(C.context_dims)
		bbox = env.get_bbox(action)
		sl = [slice(bbox[i], bbox[i+1]) for i in [0,2,4]]
		a_Tx[sl] = 1

		return np.concatenate([a_Tx.flatten(), action[6:]])

	def replay_Q(self, g_step, GAMMA=.99):
		batch, _, ixs = self.q_buff.sample(g_step)
		states = np.asarray([e[0] for e in batch])
		actions = np.asarray([e[1] for e in batch])
		rewards = np.asarray([e[2] for e in batch])
		new_states = np.asarray([e[3] for e in batch])
		dones = np.asarray([e[4] for e in batch])
		y_t = np.zeros(actions.shape[0])

		target_a = self.actor.target_model.predict(new_states)
		#self.transform_action(self.actor.target_model.predict(new_states))
		target_q_values = self.critic.target_model.predict([new_states, target_a]).flatten()

		#temporal difference error for prioritized exp replay
		TD = self.critic.model.predict([states, actions]).flatten()
		for k in range(len(batch)):
			if dones[k]:
				y_t[k] = rewards[k]
			else:
				y_t[k] = rewards[k] + GAMMA*target_q_values[k]
		TD = np.abs(TD - y_t)
		self.q_buff.update_priority(ixs, TD)

		closs = self.critic.model.train_on_batch([states,actions], np.expand_dims(y_t,-1))
		a_for_grad = self.actor.model.predict(states)
		grads = self.critic.gradients(states, a_for_grad)
		self.actor.train(states, grads)
		self.actor.target_train()
		self.critic.target_train()

	def train(self, dqn_generator, verbose=False):    #1 means Train, 0 means simply Run
		max_steps = 50
		noise_gen = OU()       #Ornstein-Uhlenbeck Process
		EXPLORE = 10000.
		episode_count = 2000

		replay_conf = {'size': 10000,
				'learn_start': 100,
				'partition_num': 100,
				'total_step': 10000,
				'batch_size': 4}
		BATCH_SIZE = replay_conf["batch_size"]

		self.load_models(replay_conf)

		if exists(join(C.model_dir, "replay_buffer.bin")):
			self.q_buff = hf.pickle_load(join(C.model_dir, "replay_buffer.bin"))
		else:
			self.q_buff = rrep.Experience(replay_conf)
		if exists(join(C.model_dir, "unet_buffer.bin")):
			self.u_buff = hf.pickle_load(join(C.model_dir, "unet_buffer.bin"))
		else:
			self.u_buff = ln.UniformReplay(1000)

		g_step = 1
		for i in range(episode_count):
			print("Episode %d" % i, "Replay self.q_buffer %d" % self.q_buff.record_size)

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
				for ix in list(range(6))+[-1]:
					noise_t[ix] = self.epsilon * noise_gen.function(a_log[ix], 0,.7,1.5) #shift, scale, gaussian
				for ix in [6,7,8]:
					noise_t[ix] = self.epsilon * noise_gen.function(a_log[ix], 0,.5,1.5) #shift, scale, gaussian
				a_t = 1/(1+np.exp(-a_log-noise_t))

				s_t1, r_t, done, cropI, crop_true_seg = self.env.step(a_t, get_crops=True)

				if crop_true_seg.sum() > 0:
					self.u_buff.store(cropI, crop_true_seg, self.env.true_cls)
				self.q_buff.store(s_t, a_t, r_t, s_t1, done)

				#"Replay" for UNet
				if j % 2 == 0:
					batch = self.u_buff.sample(BATCH_SIZE)
					cropI = np.asarray([e[0] for e in batch])
					crop_true_seg = np.asarray([e[1] for e in batch])
					true_cls = np.asarray([e[2] for e in batch])

					cropI, crop_true_seg, true_cls = augment_crops(cropI, crop_true_seg, true_cls)
					uloss = self.env.train_model.train_on_batch([cropI, crop_true_seg, true_cls], None)

				#Replay for Q learning
				if self.q_buff.record_size > self.q_buff.learn_start:
					self.replay_Q(g_step)

				if verbose:
					print("\tAction", a_t, "Reward %.1f" % r_t)
			
				total_reward += r_t
				s_t = s_t1
				g_step += 1
				if done:
					steps = j+1
					break

			if self.q_buff.record_size > self.q_buff.learn_start:
				self.q_buff.rebalance()

				memory()
				if i % 2 == 0:
					self.save_models()

			print("TOTAL REWARD: %.1f" % total_reward, "(%d steps)\n" % steps)

				
		self.save_models()
		K.clear_session()

class OU(object):
	def function(self, x, mu, theta, sigma):
		return theta * (mu - x) + sigma * np.random.randn(1)
