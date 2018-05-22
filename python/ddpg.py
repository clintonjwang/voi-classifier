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
