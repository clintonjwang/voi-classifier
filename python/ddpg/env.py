"""
Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

import copy
import config
import importlib
import math
from math import log, ceil
import numpy as np
import operator
import os
import pandas as pd
import random
import time
import tensorflow as tf

import keras.backend as K
import keras.layers as layers
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils

import ddpg.task_nets as tn
import niftiutils.cnn_components as cnnc
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr

C = config.Config()
class DQNEnv:
	def __init__(self):
		self.pred_model, train_model = tn.get_models()

	def reset(self, img, true_seg=None, true_cls=None, save_path=None):
		self.img = tr.rescale_img(img, C.context_dims)
		self.save_path = save_path
		self.true_seg = true_seg
		self.true_cls = true_cls
		self.last_loss = 0
		self.center = np.array([(self.bbox[i] + self.bbox[i+1])/2 for i in [0,2,4]])
		self.dx = np.array([self.bbox[i+1] - self.bbox[i] for i in [0,2,4]], float)

		self.train_unet_cls()

		sl = [slice(self.bbox[i], self.bbox[i+1]) for i in [0,2,4]]
		self.cropI = tr.rescale_img(self.img[sl], C.dims)

	def update_bbox(self, action):
		self.center += action[:3] * (self.dx/2) # action of 1 traverses 1/2 of the dimension
		self.center -= action[3:6] * (self.dx/2)
		self.dx += action[6] * (self.dx/2)
		self.dx -= action[7] * (self.dx/2)
		self.bbox = [[round(self.center[i]-self.dx[i]/2),
			round(self.center[i]+self.dx[i]/2)] for i in range(3)]
		for i in range(3):
			self.bbox[i][1] = max(self.min+1, min(self.img.shape[i] - 1, self.bbox[i][1]))
			self.bbox[i][0] = max(1, min(self.bbox[i][1] - self.min, self.bbox[i][0]))
		self.bbox = np.array(self.bbox, int).flatten()

		self.center = np.array([(self.bbox[i] + self.bbox[i+1])/2 for i in [0,2,4]])
		self.dx = np.array([self.bbox[i+1] - self.bbox[i] for i in [0,2,4]], float)

	def train_unet_cls(bbox=None):
		self.train_model.fit([self.cropI, self.crop_true_seg, self.true_cls])

	def run_unet_cls(bbox=None):
		self.pred_seg, self.pred_cls = self.pred_model.predict(self.cropI)

	def step(self, action):
		self.update_bbox(action)

		if self.true_seg is None:
			return next_state, action[-1] == 1

		sl = [slice(self.bbox[i], self.bbox[i+1]) for i in [0,2,4]]
		try:
			cropI = tr.rescale_img(self.img[sl], self.state_size)
		except:
			print(self.bbox, self.img.shape)
			raise ValueError()
		done = False

		if action[-1] == 1:
			done = True
			reward = 0
		elif dice > self.best_dice:
			reward = 10*(self.last_loss - cur_loss)

		if dice > self.dice_thresh:
			np.save(self.save_path+str(self.ix), cropI)
			#seg_df = get_seg(self.bbox, self.img)
			self.ix += 1

		next_state = np.expand_dims(cropI, 0)
		return next_state, reward, done
