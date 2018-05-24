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
import copy
import config
import dr_methods as drm
import feature_interpretation as cnna
import importlib
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
import tensorflow as tf
import voi_methods as vm

def get_DICE(A, B):
	dice = 1
	for i in [0,2,4]:
		num = (B[i+1]-A[i]) * (A[i]>B[i] and A[i]<B[i+1]) + (A[i+1]-B[i]) * (B[i]>A[i] and B[i]<A[i+1])
		if num == 0:
			return 0
		frac = num / (A[i+1]-A[i] + B[i+1]-B[i])
		dice *= frac
	return dice

class DQNEnv:
	def __init__(self, state_size):
		self.state_size = state_size
		self.dice_thresh = 0.5
		self.min = 16

	def set_img(self, img, true_bbox=None, save_path=None, true_seg=None):
		self.img = img[0]
		self.ix = 0
		self.save_path = save_path
		self.true_bbox = true_bbox[0]
		self.true_seg = true_seg
		self.best_dice = 0.001
		self.pred_bbox = np.array([[self.img.shape[i]//4, self.img.shape[i]*3//4] for i in range(3)]).flatten()
		self.center = np.array([(self.pred_bbox[i] + self.pred_bbox[i+1])/2 for i in [0,2,4]])
		self.dx = np.array([self.pred_bbox[i+1] - self.pred_bbox[i] for i in [0,2,4]], float)

		sl = [slice(self.pred_bbox[i], self.pred_bbox[i+1]) for i in [0,2,4]]
		cropI = tr.rescale_img(self.img[sl], self.state_size)
		return np.expand_dims(cropI, 0)

	def get_bbox(self, action):
		self.center += action[:3] * (self.dx/2) # action of 1 traverses 1/2 of the dimension
		self.center -= action[3:6] * (self.dx/2)
		self.dx += action[6] * (self.dx/2)
		self.dx -= action[7] * (self.dx/2)
		self.pred_bbox = [[round(self.center[i]-self.dx[i]/2),
			round(self.center[i]+self.dx[i]/2)] for i in range(3)]
		for i in range(3):
			self.pred_bbox[i][1] = max(self.min+1, min(self.img.shape[i] - 1, self.pred_bbox[i][1]))
			self.pred_bbox[i][0] = max(1, min(self.pred_bbox[i][1] - self.min, self.pred_bbox[i][0]))
		self.pred_bbox = np.array(self.pred_bbox, int).flatten()

		self.center = np.array([(self.pred_bbox[i] + self.pred_bbox[i+1])/2 for i in [0,2,4]])
		self.dx = np.array([self.pred_bbox[i+1] - self.pred_bbox[i] for i in [0,2,4]], float)

	def step(self, action):
		self.get_bbox(action)

		if self.true_bbox is None:
			return next_state, action[-1] == 1

		dice = get_DICE(self.true_bbox, self.pred_bbox)
		sl = [slice(self.pred_bbox[i], self.pred_bbox[i+1]) for i in [0,2,4]]
		try:
			cropI = tr.rescale_img(self.img[sl], self.state_size)
		except:
			print(self.pred_bbox, self.img.shape)
			raise ValueError()
		done = False

		if action[-1] == 1:
			done = True
			if dice > self.dice_thresh:
				reward = 25
			elif dice > .8:
				reward = 100
			else:
				reward = -10
		elif dice > self.best_dice:
			reward = 10*(self.best_dice - dice)
			self.best_dice = dice
		elif dice < .1:
			reward = -1
		else:
			reward = -.1

		if dice > self.dice_thresh:
			np.save(self.save_path+str(self.ix), cropI)
			#seg_df = get_seg(self.pred_bbox, self.img)
			self.ix += 1

		next_state = np.expand_dims(cropI, 0)
		return next_state, reward, done
