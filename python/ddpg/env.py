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
importlib.reload(tn)
class Env(object):
	def __init__(self, lr=.0001):
		self.min_x = 5
		self.pred_model, self.train_model = tn.get_models(lr)

	def reset(self, img, true_seg=None, true_cls=None):
		"""Set new image to train on or predict.
		If training, include ground truth segs and cls."""
		self.img = tr.rescale_img(img, C.context_dims)
		self.pred_seg = np.zeros((*C.context_dims, 2))
		self.pred_seg_var = np.ones(C.context_dims)
		self.pred_cls = np.zeros(len(C.cls_names))
		self.pred_cls_var = 1
		self.sum_w_seg = np.ones(C.context_dims) * 1e-5
		self.sum_w_cls = 1e-5
		self.art_small = tr.rescale_img(self.img[...,0], C.dims)

		# Training parameters
		self.true_seg = true_seg
		self.true_cls = true_cls
		self.last_loss = 1e50

		init_action = [.5]*3+[1]*3+[.1,0]
		self.step(init_action)

	def get_bbox(self, action):
		"""Set focus bounding box based on the action."""
		center = action[:3] * np.array(self.img.shape[:3])
		dx = action[3:6] * np.array(self.img.shape[:3]) / 2
		bbox = [[round(center[i]-dx[i]),
			round(center[i]+dx[i])] for i in range(3)]
		for i in range(3):
			bbox[i][1] = max(self.min_x+1, min(self.img.shape[i] - 1, bbox[i][1]))
			bbox[i][0] = max(1, min(bbox[i][1] - self.min_x, bbox[i][0]))
		return np.array(bbox, int).flatten()

	def run_unet_cls(self, action, train_unet=False):
		# For now, w_seg and w_cls formulas don't make sense. Needs Bayesian treatment
		sl = [slice(self.bbox[i], self.bbox[i+1]) for i in [0,2,4]]
		D = [self.bbox[i+1] - self.bbox[i] for i in [0,2,4]]
		cropI = tr.rescale_img(self.img[sl], C.dims)
		#action[6] is avg(T1,T2) in [-.5,.5], action[7] is T2-T1
		T = [max(action[6] - action[7] - .55, -1), min(action[6] + action[7] - .45, 1)]
		cropI[cropI < T[0]] = T[0]
		cropI[cropI > T[1]] = T[1]
		cropI = tr.normalize_intensity(cropI,1,-1, ignore_empty=True)

		crop_pred_seg, crop_pred_cls = self.pred_model.predict(np.expand_dims(cropI, 0))
		crop_pred_seg = tr.rescale_img(crop_pred_seg[0], D)

		crop_pred_seg_var = np.exp(np.clip(crop_pred_seg[..., -1], -10, 50))
		crop_pred_cls_var = np.exp(np.clip(crop_pred_cls[0, -1], -10, 50))
		crop_pred_seg = crop_pred_seg[..., :-1] #logits
		crop_pred_cls = crop_pred_cls[0, :-1] #logits
		w_seg = 1/crop_pred_seg_var
		w_cls = action[-2] / crop_pred_cls_var
		#self.pred_seg[sl] += crop_pred_seg * np.tile(w_seg, (2,1,1,1)).transpose((1,2,3,0))
		#self.pred_cls += crop_pred_cls * w_cls

		# Update variances of the mean
		self.pred_seg[sl] = self.pred_seg[sl] * np.tile(self.sum_w_seg[sl], (2,1,1,1)).transpose((1,2,3,0))
		self.pred_seg_var[sl] = self.pred_seg_var[sl] * self.sum_w_seg[sl]
		self.pred_cls *= self.sum_w_cls
		self.pred_cls_var *= self.sum_w_cls
		self.sum_w_seg[sl] += w_seg
		self.sum_w_cls += w_cls
		self.pred_seg[sl] = (self.pred_seg[sl] + crop_pred_seg * \
						np.tile(w_seg, (2,1,1,1)).transpose((1,2,3,0))) / \
						np.tile(self.sum_w_seg[sl], (2,1,1,1)).transpose((1,2,3,0))
		self.pred_seg_var[sl] = (self.pred_seg_var[sl] + crop_pred_seg_var * w_seg) / self.sum_w_seg[sl]
		self.pred_cls = (self.pred_cls + crop_pred_cls * w_cls) / self.sum_w_cls
		self.pred_cls_var = (self.pred_cls_var + crop_pred_cls_var * w_cls) / self.sum_w_cls

		self.pred_seg_var = np.clip(self.pred_seg_var, 1e-3, 1e5)
		self.pred_cls_var = np.clip(self.pred_cls_var, 1e-3, 1e5)

		if train_unet:
			return sl, cropI

	def get_state(self):
		return np.stack([self.art_small, tr.rescale_img(self.pred_seg[...,-1] / 10, C.dims),
			tr.rescale_img(np.log(self.pred_seg_var), C.dims)], -1)

	def get_loss(self):
		log_vars = [K.get_value(x)[0] for x in self.train_model.layers[-1].log_vars]

		y_true = self.true_seg
		y_pred = np.concatenate([self.pred_seg, np.expand_dims(np.log(self.pred_seg_var), -1)], -1)
		num_classes = 2
		loss = np.sum(np.exp(-log_vars[0]) * K.get_value(tn.hetero_cls_loss(y_true, y_pred,
						num_classes=num_classes)) + log_vars[0], -1)

		y_true = self.true_cls
		y_pred = np.array(list(self.pred_cls) + [math.log(self.pred_cls_var)])
		num_classes = len(C.cls_names)
		loss += np.sum(np.exp(-log_vars[1]) * K.get_value(tn.hetero_cls_loss(y_true, y_pred,
						num_classes=num_classes)) + log_vars[1], -1)

		return np.mean(loss)

	def step(self, action, get_crops=False):
		self.bbox = self.get_bbox(action)
		if self.true_seg is None:
			self.run_unet_cls(action)
			return self.get_state(), action[-1] > .95

		sl, cropI = self.run_unet_cls(action, train_unet=True)
		crop_true_seg = tr.rescale_img(self.true_seg[sl], C.dims)
		self.train_model.train_on_batch([np.expand_dims(cropI,0),
			np.expand_dims(crop_true_seg,0), np.expand_dims(self.true_cls,0)], None)

		cur_loss = self.get_loss()

		if action[-1] > .95:
			done = True
			reward = 0
		else:
			done = False
			reward = np.clip(2*(self.last_loss - cur_loss) - .5, -15,15)
			self.last_loss = cur_loss

		next_state = self.get_state()
		if get_crops:
			return next_state, reward, done, cropI, crop_true_seg

		return next_state, reward, done
