"""
Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

from keras.models import Model
import keras.models
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from keras import backend as K
import tensorflow as tf

import cnn_builder as cbuild
import config
import copy
import csv
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import importlib
from math import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from numba import njit
import numpy as np
import operator
import os
from os.path import *
import pandas as pd
import random
import math
from sklearn.manifold import TSNE
import scipy.stats
import dr_methods as drm
import voi_methods as vm
import time

###########################
### Higher-level methods
###########################

class InfluenceAnalyzer:
	def __init__(self, M, voi_df, all_imgs, all_cls):
		self.M = M
		self.voi_df = voi_df
		self.all_imgs = all_imgs
		self.all_cls = all_cls

		W = M.get_weights()
		del W[20:22]
		del W[14:16]
		del W[8:10]
		del W[-4:-2]
		self.W = W

	def perturb_weights(self, t):
		W = copy.deepcopy(self.W)
		t_ix = 0
		
		for w_ix in range(len(self.W)):
			W[w_ix] += np.reshape(t[t_ix:t_ix+W[w_ix].size], W[w_ix].shape)
			t_ix += W[w_ix].size
			
		return W

	def get_grad(self, lesion_id, perturb_W=None, pred_cls=None):
		C = config.Config()
		
		cls = self.voi_df.loc[lesion_id]["cls"]
		img = np.load(join(C.orig_dir, cls, lesion_id+".npy"))
		img = np.expand_dims(img,0)

		if pred_cls is not None:
			cls = pred_cls

		y_true = np_utils.to_categorical(C.cls_names.index(cls), 6)
		loss = K.categorical_crossentropy(y_true, self.M.output)
		
		#with tf.device('/gpu:0'):
		g = K.gradients(loss, self.M.trainable_weights)
		#with tf.Session() as sess:
		#	sess.run(tf.global_variables_initializer())
		g_i = K.function([self.M.input, K.learning_phase()], g)([img, 0])

		#g_i = sess.run(g, feed_dict={self.M.input:img, K.learning_phase():0})
		#if perturb_W is not None:
		#	feed_dict_plus = {M.trainable_weights[i]:perturb_W[i] for i in range(len(perturb_W))}
		#	g_i_plus = sess.run(g, feed_dict={**feed_dict_plus, self.M.input:img, K.learning_phase():0})
			
		g_i = np.concatenate([x.flatten() for x in g_i], 0)
		if perturb_W is not None:
			g_i_plus = np.concatenate([x.flatten() for x in g_i_plus], 0)
			return g_i, g_i_plus
		
		return g_i

	def get_HVP(self, perturb_W, g_shape, eps=1e-5, verbose=False):
		Ht = np.zeros(g_shape)
		feed_dict_plus = {self.M.trainable_weights[i]:perturb_W[i] for i in range(len(perturb_W))}
		losses = [K.categorical_crossentropy(y_true, self.M.output) for y_true in \
				  [np_utils.to_categorical(i, 6) for i in range(6)]]
			
		t = time.time()
		with tf.device('/gpu:0'):
			g = [K.gradients(loss, self.M.trainable_weights) for loss in losses]
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for ix in range(len(self.all_cls)):
				g_i = sess.run(g[self.all_cls[ix]], feed_dict={self.M.input: self.all_imgs[ix], K.learning_phase():0})
				g_i_plus = sess.run(g[self.all_cls[ix]], feed_dict={**feed_dict_plus, self.M.input: self.all_imgs[ix], K.learning_phase():0})
		
				g_i = np.concatenate([x.flatten() for x in g_i], 0)
				g_i_plus = np.concatenate([x.flatten() for x in g_i_plus], 0)

				Ht += (g_i_plus - g_i)/eps

			sess.close()

		if verbose:
			print(time.time()-t)
		
		return Ht / len(self.all_cls)

	def get_stest(self, g_test, max_iters=50, tol=.05, eps=1e-5):
		t_k = np.zeros(g_test.shape)
		r_k = g_test#-Ht
		p_k = r_k
		d_0 = d_new = np.dot(r_k, r_k)
		i = 0

		while d_new > tol**2 * d_0 and i < range(max_iters):
			W_new = self.perturb_weights(p_k*eps)
			Hp = self.get_HVP(W_new, g_test.shape, eps)

			alpha = d_new / (np.dot(p_k, Hp) + 1e-10)
			t_k += alpha * p_k
			if i % 25 == 1:
				Ht = self.get_HVP(self.perturb_weights(t_k*eps), g_test.shape, eps)
				r_k2 = b - Ht
			else:
				r_k2 = r_k - alpha * Hp
			#phi_hist = .5*np.dot(t_k, Ht) - g_test

			beta = np.dot(r_k2, r_k2) / (d_new + 1e-10)
			r_k = r_k2
			p_k = r_k + beta*p_k
			d_new = np.dot(r_k, r_k)

			i += 1

		return t_k

	def get_avg_influence(self, Z_sample, s_test, verbose=False):
		C = config.Config()
		eps = 1e-5
		imgs = []
		classes = []

		for lesion_id in Z_sample:
			cls = self.voi_df.loc[lesion_id]["cls"]
			img = np.load(join(C.orig_dir, cls, lesion_id+".npy"))
			imgs.append(np.expand_dims(img,0))
			classes.append(C.cls_names.index(cls))

		Ht = np.zeros(s_test.shape)
		losses = [K.categorical_crossentropy(y_true, self.M.output) for y_true in \
				  [np_utils.to_categorical(i, 6) for i in range(6)]]
			
		I = 0
		t = time.time()
		with tf.device('/gpu:0'):
			g = [K.gradients(loss, self.M.trainable_weights) for loss in losses]
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for ix in range(len(Z_sample)):
				g_i = sess.run(g[classes[ix]], feed_dict={self.M.input:imgs[ix], K.learning_phase():0})
				g_i = np.concatenate([x.flatten() for x in g_i], 0)
				I -= np.dot(g_i, s_test)
				
			sess.close()

		if verbose:
			print(time.time()-t)
		
		return I / len(Z_sample)

