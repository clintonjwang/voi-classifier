"""
Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

import keras.backend as K
import keras.layers as layers
from keras.layers import Input, Dense, Concatenate, Flatten, Dropout, Lambda, SimpleRNN, Activation, ELU, TimeDistributed, Permute, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils

import importlib
import feature_interpretation as cnna
import cnn_builder as cbuild
import copy
import glob
import config
import niftiutils.helper_fxns as hf
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

import dr_methods as drm
import voi_methods as vm

####################################
### OVERNIGHT PROCESSES
####################################

importlib.reload(config)
importlib.reload(cbuild)
C = config.Config()

def hyperband():
	# https://arxiv.org/abs/1603.06560
	max_iter = 30  # maximum iterations/epochs per configuration
	eta = 3 # defines downsampling rate (default=3)
	logeta = lambda x: log(x)/log(eta)
	s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
	B = (s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

	#### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
	for s in reversed(range(s_max+1)):
		n = int(ceil(B/max_iter/(s+1)*eta**s)) # initial number of configurations
		r = max_iter*eta**(-s) # initial number of iterations to run configurations for

		#### Begin Finite Horizon Successive Halving with (n,r)
		T = [ get_random_hyperparameter_configuration() for i in range(n) ]
		for i in range(s+1):
			# Run each of the n_i configs for r_i iterations and keep best n_i/eta
			n_i = n*eta**(-i)
			r_i = r*eta**(i)
			val_losses = [ run_then_return_val_loss(num_iters=r_i, hyperparams=t) for t in T ]
			T = [ T[i] for i in argsort(val_losses)[0:int( n_i/eta )] ]
		#### End Finite Horizon Successive Halving with (n,r)
	return val_losses, T

def get_run_stats_csv():
	C = config.Config()
	try:
		running_stats = pd.read_csv(C.run_stats_path)
		index = len(running_stats)
	except FileNotFoundError:
		running_stats = pd.DataFrame(columns = ["n", "steps_per_epoch", "epochs",
			"test_num", "augment_factor", "non_imaging_inputs",
			"kernel_size", "conv_filters", "conv_padding",
			"dropout", "dense_units", "pooling",
			"acc6cls", "acc3cls", "time_elapsed(s)", "loss_hist"] + \
			C.cls_names + \
			['confusion_matrix', 'timestamp',
			'miscls_test', 'miscls_train', 'model_num',
			'y_true', 'y_pred_raw', 'z_test'])

	return running_stats

def run_fixed_hyperparams(overwrite=False, max_runs=999, T=None, model_name='models_'):
	"""Runs the CNN for max_runs times, saving performance metrics."""

	def _get_hyperparams_as_list(T):
		return [T.n, T.steps_per_epoch, T.epochs,
				C.test_num, C.aug_factor, C.non_img_inputs,
				T.kernel_size, T.f, T.padding,
				T.dropout, T.dense_units, T.pool_sizes]

	if overwrite and exists(C.run_stats_path):
		os.remove(C.run_stats_path)
	running_stats = get_run_stats_csv()
	index = len(running_stats)

	model_names = glob.glob(join(C.model_dir, model_name+"*"))
	if len(model_names) > 0:
		model_num = max([int(x[x.find('_')+1:x.find('.')]) for x in model_names if 'reader' not in x]) + 1
	else:
		model_num = 0

	running_acc_6 = []
	running_acc_3 = []
	early_stopping = EarlyStopping(monitor='loss', min_delta=0.001, patience=5)

	C_index = 0
	while index < max_runs:
		#run_then_return_val_loss(num_iters=1, hyperparams=None)

		if T is None:
			T = config.Hyperparams()

		X_test, Y_test, train_generator, num_samples, train_orig, Z = cbuild.get_cnn_data(n=T.n)
		Z_test, Z_train_orig = Z
		X_train_orig, Y_train_orig = train_orig
		model = cbuild.build_cnn_hyperparams(T)
		#print(model.summary())
		#return
		t = time.time()
		hist = model.fit_generator(train_generator, steps_per_epoch=T.steps_per_epoch,
				epochs=T.epochs, callbacks=[T.early_stopping], verbose=False)
		loss_hist = hist.history['loss']

		Y_pred = model.predict(X_train_orig)
		y_true = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_train_orig])
		y_pred = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_pred])
		miscls_train = list(Z_train_orig[~np.equal(y_pred, y_true)])

		Y_pred = model.predict(X_test)
		y_true = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_test])
		y_pred = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_pred])
		miscls_test = list(Z_test[~np.equal(y_pred, y_true)])
		cm = confusion_matrix(y_true, y_pred)
		f1 = f1_score(y_true, y_pred, average="weighted")

		running_acc_6.append(accuracy_score(y_true, y_pred))
		print("Accuracy: %d%% (avg: %d%%)" % (running_acc_6[-1]*100, np.mean(running_acc_6)*100))

		y_true_simp, y_pred_simp, _ = cnna.merge_classes(y_true, y_pred, C.cls_names)
		running_acc_3.append(accuracy_score(y_true_simp, y_pred_simp))
		#print("3cls accuracy:", running_acc_3[-1], " - average:", np.mean(running_acc_3))

		running_stats.loc[index] = _get_hyperparams_as_list(T) + [running_acc_6[-1], running_acc_3[-1], time.time()-t, loss_hist] +\
							[num_samples[k] for k in C.cls_names] + \
							[cm, time.time(), #C.run_num,
							miscls_test, miscls_train, model_num, y_true, str(Y_pred), list(Z_test)]

		running_stats.to_csv(C.run_stats_path, index=False)

		model.save(join(C.model_dir, model_name+'%d.hdf5' % model_num))
		model_num += 1
		index += 1
		#end cycle_len
		C_index += 1

####################################
### BUILD CNNS
####################################

def run_then_return_val_loss(num_iters=1, hyperparams=None):
	"""Runs the CNN indefinitely, saving performance metrics."""
	C = config.Config()
	running_stats = pd.read_csv(C.run_stats_path)
	index = len(running_stats)

	X_test, Y_test, train_generator, num_samples, train_orig, Z = get_cnn_data(n=T.n,
				n_art=T.n_art, run_2d=T.run_2d)
	#Z_test, Z_train_orig = Z
	#X_train_orig, Y_train_orig = train_orig

	T = hyperparams
	model = build_cnn_hyperparams(T)

	t = time.time()
	hist = model.fit_generator(train_generator, steps_per_epoch=T.steps_per_epoch,
			epochs=num_iters, callbacks=[T.early_stopping], verbose=False, validation=[X_test, Y_test])
	loss_hist = hist.history['val_loss']

	return loss_hist

def get_random_hyperparameter_configuration():
	T = config.Hyperparams()

	return T
