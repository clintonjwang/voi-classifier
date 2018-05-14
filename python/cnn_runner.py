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

import argparse
import feature_interpretation as cnna
import cnn_builder as cbuild
import copy
import config
import niftiutils.helper_fxns as hf
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

import dr_methods as drm
import voi_methods as vm

####################################
### OVERNIGHT PROCESSES
####################################

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
			C.classes_to_include + \
			['confusion_matrix', 'timestamp',
			'misclassified_test', 'misclassified_train', 'model_num',
			'y_true', 'y_pred_raw', 'z_test'])

	return running_stats

def run_fixed_hyperparams(overwrite=False, max_runs=999, hyperparams=None):
	"""Runs the CNN for max_runs times, saving performance metrics."""
	C_list = [config.Config()]

	running_stats = get_run_stats_csv()
	index = len(running_stats)

	model_names = os.listdir(C_list[0].model_dir)
	if len(model_names) > 0:
		model_num = max([int(x[x.find('_')+1:x.find('.')]) for x in model_names if 'reader' not in x]) + 1
	else:
		model_num = 0

	running_acc_6 = []
	running_acc_3 = []
	"""n = [4]
			n_art = [0]
			steps_per_epoch = [750]
			epochs = [25]
			run_2d = False
			f = [[64,128,128]]
			padding = [['valid','valid']]
			dropout = [[0.1,0.1]]
			dense_units = [128]
			dilation_rate = [(1,1,1)]
			kernel_size = [(3,3,2)]
			pool_sizes = [(2,2,1),(1,1,2)]
			activation_type = ['elu']
			merge_layer = [0]
			cycle_len = 1"""
	early_stopping = EarlyStopping(monitor='loss', min_delta=0.002, patience=3)
	time_dist = True

	C_index = 0
	while index < max_runs:
		C = C_list[C_index % len(C_list)]
		#C.hard_scale = False

		#run_then_return_val_loss(num_iters=1, hyperparams=None)

		if hyperparams is not None:
			T = hyperparams

			X_test, Y_test, train_generator, num_samples, train_orig, Z = cbuild.get_cnn_data(n=T.n,
						n_art=T.n_art, run_2d=T.run_2d)
			Z_test, Z_train_orig = Z
			X_train_orig, Y_train_orig = train_orig
			model = cbuild.build_cnn_hyperparams(T)
			#print(model.summary())
			#return
			t = time.time()
			hist = model.fit_generator(train_generator, steps_per_epoch=T.steps_per_epoch,
					epochs=T.epochs, callbacks=[T.early_stopping], verbose=False)
			loss_hist = hist.history['loss']

		else:
			X_test, Y_test, train_generator, num_samples, train_orig, Z = cbuild.get_cnn_data(n=n[C_index % len(n)],
						n_art=n_art[C_index % len(n_art)], run_2d=run_2d)
			Z_test, Z_train_orig = Z
			X_train_orig, Y_train_orig = train_orig
		#for _ in range(cycle_len):
			model = cbuild.build_cnn('adam', activation_type=activation_type[index % len(activation_type)],
					f=f[index % len(f)], pool_sizes=pool_sizes,
					padding=padding[index % len(padding)], dropout=dropout[index % len(dropout)],
					dense_units=dense_units[index % len(dense_units)], kernel_size=kernel_size[index % len(kernel_size)],
					merge_layer=merge_layer[index % len(merge_layer)], dual_inputs=C.non_imaging_inputs)
		
			t = time.time()
			hist = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch[index % len(steps_per_epoch)],
					epochs=epochs[index % len(epochs)], callbacks=[early_stopping], verbose=False)
			loss_hist = hist.history['loss']

		Y_pred = model.predict(X_train_orig)
		y_true = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_train_orig])
		y_pred = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_pred])
		misclassified_train = list(Z_train_orig[~np.equal(y_pred, y_true)])

		Y_pred = model.predict(X_test)
		y_true = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_test])
		y_pred = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_pred])
		misclassified_test = list(Z_test[~np.equal(y_pred, y_true)])
		cm = confusion_matrix(y_true, y_pred)
		f1 = f1_score(y_true, y_pred, average="weighted")

		running_acc_6.append(accuracy_score(y_true, y_pred))
		print("6cls accuracy:", running_acc_6[-1], " - average:", np.mean(running_acc_6))

		y_true_simp, y_pred_simp, _ = cnna.merge_classes(y_true, y_pred, C.classes_to_include)
		running_acc_3.append(accuracy_score(y_true_simp, y_pred_simp))
		#print("3cls accuracy:", running_acc_3[-1], " - average:", np.mean(running_acc_3))

		if hyperparams is not None:
			running_stats.loc[index] = _get_hyperparams_as_list(C, T) + [running_acc_6[-1], running_acc_3[-1], time.time()-t, loss_hist] +\
								[num_samples[k] for k in C.classes_to_include] + \
								[cm, time.time(), #C.run_num,
								misclassified_test, misclassified_train, model_num, y_true, str(Y_pred), list(Z_test)]

		else:
			running_stats.loc[index] = [n[C_index % len(n)], steps_per_epoch[index % len(steps_per_epoch)], epochs[index % len(epochs)],
								C.train_frac, C.test_num, C.aug_factor, C.non_imaging_inputs,
								kernel_size[index % len(kernel_size)], f[index % len(f)], padding[index % len(padding)],
								dropout[index % len(dropout)], dense_units[index % len(dense_units)],
								running_acc_6[-1], running_acc_3[-1], time.time()-t, loss_hist] +\
								[num_samples[k] for k in C.classes_to_include] + \
								[cm, time.time(), #C.run_num,
								misclassified_test, misclassified_train, model_num, y_true, str(Y_pred), list(Z_test)]
		running_stats.to_csv(C.run_stats_path, index=False)

		model.save(C.model_dir+'models_%d.hdf5' % model_num)
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

	"""	Y_pred = model.predict(X_train_orig)
	y_true = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_train_orig])
	y_pred = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_pred])
	misclassified_train = list(Z_train_orig[~np.equal(y_pred, y_true)])

	Y_pred = model.predict(X_test)
	y_true = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_test])
	y_pred = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_pred])
	misclassified_test = list(Z_test[~np.equal(y_pred, y_true)])
	cm = confusion_matrix(y_true, y_pred)
	f1 = f1_score(y_true, y_pred, average="weighted")
	acc_6cl = accuracy_score(y_true, y_pred)

	y_true_simp, y_pred_simp, _ = condense_cm(y_true, y_pred, C.classes_to_include)
	acc_3cl = accuracy_score(y_true_simp, y_pred_simp)

	running_stats.loc[index] = _get_hyperparams_as_list(C, T) + \
			[acc_6cl, acc_3cl, time.time()-t, loss_hist,
			num_samples['hcc'], num_samples['cholangio'], num_samples['colorectal'], num_samples['cyst'], num_samples['hemangioma'], num_samples['fnh'],
			cm, time.time(), misclassified_test, misclassified_train, model_num, y_true, str(Y_pred), list(Z_test)]
	running_stats.to_csv(C.run_stats_path, index=False)"""

	return loss_hist

def _get_hyperparams_as_list(C=None, T=None):
	if T is None:
		T = config.Hyperparams()
	if C is None:
		C = config.Config()
	
	return [T.n, T.steps_per_epoch, T.epochs,
			C.test_num, C.aug_factor, C.non_imaging_inputs,
			T.kernel_size, T.f, T.padding,
			T.dropout, T.dense_units, T.pool_sizes]

def get_random_hyperparameter_configuration():
	T = config.Hyperparams()

	return T
