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
import copy
import config
import glob
import math
import numpy as np
import operator
import os
from os.path import *
import pandas as pd
import random
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import time

import dr_methods as drm
import voi_methods as vm
import feature_interpretation as cnna
import cnn_builder as cbuild
import niftiutils.helper_fxns as hf

importlib.reload(config)
importlib.reload(cbuild)

####################################
### RUN!
####################################

class CNNRunner():
	def __init__(self, C=None, T=None):
		if C is None:
			self.C = config.Config()
		else:
			self.C = C
		if T is None:
			self.T = config.Hyperparams()
			self.T.get_best_hyperparams(C.dataset)
		else:
			self.T = T

	def run_fixed_hyperparams(self, overwrite=False, max_runs=999, Z_test=None, model_name='models_'):
		"""Runs the CNN for max_runs times, saving performance metrics."""
		if overwrite and exists(self.C.run_stats_path):
			os.remove(self.C.run_stats_path)
		running_stats = get_run_stats_csv(self.C)
		index = len(running_stats)

		model_names = glob.glob(join(self.C.model_dir, model_name+"*"))
		if len(model_names) > 0:
			model_num = max([int(x[x.find('_')+1:x.find('.')]) for x in model_names]) + 1
		else:
			model_num = 0

		running_acc_6 = []

		for _ in range(max_runs):
			if self.C.aleatoric:
				self.pred_model, self.train_model = cbuild.build_cnn_hyperparams(self.T)
			else:
				self.pred_model = cbuild.build_cnn_hyperparams(self.T)
				self.train_model = self.pred_model

			X_test, Y_test, train_gen, num_samples, train_orig, Z = cbuild.get_cnn_data(n=self.T.n,
					Z_test_fixed=Z_test)
			self.Z_test, self.Z_train_orig = Z
			X_train_orig, Y_train_orig = train_orig

			t = time.time()
			hist = self.train_model.fit_generator(train_gen, self.T.steps_per_epoch,
					self.T.epochs, verbose=False)
			loss_hist = hist.history['loss']

			Y_pred = self.pred_model.predict(X_train_orig)
			y_true = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_train_orig])
			y_pred = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_pred])
			miscls_train = list(self.Z_train_orig[~np.equal(y_pred, y_true)])

			if self.C.aug_pred:
				x = np.empty((self.C.aug_factor, *self.C.dims, self.C.nb_channels))
				Y_pred = []
				for z in self.Z_test:
					x = np.stack([np.load(fn) for fn in glob.glob(join(C.aug_dir,"*")) if basename(fn).startswith(z)], 0)
					y = self.pred_model.predict(x)
					Y_pred.append(np.median(y, 0))
				Y_pred = np.array(Y_pred)
			elif self.T.mc_sampling:
				Y_pred = []
				for ix in range(len(self.Z_test)):
					x = np.tile(X_test[ix], (100, 1,1,1,1))
					y = self.pred_model.predict(x)
					Y_pred.append(np.median(y, 0))
				Y_pred = np.array(Y_pred)
			else:
				Y_pred = self.pred_model.predict(X_test)
			y_true = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_test])
			y_pred = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_pred])
			miscls_test = list(self.Z_test[~np.equal(y_pred, y_true)])

			cm = confusion_matrix(y_true, y_pred)
			#f1 = f1_score(y_true, y_pred, average="weighted")
			running_acc_6.append(accuracy_score(y_true, y_pred))
			print("Accuracy: %d%% (avg: %d%%), time: %ds" % (running_acc_6[-1]*100, np.mean(running_acc_6)*100, time.time()-t))

			if hasattr(self.C,'simplify_map'):
				y_true_simp, y_pred_simp = merge_classes(y_true, y_pred, self.C)
				acc_3 = accuracy_score(y_true_simp, y_pred_simp)
				row = _get_hyperparams_as_list(self.C, self.T) + [num_samples[k] for k in self.C.cls_names] + [running_acc_6[-1], acc_3]
			else:
				row = _get_hyperparams_as_list(self.C, self.T) + [num_samples[k] for k in self.C.cls_names] + [running_acc_6[-1]]
			
			running_stats.loc[index] = row + [loss_hist, cm, time.time()-t, time.time(),
							miscls_test, miscls_train, model_name+str(model_num), y_true, str(Y_pred), list(self.Z_test)]

			running_stats.to_csv(self.C.run_stats_path, index=False)
			self.pred_model.save(join(self.C.model_dir, model_name+'%d.hdf5' % model_num))
			model_num += 1
			index += 1

	def run_ensemble(self, overwrite=False, max_runs=999, Z_test=None, model_name='ensembles_'):
		"""Runs the CNN for max_runs times, saving performance metrics."""
		if overwrite and exists(self.C.run_stats_path):
			os.remove(self.C.run_stats_path)
		running_stats = get_run_stats_csv(self.C)
		index = len(running_stats)

		model_names = glob.glob(join(self.C.model_dir, model_name+"*"))
		if len(model_names) > 0:
			model_num = max([int(x[x.find('_')+1:x.rfind('_')]) for x in model_names]) + 1
		else:
			model_num = 0

		running_acc_6 = []

		for _ in range(max_runs):
			t = time.time()
			X_test, Y_test, train_gen, num_samples, self.train_orig, Z = cbuild.get_cnn_data(n=self.T.n,
					Z_test_fixed=Z_test)
			self.Z_test, self.Z_train_orig = Z
			X_train_orig, Y_train_orig = self.train_orig

			self.M = []
			for _ in range(self.C.ensemble_num):
				ensemble_train_gen = cbuild.train_gen_ensemble(n=self.T.n, Z_exc=list(self.Z_test) + \
						random.sample(list(self.Z_train_orig), round(len(self.Z_train_orig) * (1 - self.C.ensemble_frac))))

				if self.C.aleatoric:
					pred_model, train_model = cbuild.build_cnn_hyperparams(self.T)
				else:
					pred_model = cbuild.build_cnn_hyperparams(self.T)
					train_model = pred_model
				hist = train_model.fit_generator(ensemble_train_gen, self.T.steps_per_epoch,
						self.T.epochs, verbose=False)
				self.M.append(pred_model)

			Y_pred = []
			for ix in range(len(self.Z_train_orig)):
				y = np.concatenate([m.predict(X_train_orig[ix:ix+1]) for m in self.M], 0)
				Y_pred.append(np.median(y, 0))
			Y_pred = np.array(Y_pred)
			y_pred = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_pred])
			y_true = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_train_orig])
			miscls_train = list(self.Z_train_orig[~np.equal(y_pred, y_true)])

			if self.C.aug_pred:
				x = np.empty((self.C.aug_factor, *self.C.dims, self.C.nb_channels))
				Y_pred = []
				for z in self.Z_test:
					x = np.stack([np.load(fn) for fn in glob.glob(join(C.aug_dir,"*")) if basename(fn).startswith(z)], 0)
					y = np.concatenate([m.predict(x) for m in self.M], 0)
					Y_pred.append(np.median(y, 0))
				Y_pred = np.array(Y_pred)
			elif self.T.mc_sampling:
				Y_pred = []
				for ix in range(len(self.Z_test)):
					x = np.tile(X_test[ix], (100, 1,1,1,1))
					y = np.concatenate([m.predict(x) for m in self.M], 0)
					Y_pred.append(np.median(y, 0))
				Y_pred = np.array(Y_pred)
			else:
				for ix in range(len(self.Z_test)):
					y = np.concatenate([m.predict(X_test[ix:ix+1]) for m in self.M], 0)
					Y_pred.append(np.median(y, 0))
				Y_pred = np.array(Y_pred)
			y_true = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_test])
			y_pred = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_pred])
			miscls_test = list(self.Z_test[~np.equal(y_pred, y_true)])

			cm = confusion_matrix(y_true, y_pred)
			running_acc_6.append(accuracy_score(y_true, y_pred))
			print("Accuracy: %d%% (avg: %d%%), time: %ds" % (running_acc_6[-1]*100, np.mean(running_acc_6)*100, time.time()-t))

			if hasattr(self.C,'simplify_map'):
				y_true_simp, y_pred_simp, _ = cnna.merge_classes(y_true, y_pred, self.C.cls_names)
				acc_3 = accuracy_score(y_true_simp, y_pred_simp)
				row = _get_hyperparams_as_list(self.C, self.T) + [num_samples[k] for k in self.C.cls_names] + [running_acc_6[-1], acc_3]
			else:
				row = _get_hyperparams_as_list(self.C, self.T) + [num_samples[k] for k in self.C.cls_names] + [running_acc_6[-1]]
			
			running_stats.loc[index] = row + ['loss_hist', cm, time.time()-t, time.time(),
							miscls_test, miscls_train, model_name+str(model_num), y_true, str(Y_pred), list(self.Z_test)]

			running_stats.to_csv(self.C.run_stats_path, index=False)
			for ix in range(self.C.ensemble_num):
				self.M[ix].save(join(self.C.model_dir, model_name+'%d_%d.hdf5' % (model_num, ix)))
			model_num += 1
			index += 1

	def run_hyperparam_seq(overwrite=False, max_runs=999, T=None, model_name='models_'):
		"""Runs the CNN for max_runs times, saving performance metrics."""
		pass

def _get_hyperparams_as_list(C, T):
	return [T.n, T.steps_per_epoch, T.epochs,
			C.test_num, C.aug_factor, C.clinical_inputs,
			T.kernel_size, T.f, T.padding, T.dropout, T.dense_units,
			T.pool_sizes,
			T.cnn_type+C.aleatoric*'-al'+C.aug_pred*'-aug'+T.mc_sampling*'-mc'+'-foc%.1f'%C.focal_loss,
			C.ensemble_num,
			T.optimizer.get_config()['lr']]

def get_run_stats_csv(C):
	try:
		running_stats = pd.read_csv(C.run_stats_path)
		index = len(running_stats)
	except FileNotFoundError:
		running_stats = pd.DataFrame(columns = ["n", "steps_per_epoch", "epochs",
			"test_num", "augment_factor", "clinical_inputs",
			"kernel_size", "conv_filters", "conv_padding",
			"dropout", "dense_units", "pooling", "cnn_type", "ensemble_num", "learning_rate"] + \
			C.cls_names + ["acc6cls"] + ["acc3cls"]*(hasattr(C,'simplify_map')) + \
			["loss_hist", 'confusion_matrix', "time_elapsed(s)", 'timestamp',
			'miscls_test', 'miscls_train', 'model_num',
			'y_true', 'y_pred_raw', 'z_test'])

	return running_stats

def merge_classes(y_true, y_pred, C):
	"""From lists y_true and y_pred with class numbers, """
	y_true_simp = np.array([C.simplify_map[C.cls_names[y]] for y in y_true])
	y_pred_simp = np.array([C.simplify_map[C.cls_names[y]] for y in y_pred])
	
	return y_true_simp, y_pred_simp

####################################
### HYPERBAND
####################################

"""def hyperband():
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
	return val_losses, T"""

"""def run_then_return_val_loss(num_iters=1, hyperparams=None):
	C = config.Config()
	running_stats = pd.read_csv(C.run_stats_path)
	index = len(running_stats)

	X_test, Y_test, train_gen, num_samples, train_orig, Z = get_cnn_data(n=T.n,
				n_art=T.n_art, run_2d=T.run_2d)
	#Z_test, Z_train_orig = Z
	#X_train_orig, Y_train_orig = train_orig

	T = hyperparams
	model = build_cnn_hyperparams(T)

	t = time.time()
	hist = model.fit_generator(train_gen, steps_per_epoch=T.steps_per_epoch,
			epochs=num_iters, callbacks=[T.early_stopping], verbose=False, validation=[X_test, Y_test])
	loss_hist = hist.history['val_loss']

	return loss_hist"""

"""def get_random_hyperparameter_configuration():
	T = config.Hyperparams()

	return T"""
