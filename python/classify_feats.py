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

import cnn_builder as cbuild
import config
import copy
import csv
import niftiutils.io as io
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
from math import log, exp, sqrt
import random
import math
from sklearn.manifold import TSNE
import scipy.stats

import dr_methods as drm
import voi_methods as vm
import niftiutils.tf_nn.common as common

C = config.Config()
T = config.Hyperparams()
T.get_best_hyperparams(C.dataset)

###########################
### Higher-level methods
###########################

def feature_id_bulk(model_nums, model_prefix='fixZ-ens_'):
	Z_test = ['E103312835_1','12823036_0','12569915_0','E102093118_0','E102782525_0','12799652_0','E100894274_0','12874178_3','E100314676_0','12842070_0','13092836_2','12239783_0','12783467_0','13092966_0','E100962970_0','E100183257_1','E102634440_0','E106182827_0','12582632_0','E100121654_0','E100407633_0','E105310461_0','12788616_0','E101225606_0','12678910_1','E101083458_1','12324408_0','13031955_0','E101415263_0','E103192914_0','12888679_2','E106096969_0','E100192709_1','13112385_1','E100718398_0','12207268_0','E105244287_0','E102095465_0','E102613189_0','12961059_0','11907521_0','E105311123_0','12552705_0','E100610622_0','12975280_0','E105918926_0','E103020139_1','E101069048_1','E105427046_0','13028374_0','E100262351_0','12302576_0','12451831_0','E102929168_0','E100383453_0','E105344747_0','12569826_0','E100168661_0','12530153_0','E104697262_0']
	orig_data_dict, num_samples = cbuild._collect_unaug_data()
	num_annotations = 10
	
	features_by_cls, feat_count = collect_features()
	feat_count.pop("homogeneous texture")
	all_features = sorted(list(feat_count.keys()))
	cls_features = {f: [c for c in C.cls_names if f in features_by_cls[c]] for f in all_features}

	Z_features = get_annotated_files(features_by_cls)
	Z_features.pop("homogeneous texture")

	num_features = len(all_features) # number of features

	all_imgs = [orig_data_dict[cls][0] for cls in C.cls_names]
	all_imgs = np.array(io.flatten(all_imgs))

	all_lesionids = [orig_data_dict[cls][1] for cls in C.cls_names]
	all_lesionids = np.array(io.flatten(all_lesionids))
	test_indices = np.where(np.isin(all_lesionids, Z_test))[0]

	x_test = all_imgs[test_indices]
	z_test = all_lesionids[test_indices]

	train_indices = np.where(~np.isin(all_lesionids, C.Z_reader))[0]
	z_train = all_lesionids[train_indices]

	full_dfs = []

	for model_ix in model_nums:
		if C.ensemble_num > 0:
			for e_ix in range(C.ensemble_num):
				fullM = keras.models.load_model(join(C.model_dir, model_prefix+"%d_%d.hdf5" % (model_ix,e_ix))) #models_305
				M = keras.models.load_model(join(C.model_dir, model_prefix+"%d_%d.hdf5" % (model_ix,e_ix))) #models_305
				M = M.layers[1]
				model_fc, _ = cbuild.build_cnn_hyperparams(T)
				model_fc = model_fc.layers[1]
				for l in range(len(model_fc.layers)):
					model_fc.layers[l].set_weights(M.layers[l].get_weights())
				model_fc = common.pop_n_layers(model_fc, 2)

				all_dense = get_overall_activations(z_train, model_fc)
				feature_dense = get_feature_activations(Z_features, all_features, model_fc)
				df = predict_test_features(fullM, model_fc, all_dense, feature_dense, x_test, z_test)#, size_cutoff, lesion_sizes)
				full_dfs.append(df)
		else:
			fullM = keras.models.load_model(join(C.model_dir, model_prefix+"%d.hdf5" % model_ix)) #models_305
			M = keras.models.load_model(join(C.model_dir, model_prefix+"%d.hdf5" % model_ix)) #models_305
			model_fc = cbuild.build_cnn_hyperparams(T)
			if C.aleatoric:
				M = M.layers[1]
				model_fc = model_fc[0]
				model_fc = model_fc.layers[1]
			for l in range(len(model_fc.layers)):
				model_fc.layers[l].set_weights(M.layers[l].get_weights())
			model_fc = common.pop_n_layers(model_fc, 2)

			all_dense = get_overall_activations(model_fc, z_train)
			feature_dense = get_feature_activations(model_fc, Z_features, all_features)
			df = predict_test_features(fullM, model_fc, all_dense, feature_dense, x_test, z_test)#, size_cutoff, lesion_sizes)
			full_dfs.append(df)

	return full_dfs

def process_feat_id_dfs(all_features, DFs):
	num_features = len(all_features) # number of features
	data_dir = join(C.base_dir, 'excel')
	answer_key = join(data_dir, "ground_truth.xlsx")
	answer_key = pd.read_excel(answer_key, index_col=0)

	FEAT_DATA = np.zeros((num_features,3))
	CLS_DATA = np.zeros((C.nb_classes,3))
	LbyL = np.zeros(2)
	MISCLS = np.zeros(4)
	FIRST = np.zeros(2)
	MISFIRST = np.zeros(2)

	prec_history = {"total":[]}
	recall_history = {"total":[]}
	for f in sorted(all_features):
		prec_history[f] = []
		recall_history[f] = []
	for cls in sorted(C.cls_names):
		prec_history[cls] = []
		recall_history[cls] = []

	for df in DFs:
		acc_df = pd.DataFrame(columns=["num_correct", "pred_freq", "true_freq"])

		for f in sorted(all_features):
			num, prec_den, rec_den = 0,0,0
			for key, row in answer_key.iterrows():
				answer_features = row[['feature_1', 'feature_2', 'feature_3', 'feature_4']].dropna().values
				ignore_features = row[['ignore_1', 'ignore_2']].dropna().values
				pred_features = df.loc[key][['feature_1', 'feature_2', 'feature_3', 'feature_4']].dropna().values
				if f in pred_features and f in answer_features:
					num += 1
				if f in pred_features and f not in ignore_features:
					prec_den += 1
				if f in answer_features:
					rec_den += 1
			acc_df.loc[f] = [num, prec_den, rec_den]
			prec_history[f].append(num/(prec_den+1e-4))
			recall_history[f].append(num/rec_den)
		FEAT_DATA += acc_df.values.astype(float)

		prec_history["total"].append(acc_df["num_correct"].sum()/acc_df["pred_freq"].sum())
		recall_history["total"].append(acc_df["num_correct"].sum()/acc_df["true_freq"].sum())

		# by lesion class
		acc_df = pd.DataFrame(columns=["num_correct", "pred_freq", "true_freq"])
		for cls in sorted(C.cls_names):
			num, prec_den, rec_den = 0,0,0
			for key, row in answer_key.iterrows():
				if row['true_cls'] != cls:
					continue
				answer_features = row[['feature_1', 'feature_2', 'feature_3', 'feature_4']].dropna().values
				ignore_features = row[['ignore_1', 'ignore_2']].dropna().values
				pred_features = df.loc[key, ['feature_1', 'feature_2', 'feature_3', 'feature_4']].dropna().values
				num += len([f for f in answer_features if f in pred_features])
				prec_den += len([f for f in pred_features if f not in ignore_features])
				rec_den += len(answer_features)
			acc_df.loc[cls] = [num, prec_den, rec_den]
			prec_history[cls].append(num/(prec_den+1e-4))
			recall_history[cls].append(num/rec_den)
		CLS_DATA += acc_df.values.astype(float)

		# lesion by lesion
		num, prec, rec = 0,0,0
		n=0
		for key, row in answer_key.iterrows():
			answer_features = row[['feature_1', 'feature_2', 'feature_3', 'feature_4']].dropna().values
			ignore_features = row[['ignore_1', 'ignore_2']].dropna().values
			pred_features = df.loc[key][['feature_1', 'feature_2', 'feature_3', 'feature_4']].dropna().values
			num = len([f for f in answer_features if f in pred_features])
			rec += num/len(answer_features)
			prec_den = len([f for f in pred_features if f not in ignore_features])
			if prec_den > 0:
				prec += num/prec_den
				n+=1
		LbyL += np.array([prec/n, rec/len(answer_key)], float)

		# misclassified lesions only
		num, prec_den, rec_den = 0,0,0
		n=0
		for key, row in answer_key.iterrows():
			answer_features = row[['feature_1', 'feature_2', 'feature_3', 'feature_4']].dropna().values
			ignore_features = row[['ignore_1', 'ignore_2']].dropna().values
			pred_features = df.loc[key][['feature_1', 'feature_2', 'feature_3', 'feature_4']].dropna().values
			if df.loc[key, 'true_cls'] == df.loc[key, 'pred_cls']:
				continue
			num += len([f for f in answer_features if f in pred_features])
			prec_den += len([f for f in pred_features if f not in ignore_features])
			rec_den += len(answer_features)
			n+=1
		MISCLS += np.array([n, num, prec_den, rec_den], float)

		num, rec_den = 0,0
		for key, row in answer_key.iterrows():
			answer_features = row[['feature_1', 'feature_2', 'feature_3', 'feature_4']].dropna().values
			ignore_features = row[['ignore_1', 'ignore_2']].dropna().values
			pred_features = df.loc[key][['feature_1', 'feature_2', 'feature_3', 'feature_4']].dropna().values
			if pred_features[0] in answer_features:
				num += 1
			rec_den += 1
		FIRST += np.array([num, rec_den], float)

		num, rec_den = 0,0
		for key, row in answer_key.iterrows():
			answer_features = row[['feature_1', 'feature_2', 'feature_3', 'feature_4']].dropna().values
			ignore_features = row[['ignore_1', 'ignore_2']].dropna().values
			pred_features = df.loc[key][['feature_1', 'feature_2', 'feature_3', 'feature_4']].dropna().values
			if df.loc[key, 'true_cls'] == df.loc[key, 'pred_cls']:
				continue
			if pred_features[0] in answer_features:
				num += 1
			rec_den += 1
		MISFIRST += np.array([num, rec_den], float)

	feat_df = pd.DataFrame(FEAT_DATA, index=sorted(all_features), columns=["num_correct", "pred_freq", "true_freq"])
	cls_df = pd.DataFrame(CLS_DATA, index=sorted(C.cls_names), columns=["num_correct", "pred_freq", "true_freq"])

	return feat_df, cls_df, LbyL, MISCLS, FIRST, MISFIRST, prec_history, recall_history

###########################
### Feature ID subroutines
###########################

def get_overall_activations(Z, model_fc=None, models_conv=None, samples=1000):
	num_samples = samples*6

	all_dense = np.empty([num_samples,T.dense_units])
	all_conv3_sh = np.empty([num_samples,T.f[2]*4])
	all_conv2_sh = np.empty([num_samples,T.f[1]*4])
	all_conv1_sh = np.empty([num_samples,T.f[0]*12])
	all_conv3_ch = np.empty([num_samples,T.f[2]])
	all_conv2_ch = np.empty([num_samples,T.f[1]])
	all_conv1_ch = np.empty([num_samples,T.f[0]*3])

	ix = 0
	for cls in C.cls_names:
		for _ in range(samples):
			img_id = random.choice(Z[cls])
			aug_id = random.randint(0, C.aug_factor-1)
			img = np.load(join(C.aug_dir, "%s_%d.npy" % (img_id, aug_id)))
			img = np.expand_dims(img, 0)
			
			if model_fc is not None:
				all_dense[ix] = model_fc.predict(img)[0]
			
			if models_conv is not None:
				activ = model_conv3.predict(img)[0]
				all_conv3_ch[ix] = activ.mean((0,1,2))
				all_conv3_sh[ix] = get_shells(activ, D)
				
				activ = model_conv2.predict(img)[0]
				all_conv2_ch[ix] = activ.mean((0,1,2))
				all_conv2_sh[ix] = get_shells(activ, D)
				
				activ = model_conv1.predict(img)[0]
				all_conv1_ch[ix] = activ.mean((0,1,2))
				all_conv1_sh[ix] = get_shells(activ, D)
			ix += 1

	if models_conv is None:
		return all_dense
	else:
		return all_dense, all_conv3_ch, all_conv3_sh, all_conv2_ch, all_conv2_sh, all_conv1_ch, all_conv1_sh

def get_feature_activations(Z_features, all_features, model_fc=None, models_conv=None):
	feature_dense = {f:np.empty([0,T.dense_units]) for f in all_features}
	feature_conv3_sh = {f:np.empty([0,T.f[2]*4]) for f in all_features}
	feature_conv3_ch = {f:np.empty([0,T.f[2]]) for f in all_features}
	feature_conv2_ch = {f:np.empty([0,T.f[1]]) for f in all_features}
	feature_conv1_ch = {f:np.empty([0,T.f[0]*3]) for f in all_features}

	for f in all_features:
		Z = Z_features[f]
		for img_id in range(len(Z)):
			for aug_id in range(C.aug_factor):
				img = np.load(join(C.aug_dir, "%s_%d.npy" % (Z[img_id], aug_id)))
				
				if model_fc is not None:
					activ = model_fc.predict(np.expand_dims(img, 0))
					feature_dense[f] = np.concatenate([feature_dense[f], activ], axis=0)
			
				if models_conv is not None:
					activ = model_conv3.predict(np.expand_dims(img, 0))
					feature_conv3_ch[f] = np.concatenate([feature_conv3_ch[f], activ.mean(axis=(1,2,3))], axis=0)
					feature_conv3_sh[f] = np.concatenate([feature_conv3_sh[f], get_shells(activ, D)], axis=0)
				
					activ = model_conv2.predict(np.expand_dims(img, 0))
					feature_conv2_ch[f] = np.concatenate([feature_conv2_ch[f], activ.mean(axis=(1,2,3))], axis=0)

					activ = model_conv1.predict(np.expand_dims(img, 0))
					feature_conv1_ch[f] = np.concatenate([feature_conv1_ch[f], activ.mean(axis=(1,2,3))], axis=0)

	if models_conv is None:
		return feature_dense
	else:
		return feature_dense, feature_conv3_ch, feature_conv3_sh, feature_conv2_ch, feature_conv1_ch

def predict_test_features(full_model, model_fc, all_neurons, feature_neurons, x_test, z_test, Z_features=None, priors=None, models_conv=None, num_samples=15):
	all_features = list(feature_neurons.keys())
	num_features = len(all_features)

	df = pd.DataFrame(columns=['true_cls', 'pred_cls'] + all_features)

	lesion_ids = {}
	for cls in C.cls_names:
		src_data_df = drm.get_coords_df(cls)
		accnums = src_data_df["acc #"].values
		lesion_ids[cls] = [x[:-4] for x in os.listdir(C.crops_dir) if x[:x.find('_')] in accnums]

	num_neurons = all_neurons.shape[-1]

	if priors is None: #Use uniform distribution
		pf = np.ones(num_features)
	else:
		pf = priors

	for img_ix in range(len(z_test)):
		print('.',end='')
		test_dense = np.empty([0,T.dense_units])
		test_conv3_ch = np.empty([0,T.f[2]])
		test_conv3_sh = np.empty([0,T.f[2]*4])
		test_conv2_ch = np.empty([0,T.f[1]])
		test_conv1_ch = np.empty([0,T.f[0]*3])
		z = z_test[img_ix]
		for cls in C.cls_names:
			if z in lesion_ids[cls]:
				row = [cls]
				break

		x = np.expand_dims(x_test[img_ix], axis=0)
		preds = full_model.predict(x, verbose=False)[0]
		row.append(C.cls_names[list(preds).index(max(preds))])

		for aug_id in range(num_samples):
			img = np.load(join(C.aug_dir, "%s_%d.npy" % (z, aug_id)))

			activ = model_fc.predict(np.expand_dims(img, 0))
			test_dense = np.concatenate([test_dense, activ], axis=0)
			
			if models_conv is not None:
				activ = model_conv3.predict(np.expand_dims(img, 0))
				test_conv3_ch = np.concatenate([test_conv3_ch, activ.mean(axis=(1,2,3))], axis=0)
				test_conv3_sh = np.concatenate([test_conv3_sh, get_shells(activ, D)], axis=0)
				
				activ = model_conv2.predict(np.expand_dims(img, 0))
				test_conv2_ch = np.concatenate([test_conv2_ch, activ.mean(axis=(1,2,3))], axis=0)
				
				activ = model_conv1.predict(np.expand_dims(img, 0))
				test_conv1_ch = np.concatenate([test_conv1_ch, activ.mean(axis=(1,2,3))], axis=0)
		
		#np.concatenate([all_conv1_ch, all_conv2_ch, all_conv3_ch, all_dense], axis=1)
		pf_x = np.zeros(num_features)
		pH_f = np.zeros(num_features)
		for x_ix in range(test_dense.shape[0]):
			pH = kde_rodeo_local(test_dense[x_ix], all_dense)
			for f_ix in range(num_features):
				if Z_features is None or z in Z_features[all_features[f_ix]]:
					pH_f[f_ix] = kde_rodeo_local(test_dense[x_ix], feature_neurons[all_features[f_ix]])
				else:
					pH_f[f_ix] = 0
			pf_x += pH_f/pH
		pf_x = pf_x/test_dense.shape[0]*pf
		row += list(pf_x)
			
		df.loc[z] = row

	return df

def predict_features(full_model, model_fc, all_dense, feature_dense, x_test, z_test, Z_features=None, priors=None, models_conv=None, num_samples=15):
	all_features = list(feature_dense.keys())
	num_features = len(all_features)

	df = pd.DataFrame(columns=['true_cls', 'pred_cls'] + all_features)

	lesion_ids = {}
	for cls in C.cls_names:
		src_data_df = drm.get_coords_df(cls)
		accnums = src_data_df["acc #"].values
		lesion_ids[cls] = [x[:-4] for x in os.listdir(C.crops_dir) if x[:x.find('_')] in accnums]

	all_neurons = all_dense #np.concatenate([all_conv1_ch, all_conv2_ch, all_conv3_ch, all_dense], axis=1)
	m = all_neurons.mean(axis=0)
	all_cov = np.cov(all_neurons.T)

	num_neurons = all_neurons.shape[-1]

	if priors is None: #Use uniform distribution
		pf = np.ones(num_features)
	else:
		pf = priors

	for img_ix in range(len(z_test)):
		print('.',end='')
		test_dense = np.empty([0,T.dense_units])
		test_conv3_ch = np.empty([0,T.f[2]])
		test_conv3_sh = np.empty([0,T.f[2]*4])
		test_conv2_ch = np.empty([0,T.f[1]])
		test_conv1_ch = np.empty([0,T.f[0]*3])
		z = z_test[img_ix]
		for cls in C.cls_names:
			if z in lesion_ids[cls]:
				row = [cls]
				break

		x = np.expand_dims(x_test[img_ix], axis=0)
		preds = full_model.predict(x, verbose=False)[0]
		row.append(C.cls_names[list(preds).index(max(preds))])

		for aug_id in range(num_samples):
			img = np.load(join(C.aug_dir, "%s_%d.npy" % (z, aug_id)))

			activ = model_fc.predict(np.expand_dims(img, 0))
			test_dense = np.concatenate([test_dense, activ], axis=0)
			
			if models_conv is not None:
				activ = model_conv3.predict(np.expand_dims(img, 0))
				test_conv3_ch = np.concatenate([test_conv3_ch, activ.mean(axis=(1,2,3))], axis=0)
				test_conv3_sh = np.concatenate([test_conv3_sh, get_shells(activ, D)], axis=0)
				
				activ = model_conv2.predict(np.expand_dims(img, 0))
				test_conv2_ch = np.concatenate([test_conv2_ch, activ.mean(axis=(1,2,3))], axis=0)
				
				activ = model_conv1.predict(np.expand_dims(img, 0))
				test_conv1_ch = np.concatenate([test_conv1_ch, activ.mean(axis=(1,2,3))], axis=0)

		pf_x = np.zeros(num_features)
		pH_f = np.zeros(num_features)
		for x_ix in range(test_dense.shape[0]):
			pH = kde_rodeo_local(test_dense[x_ix], all_dense)
			for f_ix in range(num_features):
				if Z_features is None or z in Z_features[all_features[f_ix]]:
					pH_f[f_ix] = kde_rodeo_local(test_dense[x_ix], feature_dense[all_features[f_ix]])
				else:
					pH_f[f_ix] = 0
			pf_x += pH_f/pH
		pf_x = pf_x/test_dense.shape[0]*pf
		row += list(pf_x)
			
		df.loc[z] = row

	return df

@njit
def kde_rodeo_global(x, X):
	d = X.shape[-1] # number of dimensions in the hidden layer
	n = X.shape[0] # number of samples from the population
	m = x.shape[0] # number of samples for the global bandwidth
	beta = .9
	c0 = 1000.
	cn = log(d)
	h = np.ones(d) * c0/log(log(n))

	for k in range(m):
		K = np.zeros(n)
		for i in range(n):
			K[i] = exp(-np.sum(((x[k]-X[i])/h)**2/2))

		A = list(range(d))

		Z = np.zeros((d,n))
		while len(A) > 0:
			rm_set = []
			for j in A:
				for i in range(n):
					Z[j,i] = ((x[j] - X[i,j])**2 - h[j]**2) * K[i]
				sj = np.std(Z[j])
				Zj = np.mean(Z[j])
				lj = sj**2 + 2*sj*sqrt(log(n*cn))
				if np.abs(Zj) > lj:
					h[j] *= beta
				else:
					rm_set.append(j)

			ix = 0
			while ix < len(A):
				if A[ix] in rm_set:
					A.pop(ix);
				else:
					ix += 1
		H = 1
		for hi in h:
			H *= hi

	return np.mean(K)/H

@njit
def kde_rodeo_local(x, X):
	#uniform, local rodeo http://www.cs.cmu.edu/~hanliu/papers/drodeo_aistats.pdf
	d = X.shape[-1] # number of dimensions in the hidden layer
	n = X.shape[0] # number of samples from the population
	beta = .9
	c0 = 50.
	cn = log(d)
	h = np.ones(d) * c0#/log(log(n))

	K = np.zeros(n)
	for i in range(n):
		K[i] = exp(-np.sum(((x-X[i])/h)**2/2))

	A = list(range(d))

	Z = np.zeros((d,n))
	while len(A) > 0:
		rm_set = []
		for j in A:
			for i in range(n):
				Z[j,i] = ((x[j] - X[i,j])**2 - h[j]**2) * K[i]
			sj = np.std(Z[j])
			Zj = np.mean(Z[j])
			lj = sj*sqrt(2*log(n*cn))
			if np.abs(Zj) > lj:
				h[j] *= beta
			else:
				rm_set.append(j)

		ix = 0
		while ix < len(A):
			if A[ix] in rm_set:
				A.pop(ix);
			else:
				ix += 1
	H = 1
	for hi in h:
		H *= hi
		if hi > c0 - 1e-4:
			print('c0 not high enough', hi)

	return np.mean(K)/H

def predict_features_gaussian(full_model, model_fc, all_dense, feature_dense, x_test, z_test, priors=None,
		size_cutoff=None, lesion_sizes=None, models_conv=None):
	
	df = pd.DataFrame(columns=['true_cls', 'pred_cls'] + \
				[s for i in range(1,5) for s in ['feature_%d' % i,'strength_%d' % i]])

	lesion_ids = {}
	for cls in C.cls_names:
		src_data_df = drm.get_coords_df(cls)
		accnums = src_data_df["acc #"].values
		lesion_ids[cls] = [x[:-4] for x in os.listdir(C.crops_dir) if x[:x.find('_')] in accnums]

	all_features = list(feature_dense.keys())
	num_features = len(all_features)

	all_neurons = all_dense #np.concatenate([all_conv1_ch, all_conv2_ch, all_conv3_ch, all_dense], axis=1)
	m = all_neurons.mean(axis=0)
	all_cov = np.cov(all_neurons.T)

	num_neurons = all_neurons.shape[-1]

	lnZ = np.empty(num_features)
	f_m = np.empty((num_features, num_neurons))
	f_cov = np.empty((num_features, num_neurons, num_neurons))
	"""for f_ix in range(num_features):
		#f_neurons = np.concatenate([feature_conv1_ch[all_features[f_ix]],
		#                            feature_conv2_ch[all_features[f_ix]],
		#                            feature_conv3_ch[all_features[f_ix]],
		#                            feature_dense[all_features[f_ix]]], axis=1)
		f_neurons = feature_dense[all_features[f_ix]]
		f_m[f_ix] = f_neurons.mean(0)
		f_cov[f_ix] = np.cov(f_neurons.T)

		lnZ_f = -scipy.stats.multivariate_normal.logpdf(f_neurons, m, all_cov, allow_singular=True)
		adj = np.amax(lnZ_f)
		lnZ[f_ix] = np.log(np.mean(np.exp(lnZ_f - adj))) + adj"""

	if priors is None: #Use uniform distribution
		lnpf = np.log(np.ones(num_features) / num_features)
	else:
		lnpf = np.log(priors / np.max(priors))

	for img_ix in range(len(z_test)):
		test_dense = np.empty([0,T.dense_units])
		test_conv3_ch = np.empty([0,T.f[2]])
		test_conv3_sh = np.empty([0,T.f[2]*4])
		test_conv2_ch = np.empty([0,T.f[1]])
		test_conv1_ch = np.empty([0,T.f[0]*3])
		z = z_test[img_ix]
		for cls in C.cls_names:
			if z in lesion_ids[cls]:
				row = [cls]
				break

		x = np.expand_dims(x_test[img_ix], axis=0)
		preds = full_model.predict(x, verbose=False)[0]
		#for pred_cls, _ in sorted(zip(C.cls_names, preds), key=lambda x:x[1], reverse=True)[:1]:
		row.append(C.cls_names[list(preds).index(max(preds))])

		p_f = np.empty(num_features)
		for aug_id in range(25):
			img = np.load(join(C.aug_dir, "%s_%d.npy" % (z, aug_id)))

			activ = model_fc.predict(np.expand_dims(img, 0))
			test_dense = np.concatenate([test_dense, activ], axis=0)
			
			if models_conv is not None:
				activ = model_conv3.predict(np.expand_dims(img, 0))
				test_conv3_ch = np.concatenate([test_conv3_ch, activ.mean(axis=(1,2,3))], axis=0)
				test_conv3_sh = np.concatenate([test_conv3_sh, get_shells(activ, D)], axis=0)
				
				activ = model_conv2.predict(np.expand_dims(img, 0))
				test_conv2_ch = np.concatenate([test_conv2_ch, activ.mean(axis=(1,2,3))], axis=0)
				
				activ = model_conv1.predict(np.expand_dims(img, 0))
				test_conv1_ch = np.concatenate([test_conv1_ch, activ.mean(axis=(1,2,3))], axis=0)

		test_neurons = test_dense#np.concatenate([test_conv3_ch, test_dense], axis=1) 
		#m_test = test_neurons.mean(axis=0)
		#test_cov = np.cov(test_neurons.T)

		p_f = np.empty(num_features)
		for f_ix in range(num_features):  
			#indices = np.random.randint(0,test_neurons.shape[0], 1000)
			samp = test_neurons#[indices] #scipy.random.multivariate_normal(m_test, test_cov, size=10000)
			lnph_f = scipy.stats.multivariate_normal.logpdf(samp, f_m[f_ix], f_cov[f_ix], allow_singular=True)
			lnphf = lnph_f + lnpf[f_ix]
			lnph = scipy.stats.multivariate_normal.logpdf(samp, m, all_cov, allow_singular=True)

			adj = np.max(lnphf - lnph)
			p_f[f_ix] = np.log(np.mean(np.exp(lnphf - lnph - adj))) + adj# + lnZ[f_ix]
			
		evidence = {all_features[f_ix]: p_f[f_ix] for f_ix in range(num_features)}
		
		if False:
			f1='infiltrative growth'
			f2='nodular growth'
			evidence[f1] -= 40
			evidence[f2] -= 40
			if evidence[f1] < evidence[f2]:
				evidence[f1] -= 40
			else:
				evidence[f2] -= 20

			f3='central scar'
			if lesion_sizes[img_ix] < size_cutoff:
				evidence[f3] -= 150
			else:
				evidence[f3] -= 35
			
			f4='isointense on venous/delayed phase'
			f5='washout'
			if evidence[f4] < evidence[f5]:
				evidence.pop(f4)
			else:
				evidence.pop(f5)

		#top4 = np.array(sorted(evidence.items(), key=lambda x:x[1], reverse=True)[:4])[:,0]
		#if 'heterogeneous lesion' not in top4:
		#    evidence.pop(f1)
		#    evidence.pop(f2)

		for f,strength in sorted(evidence.items(), key=lambda x:x[1], reverse=True)[:4]:
			row += [f, strength]
			
		#if np.mean([row[-7], row[-5], row[-3]]) / 3 > row[-1]:
		#	row[-2] = ''
			
		#if np.mean([row[-7], row[-5]]) / 3 > row[-3]:
		#	row[-4] = ''
			
		df.loc[z] = row

	return df

###########################
### Analyze annotations
###########################

def collect_features():
	try:
		feature_sheet = pd.read_excel(C.coord_xls_path, "Descriptions")
	except:
		feature_sheet = pd.read_excel(r"D:\Etiology\excel\features.xlsx")

	features_by_cls = {}
	feat_count = {}
	for cls in C.cls_names:
		features_by_cls[cls] = list(feature_sheet["evidence1"+cls].dropna().values)
		features_by_cls[cls] = features_by_cls[cls] + list(feature_sheet["evidence2"+cls].dropna().values)

	feat_count = dict(zip(*np.unique([f for cls in features_by_cls for f in features_by_cls[cls]], return_counts=True)))
	for cls in C.cls_names:
		features_by_cls[cls] = list(set(features_by_cls[cls]))

	return features_by_cls, feat_count

def get_annotated_files(features_by_cls, num_samples=None):
	try:
		feature_sheet = pd.read_excel(C.coord_xls_path, "Descriptions")
	except:
		feature_sheet = pd.read_excel(r"D:\Etiology\excel\features.xlsx")

	Z_features_by_cls = {cls: {} for cls in features_by_cls}
	Z_features = {}
	for cls in C.cls_names:
		for f in features_by_cls[cls]:
			if f not in Z_features:
				Z_features[f] = []
				
			Z_features_by_cls[cls][f] = [x for x in feature_sheet[(feature_sheet["evidence1"+cls] == f) & ~(feature_sheet["test"+cls] >= 0)][cls].values]
			Z_features[f] += Z_features_by_cls[cls][f]
			if feature_sheet["evidence2"+cls].dropna().size > 0:
				X = [x for x in \
						feature_sheet[(feature_sheet["evidence2"+cls] == f) & ~(feature_sheet["test"+cls] >= 0)][cls].values]
				Z_features_by_cls[cls][f] += X
				Z_features[f] += X

	for f in Z_features:
		if len(Z_features[f]) < 10:
			print(f, Z_features[f])
		if num_samples is not None:
			Z_features[f] = np.random.choice(Z_features[f], num_samples, replace=False)

	return Z_features

def get_evidence_strength(feature_filters, pred_filters):
	"""A good pred_filter has high values for all the key (non-zero) features of feature_filter.
	These values should be unscaled.
	Returns average percentage of the mean value of the key filters (capped at 100%)"""
	
	strength = 0
	num_key_filters = sum(feature_filters > 0)
	
	for i in range(len(pred_filters)):
		t = feature_filters[i]
		p = pred_filters[i]
		
		if t == 0:
			continue
			
		strength += min(p/t, 1.1)#t*p / filter_avgs[i]**.7
	return (strength / num_key_filters / 1.1)**.3
	
###########################
### Misc
###########################

def save_output(Z, y_pred, y_true, save_dir=None):
	"""Saves large and small cropped images of all lesions in Z.
	Uses y_true and y_pred to separate correct and incorrect predictions.
	Requires C.cls_names, C.output_img_dir, C.crops_dir, C.orig_dir"""

	if save_dir is None:
		save_dir = C.output_img_dir

	cls_mapping = C.cls_names

	for cls in cls_mapping:
		if not exists(save_dir + "\\correct\\" + cls):
			os.makedirs(save_dir + "\\correct\\" + cls)
		if not exists(save_dir + "\\incorrect\\" + cls):
			os.makedirs(save_dir + "\\incorrect\\" + cls)

	for i in range(len(Z)):
		if y_pred[i] != y_true[i]:
			vm.save_img_with_bbox(cls=y_true[i], lesion_nums=[Z[i]],
				fn_suffix = " (bad_pred %s).png" % cls_mapping[y_pred[i]],
				save_dir=save_dir + "\\incorrect\\" + cls_mapping[y_true[i]])
		else:
			vm.save_img_with_bbox(cls=y_true[i], lesion_nums=[Z[i]],
				fn_suffix = " (good_pred %s).png" % cls_mapping[y_pred[i]],
				save_dir=save_dir + "\\correct\\" + cls_mapping[y_true[i]])