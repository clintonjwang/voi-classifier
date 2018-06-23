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
import feature_interpretation as fin
import cnn_builder as cbuild
import copy
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
import scipy
from skimage.transform import rescale
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import time


####################################
### Epistemic uncertainty
####################################
def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0)
	
def aug_uncertainty(lesion_id, model, aug_factor=100):
	logits = np.empty((aug_factor, *model.output_shape[1:]))
	#uncertainties = np.empty(aug_factor)

	for aug_id in range(aug_factor):
		path = join(C.aug_dir, "%s_%d.npy" % (lesion_id, aug_id))
		img = np.load(path)
		logits[aug_id] = model.predict(np.expand_dims(img, 0))
		#logits[aug_id], uncertainties[aug_id] = model.predict(np.expand_dims(img, 0))
	
	return np.argmax(np.mean(logits,0))
	#return softmax(np.mean(logits,0))

	#scipy.stats.percentileofscore(logits, 0)
	#return logits, uncertainties

"""def voi_uncertainty(lesion_id, model, voi_df, small_voi_df, shift_combinations=None):
	if shift_combinations is None:
		shift_combinations = tuple([seq for seq in itertools.product([-5,0,5], repeat=6) if \
									seq[1] >= seq[0] and seq[3] >= seq[2] and seq[5] >= seq[4]])
	aug_factor=len(shift_combinations)

	coords = vm._get_voi_coords(small_voi_df[small_voi_df["id"] == lesion_id])
	voi_row = voi_df.loc[test_names[test_id]]
	
	logits = np.empty(aug_factor)
	uncertainties = np.empty(aug_factor)
	
	dims = dims_df[dims_df["AccNum"] == lesion_id[:lesion_id.rfind('_')]].iloc[0]
	shifts = [hf.flatten([x[:2]//dims['x'], x[2:4]//dims['y'], x[4:]//dims['z']]) for x in shift_combinations]
	for voi_id in range(aug_factor):
		dx = shifts[voi_id]#scipy.random.normal(0, 2, 3)
		#voi = list(map(int, hf.flatten([coords[:2]+dx[0], coords[2:4]+dx[1], coords[4:]+dx[2]])))
		voi = list(map(int, [coords[i]+dx[i] for i in range(6)]))
		img = vm.save_unaugmented_set(cls=voi_row["cls"], lesion_ids=[lesion_id], custom_vois=[voi],
									  return_img_only=True)[0]
		logits[aug_id], uncertainties[aug_id] = model.predict(np.expand_dims(img, 0))"""

#MC DROPOUT SAMPLING
def mc_dropout(mc_model, img, T=50):
	img = np.expand_dims(img, 0)
	logits = np.empty(T)
	uncertainties = np.empty(T)
	for ix in range(T):
		logits[ix], uncertainties[ix] = mc_model.predict(img)

	return logits, uncertainties

def mc_augment(lesion_id, model, voi_df, aug_factor=100, T=50):
	voi_row = voi_df.loc[lesion_id]
	logits = np.empty((aug_factor, T))
	uncertainties = np.empty((aug_factor, T))

	for aug_id in range(aug_factor):
		img = np.load(os.path.join(C.aug_dir, voi_row['cls'], "%s_%d.npy" % (lesion_id, aug_id)))
		logits[aug_id], uncertainties[aug_id] = mc_dropout(model, img, T)
	
	return logits, uncertainties
