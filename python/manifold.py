"""
Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

import copy
import glob
import importlib
import math
import os
from os.path import *
import random
import time

import keras
import keras.backend as K
import keras.layers as layers
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils

import config
import dr_methods as drm
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import niftiutils.deep_learning.cnn_components as cnnc
import niftiutils.deep_learning.dcgan as dcgan
import niftiutils.deep_learning.densenet as densenet
import niftiutils.deep_learning.uncertainty as uncert
import voi_methods as vm

importlib.reload(config)
importlib.reload(dcgan)
importlib.reload(cnnc)
C = config.Config()

class DiseaseManifold():
	def __init__(self, T=None, Z_len=128, spatial_sparsity=.8):
		if T is None:
			T = config.Hyperparams()
		self.Z_len = Z_len
		self.sparsity = int(Z_len*spatial_sparsity) #80% spatial sparsity
		self.init_components(T)

	def init_components(self, T):
		"""Latent vector is a probability distribution (histogram) over regions of the manifold."""
		self.mr_to_Z = densenet.DenseNet((*C.dims, C.nb_channels),
			self.Z_len, depth=19, dropout_rate=T.dropout)
		#self.wholemr_to_Z = densenet.DenseNet((*C.context_dims, C.nb_channels),
		#	self.Z_len, depth=19, dropout_rate=T.dropout)

		labs = layers.Input((C.non_img_inputs,))
		x = cnnc.bn_relu_etc(labs, drop=T.dropout, fc_u=32)
		x = cnnc.bn_relu_etc(x, drop=T.dropout, fc_u=64)
		z = layers.Dense(self.Z_len, activation='softmax')(x)
		self.labs_to_Z = Model(labs, z)

		z = layers.Input((self.Z_len,))
		x = self.z_to_cls(z, self.sparsity)
		labs_out = layers.Dense(C.non_img_inputs, activation='softmax')(x)
		self.Z_to_labs = Model(z, labs_out)
		#Z_to_mr = uncert.add_aleatoric_var_reg(Z_to_cls)

		z = layers.Input((self.Z_len,))
		x = self.z_to_cls(z, self.sparsity)
		cls_out = layers.Dense(C.nb_classes, activation='softmax')(x)
		self.Z_to_cls = Model(z, cls_out)

		self.mr_gan = dcgan.DCGAN("E:\\DiseaseManifold\\gan",
				self.Z_len, self.build_generator, self.build_discriminator)
		self.Z_to_mr = self.mr_gan.generator

		#model = Model(img, [pred_sex, pred_age])
		#model.compile(optimizer=optimizer, loss=['binary_crossentropy', 'mse'],
		#				loss_weights=[1., 2.], metrics=['accuracy'])

	def build_generator(self, mid_len=3, drop=.1):
		z = layers.Input((self.Z_len,))
		x = self.z_to_cls(z, self.sparsity)
		x = layers.Dense(128, use_bias=False)(x)
		x = cnnc.bn_relu_etc(x, drop=drop)
		x = cnnc.bn_relu_etc(x, drop=drop, fc_u=mid_len**3*16)
		x = layers.Reshape((*[mid_len]*3, -1))(x)
		#x = layers.UpSampling3D((2,2,2))(x)
		x = layers.Conv3DTranspose(64, 3, strides=2, kernel_initializer='he_uniform', padding='same')(x)
		x = cnnc.bn_relu_etc(x, drop=drop)
		x = layers.Conv3DTranspose(64, 3, strides=2, kernel_initializer='he_uniform', padding='same')(x)
		x = cnnc.bn_relu_etc(x, drop=drop)
		x = layers.Conv3DTranspose(64, 3, strides=2, kernel_initializer='he_uniform', padding='same')(x)
		x = cnnc.bn_relu_etc(x, drop=drop)
		img = layers.Conv3DTranspose(3, (3,3,2), activation='sigmoid', padding='same')(x)
		model = Model(z, img)

		return model

	def build_discriminator(self):
		C = config.Config()
		img = layers.Input((*C.dims, C.nb_channels))
		x = cnnc.bn_relu_etc(img, cv_u=64, pool=2)
		x = cnnc.bn_relu_etc(x, cv_u=64, pool=2)
		x = cnnc.bn_relu_etc(x, cv_u=64, pool=2)
		x = layers.Flatten()(x)
		discrim = layers.Dense(1, activation='sigmoid')(x)
		model = Model(img, discrim)

		return model

	def z_to_cls(self, in_layer, k, drop=.1):
		x = cnnc.spatial_sparsity(in_layer, k)
		x = layers.Dense(64, use_bias=False)(x)
		x = cnnc.bn_relu_etc(x, drop=drop)
		x = cnnc.bn_relu_etc(x, drop=drop, fc_u=32)

		return x

	def train_mr_dec_gan(self):
		train_gen = _train_gen_encoder(inputs=['mr'], outputs=None)
		self.mr_gan.train(epochs=50)
		pass

	def train_mr_enc_Zmse(self):
		pass

	def train_lab_autoenc(self):
		labs_to_Z
		Z_to_labs
		model = Model(labs_in, Z_to_labs)

	def train_lab_enc_Zmse(self): #could train dec too
		pass

	def train_cls_autoenc(self):
		pass

	def train_cls_enc_Zmse(self):
		pass

	def train_mrlab_to_cls(self):
		mr = layers.Input((*C.dims, C.nb_channels))
		labs = layers.Input((C.non_img_inputs,))
		z1 = self.mr_to_Z(mr)
		z2 = self.labs_to_Z(labs)
		w = cnnc._expand_dims(z1)

		w1 = tf.Variable(1., name='learned_scalar')
		z_post = layers.Lambda(lambda x: x * w1)(z1)

		w = layers.LocallyConnected1D(1, 1, use_bias=False, kernel_initializer=keras.initializers.Ones(),
			kernel_constraint=keras.constraints.non_neg())(w)
		w = layers.Flatten()(w)
		z_post = layers.Add()([w,z2])
		z_post = layers.Lambda(lambda x: x / K.maximum(x,-1))(z_post)
		cls = self.Z_to_cls(z_post)
		model = Model([mr,labs], cls)

	def _train_gen_encoder(self, test_accnums=[], n=4, inputs=['mr','labs'], outputs=['cls']):
		img_fns = [fn for fn in glob.glob(join(C.full_img_dir, "*.npy")) if not fn.endswith("seg.npy") \
					and basename(fn)[:-4] not in test_accnums]

		test_path="E:\\LIRADS\\excel\\clinical_data_test.xlsx"
		clinical_df = pd.read_excel(test_path, index_col=0)
		clinical_df.index = clinical_df.index.astype(str)

		while True:
			tmp = [[],[]]
			for ix,arr in enumerate([inputs, outputs]):
				for elem in arr:
					if elem == 'mr':
						tmp[ix].append(np.empty((n, *C.dims, C.nb_channels)))
					elif elem == 'labs':
						tmp[ix].append(np.empty((n, C.non_img_inputs)))
					elif elem == 'cls':
						tmp[ix].append(np.empty((n, C.nb_classes)))

			for train_ix in range(n):
				img_fn = random.choice(img_fns)
				accnum = img_fn[:-4]

				if not exists(accnum+"_tumorseg.npy") or not exists(accnum+"_liverseg.npy"):
					continue

				if 'mr' in inputs + outputs:
					img = np.load(img_fn)
					img = tr.rescale_img(img, C.context_dims)
					img = tr.normalize_intensity(img, 1, -1)
				elif 'labs' in inputs + outputs:
					labs = clinical_df.loc[accnum].values[:C.clinical_inputs]
				elif 'cls' in inputs + outputs:
					cls = cls_df.loc[accnum].values[0]
					cls = C.cls_names.index(cls)
					cls = np_utils.to_categorical(cls_num, len(C.cls_names)).astype(int)

				for ix,arr in enumerate([inputs, outputs]):
					for elem in arr:
						if elem == 'mr':
							tmp[ix][train_ix] = img
						elif elem == 'labs':
							tmp[ix][train_ix] = labs
						elif elem == 'cls':
							tmp[ix][train_ix] = cls

			ins = tmp[0]
			outs = tmp[1]

			yield ins, outs
