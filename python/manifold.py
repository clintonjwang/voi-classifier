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
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import niftiutils.deep_learning.cnn_components as cnnc
import niftiutils.deep_learning.dcgan as dcgan
import niftiutils.deep_learning.densenet as densenet
import niftiutils.deep_learning.uncertainty as uncert

importlib.reload(config)
importlib.reload(dcgan)
importlib.reload(cnnc)
C = config.Config()

class DiseaseManifold():
	"""N is the latent multivariate normal distribution.
	Z is a categorical distribution to select"""

	def __init__(self, T=None, num_states=128, latent_dims=64, spatial_sparsity=.8):
		#num_states is the number of clinically distinct states on the manifold
		#latent_dims is the dimensionality of the latent vector

		if T is None:
			T = config.Hyperparams()
		self.Z_len = num_states
		self.N_len = latent_dims
		self.sparsity = int(Z_len*spatial_sparsity) #80% spatial sparsity
		self.init_components(T)

	def init_components(self, T):
		"""Latent vector is a probability distribution (histogram) over regions of the manifold."""
		self.mr_to_Z = densenet.DenseNet((*C.dims, C.nb_channels),
			self.N_len, depth=19, dropout_rate=T.dropout)
		#self.wholemr_to_Z = densenet.DenseNet((*C.context_dims, C.nb_channels),
		#	self.Z_len, depth=19, dropout_rate=T.dropout)

		labs = layers.Input((C.clinical_inputs,))
		x = cnnc.bn_relu_etc(labs, drop=T.dropout, fc_u=32)
		x = cnnc.bn_relu_etc(x, drop=T.dropout, fc_u=64)
		z = layers.Dense(self.Z_len, activation='softmax')(x)
		self.labs_to_Z = Model(labs, z)

		z = layers.Input((self.Z_len,))
		x = self.z_to_cls(z, self.sparsity)
		labs_out = layers.Dense(C.clinical_inputs, activation='softmax')(x)
		self.Z_to_labs = Model(z, labs_out)
		#Z_to_mr = uncert.add_aleatoric_var_reg(Z_to_cls)

		z = layers.Input((self.Z_len,))
		x = self.z_to_cls(z, self.sparsity)
		cls_out = layers.Dense(C.nb_classes, activation='softmax')(x)
		self.Z_to_cls = Model(z, cls_out)

		self.mr_gan = dcgan.DCGAN("E:\\DiseaseManifold\\gan",
				self.N_len, self.build_generator, self.build_discriminator)

		self.Z_to_mr = self.mr_gan.generator
		#z = layers.Input((self.Z_len,))
		#x = self.z_to_n(z, self.sparsity)

		#model = Model(img, [pred_sex, pred_age])
		#model.compile(optimizer=optimizer, loss=['binary_crossentropy', 'mse'],
		#				loss_weights=[1., 2.], metrics=['accuracy'])

	def build_generator(self, mid_len=4, drop=.1):
		z = layers.Input((self.N_len,))
		x = cnnc.bn_relu_etc(z, drop=drop, fc_u=256)
		x = cnnc.bn_relu_etc(x, drop=drop, fc_u=1024)
		x = cnnc.bn_relu_etc(x, drop=drop, fc_u=mid_len**3*16)
		x = layers.Reshape((*[mid_len]*3, -1))(x)
		x = layers.UpSampling3D(2)(x)
		x = layers.Conv3DTranspose(64, 5, kernel_initializer='he_uniform', padding='same')(x)
		#x = cnnc.bn_relu_etc(x)
		x = layers.Activation('tanh')(x)
		x = layers.UpSampling3D(2)(x)
		x = layers.Conv3D(64, 5, kernel_initializer='he_uniform')(x)
		x = cnnc.bn_relu_etc(x)
		x = layers.SpatialDropout3D(drop)(x)
		x = layers.Conv3DTranspose(64, 5, strides=(2,2,1), kernel_initializer='he_uniform', padding='same')(x)
		x = cnnc.bn_relu_etc(x)
		img = layers.Conv3DTranspose(3, (5,5,3), activation='sigmoid', padding='same')(x)
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

	def z_to_n(self, in_layer, k, drop=.1):
		x = cnnc.spatial_sparsity(in_layer, k)
		x = layers.Dense(self.N_len, use_bias=False, kernel_initializer=keras.initializers.RandomUniform(minval=-1, maxval=1))(x)

		return x

	def train_mr_dec_vae(self):
		train_gen = self._train_gen_encoder(inputs=['mr'], outputs=['mr'])
		self.mr_gan.train(epochs=50, gen=train_gen)

	"""def train_mr_dec_gan(self):
		train_gen = self._train_gen_encoder(inputs=['z'], outputs=['mr'])
		self.mr_gan.train(epochs=50, gen=train_gen)"""

	def _train_gen_encoder(self, test_accnums=[], n=6, inputs=['mr','labs'], outputs=['cls']):
		img_fns = [fn for fn in glob.glob(join(C.unaug_dir, "*.npy")) \
				if basename(fn)[:basename(fn).find('_')] not in test_accnums]

		test_path="E:\\LIRADS\\excel\\clinical_data_test.xlsx"
		clinical_df = pd.read_excel(test_path, index_col=0)
		clinical_df.index = clinical_df.index.astype(str)

		while True:
			tmp = [[],[]]
			for ix,arr in enumerate([inputs, outputs]):
				for elem in arr:
					if elem == 'mr':
						tmp[ix].append(np.zeros((n, *C.dims, C.nb_channels)))
					elif elem == 'labs':
						tmp[ix].append(np.zeros((n, C.clinical_inputs)))
					elif elem == 'cls':
						tmp[ix].append(np.zeros((n, C.nb_classes)))
					elif elem == 'z':
						tmp[ix].append(np.zeros((n, self.Z_len)))

			for train_ix in range(n):
				img_fn = random.choice(img_fns)
				lesion_id = basename(img_fn)
				accnum = lesion_id[:lesion_id.find('_')]

				if 'mr' in inputs + outputs:
					img = np.load(img_fn)
					#img = tr.rescale_img(img, C.dims)
					#img = tr.normalize_intensity(img, 1, -1)
				if 'labs' in inputs + outputs:
					labs = clinical_df.loc[accnum].values[:C.clinical_inputs]
				if 'cls' in inputs + outputs:
					cls = cls_df.loc[lesion_id].values[0]
					cls = C.cls_names.index(cls)
					cls = np_utils.to_categorical(cls_num, len(C.cls_names)).astype(int)
				if 'z' in inputs + outputs:
					z = np.random.uniform(-10,10,size=5)
					z = np.exp(z - np.max(z))
					z /= z.sum()
					z = np.concatenate((z, np.zeros(self.Z_len-5)))
					np.random.shuffle(z)
				if 'n' in inputs + outputs:
					n = np.random.normal(size=self.N_len)

				for ix,arr in enumerate([inputs, outputs]):
					for jx,elem in enumerate(arr):
						if elem == 'mr':
							if ix == 0:
								tmp[ix][jx][train_ix] = add_noise(img)
							else:
								tmp[ix][jx][train_ix] = img
						elif elem == 'labs':
							tmp[ix][jx][train_ix] = labs
						elif elem == 'cls':
							tmp[ix][jx][train_ix] = cls
						elif elem == 'z':
							tmp[ix][jx][train_ix] = z

			if len(tmp[0]) == 1:
				ins = tmp[0][0]
			else:
				ins = tmp[0]
			if len(tmp[1]) == 1:
				outs = tmp[1][0]
			else:
				outs = tmp[1]

			yield ins, outs

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
		labs = layers.Input((C.clinical_inputs,))
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

def add_noise(img):
	return img + np.random.normal(size=img.shape)