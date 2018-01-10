import keras.backend as K
from keras.layers import Input, Dense, Concatenate, Flatten, Dropout, Lambda, Conv3D, MaxPooling3D, LSTM
from keras.layers import SimpleRNN, Conv2D, MaxPooling2D, ZeroPadding3D, Activation, ELU, TimeDistributed, Permute, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
#from keras.utils import np_utils

import copy
import config
import helper_fxns as hf
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
import voi_methods as vm

####################################
### OVERNIGHT PROCESSES
####################################

def run_all():
	"""Reruns everything except dimensions. Meant for overnight runs."""
	
	import dr_methods as drm
	import voi_methods as vm
	import artif_gen_methods as agm
	
	C = config.Config()
	drm.load_all_vois(C)
	
	intensity_df = drm.load_ints(C)
	intensity_df.to_csv(C.int_df_path, index=False)
	
	n = 1500
	for cls in C.classes_to_include:
		agm.gen_imgs(cls, C, n)
		if not os.path.exists(C.orig_dir + cls):
			os.makedirs(C.orig_dir + cls)
		if not os.path.exists(C.aug_dir + cls):
			os.makedirs(C.aug_dir + cls)
		if not os.path.exists(C.crops_dir + cls):
			os.makedirs(C.crops_dir + cls)
			
	final_size = C.dims

	voi_df_art = pd.read_csv(C.art_voi_path)
	voi_df_ven = pd.read_csv(C.ven_voi_path)
	voi_df_eq = pd.read_csv(C.eq_voi_path)
	intensity_df = pd.read_csv(C.int_df_path)
	
	small_vois = {}
	small_vois = vm.extract_vois(small_vois, C, voi_df_art, voi_df_ven, voi_df_eq, intensity_df)

	# scaled imgs
	t = time.time()
	for cls in C.classes_to_include:
		for fn in os.listdir(C.crops_dir + cls):
			img = np.load(C.crops_dir + cls + "\\" + fn)
			unaug_img = vm.resize_img(img, C.dims, small_vois[fn[:-4]])
			np.save(C.orig_dir + cls + "\\" + fn, unaug_img)
	print(time.time()-t)
	
	# augmented imgs
	t = time.time()
	for cls in C.classes_to_include:
		vm.parallel_augment(cls, small_vois, C)
		print(cls, time.time()-t)
		
	for cls in C.classes_to_include:
		vm.save_vois_as_imgs(cls, C)
		
	run_fixed_hyperparams(C)

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

def run_fixed_hyperparams(C_list=None, overwrite=False, max_runs=999, hyperparams=None):
	"""Runs the CNN indefinitely, saving performance metrics."""
	if C_list is None:
		C_list = [config.Config()]
	if overwrite:
		running_stats = pd.DataFrame(columns = ["n", "n_art", "steps_per_epoch", "epochs",
			"num_phases", "input_res", "training_fraction", "test_num", "augment_factor", "non_imaging_inputs",
			"kernel_size", "conv_filters", "conv_padding",
			"dropout", "time_dist", "dilation", "dense_units",
			"acc6cls", "acc3cls", "time_elapsed(s)", "loss_hist",
			'hcc', 'cholangio', 'colorectal', 'cyst', 'hemangioma', 'fnh',
			'confusion_matrix', 'f1', 'timestamp', 'run_num',
			'misclassified_test', 'misclassified_train', 'model_num',
			'y_true', 'y_pred_raw', 'z_test'])
		index = 0
	else:
		running_stats = pd.read_csv(C_list[0].run_stats_path)
		index = len(running_stats)

	model_names = os.listdir("E:\\models\\")
	if len(model_names) > 0:	
		model_num = max([int(x[x.find('_')+1:x.find('.')]) for x in model_names]) + 1
	else:
		model_num = 0

	running_acc_6 = []
	running_acc_3 = []
	n = [4]
	n_art = [0]
	steps_per_epoch = [750]
	epochs = [30]
	run_2d = False
	f = [[64,128,128]]
	padding = [['same','valid']]
	dropout = [[0.1,0.1]]
	dense_units = [100]
	dilation_rate = [(1,1,1)]
	kernel_size = [(3,3,2)]
	pool_sizes = [(2,2,1),(2,2,2)]
	activation_type = ['relu']
	merge_layer = [1]
	cycle_len = 1
	early_stopping = EarlyStopping(monitor='loss', min_delta=0.002, patience=3)
	time_dist = False

	C_index = 0
	while index < max_runs:
		C = C_list[C_index % len(C_list)]
		#C.hard_scale = False

		X_test, Y_test, train_generator, num_samples, train_orig, Z = get_cnn_data(n=n[C_index % len(n)],
					n_art=n_art[C_index % len(n_art)], run_2d=run_2d, C=C)
		Z_test, Z_train_orig = Z
		X_train_orig, Y_train_orig = train_orig

		for _ in range(cycle_len):
			model = build_cnn(C, 'adam', activation_type=activation_type[index % len(activation_type)],
					dilation_rate=dilation_rate[index % len(dilation_rate)], f=f[index % len(f)], pool_sizes=pool_sizes,
					padding=padding[index % len(padding)], dropout=dropout[index % len(dropout)],
					dense_units=dense_units[index % len(dense_units)], kernel_size=kernel_size[index % len(kernel_size)],
					merge_layer=merge_layer[index % len(merge_layer)], non_imaging_inputs=C.non_imaging_inputs, time_dist=time_dist)
			
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

			y_true_simp, y_pred_simp, _ = condense_cm(y_true, y_pred, C.classes_to_include)
			running_acc_3.append(accuracy_score(y_true_simp, y_pred_simp))
			#print("3cls accuracy:", running_acc_3[-1], " - average:", np.mean(running_acc_3))

			running_stats.loc[index] = [n[C_index % len(n)], n_art[C_index % len(n_art)], steps_per_epoch[index % len(steps_per_epoch)], epochs[index % len(epochs)],
								C.dims, C.train_frac, C.test_num, C.aug_factor, C.non_imaging_inputs,
								kernel_size[index % len(kernel_size)], f[index % len(f)], padding[index % len(padding)],
								dropout[index % len(dropout)], time_dist, dilation_rate[index % len(dilation_rate)], dense_units[index % len(dense_units)],
								running_acc_6[-1], running_acc_3[-1], time.time()-t, loss_hist,
								num_samples['hcc'], num_samples['cholangio'], num_samples['colorectal'], num_samples['cyst'], num_samples['hemangioma'], num_samples['fnh'],
								cm, time.time(), #C.run_num,
								misclassified_test, misclassified_train, model_num, y_true, str(Y_pred), list(Z_test)]
			running_stats.to_csv(C.run_stats_path, index=False)

			model.save(C.model_save_dir+'models_%d.hdf5' % model_num)
			model_num += 1

			index += 1

		C_index += 1

####################################
### BUILD CNNS
####################################

def run_then_return_val_loss(num_iters=1, hyperparams=None):
	"""Runs the CNN indefinitely, saving performance metrics."""
	C = config.Config()
	if overwrite:
		running_stats = pd.DataFrame(columns = ["n", "n_art", "steps_per_epoch", "epochs",
			"num_phases", "input_res", "training_fraction", "test_num", "augment_factor", "non_imaging_inputs",
			"kernel_size", "conv_filters", "conv_padding",
			"dropout", "time_dist", "dilation", "dense_units",
			"acc6cls", "acc3cls", "time_elapsed(s)", "loss_hist",
			'hcc', 'cholangio', 'colorectal', 'cyst', 'hemangioma', 'fnh',
			'confusion_matrix', 'f1', 'timestamp', 'run_num',
			'misclassified_test', 'misclassified_train', 'model_num',
			'y_true', 'y_pred_raw', 'z_test'])
		index = 0
	else:
		running_stats = pd.read_csv(C.run_stats_path)
		index = len(running_stats)

	model_names = os.listdir("E:\\models\\")
	if len(model_names) > 0:	
		model_num = max([int(x[x.find('_')+1:x.find('.')]) for x in model_names]) + 1
	else:
		model_num = 0

	X_test, Y_test, train_generator, num_samples, train_orig, Z = get_cnn_data(n=T.n,
				n_art=T.n_art, run_2d=T.run_2d, C=C)
	Z_test, Z_train_orig = Z
	X_train_orig, Y_train_orig = train_orig

	T = hyperparams
	model = build_cnn(T)

	t = time.time()
	hist = model.fit_generator(train_generator, steps_per_epoch=T.steps_per_epoch,
			epochs=num_iters, callbacks=[T.early_stopping], verbose=False)
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
	acc_6cl = accuracy_score(y_true, y_pred)

	y_true_simp, y_pred_simp, _ = condense_cm(y_true, y_pred, C.classes_to_include)
	acc_3cl = accuracy_score(y_true_simp, y_pred_simp)

	running_stats.loc[index] = _get_hyperparams_as_list(C, T) + \
			[acc_6cl, acc_3cl, time.time()-t, loss_hist,
			num_samples['hcc'], num_samples['cholangio'], num_samples['colorectal'], num_samples['cyst'], num_samples['hemangioma'], num_samples['fnh'],
			cm, time.time(), misclassified_test, misclassified_train, model_num, y_true, str(Y_pred), list(Z_test)]
	running_stats.to_csv(C.run_stats_path, index=False)

	model.save(C.model_save_dir+'models_%d.hdf5' % model_num)

def _get_hyperparams_as_list(C=None, T=None):
	if T is None:
		T = config.Hyperparams()
	if C is None:
		C = config.Config()
	
	return [T.n, T.n_art, T.steps_per_epoch, T.epochs,
			C.dims, C.train_frac, C.test_num, C.aug_factor, C.non_imaging_inputs,
			T.kernel_size, T.f, T.padding,
			T.dropout, T.time_dist, T.dilation_rate, T.dense_units]

def get_random_hyperparameter_configuration():
	n = [4]
	n_art = [0]
	steps_per_epoch = [750]
	epochs = [30]
	run_2d = False
	f = [[64,128,128]]
	padding = [['same','valid']]
	dropout = [[0.1,0.1]]
	dense_units = [100]
	dilation_rate = [(2, 2, 2)]
	kernel_size = [(3,3,2)]
	activation_type = ['relu']
	merge_layer = [-1]
	cycle_len = 1
	early_stopping = EarlyStopping(monitor='loss', min_delta=0.002, patience=3)
	time_dist = True
	T = []

	return T

def build_cnn_hyperparams(hyperparams):
	C = config.Config()
	return build_cnn(optimizer=hyperparams.optimizer, dilation_rate=hyperparams.dilation_rate,
		padding=hyperparams.padding, pool_sizes=hyperparams.pool_sizes, dropout=hyperparams.dropout,
		activation_type=hyperparams.activation_type, f=hyperparams.f, dense_units=hyperparams.dense_units,
		kernel_size=hyperparams.kernel_size, merge_layer=hyperparams.merge_layer,
		non_imaging_inputs=C.non_imaging_inputs, run_2d=hyperparams.run_2d, time_dist=hyperparams.time_dist)

def build_cnn(C=None, optimizer='adam', dilation_rate=(1,1,1), padding=['same', 'valid'], pool_sizes = [(2,2,2), (2,2,2)],
	dropout=[0.1,0.1], activation_type='relu', f=[64,128,128], dense_units=100, kernel_size=(3,3,2), merge_layer=1,
	non_imaging_inputs=False, run_2d=False, time_dist=True):
	"""Main class for setting up a CNN. Returns the compiled model."""

	if C is None:
		C = config.Config()
	if activation_type == 'elu':
		ActivationLayer = ELU
		activation_args = 1
	elif activation_type == 'relu':
		ActivationLayer = Activation
		activation_args = 'relu'

	nb_classes = len(C.classes_to_include)

	if not run_2d:
		img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 3))
	else:
		img = Input(shape=(C.dims[0], C.dims[1], 3))

	if merge_layer == 1:
		art_x = Lambda(lambda x : K.expand_dims(x[:,:,:,:,0], axis=4))(img)
		art_x = Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0])(art_x)
		art_x = BatchNormalization()(art_x)
		art_x = ActivationLayer(activation_args)(art_x)

		ven_x = Lambda(lambda x : K.expand_dims(x[:,:,:,:,1], axis=4))(img)
		ven_x = Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0])(ven_x)
		ven_x = BatchNormalization()(ven_x)
		ven_x = ActivationLayer(activation_args)(ven_x)

		eq_x = Lambda(lambda x : K.expand_dims(x[:,:,:,:,2], axis=4))(img)
		eq_x = Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0])(eq_x)
		eq_x = BatchNormalization()(eq_x)
		eq_x = ActivationLayer(activation_args)(eq_x)

		x = Concatenate(axis=4)([art_x, ven_x, eq_x])
		x = MaxPooling3D(pool_sizes[0])(x)
		#x = Dropout(dropout[0])(x)

		if time_dist:
			x = Reshape()
			x = Permute((4,1,2,3))(x)

			for layer_num in range(1,len(f)):
				x = TimeDistributed(Conv3D(filters=f[layer_num], kernel_size=kernel_size, padding=padding[1]))(x)
				x = TimeDistributed(BatchNormalization())(x)
				x = TimeDistributed(ActivationLayer(activation_args))(x)
				x = Dropout(dropout[0])(x)
			
			x = Permute((2,3,4,1))(x)
		else:
			for layer_num in range(1,len(f)):
				x = Conv3D(filters=f[layer_num], kernel_size=kernel_size, padding=padding[1])(x)
				x = BatchNormalization()(x)
				x = ActivationLayer(activation_args)(x)
				x = Dropout(dropout[0])(x)

	elif merge_layer == 0:
		x = img

		if time_dist:
			x = Reshape((C.dims[0], C.dims[1], C.dims[2], 3, 1))(x)
			x = Permute((4,1,2,3,5))(x)

			for layer_num in range(len(f)):
				if layer_num == 1:
					x = TimeDistributed(Conv3D(filters=f[layer_num], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[1]))(x) #, kernel_regularizer=l2(.01)
				#elif layer_num == 0:
				#	x = TimeDistributed(Conv3D(filters=f[layer_num], kernel_size=kernel_size, strides=(2,2,1), padding=padding[1]))(x) #, kernel_regularizer=l2(.01)
				else:
					x = TimeDistributed(Conv3D(filters=f[layer_num], kernel_size=kernel_size, padding=padding[1]))(x) #, kernel_regularizer=l2(.01)
				#x = BatchNormalization()(x)
				x = TimeDistributed(Dropout(dropout[0]))(x)
				x = ActivationLayer(activation_args)(x)
				if layer_num == 0:
					x = TimeDistributed(MaxPooling3D(pool_sizes[0]))(x)
			
			#x = Permute((2,3,4,1,5))(x)

			#if padding[1] == "valid":
			#	dims = [(C.dims[i]-len(f)*math.ceil(kernel_size[i]/2))//2 for i in range(3)]
			#else:
			#	dims = C.dims
			#x = Reshape((dims[0], dims[1], dims[2], -1))(x)
		else:
			for layer_num in range(len(f)):
				x = Conv3D(filters=f[layer_num], kernel_size=kernel_size, padding=padding[1])(x)
				x = BatchNormalization()(x)
				x = ActivationLayer(activation_args)(x)
				x = Dropout(dropout[0])(x)
	else:
		raise ValueError("invalid settings")

	if time_dist:
		x = TimeDistributed(MaxPooling3D(pool_sizes[1]))(x)
		x = TimeDistributed(Flatten())(x)

		#x = TimeDistributed(Dense(dense_units))(x) #, kernel_regularizer=l2(.01)
		#x = BatchNormalization()(x)
		#x = TimeDistributed(Dropout(dropout[1]))(x)
		#x = ActivationLayer(activation_args)(x)

		#x = SimpleRNN(128, return_sequences=True)(x)
		x = SimpleRNN(dense_units)(x)
		x = Dropout(dropout[1])(x)
	else:
		x = MaxPooling3D(pool_sizes[1])(x)
		x = Flatten()(x)

		if non_imaging_inputs:
			img_traits = Input(shape=(2,)) #bounding volume and aspect ratio of lesion
			x = Concatenate(axis=1)([x, img_traits])

		x = Dense(dense_units)(x)#, kernel_initializer='normal', kernel_regularizer=l2(.01), kernel_constraint=max_norm(3.))(x)
		x = BatchNormalization()(x)
		x = Dropout(dropout[1])(x)
		x = ActivationLayer(activation_args)(x)

	x = Dense(nb_classes)(x)
	x = BatchNormalization()(x)
	pred_class = Activation('softmax')(x)

	if not non_imaging_inputs:
		model = Model(img, pred_class)
	else:
		model = Model([img, img_traits], pred_class)
	
	#optim = Adam(lr=0.01)#5, decay=0.001)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	return model

def build_pretrain_model(trained_model, C=None):
	"""Sets up CNN with pretrained weights"""

	if C is None:
		C = config.Config()
	nb_classes = len(C.classes_to_include)

	voi_img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], C.nb_channels))
	x = voi_img
	x = Conv3D(filters=128, kernel_size=(3,3,2), activation='relu', trainable=False)(x)
	x = Dropout(0.5)(x)
	x = Conv3D(filters=128, kernel_size=(3,3,2), activation='relu', trainable=False)(x)
	x = MaxPooling3D((2, 2, 2))(x)
	x = Dropout(0.5)(x)
	x = Conv3D(filters=64, kernel_size=(3,3,2), activation='relu', trainable=False)(x)
	x = MaxPooling3D((2, 2, 1))(x)
	x = Dropout(0.5)(x)
	x = Flatten()(x)

	img_traits = Input(shape=(2,)) #bounding volume and aspect ratio of lesion

	intermed = Concatenate(axis=1)([x, img_traits])
	x = Dense(64, activation='relu')(intermed)#, kernel_initializer='normal', kernel_regularizer=l1(.01), kernel_constraint=max_norm(3.))(x)
	x = Dropout(0.5)(x)
	pred_class = Dense(nb_classes, activation='softmax')(x)


	model_pretrain = Model([voi_img, img_traits], pred_class)
	model_pretrain.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	for l in range(1,5):
		if type(model_pretrain.layers[l]) == Conv3D:
			model_pretrain.layers[l].set_weights(trained_model.layers[l].get_weights())

	return model_pretrain

def get_cnn_data(n=4, n_art=4, run_2d=False, verbose=False, C=None):
	"""Subroutine to run CNN"""

	if C is None:
		C = config.Config()

	nb_classes = len(C.classes_to_include)
	intensity_df = pd.read_csv(C.int_df_path)
	orig_data_dict, num_samples = _collect_unaug_data(C)

	#avg_X2 = {}
	train_ids = {} #filenames of training set originals
	test_ids = {} #filenames of test set
	X_test = []
	X2_test = []
	Y_test = []
	Z_test = []
	X_train_orig = []
	X2_train_orig = []
	Y_train_orig = []
	Z_train_orig = []

	train_samples = {}

	for cls in orig_data_dict:
		cls_num = C.classes_to_include.index(cls)
		#avg_X2[cls] = np.mean(orig_data_dict[cls][1], axis=0)

		if C.train_frac is None:
			train_samples[cls] = num_samples[cls] - C.test_num
		else:
			train_samples[cls] = round(num_samples[cls]*C.train_frac)
		
		order = np.random.permutation(list(range(num_samples[cls])))
		train_ids[cls] = list(orig_data_dict[cls][1][order[:train_samples[cls]]])
		test_ids[cls] = list(orig_data_dict[cls][1][order[train_samples[cls]:]])
		
		X_test = X_test + list(orig_data_dict[cls][0][order[train_samples[cls]:]])
		#X2_test = X2_test + list(orig_data_dict[cls][1][order[train_samples[cls]:]])
		Y_test = Y_test + [[0] * cls_num + [1] + [0] * (nb_classes - cls_num - 1)] * \
							(num_samples[cls] - train_samples[cls])
		Z_test = Z_test + test_ids[cls]
		
		X_train_orig = X_train_orig + list(orig_data_dict[cls][0][order[:train_samples[cls]]])
		#X2_train_orig = X2_train_orig + list(orig_data_dict[cls][1][order[:train_samples[cls]]])
		Y_train_orig = Y_train_orig + [[0] * cls_num + [1] + [0] * (nb_classes - cls_num - 1)] * \
							(train_samples[cls])
		Z_train_orig = Z_train_orig + train_ids[cls]
		
		if verbose:
			print("%s has %d samples for training (%d after augmentation) and %d for testing" %
				  (cls, train_samples[cls], train_samples[cls] * C.aug_factor, num_samples[cls] - train_samples[cls]))
		
	#Y_test = np_utils.to_categorical(Y_test, nb_classes)
	#Y_train_orig = np_utils.to_categorical(Y_train_orig, nb_classes)
	if C.non_imaging_inputs:
		X_test = [np.array(X_test), np.array(X2_test)]
		X_train_orig = [np.array(X_train_orig), np.array(X2_train_orig)]

	Y_test = np.array(Y_test)
	Y_train_orig = np.array(Y_train_orig)

	Z_test = np.array(Z_test)
	Z_train_orig = np.array(Z_train_orig)

	X_test = np.array(X_test)#_separate_phases(X_test, C.non_imaging_inputs)
	X_train_orig = np.array(X_train_orig)#_separate_phases(X_train_orig, C.non_imaging_inputs)

	if run_2d:
		train_generator = _train_generator_func_2d(train_ids, n=n, n_art=n_art)
	else:
		train_generator = _train_generator_func(train_ids, n=n, n_art=n_art)

	return X_test, Y_test, train_generator, num_samples, [X_train_orig, Y_train_orig], [Z_test, Z_train_orig]

###########################
### FOR OUTPUTTING IMAGES AFTER TRAINING
###########################

def save_output(Z, y_pred, y_true, C=None, save_dir=None):
	"""Saves large and small cropped images of all lesions in Z.
	Uses y_true and y_pred to separate correct and incorrect predictions.
	Requires C.classes_to_include, C.output_img_dir, C.crops_dir, C.orig_dir"""

	if C is None:
		C = config.Config()
	if save_dir is None:
		save_dir = C.output_img_dir
	
	small_voi_df = pd.read_csv(C.small_voi_path)

	cls_mapping = C.classes_to_include

	for cls in cls_mapping:
		if not os.path.exists(save_dir + "\\correct\\" + cls):
			os.makedirs(save_dir + "\\correct\\" + cls)
		if not os.path.exists(save_dir + "\\incorrect\\" + cls):
			os.makedirs(save_dir + "\\incorrect\\" + cls)

	for i in range(len(Z)):
		if y_pred[i] != y_true[i]:
			_plot_multich_with_bbox(Z[i], cls_mapping[y_pred[i]], small_voi_df,
					save_dir=save_dir + "\\incorrect\\" + cls_mapping[y_true[i]])
		else:
			_plot_multich_with_bbox(Z[i], cls_mapping[y_pred[i]], small_voi_df,
					save_dir=save_dir + "\\correct\\" + cls_mapping[y_true[i]])

def condense_cm(y_true, y_pred, cls_mapping):
	"""From lists y_true and y_pred with class numbers, """
	C = config.Config()
	
	y_true_simp = np.array([C.simplify_map[cls_mapping[y]] for y in y_true])
	y_pred_simp = np.array([C.simplify_map[cls_mapping[y]] for y in y_pred])
	
	return y_true_simp, y_pred_simp, ['hcc', 'benign', 'malignant non-hcc']

####################################
### Training Submodules
####################################

def _train_generator_func(train_ids, voi_df=None, n=12, n_art=0, C=None):
	"""n is the number of samples from each class, n_art is the number of artificial samples"""

	import voi_methods as vm
	if C is None:
		C = config.Config()
	if voi_df is None:
		voi_df = pd.read_csv(C.art_voi_path)
	classes_to_include = C.classes_to_include
	
	num_classes = len(classes_to_include)
	while True:
		x1 = np.empty(((n+n_art)*num_classes, C.dims[0], C.dims[1], C.dims[2], C.nb_channels))
		#x2 = np.empty(((n+n_art)*num_classes, 2))
		y = np.zeros(((n+n_art)*num_classes, num_classes))

		train_cnt = 0
		for cls in classes_to_include:
			if n_art > 0:
				img_fns = os.listdir(C.artif_dir+cls)
				for _ in range(n_art):
					img_fn = random.choice(img_fns)
					x1[train_cnt] = np.load(C.artif_dir + cls + "\\" + img_fn)
					#x2[train_cnt] = avg_X2[cls]
					y[train_cnt][C.classes_to_include.index(cls)] = 1

					train_cnt += 1

			img_fns = os.listdir(C.aug_dir+cls)
			while n > 0:
				img_fn = random.choice(img_fns)
				if img_fn[:img_fn.rfind('_')] + ".npy" in train_ids[cls]:
					x1[train_cnt] = np.load(C.aug_dir+cls+"\\"+img_fn)
					if C.hard_scale:
						x1[train_cnt] = vm.scale_intensity(x1[train_cnt], 1, max_int=2)#, keep_min=True)

					row = voi_df[(voi_df["Filename"] == img_fn[:img_fn.find('_')] + ".npy") &
								 (voi_df["lesion_num"] == int(img_fn[img_fn.find('_')+1:img_fn.rfind('_')]))]
					#x2[train_cnt] = [(float(row["real_dx"]) * float(row["real_dy"]) * float(row["real_dz"])) ** (1/3) / 50,
					#					max(float(row["real_dx"]), float(row["real_dy"])) / float(row["real_dz"])]
					
					y[train_cnt][C.classes_to_include.index(cls)] = 1
					
					train_cnt += 1
					if train_cnt % (n+n_art) == 0:
						break

		if C.non_imaging_inputs:
			yield _separate_phases([np.array(x1), np.array(x2)]), np.array(y) #[np.array(x1), np.array(x2)], np.array(y) #
		else:
			yield np.array(x1), np.array(y) #[np.array(x1), np.array(x2)], np.array(y) #

def _train_generator_func_2d(train_ids, voi_df, avg_X2, n=12, n_art=0, C=None):
	"""n is the number of samples from each class, n_art is the number of artificial samples"""

	classes_to_include = C.classes_to_include
	if C is None:
		C = config.Config()
	
	num_classes = len(classes_to_include)
	while True:
		x1 = np.empty(((n+n_art)*num_classes, C.dims[0], C.dims[1], C.nb_channels))
		x2 = np.empty(((n+n_art)*num_classes, 2))
		y = np.zeros(((n+n_art)*num_classes, num_classes))

		train_cnt = 0
		for cls in classes_to_include:
			if n_art>0:
				img_fns = os.listdir(C.artif_dir+cls)
				for _ in range(n_art):
					img_fn = random.choice(img_fns)
					temp = np.load(C.artif_dir + cls + "\\" + img_fn)
					x1[train_cnt] = temp[:,:,temp.shape[2]//2,:]
					x2[train_cnt] = avg_X2[cls]
					y[train_cnt][C.classes_to_include.index(cls)] = 1

					train_cnt += 1

			img_fns = os.listdir(C.aug_dir+cls)
			while n>0:
				img_fn = random.choice(img_fns)
				if img_fn[:img_fn.rfind('_')] + ".npy" in train_ids[cls]:
					temp = np.load(C.aug_dir+cls+"\\"+img_fn)
					x1[train_cnt] = temp[:,:,temp.shape[2]//2,:]

					row = voi_df[(voi_df["Filename"] == img_fn[:img_fn.find('_')] + ".npy") &
								 (voi_df["lesion_num"] == int(img_fn[img_fn.find('_')+1:img_fn.rfind('_')]))]
					x2[train_cnt] = [(float(row["real_dx"]) * float(row["real_dy"]) * float(row["real_dz"])) ** (1/3) / 50,
										max(float(row["real_dx"]), float(row["real_dy"])) / float(row["real_dz"])]
					
					y[train_cnt][C.classes_to_include.index(cls)] = 1
					
					train_cnt += 1
					if train_cnt % (n+n_art) == 0:
						break
			
		
		yield _separate_phases([np.array(x1), np.array(x2)]), np.array(y) #[np.array(x1), np.array(x2)], np.array(y) #

def _separate_phases(X, non_imaging_inputs=False):
	"""Assumes X[0] contains imaging and X[1] contains dimension data.
	Reformats such that X[0:2] has 3 phases and X[3] contains dimension data.
	Image data still is 5D (nb_samples, 3D, 1 channel).
	Handles both 2D and 3D images"""
	
	if non_imaging_inputs:
		dim_data = copy.deepcopy(X[1])
		img_data = X[0]
		
		if len(X[0].shape)==5:
			X[1] = np.expand_dims(X[0][:,:,:,:,1], axis=4)
			X += [np.expand_dims(X[0][:,:,:,:,2], axis=4)]
			X += [dim_data]
			X[0] = np.expand_dims(X[0][:,:,:,:,0], axis=4)
		
		else:
			X[1] = np.expand_dims(X[0][:,:,:,1], axis=3)
			X += [np.expand_dims(X[0][:,:,:,2], axis=3)]
			X += [dim_data]
			X[0] = np.expand_dims(X[0][:,:,:,0], axis=3)
	
	else:
		X = np.array(X)
		if len(X.shape)==5:
			X = [np.expand_dims(X[:,:,:,:,0], axis=4), np.expand_dims(X[:,:,:,:,1], axis=4), np.expand_dims(X[:,:,:,:,2], axis=4)]
		else:
			X = [np.expand_dims(X[:,:,:,0], axis=3), np.expand_dims(X[:,:,:,1], axis=3), np.expand_dims(X[:,:,:,2], axis=3)]

	return X

def _collect_unaug_data(C, verbose=False):
	"""Return dictionary pointing to X (img data) and Z (filenames) and dictionary storing number of samples of each class."""

	orig_data_dict = {}
	num_samples = {}
	voi_df = pd.read_csv(C.art_voi_path)

	for cls in C.classes_to_include:
		if verbose:
			print("\n"+cls)

		x = np.empty((10000, C.dims[0], C.dims[1], C.dims[2], C.nb_channels))
		#x2 = np.empty((10000, 2))
		z = []

		for index, img_fn in enumerate(os.listdir(C.orig_dir+cls)):
			try:
				x[index] = np.load(C.orig_dir+cls+"\\"+img_fn)
				if C.hard_scale:
					x[index] = vm.scale_intensity(x[index], 1, max_int=2)#, keep_min=True)
			except:
				raise ValueError(C.orig_dir+cls+"\\"+img_fn + " not found")
			z.append(img_fn)
			
			row = voi_df[(voi_df["Filename"] == img_fn[:img_fn.find('_')] + ".npy") &
						 (voi_df["lesion_num"] == int(img_fn[img_fn.find('_')+1:-4]))]
			
			try:
				skip=False
				#x2[index] = [(float(row["real_dx"]) * float(row["real_dy"]) * float(row["real_dz"])) ** (1/3) / 50,
				#			max(float(row["real_dx"]), float(row["real_dy"])) / float(row["real_dz"])]
			except TypeError:
				print(img_fn[:img_fn.find('_')], end=",")
				skip=True
				continue
				#raise ValueError(img_fn + " is probably missing a voi_df entry.")

		if not skip:
			x.resize((index+1, C.dims[0], C.dims[1], C.dims[2], C.nb_channels)) #shrink first dimension to fit
			#x2.resize((index+1, 2)) #shrink first dimension to fit
			orig_data_dict[cls] = [x,np.array(z)]
			num_samples[cls] = index + 1
		
	return orig_data_dict, num_samples

###########################
### Output Submodules
###########################

def _plot_multich_with_bbox(fn, pred_class, small_voi_df, num_ch=3, save_dir=None, normalize=False, C=None):
	if C is None:
		C = config.Config()

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
		
	img_fn = fn[:fn.find('_')] + ".npy"
	cls = small_voi_df.loc[small_voi_df["id"] == fn[:-4], "cls"]
	
	img = np.load(C.crops_dir + cls + "\\" + fn)
	img_slice = img[:,:, img.shape[2]//2, :].astype(float)
	#for ch in range(img_slice.shape[-1]):
	#	img_slice[:, :, ch] *= 255/np.amax(img_slice[:, :, ch])
	if normalize:
		img_slice[0,0,:]=-1
		img_slice[0,-1,:]=.8

	img_slice = np.stack([img_slice, img_slice, img_slice], axis=2)
	
	img_slice = _draw_bbox(img_slice, small_voi_df.loc[small_voi_df["id"] == fn[:-4], "coords"])
		
	ch1 = np.transpose(img_slice[:,::-1,:,0], (1,0,2))
	ch2 = np.transpose(img_slice[:,::-1,:,1], (1,0,2))
	
	if num_ch == 2:
		ret = np.empty([ch1.shape[0]*2, ch1.shape[1], 3])
		ret[:ch1.shape[0],:,:] = ch1
		ret[ch1.shape[0]:,:,:] = ch2
		
	elif num_ch == 3:
		ch3 = np.transpose(img_slice[:,::-1,:,2], (1,0,2))

		ret = np.empty([ch1.shape[0]*3, ch1.shape[1], 3])
		ret[:ch1.shape[0],:,:] = ch1
		ret[ch1.shape[0]:ch1.shape[0]*2,:,:] = ch2
		ret[ch1.shape[0]*2:,:,:] = ch3
		
	else:
		raise ValueError("Invalid num channels")
		
	imsave("%s\\large-%s (pred %s).png" % (save_dir, fn[:-4], pred_class), ret)


	rescale_factor = 3
	img = np.load(C.orig_dir + cls + "\\" + fn)

	img_slice = img[:,:, img.shape[2]//2, :].astype(float)

	if normalize:
		img_slice[0,0,:]=-1
		img_slice[0,-1,:]=.8
		
	ch1 = np.transpose(img_slice[:,::-1,0], (1,0))
	ch2 = np.transpose(img_slice[:,::-1,1], (1,0))
	
	if num_ch == 2:
		ret = np.empty([ch1.shape[0]*2, ch1.shape[1]])
		ret[:ch1.shape[0],:] = ch1
		ret[ch1.shape[0]:,:] = ch2
		
	elif num_ch == 3:
		ch3 = np.transpose(img_slice[:,::-1,2], (1,0))

		ret = np.empty([ch1.shape[0]*3, ch1.shape[1]])
		ret[:ch1.shape[0],:] = ch1
		ret[ch1.shape[0]:ch1.shape[0]*2,:] = ch2
		ret[ch1.shape[0]*2:,:] = ch3

	imsave("%s\\small-%s (pred %s).png" % (save_dir, fn[:fn.find('.')], pred_class), rescale(ret, rescale_factor, mode='constant'))

def _draw_bbox(img_slice, voi, C=None):
	"""Draw a colored box around the voi of an image slice showing how it would be cropped."""

	if C is None:
		C = config.Config()

	scale_ratios = vm.get_scale_ratios(voi)
	
	crop = [img_slice.shape[i] - round(C.dims[i]/scale_ratios[i]) for i in range(2)]
	
	x1 = crop[0]//2
	x2 = -crop[0]//2
	y1 = crop[1]//2
	y2 = -crop[1]//2

	img_slice[x1:x2, y2, 2, :] = 1
	img_slice[x1:x2, y2, :2, :] = -1

	img_slice[x1:x2, y1, 2, :] = 1
	img_slice[x1:x2, y1, :2, :] = -1

	img_slice[x1, y1:y2, 2, :] = 1
	img_slice[x1, y1:y2, :2, :] = -1

	img_slice[x2, y1:y2, 2, :] = 1
	img_slice[x2, y1:y2, :2, :] = -1
	
	return img_slice

if __name__ == '__main__':
	run_fixed_hyperparams()
	#import doctest
	#doctest.testmod()