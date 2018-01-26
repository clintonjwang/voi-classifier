"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
	   python capsulenet.py
	   python capsulenet.py --epochs 50
	   python capsulenet.py --epochs 50 --routings 3
	   ... ...
	   
Result:
	Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
	About 110 seconds per epoch on a single GTX1070 GPU card
	
Author 1: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
Author 2: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import os
import argparse
from keras import callbacks
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from keras_contrib.layers.normalization import InstanceNormalization
import matplotlib.pyplot as plt
import utils
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

import cnn_builder as cbuild
import config
import pandas as pd

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings, T=None):
	"""
	A Capsule Network on MNIST.
	:param input_shape: data shape, 4d, [width, height, channels]
	:param n_class: number of classes
	:param routings: number of routing iterations
	:return: Two Keras Models, the first one used for training, and the second one for evaluation.
			`eval_model` can also be used for training.
	"""
	dense_layers = False
	dim_capsule = [8, 8]
	dense_units = 256 #512
	n_channels = 16 # 32
	time_dist = False

	if T is not None:
		dense_layers = T.dense_layers
		dim_capsule = T.dim_capsule
		dense_units = T.dense_units
		n_channels = T.n_channels
		time_dist = T.time_dist

	x = layers.Input(shape=input_shape)

	# Layer 1: Just a conventional Conv3D layer
	if time_dist:
		conv1 = layers.Reshape(target_shape=(24,24,12,3,1))(x)
		conv1 = layers.Permute((4,1,2,3,5))(conv1)
		conv1 = layers.TimeDistributed(layers.Conv3D(filters=256, kernel_size=8, strides=1,
			padding='valid', activation='relu', name='conv1'))(conv1)
	else:
		conv1 = layers.Conv3D(filters=256, kernel_size=8, strides=1, padding='valid', name='conv1')(x) #[9,9,9]
		conv1 = layers.BatchNormalization(axis=4)(conv1)
		conv1 = layers.Activation('relu')(conv1)

	# Layer 2: Conv3D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
	primarycaps = PrimaryCap(conv1, dim_capsule=dim_capsule[0], n_channels=n_channels, 
		kernel_size=5, strides=2, padding='valid', time_dist=time_dist) #[6,6,3]
	# Layer 3: Capsule layer. Routing algorithm works here.
	digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=dim_capsule[1], routings=routings,
							 name='digitcaps')(primarycaps)

	# Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
	# If using tensorflow, this will not be necessary. :)
	out_caps = Length(name='capsnet')(digitcaps)

	# Decoder network.
	y = layers.Input(shape=(n_class,))
	masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
	masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

	# Shared Decoder model in training and prediction
	decoder = models.Sequential(name='decoder')
	if not dense_layers:
		decoder.add(layers.Dense(dense_units, activation='relu', input_dim=dim_capsule[1]*n_class))
		decoder.add(layers.Reshape(target_shape=(1,1,1,dense_units))) #(3,4,4,1,1)
		decoder.add(layers.Conv3DTranspose(filters=64, kernel_size=[11,11,6], strides=1, padding='valid'))
		decoder.add(layers.BatchNormalization(axis=4))
		decoder.add(layers.Activation('relu'))
		decoder.add(layers.Conv3DTranspose(filters=64, kernel_size=[4,4,2], strides=2, padding='valid'))
		decoder.add(layers.BatchNormalization(axis=4))
		decoder.add(layers.Activation('relu'))
		#decoder.add(layers.TimeDistributed(layers.Conv3DTranspose(filters=256, kernel_size=[8,8,6], strides=1, padding='valid', activation='relu')))
		#decoder.add(layers.TimeDistributed(layers.Conv3DTranspose(filters=128, kernel_size=[4,4,2], strides=2, padding='valid', activation='relu')))
		#decoder.add(layers.TimeDistributed(layers.Conv3DTranspose(filters=256, kernel_size=[14,14,10], strides=1, padding='valid', activation='relu')))
		#decoder.add(layers.TimeDistributed(layers.Conv3DTranspose(filters=128, kernel_size=[8,8,3], strides=1, padding='valid', activation='relu')))
		decoder.add(layers.Dense(3, activation='sigmoid', name='out_recon'))
		#decoder.add(layers.Reshape((3, 24, 24, 12)))
		#decoder.add(layers.Permute((2,3,4,1), name='out_recon'))
	else:
		#decoder.add(layers.Dense(dense_units, activation='relu', input_dim=dim_capsule*n_class))
		decoder.add(layers.Dense(dense_units, activation='relu', input_dim=dim_capsule[1]*n_class)) #1024
		#decoder.add(layers.BatchNormalization())
		#decoder.add(layers.Activation('relu'))
		decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
		decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

	# Models for training and evaluation (prediction)
	train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
	eval_model = models.Model(x, [out_caps, decoder(masked)])

	# manipulate model
	noise = layers.Input(shape=(n_class, dim_capsule[1]))
	noised_digitcaps = layers.Add()([digitcaps, noise])
	masked_noised_y = Mask()([noised_digitcaps, y])
	manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
	return train_model, eval_model, manipulate_model

def run_hyperparam_cycle(max_cycles=50, T=None):
	"""Runs the CNN for max_runs times, saving performance metrics."""
	T_list = [config.hyperparams()]
	running_stats = cbuild.get_run_stats_csv()
	index = len(running_stats)

	model_names = os.listdir(C_list[0].model_dir)
	if len(model_names) > 0:	
		model_num = max([int(x[x.find('_')+1:x.find('.')]) for x in model_names]) + 1
	else:
		model_num = 0

	running_acc_6 = []
	running_acc_3 = []
	early_stopping = EarlyStopping(monitor='loss', min_delta=0.002, patience=3)


	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)


	T_index = 0
	while index < max_cycles:
		# load data
		if data is None:
			train_generator, (x_test, y_test), _ = cbuild.load_data_capsnet(n=T_list[0].n)
		else:
			train_generator, (x_test, y_test) = data
		(x_train, y_train), _ = next(train_generator)

		for T in T_list:
			model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
														  n_class=len(np.unique(np.argmax(y_train, 1))),
														  routings=args.routings, T=T)

			t = time.time()
			hist = train(model=model, data=(train_generator, (x_test, y_test)), args=args, T=T)
			loss_hist = hist.history['loss']

			test(model=eval_model, data=(x_test, y_test), args=args, T=T)


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

			#running_stats.loc[index] = _get_hyperparams_as_list(C, T) + [running_acc_6[-1], running_acc_3[-1], time.time()-t, loss_hist,
			#					num_samples['hcc'], num_samples['cholangio'], num_samples['colorectal'], num_samples['cyst'], num_samples['hemangioma'], num_samples['fnh'],
			#					cm, time.time(), #C.run_num,
			#					misclassified_test, misclassified_train, model_num, y_true, str(Y_pred), list(Z_test)]

			running_stats.to_csv(C.run_stats_path, index=False)

			model.save(C.model_dir+'models_%d.hdf5' % model_num)
			model_num += 1

			T_index += 1

def margin_loss(y_true, y_pred):
	"""
	Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not tested.
	:param y_true: [None, n_classes]
	:param y_pred: [None, num_capsule]
	:return: a scalar loss value.
	"""
	L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
		0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

	return K.mean(K.sum(L, 1))

def train(model, data, args, T=None):
	"""
	Training a CapsuleNet
	:param model: the CapsuleNet model
	:param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
	:param args: arguments
	:return: The trained model
	"""
	# unpacking the data
	steps_per_epoch=1000
	epochs=50
	lr=args.lr
	if T is not None:
		steps_per_epoch = T.steps_per_epoch
		epochs = T.epochs
		lr = T.lr

	train_generator, (x_test, y_test) = data

	# callbacks
	log = callbacks.CSVLogger(args.save_dir + '/log.csv')
	tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
							   batch_size=args.batch_size, histogram_freq=int(args.debug))
	checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
										   save_best_only=True, save_weights_only=True, verbose=1)
	lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

	# compile the model
	model.compile(optimizer=optimizers.Adam(lr=lr),
				  loss=['categorical_crossentropy', 'mse'], #[margin_loss, 'mse']
				  loss_weights=[1., args.lam_recon],
				  metrics={'capsnet': 'accuracy'})

	"""
	# Training without data augmentation:
	model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
			  validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
	"""

	# Begin: Training with data augmentation ---------------------------------------------------------------------#
	"""def train_generator(x, y, batch_size, shift_fraction=0.):
					train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
													   height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
					generator = train_datagen.flow(x, y, batch_size=batch_size)
					while 1:
						x_batch, y_batch = generator.next()
						yield ([x_batch, y_batch], [y_batch, x_batch])"""

	# Training with data augmentation. If shift_fraction=0., also no augmentation.
	(_, y_train), _ = next(train_generator)
	hist = model.fit_generator(train_generator,
						steps_per_epoch=steps_per_epoch, #int(y_train.shape[0] / args.batch_size)
						epochs=epochs, validation_data=[[x_test, y_test], [y_test, x_test]], #args.epochs
						callbacks=[log, tb, checkpoint, lr_decay])
	# End: Training with data augmentation -----------------------------------------------------------------------#

	model.save_weights(args.save_dir + '/trained_model.h5')
	print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

	from utils import plot_log
	plot_log(args.save_dir + '/log.csv', show=True)

	return model

def test(model, data, args, T=None):
	import importlib
	import utils
	importlib.reload(utils)
	x_test, y_test = data
	y_pred, x_recon = model.predict(x_test, batch_size=100)
	print('-'*30 + 'Begin: test' + '-'*30)
	print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

	img = utils.combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
	for i in range(3):
		plt.subplot(131+i)
		plt.imshow(img[:,:,i], cmap='gray')
	plt.show()
	img = np.concatenate([img[:,:,0], img[:,:,1], img[:,:,2]], axis=0)
	image = (img - np.min(img)) * 255 / (np.max(img) - np.min(img))
	Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
	print()
	print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
	print('-' * 30 + 'End: test' + '-' * 30)
	#plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"), cmap='gray')
	#plt.show()

def manipulate_latent(model, data, args, cls=None):
	C = config.Config()
	if cls is not None:
		args.digit = cls

	dim_capsule = 8
	nb_classes = 6

	print('-'*30 + 'Begin: manipulate' + '-'*30)
	x_test, y_test = data
	index = np.argmax(y_test, 1) == args.digit
	number = np.random.randint(low=0, high=sum(index) - 1)
	x, y = x_test[index][number], y_test[index][number]
	x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
	noise = np.zeros([1, nb_classes, dim_capsule])
	x_recons = []
	for dim in range(dim_capsule):
		for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
			tmp = np.copy(noise)
			tmp[:,:,dim] = r
			x_recon = model.predict([x, y, tmp])
			x_recons.append(x_recon)

	x_recons = np.concatenate(x_recons)

	img = utils.combine_images(x_recons, height=dim_capsule, multislice=True)
	#img = restack(img)
	img = np.concatenate([img[:,:,:,0], img[:,:,:,1], img[:,:,:,2]], axis=1)
	img = np.concatenate([img[:,:,0], img[:,:,1], img[:,:,2]], axis=0)
	image = (img - np.min(img)) * 255 / (np.max(img) - np.min(img))
	Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%s.png' % C.classes_to_include[args.digit])
	print('manipulated result saved to %s/manipulate-%s.png' % (args.save_dir, C.classes_to_include[args.digit]))
	print('-' * 30 + 'End: manipulate' + '-' * 30)

def restack(img):
	new_img = np.zeros([img.shape[0]*3, img.shape[1]])
	new_img[::3,:] = img[:,:,0]
	new_img[1::3,:] = img[:,:,1]
	new_img[2::3,:] = img[:,:,2]
	return new_img

def main(args, data=None):
	import importlib
	importlib.reload(utils)
	importlib.reload(cbuild)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	# load data
	if data is None:
		train_generator, (x_test, y_test), _ = cbuild.load_data_capsnet(n=2)
	else:
		train_generator, (x_test, y_test) = data
	(x_train, y_train), _ = next(train_generator)

	# define model
	model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
												  n_class=len(np.unique(np.argmax(y_train, 1))),
												  routings=args.routings)
	model.summary()

	# train or test
	if args.weights is not None:  # init the model weights with provided one
		model.load_weights(args.weights)
	if not args.testing:
		train(model=model, data=(train_generator, (x_test, y_test)), args=args)
	else:  # as long as weights are given, will run testing
		if args.weights is None:
			print('No weights are provided. Will test using random initialized weights.')
		manipulate_latent(manipulate_model, (x_test, y_test), args)
		test(model=eval_model, data=(x_test, y_test), args=args)

	return model, eval_model, manipulate_model

if __name__ == "__main__":
	import win_unicode_console
	win_unicode_console.enable()

	# setting the hyper parameters
	parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
	parser.add_argument('--epochs', default=50, type=int)
	parser.add_argument('--batch_size', default=100, type=int)
	parser.add_argument('--lr', default=0.001, type=float,
						help="Initial learning rate")
	parser.add_argument('--lr_decay', default=0.9, type=float,
						help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
	parser.add_argument('--lam_recon', default=0.392, type=float,
						help="The coefficient for the loss of decoder")
	parser.add_argument('-r', '--routings', default=3, type=int,
						help="Number of iterations used in routing algorithm. should > 0")
	parser.add_argument('--shift_fraction', default=0.1, type=float,
						help="Fraction of pixels to shift at most in each direction.")
	parser.add_argument('--debug', action='store_true',
						help="Save weights by TensorBoard")
	parser.add_argument('--save_dir', default='./result')
	parser.add_argument('-t', '--testing', action='store_true',
						help="Test the trained model on testing dataset")
	parser.add_argument('--digit', default=5, type=int,
						help="Digit to manipulate")
	parser.add_argument('-w', '--weights', default=None,
						help="The path of the saved weights. Should be specified when testing")
	args = parser.parse_args()

	main(args)
