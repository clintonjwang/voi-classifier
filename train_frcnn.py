#!/usr/bin/env python3

# TODO: fix inconsistencies in width/height ordering

from __future__ import division
import random
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.callbacks import TensorBoard
from keras_frcnn import config, data_generators
from keras_frcnn import losses as klosses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils

import tensorflow as tf
from tensorflow.python import debug as tf_debug
sess = tf.Session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

DIMS=3

def train(parser, epoch_length = 2): #1000
	"""For some reason some versions of keras use theano ordering for tensorflow.
	https://github.com/fchollet/keras/issues/3945
	"""

	def get_config_obj(class_mapping, config_filename):
		C = config.Config()
		C.rot_90 = bool(options.rot_90)
		C.model_path = options.output_weight_path
		C.num_rois = int(options.num_rois)
		C.network = options.network
		C.base_net_weights = options.input_weight_path
		C.class_mapping = class_mapping

		with open(config_filename, 'wb') as config_f:
			pickle.dump(C,config_f)
			print('Config has been written to %s, and can be loaded when testing' % config_filename)

		return C

	def get_data_gens(all_imgs, classes_count, C):
		"""Get train/val generators"""
		random.shuffle(all_imgs)

		train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
		val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

		print('Num train samples {}'.format(len(train_imgs)))
		print('Num val samples {}'.format(len(val_imgs)))

		data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, mode='train')
		data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length, mode='val')

		return data_gen_train, data_gen_val

	def setup_models(classes_count, C):
		if nn.CHANNEL_AXIS==4:
			img_input = Input(shape=(None, None, None, C.nb_channels))
		else:
			img_input = Input(shape=(C.nb_channels, None, None, None))
		roi_input = Input(shape=(None, DIMS*2))

		shared_layers = nn.nn_base(input_tensor=img_input, trainable=True) # define the base network (resnet here, can be VGG, Inception, etc)

		num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
		rpn = nn.rpn(shared_layers, num_anchors) # define the RPN, built on the base layers

		classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

		model_rpn = Model(img_input, rpn[:2])
		model_classifier = Model([img_input, roi_input], classifier)

		# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
		model_all = Model([img_input, roi_input], rpn[:2] + classifier)

		if C.base_net_weights is None:
			print("No pretrained model available yet.")
		else:
			try:
				print('loading weights from {}'.format(C.base_net_weights))
				model_rpn.load_weights(C.base_net_weights, by_name=True)
				model_classifier.load_weights(C.base_net_weights, by_name=True)
			except:
				print('Could not load pretrained model weights. Weights can be found in the keras application folder \
					https://github.com/fchollet/keras/tree/master/keras/applications')

		model_rpn.compile(optimizer=Adam(lr=1e-5),
			loss=[klosses.rpn_loss_cls(num_anchors), klosses.rpn_loss_regr(num_anchors)])
		model_classifier.compile(optimizer=Adam(lr=1e-5),
			loss=[klosses.class_loss_cls, klosses.class_loss_regr(len(classes_count)-1)],
			metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
		model_all.compile(optimizer='sgd', loss='mae')

		return model_rpn, model_classifier, model_all

	def select_samples(pos_samples, neg_samples, C):
		if C.num_rois > 1:
			if len(pos_samples) < C.num_rois//2:
				selected_pos_samples = pos_samples.tolist()
			else:
				selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
			try:
				selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
			except:
				selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

			sel_samples = selected_pos_samples + selected_neg_samples
		else:
			# in the extreme case where num_rois = 1, we pick a random pos or neg sample
			selected_pos_samples = pos_samples.tolist()
			selected_neg_samples = neg_samples.tolist()
			if np.random.randint(0, 2):
				sel_samples = random.choice(neg_samples)
			else:
				sel_samples = random.choice(pos_samples)
		
		return sel_samples

	(options, _) = parser.parse_args()

	if options.network == 'resnet50':
		from keras_frcnn import resnet as nn
	else:
		raise ValueError('%s is not a supported model' % options.network)

	all_imgs, classes_count, class_mapping = data_generators.get_data(options.train_path)

	C = get_config_obj(class_mapping, options.config_filename)

	data_gen_train, data_gen_val = get_data_gens(all_imgs, classes_count, C)

	model_rpn, model_classifier, model_all = setup_models(classes_count, C)

	num_epochs = int(options.num_epochs)
	iter_num = 0

	losses = np.zeros((epoch_length, 5))
	rpn_accuracy_rpn_monitor = []
	rpn_accuracy_for_epoch = []
	start_time = time.time()

	best_loss = np.Inf

	print('\nStarting training:')
	for epoch_num in range(num_epochs):

		progbar = generic_utils.Progbar(epoch_length)
		print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

		while True:
			X, Y, img_data = next(data_gen_train)

			#with tf.session() as sess:
			#	tf.shape(...)
			if nn.CHANNEL_AXIS==1:
				X=X.transpose([0,4,1,2,3])
				Y[0]=Y[0].transpose([0,4,1,2,3])
				Y[1]=Y[1].transpose([0,4,1,2,3])
			#print('Y shape:', Y[0].shape, Y[1].shape) # Y[0] is rpn, Y[1] is regression (6*nb_anchor_types)
			loss_rpn = model_rpn.train_on_batch(X, Y)
			#loss_rpn = model_rpn.fit(X, Y, callbacks=[TensorBoard()]) #train_on_batch(X, Y)
			#print("Checkpoint: train rpn")

			P_rpn = model_rpn.predict_on_batch(X)
			
			R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, use_regr=True, overlap_thresh=0.7, max_boxes=300, channel_axis=nn.CHANNEL_AXIS)
			if R is None:
				#print("Failed to generate any roi candidates for image " + img_data['filepath'])
				break

			# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
			X2, Y1, Y2, _ = roi_helpers.calc_iou(R, img_data, C, class_mapping)
			print(".", end="")

			if X2 is None:
				rpn_accuracy_rpn_monitor.append(0)
				rpn_accuracy_for_epoch.append(0)
				if iter_num == epoch_length:
					#print("Failed to generate any roi candidates for image " + img_data['filepath'])
					break
				continue

			neg_samples = np.where(Y1[0, :, -1] == 1)
			neg_samples = neg_samples[0] if len(neg_samples) > 0 else []
			pos_samples = np.where(Y1[0, :, -1] == 0)
			pos_samples = pos_samples[0] if len(pos_samples) > 0 else []
			rpn_accuracy_rpn_monitor.append(len(pos_samples))
			rpn_accuracy_for_epoch.append(len(pos_samples))

			sel_samples = select_samples(pos_samples, neg_samples, C)
			print ("Checkpoint: select samples")

			loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
			print ("Checkpoint: train classifier")

			losses[iter_num, 0] = loss_rpn[1]
			losses[iter_num, 1] = loss_rpn[2]
			losses[iter_num, 2] = loss_class[1]
			losses[iter_num, 3] = loss_class[2]
			losses[iter_num, 4] = loss_class[3]

			iter_num += 1
			progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
									  ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

			#gather summary stats for the epoch
			if iter_num == epoch_length:
				loss_rpn_cls = np.mean(losses[:, 0])
				loss_rpn_regr = np.mean(losses[:, 1])
				loss_class_cls = np.mean(losses[:, 2])
				loss_class_regr = np.mean(losses[:, 3])
				class_acc = np.mean(losses[:, 4])

				mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
				rpn_accuracy_for_epoch = []

				if C.verbose:
					print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
					print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
					print('Loss RPN classifier: {}'.format(loss_rpn_cls))
					print('Loss RPN regression: {}'.format(loss_rpn_regr))
					print('Loss Detector classifier: {}'.format(loss_class_cls))
					print('Loss Detector regression: {}'.format(loss_class_regr))
					print('Elapsed time: {}'.format(time.time() - start_time))

				curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
				iter_num = 0
				start_time = time.time()

				if curr_loss < best_loss:
					if C.verbose:
						print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
					best_loss = curr_loss
					model_all.save_weights(C.model_path)

				break

			if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
				mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
				rpn_accuracy_rpn_monitor = []
				print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
				if mean_overlapping_bboxes == 0:
					print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
	
	print('Training complete, exiting.')


def main():
	sys.setrecursionlimit(40000)
	parser = OptionParser()

	parser.add_option("-p", "--path", dest="train_path", help="Path to training data.",
					default="./train_list.txt")
	parser.add_option("-n", "--num_rois", dest="num_rois", help="Number of RoIs to process at once.", default=4) #32
	parser.add_option("--network", dest="network", help="Base network to use. Supports resnet50 only.", default='resnet50')
	parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
	parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
	parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
					  action="store_true", default=False)
	parser.add_option("--num_epochs", dest="num_epochs", help="Number of epochs.", default=3) #2000
	parser.add_option("--config_filename", dest="config_filename", help=
					"Location to store all the metadata related to the training (to be used when testing).",
					default="config.pickle")
	parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.",
					default='./model_frcnn.hdf5')
	parser.add_option("--input_weight_path", dest="input_weight_path",
					help="Input path for weights.",
					default=None)#'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

	train(parser)

if __name__ == "__main__":
	main()
