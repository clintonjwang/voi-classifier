# https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/DenseNet/densenet.py
from keras.models import Model
import keras.layers as layers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling3D
from keras.layers import Input, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
import config
import importlib

importlib.reload(config)
C = config.Config()

def conv_factory(x, nb_filter, dropout_rate=None, w_decay=1E-4):
	"""Apply BatchNorm, Relu 3x3Conv2D, optional dropout
	:param x: Input keras network
	:param concat_axis: int -- index of contatenate axis
	:param nb_filter: int -- number of filters
	:param dropout_rate: int -- dropout rate
	:param w_decay: int -- weight decay factor
	:returns: keras network with b_norm, relu and Conv2D added
	:rtype: keras network
	"""

	x = BatchNormalization(gamma_regularizer=l2(w_decay),
												 beta_regularizer=l2(w_decay))(x)
	x = Activation('relu')(x)
	x = layers.Conv3D(nb_filter, 3,
						 kernel_initializer="he_uniform",
						 padding="same",
						 use_bias=False,
						 kernel_regularizer=l2(w_decay))(x)
	if dropout_rate:
		x = Dropout(dropout_rate)(x)

	return x

def transition(x, nb_filter, dropout_rate=None, w_decay=1E-4, pool=(2,2,2)):
	"""Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D
	:param x: keras model
	:param concat_axis: int -- index of contatenate axis
	:param nb_filter: int -- number of filters
	:param dropout_rate: int -- dropout rate
	:param w_decay: int -- weight decay factor
	:returns: model
	:rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool
	"""

	x = BatchNormalization(gamma_regularizer=l2(w_decay), beta_regularizer=l2(w_decay))(x)
	x = Activation('relu')(x)
	x = layers.Conv3D(nb_filter, 1,
						 kernel_initializer="he_uniform",
						 padding="same",
						 use_bias=False,
						 kernel_regularizer=l2(w_decay))(x)
	if dropout_rate:
		x = Dropout(dropout_rate)(x)
	x = layers.AveragePooling3D(pool, strides=pool)(x)

	return x

def denseblock(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, w_decay=1E-4):
	"""Build a denseblock where the output of each
		 conv_factory is fed to subsequent ones
	:param x: keras model
	:returns: keras model with nb_layers of conv_factory appended
	:rtype: keras model
	"""

	list_feat = [x]

	for i in range(nb_layers):
		x = conv_factory(x, growth_rate, dropout_rate, w_decay)
		list_feat.append(x)
		x = Concatenate()(list_feat)
		nb_filter += growth_rate

	return x, nb_filter

"""def denseblock_altern(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, w_decay=1E-4):
	for i in range(nb_layers):
		merge_tensor = conv_factory(x, growth_rate,
																dropout_rate, w_decay)
		x = Concatenate()([merge_tensor, x])
		nb_filter += growth_rate

	return x, nb_filter"""

def DenseNet(depth=22, nb_dense_block=3, growth_rate=16,
			 nb_filter=32, dropout_rate=None, w_decay=1E-4):
	""" Build the DenseNet model
	:param nb_classes: int -- number of classes
	:param img_dim: tuple -- (channels, rows, columns)
	:param depth: int -- how many layers
	:param nb_dense_block: int -- number of dense blocks to add to end
	:param growth_rate: int -- number of filters to add
	:param nb_filter: int -- number of filters
	:param dropout_rate: float -- dropout rate
	:param w_decay: float -- weight decay
	:returns: keras model with nb_layers of conv_factory appended
	:rtype: keras model
	"""
	
	model_input = Input(shape=(*C.dims, C.nb_channels))

	assert (depth - 4) % 3 == 0, "Depth must be 3N+4"

	# layers in each dense block
	nb_layers = int((depth - 4) / 3)

	# Initial convolution
	x = layers.Conv3D(nb_filter, 3,
						 kernel_initializer="he_uniform",
						 padding="same",
						 name="initial_conv2D",
						 use_bias=False,
						 kernel_regularizer=l2(w_decay))(model_input)

	# Add dense blocks
	for block_idx in range(nb_dense_block - 1):
		if block_idx == 0:
			x, nb_filter = denseblock(x, 2, nb_filter, growth_rate, 
							dropout_rate=dropout_rate, w_decay=w_decay)
		else:
			x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate, 
							dropout_rate=dropout_rate, w_decay=w_decay)
		# add transition
		if block_idx == 3:
			x = transition(x, nb_filter, dropout_rate=dropout_rate,
							w_decay=w_decay, pool=(2,2,1))
		else:
			x = transition(x, nb_filter, dropout_rate=dropout_rate,
							w_decay=w_decay, pool=(2,2,2))

	# The last denseblock does not have a transition
	x, nb_filter = denseblock(x, nb_layers,
							nb_filter, growth_rate, 
							dropout_rate=dropout_rate,
							w_decay=w_decay)

	x = BatchNormalization(gamma_regularizer=l2(w_decay),
												 beta_regularizer=l2(w_decay))(x)
	x = Activation('relu')(x)
	x = GlobalAveragePooling3D(data_format=K.image_data_format())(x)
	x = Dense(C.nb_classes,
			activation='softmax',
			kernel_regularizer=l2(w_decay),
			bias_regularizer=l2(w_decay))(x)

	densenet = Model(inputs=[model_input], outputs=[x], name="DenseNet")
	densenet.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

	return densenet