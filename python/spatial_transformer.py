from keras.layers.core import Layer
import tensorflow as tf

class SpatialTransformer(Layer):
	"""Spatial Transformer Layer
	Implements a spatial transformer layer as described in [1]_.
	Borrowed from [2]_:
	downsample_factor : float
		A value of 1 will keep the orignal size of the image.
		Values larger than 1 will down sample the image. Values below 1 will
		upsample the image.
		example image: height= 100, width = 200
		downsample_factor = 2
		output image will then be 50, 100
	References
	----------
	.. [1]  Spatial Transformer Networks
			Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
			Submitted on 5 Jun 2015
	.. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

	.. [3]  https://github.com/EderSantana/seya/blob/keras1/seya/layers/attention.py
	"""

	def __init__(self,
				 localization_net,
				 downsample_factor=1,
				 return_theta=False,
				 **kwargs):
		self.locnet = localization_net
		self.downsample_factor = downsample_factor
		self.return_theta = return_theta
		super(SpatialTransformer, self).__init__(**kwargs)

	def build(self, input_shape):
		if hasattr(self, 'previous'):
			self.locnet.set_previous(self.previous)
		self.locnet.build()
		self.trainable_weights = self.locnet.trainable_weights
		#self.regularizers = self.locnet.regularizers
		#self.constraints = self.locnet.constraints
		#self.input = self.locnet.input  # This must be T.tensor4()
		#self.input_shape = input_shape

	def compute_output_shape(self, input_shape):
		return (None,
				int(input_shape[1] / self.downsample_factor),
				int(input_shape[2] / self.downsample_factor),
				int(input_shape[3] / self.downsample_factor),
				input_shape[-1])

	def get_output(self, train=False):
		X = self.get_input(train)
		theta = apply_model(self.locnet, X)
		theta = theta.reshape((X.shape[0], 3,4))
		output = self._transform(theta, X, self.downsample_factor)

		if self.return_theta:
			return theta.reshape((X.shape[0], 12))
		else:
			return output

	"""def call(self, X, mask=None):
					affine_Tx = self.locnet.call(X)
					output = self._transform(affine_Tx, X, self.output_size)
					return output"""

	def _repeat(self, x, num_repeats):
		ones = tf.ones((1, num_repeats), dtype='int32')
		x = tf.reshape(x, shape=(-1,1))
		x = tf.matmul(x, ones)
		return tf.reshape(x, [-1])

	def _interpolate(self, image, x, y, z, output_size):
		batch_size = tf.shape(image)[0]
		height = tf.shape(image)[1]
		width = tf.shape(image)[2]
		depth = tf.shape(image)[3]
		num_channels = tf.shape(image)[-1]

		x = tf.cast(x, dtype='float32')
		y = tf.cast(y, dtype='float32')
		z = tf.cast(z, dtype='float32')

		height_f = tf.cast(height, dtype='float32')
		width_f = tf.cast(width, dtype='float32')
		depth_f = tf.cast(depth, dtype='float32')

		# rescale from [-1,1] to input img coords
		x = .5*(x + 1.0)*(width_f)
		y = .5*(y + 1.0)*(height_f)
		z = .5*(z + 1.0)*(depth_f)

		x0 = tf.cast(tf.floor(x), 'int32')
		x1 = x0 + 1
		y0 = tf.cast(tf.floor(y), 'int32')
		y1 = y0 + 1
		z0 = tf.cast(tf.floor(z), 'int32')
		z1 = z0 + 1

		max_x = tf.cast(width - 1,  dtype='int32')
		max_y = tf.cast(height - 1, dtype='int32')
		max_z = tf.cast(depth - 1,  dtype='int32')
		zero = tf.zeros([], dtype='int32')

		x0 = tf.clip_by_value(x0, zero, max_x)
		x1 = tf.clip_by_value(x1, zero, max_x)
		y0 = tf.clip_by_value(y0, zero, max_y)
		y1 = tf.clip_by_value(y1, zero, max_y)
		z0 = tf.clip_by_value(z0, zero, max_z)
		z1 = tf.clip_by_value(z1, zero, max_z)

		flat_image_dims = width*height*depth
		pixels_batch = tf.range(batch_size)*flat_image_dims
		flat_output_dims = output_size[0]*output_size[1]*output_size[2]
		base = self._repeat(pixels_batch, flat_output_dims)
		base_y0z0 = base + y0*width + z0*width*height
		base_y1z0 = base + y1*width + z0*width*height
		base_y0z1 = base + y0*width + z1*width*height
		base_y1z1 = base + y1*width + z1*width*height
		ix_a = base_y0z0 + x0
		ix_b = base_y1z0 + x0
		ix_c = base_y0z0 + x1
		ix_d = base_y1z0 + x1
		ix_e = base_y0z1 + x0
		ix_f = base_y1z1 + x0
		ix_g = base_y0z1 + x1
		ix_h = base_y1z1 + x1

		flat_image = tf.reshape(image, shape=(-1, num_channels))
		flat_image = tf.cast(flat_image, dtype='float32')
		pixel_values = [tf.gather(flat_image, ix_a),
						tf.gather(flat_image, ix_b),
						tf.gather(flat_image, ix_c),
						tf.gather(flat_image, ix_d),
						tf.gather(flat_image, ix_e),
						tf.gather(flat_image, ix_f),
						tf.gather(flat_image, ix_g),
						tf.gather(flat_image, ix_h)]

		x0 = tf.cast(x0, 'float32')
		x1 = tf.cast(x1, 'float32')
		y0 = tf.cast(y0, 'float32')
		y1 = tf.cast(y1, 'float32')
		z0 = tf.cast(z0, 'float32')
		z1 = tf.cast(z1, 'float32')

		# figure out contribution of each input img pixel to the value at the output img pixel
		area_a = tf.expand_dims(((x1 - x) * (y1 - y) * (z1 - z)), 1)
		area_b = tf.expand_dims(((x1 - x) * (y - y0) * (z1 - z)), 1)
		area_c = tf.expand_dims(((x - x0) * (y1 - y) * (z1 - z)), 1)
		area_d = tf.expand_dims(((x - x0) * (y - y0) * (z1 - z)), 1)
		area_e = tf.expand_dims(((x1 - x) * (y1 - y) * (z1 - z0)), 1)
		area_f = tf.expand_dims(((x1 - x) * (y - y0) * (z1 - z0)), 1)
		area_g = tf.expand_dims(((x - x0) * (y1 - y) * (z1 - z0)), 1)
		area_h = tf.expand_dims(((x - x0) * (y - y0) * (z1 - z0)), 1)

		output = tf.add_n([area_a*pixel_values[0],
						   area_b*pixel_values[1],
						   area_c*pixel_values[2],
						   area_d*pixel_values[3],
						   area_e*pixel_values[4],
						   area_f*pixel_values[5],
						   area_g*pixel_values[6],
						   area_h*pixel_values[7]])
		return output

	def _meshgrid(self, D):
		x_linspace = tf.linspace(-1., 1., D[1])
		y_linspace = tf.linspace(-1., 1., D[0])
		z_linspace = tf.linspace(-1., 1., D[2])
		x_coords, y_coords, z_coords = tf.meshgrid(x_linspace, y_linspace, z_linspace)
		x_coords = tf.reshape(x_coords, [-1])
		y_coords = tf.reshape(y_coords, [-1])
		z_coords = tf.reshape(z_coords, [-1])
		ones = tf.ones_like(x_coords)
		ix_grid = tf.concat([x_coords, y_coords, z_coords, ones], 0)
		return ix_grid

	def _transform(self, affine_Tx, input_shape, output_size):
		batch_size = tf.shape(input_shape)[0]
		num_channels = tf.shape(input_shape)[-1]

		affine_Tx = tf.reshape(affine_Tx, shape=(batch_size,3,4))
		affine_Tx = tf.reshape(affine_Tx, (-1, 3, 4))
		affine_Tx = tf.cast(affine_Tx, 'float32')

		ix_grid = self._meshgrid(output_size)
		ix_grid = tf.expand_dims(ix_grid, 0)
		ix_grid = tf.reshape(ix_grid, [-1])
		ix_grid = tf.tile(ix_grid, tf.stack([batch_size]))
		ix_grid = tf.reshape(ix_grid, (batch_size, 4, -1))

		# grid of input img coords that correspond to the evenly spaced cartesian coords in the output img
		transformed_grid = tf.matmul(affine_Tx, ix_grid)
		x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
		y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
		z_s = tf.slice(transformed_grid, [0, 2, 0], [-1, 1, -1])
		x_s_flatten = tf.reshape(x_s, [-1])
		y_s_flatten = tf.reshape(y_s, [-1])
		z_s_flatten = tf.reshape(z_s, [-1])

		transformed_image = self._interpolate(input_shape,
												x_s_flatten,
												y_s_flatten,
												z_s_flatten,
												output_size)

		transformed_image = tf.reshape(transformed_image, shape=(batch_size,
																*output_size,
																num_channels))
		return transformed_image


