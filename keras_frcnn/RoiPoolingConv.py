from keras.engine.topology import Layer
import keras.backend as K
from scipy.ndimage import interpolation
import tensorflow as tf

class RoiPoolingConv(Layer):
    '''ROI pooling layer for 3D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        5D tensor with shape:
        `(1, rows, cols, depth, channels)`
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,z,dx,dy,dz)
    # Output shape
        4D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size, pool_size)`
    '''
    def __init__(self, pool_size, num_rois, channel_axis, **kwargs):
        self.dim_ordering = K.image_dim_ordering()
        self.pool_size = pool_size
        self.num_rois = num_rois
        self.channel_axis = channel_axis

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][self.channel_axis]

    def compute_output_shape(self, input_shape):
        if self.channel_axis == 4:
            return None, self.num_rois, self.pool_size, self.pool_size, self.pool_size, self.nb_channels
        else:
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size, self.pool_size

    def call(self, x, mask=None):

        assert(len(x) == 2)

        img = x[0]
        rois = x[1]

        outputs = []

        for roi_idx in range(self.num_rois):
            x = K.cast(rois[0, roi_idx, 0], 'int32')
            y = K.cast(rois[0, roi_idx, 1], 'int32')
            z = K.cast(rois[0, roi_idx, 2], 'int32')
            dx = K.cast(rois[0, roi_idx, 3], 'int32')
            dy = K.cast(rois[0, roi_idx, 4], 'int32')
            dz = K.cast(rois[0, roi_idx, 5], 'int32')

            # TODO: workaround since tf.image.resize_images only supports 2D images - this seems wrong
            #rs = interpolation.zoom(img[:, x:x+dx, y:y+dy, z:z+dz, :], (self.pool_size, self.pool_size, self.pool_size))
            resized_list = []
            if self.channel_axis == 4:
                img = img[:, x:x+dx, y:y+dy, z:z+dz, :]
            else:
                img = img[:, :, x:x+dx, y:y+dy, z:z+dz]

            try:
                for i in tf.unstack(img, num=self.pool_size, axis=3):
                    resized_list.append(tf.image.resize_images(i, [self.pool_size, self.pool_size]))
                stack_img = tf.stack(resized_list, axis=3)

                img = stack_img
                resized_list = []
                for i in tf.unstack(img, num=self.pool_size, axis=2):
                    resized_list.append(tf.image.resize_images(i, [self.pool_size, self.pool_size]))
                rs = tf.stack(resized_list, axis=2)
            except:
                rs = img[:, :, x:x+self.pool_size, y:y+self.pool_size, z:z+self.pool_size]

            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)
        if self.channel_axis == 4:
            return K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.pool_size, self.nb_channels))
        else:
            return K.reshape(final_output, (1, self.num_rois, self.nb_channels, self.pool_size, self.pool_size, self.pool_size))