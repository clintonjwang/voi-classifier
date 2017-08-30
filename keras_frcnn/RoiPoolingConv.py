from keras.engine.topology import Layer
import keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf

class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''
    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][4]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert(len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            z = rois[0, roi_idx, 2]
            w = rois[0, roi_idx, 3]
            h = rois[0, roi_idx, 4]
            d = rois[0, roi_idx, 5]
            
            row_length = w / float(self.pool_size)
            col_length = h / float(self.pool_size)
            slice_length = d / float(self.pool_size)

            num_pool_regions = self.pool_size

            #NOTE: the RoiPooling implementation differs between theano and tensorflow due to the lack of a resize op
            # in theano. The theano implementation is much less efficient and leads to long compile times

            if self.dim_ordering == 'th':
                raise ValueError("need to implement ROI pooling conv for theano")
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        for iz in range(num_pool_regions):
                            x1 = x + ix * row_length
                            x2 = x1 + row_length
                            y1 = y + jy * col_length
                            y2 = y1 + col_length
                            z1 = z + iz * slice_length
                            z2 = z1 + slice_length

                            x1 = K.cast(x1, 'int32')
                            x2 = K.cast(x2, 'int32')
                            z2 = K.cast(z2, 'int32')
                            y1 = K.cast(y1, 'int32')
                            y2 = K.cast(y2, 'int32')
                            z2 = K.cast(z2, 'int32')

                            x2 = x1 + K.maximum(1,x2-x1)
                            y2 = y1 + K.maximum(1,y2-y1)
                            z2 = z1 + K.maximum(1,z2-z1)
                            
                            new_shape = [input_shape[0], input_shape[1], input_shape[2],
                                         y2 - y1, x2 - x1, z2 - z1]

                            x_crop = img[:, :, :, y1:y2, x1:x2, z1:z2]
                            xm = K.reshape(x_crop, new_shape)
                            #pooled_val = K.max(xm, axis=(2, 3))
                            #outputs.append(pooled_val)

            elif self.dim_ordering == 'tf':
                x = K.cast(x, 'int32')
                y = K.cast(y, 'int32')
                z = K.cast(z, 'int32')
                w = K.cast(w, 'int32')
                h = K.cast(h, 'int32')
                d = K.cast(d, 'int32')

                #rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
                img = img[:, x:x+w, y:y+h, z:z+d, :]

                #TensorArr = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)

                #for i in range(self.pool_size):
                #    resized_list.append(tf.image.resize_images(TensorArr.unstack(img).read(i), [self.pool_size, self.pool_size]))
                #stack_img = tf.stack(resized_list, axis=3)

                img_ch=[]
                for ch in range(self.nb_channels):
                    img_ch.append(img[:,:,:,:,ch])
                    img_ch[ch] = tf.image.resize_images(img_ch[ch], [self.pool_size, self.pool_size])
                img = tf.stack(img_ch, axis=4)
                img = K.permute_dimensions(img, (0, 2, 3, 1, 4))

                img_ch=[]
                for ch in range(self.nb_channels):
                    img_ch.append(img[:,:,:,:,ch])
                    img_ch[ch] = tf.image.resize_images(img_ch[ch], [self.pool_size, self.pool_size])
                img = tf.stack(img_ch, axis=4)
                img = K.permute_dimensions(img, (0, 3, 1, 2, 4))

                #for i in tf.unstack(img, num=self.pool_size, axis=3):
                #    resized_list.append(tf.image.resize_images(i, [self.pool_size, self.pool_size]))
                #stack_img = tf.stack(resized_list, axis=3)

                #img = stack_img
                #resized_list = []
                #for i in tf.unstack(img, num=self.pool_size, axis=2):
                #    resized_list.append(tf.image.resize_images(i, [self.pool_size, self.pool_size]))
                #rs = tf.stack(resized_list, axis=2)
                #rs = img[:, x:x+self.pool_size, y:y+self.pool_size, z:z+self.pool_size, :]

                outputs.append(img)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.pool_size, self.nb_channels))

        if self.dim_ordering == 'th':
            final_output = K.permute_dimensions(final_output, (0, 1, 5, 2, 3, 4))
        else:
            final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4, 5))

        return final_output
