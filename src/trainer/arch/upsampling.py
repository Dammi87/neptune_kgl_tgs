"""Includes methods common to the different architectures."""
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.contrib.framework.python.ops import add_arg_scope


class SubpixelConv2D(tf.layers.Conv2D):
    """https://github.com/tetrachrome/subpixel."""

    def __init__(self,
                 filters,
                 kernel_size,
                 r,
                 padding='same',
                 data_format='channels_last',
                 strides=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """Init subpixel class."""
        super(SubpixelConv2D, self).__init__(
            filters=r * r * filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.r = r

    def _phase_shift(self, inp):
        """Phase shift."""
        r = self.r
        bsize, a, b, c = inp.get_shape().as_list()
        bsize = -1  # Handling Dimension(None) type for undefined batch dim
        # bsize, a, b, c/(r*r), r, r
        x = tf.reshape(inp, [bsize, a, b, int(c / (r * r)), r, r])
        x = tf.transpose(x, (0, 1, 2, 5, 4, 3))  # bsize, a, b, r, r, c/(r*r)
        # Keras backend does not support tf.split, so in future versions this could be nicer
        x = [x[:, i, :, :, :, :]
             for i in range(a)]  # a, [bsize, b, r, r, c/(r*r)
        x = tf.concat(x, 2)  # bsize, b, a*r, r, c/(r*r)
        x = [x[:, i, :, :, :] for i in range(b)]  # b, [bsize, r, r, c/(r*r)
        x = tf.concat(x, 2)  # bsize, a*r, b*r, c/(r*r)
        return x

    def call(self, inputs):
        """call."""
        return self._phase_shift(super(SubpixelConv2D, self).call(inputs))

    def compute_output_shape(self, input_shape):
        """compute_output_shape."""
        unshifted = super(
            SubpixelConv2D, self).compute_output_shape(input_shape)
        return (unshifted[0], self.r * unshifted[1], self.r * unshifted[2], unshifted[3] / (self.r * self.r))

    def get_config(self):
        """get_config."""
        config = super(tf.layers.Conv2D, self).get_config()
        config.pop('rank')
        config.pop('dilation_rate')
        config['filters'] /= self.r * self.r
        config['r'] = self.r
        return config


def deconv_bilinear(filters,
                    kernel_size=(2, 2),
                    padding='SAME',
                    strides=(2, 2),
                    reuse=None,
                    name=None,
                    activation=None):
    """Resize input using nearest neighbor then apply convolution.

    Parameters
    ----------
    inputs : tensor
        The input tensor to this operation
    filters : int
        Number of filters of the conv operation
    kernel_size : tuple, optional
        The kernel size to use
    padding : str, optional
        Padding strategy
    strides : tuple, optional
        How many steps the resize operation should take, the strides
        control how big the output tensor is
    reuse : None, optional
        Variable to control if the generated weights should be reused from somewhere else
    name : None, optional
        Desired name of this op
    activation : None, optional
        Desired activation function

    Returns
    -------
    tensor
        The output tensor that has been resized and convolved

    """
    def wrapper(x):
        shape = x.get_shape().as_list()
        height = shape[1] * strides[0]
        width = shape[2] * strides[1]
        resized = tf.image.resize_images(
            x,
            [height, width],
            method=tf.image.ResizeMethod.BILINEAR)

        return tf.layers.Conv2D(
            filters,
            (3, 3),
            padding=padding,
            strides=(1, 1),
            activation=activation)(resized)

    return wrapper


@add_arg_scope
def unpool_2d(pool,
              ind,
              stride=[1, 2, 2, 1],
              scope='unpool_2d'):
    """Adds a 2D unpooling op.
    https://arxiv.org/abs/1505.04366
    Unpooling layer after max_pool_with_argmax.
         Args:
             pool:        max pooled output tensor
             ind:         argmax indices
             stride:      stride is the same as for the pool
         Return:
             unpool:    unpooling tensor
    """
    with tf.variable_scope(scope):
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] *
                        stride[1], input_shape[2] * stride[2], input_shape[3]]

        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                 shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(
            flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] *
                            stride[1], set_input_shape[2] * stride[2], set_input_shape[3]]
        ret.set_shape(set_output_shape)
        return ret


def artifact_upsampling(x, n_first, n_out):
    """Reduce artifacts by using 3 and then 4 transpose."""
    x = tf.layers.Conv2D(n_first, (3, 3), strides=(1, 1), padding='SAME', activation=tf.nn.relu)(x)
    return tf.layers.Conv2DTranspose(n_out, kernel_size=(4, 4), padding='SAME', strides=(2, 2), activation=tf.nn.relu)(x)


def upsample(x, idx=None, upsample_type='subpixel', n_filters=None, activation=None, kernel_size=3):
    """Return the upsample method, all methods should accept x, and idx."""
    # Make sure upsample type exists
    defined = ['subpixel', 'bilinear', 'spatial_unpool']
    if upsample_type not in defined:
        raise Exception('Upsample method {} not recognized, available {}'.format(upsample_type, defined))

    if upsample_type == 'subpixel':
        return SubpixelConv2D(filters=n_filters,
                              activation=activation,
                              kernel_size=kernel_size,
                              r=2,
                              dtype=tf.float32)(x)

    elif upsample_type == 'bilinear':
        return deconv_bilinear(n_filters,
                               kernel_size=(kernel_size, kernel_size),
                               padding='SAME',
                               strides=(2, 2),
                               activation=activation)(x)

    elif upsample_type == 'spatial_unpool':
        return unpool_2d(x, y)
