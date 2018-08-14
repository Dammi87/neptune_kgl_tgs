import tensorflow as tf
from src.trainer.arch.upsampling import SubpixelConv2D, deconv_bilinear, unpool_2d, artifact_upsampling
from tensorflow.python.estimator.model_fn import ModeKeys as Modes


def up_sample(x, y, y_max_pool_idx, params, filters=None):
    """Create an upsampling stack, using the desired upsampling technique."""
    upsample_type = params.unet_upsampling_type
    if filters is None:
        filters = y.get_shape().as_list()[-1]
    # Upsample previous features
    with tf.variable_scope(upsample_type):
        if upsample_type == 'subpixel':
            x = SubpixelConv2D(filters=filters,
                               activation=None,
                               kernel_size=(3, 3),
                               r=2,
                               dtype=tf.float32)(x)

        elif upsample_type == 'bilinear':
            x = deconv_bilinear(filters,
                                kernel_size=(3, 3),
                                padding='SAME',
                                strides=(2, 2),
                                activation=None)(x)

        elif upsample_type == 'artifact_upsampling':
            n_first = int(filters * 2)
            x = artifact_upsampling(x, n_first, filters)

        elif upsample_type == 'conv_transpose':
            x = tf.layers.Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          padding='SAME',
                                          strides=(2, 2),
                                          activation=None)(x)

        elif upsample_type == 'upsample2D':
            shape = x.get_shape().as_list()
            set_height = 2 * shape[1]
            set_width = 2 * shape[2]
            x = tf.image.resize_nearest_neighbor(x, (set_height, set_width))

    # Concatenate
    return tf.concat([y, x], axis=-1)


def conv_block(x, filters, activation, is_training, params, dropout=0.0):
    y = tf.layers.Conv2D(filters, (3, 3), padding='SAME', activation=activation)(x)
    y = tf.layers.batch_normalization(y, training=is_training) if params.batchnorm else y
    y = tf.layers.dropout(y, rate=dropout, training=is_training) if dropout else y
    y = tf.layers.Conv2D(filters, (3, 3), padding='SAME', activation=activation)(y)
    y = tf.layers.batch_normalization(y, training=is_training) if params.batchnorm else y
    return tf.concat([x, y], axis=-1) if params.unet_residual else y


def down_block(x, filters, activation, is_training, params):
    y = conv_block(x, filters, activation, is_training, params)
    if params.unet_upsampling_type == 'spatial_unpool':
        x, y_max_pool_idx = tf.nn.max_pool_with_argmax(y, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
    else:
        x = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(y)
        y_max_pool_idx = None

    return x, y_max_pool_idx, y


def up_block(x, y, y_max_pool_idx, activation, is_training, params):
    x = up_sample(x, y, y_max_pool_idx, params)
    x = tf.layers.dropout(x, rate=params.dropout, training=is_training) if params.dropout else x
    # The skipt filter number control the number of out filter
    filters = y.get_shape().as_list()[-1]
    return conv_block(x, filters=filters, activation=activation, is_training=is_training, params=params)


def network(x, mode, params):
    # Encoder
    Y, Y_MAX_POOL = ([], [])
    is_training = mode == Modes.TRAIN
    with tf.variable_scope('Encoder'):
        for i in range(params.unet_depth):
            with tf.variable_scope('Stack_%d' % i):
                n_filters = params.unet_start_filters * 2 ** i
                x, y_max_pool_idx, y = down_block(x,
                                                  n_filters,
                                                  activation=tf.nn.relu,
                                                  is_training=is_training,
                                                  params=params)
                Y.append(y)
                Y_MAX_POOL.append(y_max_pool_idx)
                # print('Stack_%d' % i, x.shape)

    # Add a dropout in the middle
    with tf.variable_scope('Middle'):
        filters = params.unet_start_filters * 2 ** (i + 1)
        x = conv_block(x,
                       filters=filters,
                       activation=tf.nn.relu,
                       is_training=is_training,
                       params=params,
                       dropout=params.dropout)
    # Decoder
    R_Y, R_Y_MAX_POOL = (reversed(Y), reversed(Y_MAX_POOL))

    with tf.variable_scope('Decoder'):
        for i, (y, y_idx) in enumerate(zip(R_Y, R_Y_MAX_POOL)):
            with tf.variable_scope('Stack_%d' % i):
                x = up_block(x,
                             y,
                             y_idx,
                             activation=tf.nn.relu,
                             is_training=is_training,
                             params=params)
                # print('Stack_%d' % i, x.shape)

        with tf.variable_scope('Output'):
            x = tf.layers.dropout(x, rate=params.dropout, training=is_training) if params.dropout else x
            return tf.layers.Conv2D(params.nbr_of_classes, (1, 1), padding='same', activation=None)(x)
"""
if __name__ == "__main__":
    from src.lib.neptune import get_params
    params = get_params()
    x = tf.placeholder(shape=(1, 128, 128, 1), dtype=tf.float32)
    y = network(x, 'TRAIN', params)

    graph = tf.Graph()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir='/hdd/datasets/TGS/unet', graph=sess.graph)
        writer.flush()
"""