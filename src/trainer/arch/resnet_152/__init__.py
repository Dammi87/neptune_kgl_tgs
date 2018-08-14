import nets
import tensorflow.contrib.slim as slim
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
from src.trainer.arch.upsampling import SubpixelConv2D, deconv_bilinear, artifact_upsampling

def up_sampling_stack(in_x, bypassed, depth, filters, params):
    """Create an upsampling stack, using the desired upsampling technique."""
    upsample_type = params.resnet_upsampling_type
    # Upsample previous features
    if upsample_type == 'subpixel':
        x_up = tf.layers.Conv2D(filters, (3, 3), padding='same', activation=tf.nn.relu)(in_x)
        x_up = SubpixelConv2D(filters=filters,
                              activation=None,
                              kernel_size=(4, 4),
                              r=2,
                              dtype=tf.float32)(x_up)

    elif upsample_type == 'bilinear':
        x_up = tf.layers.Conv2D(filters, (3, 3), padding='same', activation=tf.nn.relu)(in_x)
        x_up = deconv_bilinear(filters,
                               kernel_size=(4, 4),
                               padding='SAME',
                               strides=(2, 2),
                               activation=None)(x_up)
    elif upsample_type == 'artifact_upsampling':
        x_up = artifact_upsampling(in_x, int(filters * 2), int(filters))

    else:
        raise NotImplemented

    # Concatenate
    x = tf.concat([bypassed, x_up], axis=-1)

    if params.dropout:
        tf.nn.dropout(x, params.dropout)

    # Apply rest of convolution layers
    for _ in range(depth - 1):
        x = tf.layers.Conv2D(filters, (3, 3), padding='same', activation=tf.nn.relu)(x)

    return x


def get_res_header(x, params):
    """If a header is required, set it, also change start_filters."""

    # If coloring layer is desired, then it's added to the header
    if params.resnet_add_coloring_layer:
        def very_leaky_relu(x):
            return tf.nn.leaky_relu(x, 1 / 5.5)

        # Creating a header infront of the VGG16 color network
        with tf.variable_scope('Coloring_layer'):
            x = tf.layers.Conv2D(10, (3, 3), padding='same', activation=very_leaky_relu)(x)
            x = tf.layers.Conv2D(3, (3, 3), padding='same', activation=very_leaky_relu)(x)
            tf.summary.image('colored_image', x)
    elif params.resnet_stack_input_channels:
        x = tf.concat([x] * 3, axis=-1)

    return x

def network(x, mode, params):
    start_filters = 2048
    num_filters = 32
    x = get_res_header(x, params)
    with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
        logit, le_dict = nets.resnet_v2.resnet_v2_152(x,
                                                      num_classes=None,
                                                      is_training=mode == Modes.TRAIN,
                                                      global_pool=True,
                                                      output_stride=None,
                                                      spatial_squeeze=True,
                                                      reuse=None,
                                                      scope='resnet_v2_152')

    block_0 = le_dict['resnet_v2_152/conv1']   # (?, 64, 64, 64)
    block_1 = le_dict['resnet_v2_152/block1']  # (?, 16, 16, 256)
    block_2 = le_dict['resnet_v2_152/block2']  # (?, 8, 8, 512)
    block_3 = le_dict['resnet_v2_152/block3']  # (?, 4, 4, 1024)
    block_4 = le_dict['resnet_v2_152/block4']  # (?, 4, 4, 2048)

    # Make block 4 through pool, then we can start upsampling
    # block_4 = tf.layers.MaxPooling2D((2, 2), strides=(2, 2))(block_4)  # (?, 2, 2, 2048)

    with tf.variable_scope('Decoder'):
        # point_4 = up_sampling_stack(block_4, block_3, 2, start_filters / 2, params)  # (?, 4, 4, ?)
        point_3 = up_sampling_stack(tf.concat([block_3, block_4], axis=-1), block_2, 2, num_filters * 8, params)  # (?, 8, 8, ?)
        point_2 = up_sampling_stack(point_3, block_1, 2, num_filters * 8, params)  # (?, 16, 16, ?)

        # Upsample point_2 such that it's 32
        point_2 = artifact_upsampling(point_2, int(num_filters * 8), int(num_filters * 4))  # (?, 32, 32, ?)

        point_1 = up_sampling_stack(point_2, block_0, 2, num_filters * 4, params)  # (?, 64, 64, ?)

        # Final upsampling
        point_0 = artifact_upsampling(point_1, int(num_filters * 8), int(num_filters * 4))  # (?, 128, 128, ?)

        # Output
        return tf.layers.Conv2D(params.nbr_of_classes, (1, 1), padding='same', activation=None)(point_0)
