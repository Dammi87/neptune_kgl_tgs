"""Initializes network."""
import tensorflow as tf
from src.trainer.arch.upsampling import SubpixelConv2D, deconv_bilinear, unpool_2d, artifact_upsampling


def get_vgg_16_header(x, params):
    """If a header is required, set it, also change start_filters."""
    # If vgg16 checkpoint is present, then the filters
    # need to be set accordingly. If no checkpoint, simply return
    start_filters = params.start_filters
    if params.vgg16_chkp is not None:
        start_filters = 64

    return x, start_filters


def upsample(x, idx=None, upsample_type='subpixel', n_filters=None, activation=None, kernel_size=3):
    """Return the upsample method, all methods should accept x, and idx."""
    # Make sure upsample type exists
    defined = ['subpixel', 'bilinear', 'spatial_unpool', 'artifact_upsampling']
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
        return unpool_2d(x, idx)

    elif upsample_type == 'artifact_upsampling':
        return artifact_upsampling(x)


def up_sampling_stack(in_x, bypassed, bypass_idx, depth, filters, params):
    """Create an upsampling stack, using the desired upsampling technique."""
    upsample_type = params.vgg_16_unet_upsampling_type
    x_up = tf.layers.Conv2D(filters, (3, 3), padding='same', activation=tf.nn.relu)(in_x)
    # Upsample previous features
    if upsample_type == 'subpixel':
        x_up = SubpixelConv2D(filters=filters,
                              activation=None,
                              kernel_size=(4, 4),
                              r=2,
                              dtype=tf.float32)(x_up)

    elif upsample_type == 'bilinear':
        x_up = deconv_bilinear(filters,
                               kernel_size=(4, 4),
                               padding='SAME',
                               strides=(2, 2),
                               activation=None)(x_up)

    elif upsample_type == 'spatial_unpool':
        x_up = unpool_2d(x_up, bypass_idx)

    # Concatenate
    x = tf.concat([bypassed, x_up], axis=-1)

    # Apply rest of convolution layers
    for _ in range(depth - 1):
        x = tf.layers.Conv2D(filters, (3, 3), padding='same', activation=tf.nn.relu)(x)
    return x


def get_bypass(bypass_x, params):
    """Create the by-pass bridge based on unsampling."""
    if params.vgg_16_unet_upsampling_type == 'spatial_unpool':
        output_x, idx = tf.nn.max_pool_with_argmax(bypass_x, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')
    else:
        output_x = tf.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(bypass_x)
        idx = None

    return output_x, bypass_x, idx


def vgg16(x, mode, params):
    """Create the VGG16 header."""
    # Create a list to put in all the bypass features
    x, start_filters = get_vgg_16_header(x, params)

    bypass = []
    bypass_idx = []

    # Block 1
    with tf.variable_scope('vgg_16/conv1'):
        x = tf.layers.Conv2D(start_filters, (3, 3), activation=tf.nn.relu, padding='same', name='conv1_1')(x)
        x = tf.layers.Conv2D(start_filters, (3, 3), activation=tf.nn.relu, padding='same', name='conv1_2')(x)
        x, bypass_x, indices = get_bypass(x, params)
        bypass.append(bypass_x)
        bypass_idx.append(indices)

        if params.vgg_16_stack_depth == 1:
            return x, bypass, bypass_idx

    # Block 2
    with tf.variable_scope('vgg_16/conv2'):
        x = tf.layers.Conv2D(start_filters * 2, (3, 3), activation=tf.nn.relu, padding='same', name='conv2_1')(x)
        x = tf.layers.Conv2D(start_filters * 2, (3, 3), activation=tf.nn.relu, padding='same', name='conv2_2')(x)
        x, bypass_x, indices = get_bypass(x, params)
        bypass.append(bypass_x)
        bypass_idx.append(indices)

        if params.vgg_16_stack_depth == 2:
            return x, bypass, bypass_idx

    # Block 3
    with tf.variable_scope('vgg_16/conv3'):
        x = tf.layers.Conv2D(start_filters * 4, (3, 3), activation=tf.nn.relu, padding='same', name='conv3_1')(x)
        x = tf.layers.Conv2D(start_filters * 4, (3, 3), activation=tf.nn.relu, padding='same', name='conv3_2')(x)
        x = tf.layers.Conv2D(start_filters * 4, (3, 3), activation=tf.nn.relu, padding='same', name='conv3_3')(x)
        x, bypass_x, indices = get_bypass(x, params)
        bypass.append(bypass_x)
        bypass_idx.append(indices)

        if params.vgg_16_stack_depth == 3:
            return x, bypass, bypass_idx

    # Block 4
    with tf.variable_scope('vgg_16/conv4'):
        x = tf.layers.Conv2D(start_filters * 8, (3, 3), activation=tf.nn.relu, padding='same', name='conv4_1')(x)
        x = tf.layers.Conv2D(start_filters * 8, (3, 3), activation=tf.nn.relu, padding='same', name='conv4_2')(x)
        x = tf.layers.Conv2D(start_filters * 8, (3, 3), activation=tf.nn.relu, padding='same', name='conv4_3')(x)
        x, bypass_x, indices = get_bypass(x, params)
        bypass.append(bypass_x)
        bypass_idx.append(indices)

        if params.vgg_16_stack_depth == 4:
            return x, bypass, bypass_idx

    # Block 5
    with tf.variable_scope('vgg_16/conv5'):
        x = tf.layers.Conv2D(start_filters * 8, (3, 3), activation=tf.nn.relu, padding='same', name='conv5_1')(x)
        x = tf.layers.Conv2D(start_filters * 8, (3, 3), activation=tf.nn.relu, padding='same', name='conv5_2')(x)
        x = tf.layers.Conv2D(start_filters * 8, (3, 3), activation=tf.nn.relu, padding='same', name='conv5_3')(x)
        x, bypass_x, indices = get_bypass(x, params)
        bypass.append(bypass_x)
        bypass_idx.append(indices)

        if params.vgg_16_stack_depth == 5:
            return x, bypass, bypass_idx

    return x, bypass, bypass_idx


def segmenter_unet(x, bypass, bypass_idx, stack_depths, mode, params):
    """Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.

    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param nbr_of_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Reverse the bypasses
    r_bypass = reversed(bypass)
    r_bypass_idx = reversed(bypass_idx)
    r_stack_depths = reversed(stack_depths)

    with tf.variable_scope('Decoder'):
        start_filters = x.get_shape().as_list()[-1]
        with tf.variable_scope('Middle'):
            x = tf.layers.Conv2D(start_filters * 2, (3, 3), padding='same', activation=tf.nn.relu)(x)

        with tf.variable_scope('Upsampler'):
            for i, (iby, idx, d) in enumerate(zip(r_bypass, r_bypass_idx, r_stack_depths)):
                n_filter = iby.get_shape().as_list()[-1]
                # with tf.variable_scope('upsample_%d' % i):
                x = up_sampling_stack(x, iby, idx, d, n_filter, params)

            with tf.variable_scope('Output'):
                x = tf.layers.Conv2D(params.nbr_of_classes, (1, 1), padding='same', activation=None)(x)

    return x


def network(x, mode, params):
    """Network method."""
    x, bypass, bypass_idx = vgg16(x, mode, params)

    # The stack depth of a UNET
    stack_depths = [2, 2, 3, 3, 3]
    available_depth = stack_depths[:params.vgg_16_stack_depth]
    return segmenter_unet(x, bypass, bypass_idx, available_depth, mode, params)
