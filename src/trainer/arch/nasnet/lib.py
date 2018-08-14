"""Methods that are special to Nasnet."""
import tensorflow as tf


def resize_bil(x, target_size):
    return tf.image.resize_images(
        x,
        target_size,
        method=tf.image.ResizeMethod.BILINEAR)


def deconv_bilinear(filters,
                    target_size,
                    kernel_size=(3, 3),
                    padding='SAME',
                    activation=None):
    def wrapper(x):

        return tf.layers.Conv2D(
            filters,
            kernel_size,
            padding=padding,
            strides=(1, 1),
            activation=activation)(resize_bil(x, target_size))

    return wrapper


def deconv_transpose(filters,
                     strides=(2, 2),
                     kernel_size=(3, 3),
                     padding='SAME',
                     activation=None):
    def wrapper(x):

        return tf.layers.Conv2DTranspose(
            filters,
            kernel_size,
            padding=padding,
            strides=strides,
            activation=activation)(x)

    return wrapper


def upsample_resize_concat(x0, x1, depth=2, filters=100):
    x0_shape = x0.get_shape().as_list()[1:3]
    x1_shape = x1.get_shape().as_list()[1:3]

    # If shapes are not even, then resize
    if x0_shape != x1_shape:
        # Depending on who's bigger decides how to upsample
        x1 = deconv_bilinear(100, x0_shape)(x1)

    # Then merge
    x = tf.concat([x0, x1], axis=-1)

    # Apply subsequent convolutions
    for _ in range(depth):
        x = tf.layers.Conv2D(filters=100,
                             kernel_size=(3, 3),
                             padding='SAME',
                             strides=(1, 1),
                             activation=tf.nn.relu)(x)

    return x
