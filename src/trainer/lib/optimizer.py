"""Contains all optimizers."""
import tensorflow as tf
from src.lib.neptune import get_params
from src.lib.tf_ops import GlobalStep
import logging

global_step = GlobalStep()
logger = logging.getLogger('tensorflow')
params = get_params()


def get():
    """Return the current optimize."""
    if params.optimizer == 'adam':
        return tf.train.AdamOptimizer(
            learning_rate=params.learning_rate,
            beta1=params.adam_b1,
            beta2=params.adam_b2,
        )
    elif params.optimizer == 'rmsprop':
        return tf.train.RMSPropOptimizer(
            learning_rate=params.learning_rate,
            decay=params.rms_decay,
            momentum=params.rms_momentum)
    else:
        raise Exception('Unknown optimizer.')


def get_var_list(global_step):
    """Return a var list per global step, can be used to turn on/off training.

    This is meant such that training can be scheduled in different way. For example
    training the encoder only for X training steps, then switch to both.

    Parameters
    ----------
    global_step : tf.Tensor
        The current global step of the training. Can be used to switch between
        different var lists.
    """
    # By default, train all
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # Define the rule set here
    if params.network_type == 'vgg_16_unet':
        if params.vgg16_chkp and params.vgg_16_freeze:
            # When the model is vgg16 AND has a coloring header, train only the
            # Decoder and Coloring layer for 10 epochs of the timesteps.
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')

        if params.vgg_16_add_coloring_layer:
            var_list += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Coloring_layer')

    elif params.network_type == 'nasnet':
        if params.nasnet_freeze:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')

        if params.nasnet_add_coloring_layer:
            var_list += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Coloring_layer')

    elif params.network_type == 'resnet_152':
        if params.resnet_freeze:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')

        if params.resnet_add_coloring_layer:
            var_list += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Coloring_layer')

    return var_list


def create_optimizer(total_loss, global_step):
    """Create an optimizer according to some rule."""
    optimizer = get()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        var_list = get_var_list(global_step)
        train_op = optimizer.minimize(total_loss,
                                      global_step=global_step,
                                      var_list=var_list)
    return train_op
