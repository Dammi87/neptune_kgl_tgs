"""Contains all optimizers."""
import tensorflow as tf
from src.lib.neptune import get_params
from src.lib.tf_ops import AdjustLearningRate, AdjustLayerFreeze
import logging

lr_adjuster = AdjustLearningRate()
lf_adjuster = AdjustLayerFreeze()
logger = logging.getLogger('tensorflow')
params = get_params()


def get():
    lr = lr_adjuster.get_lr()
    """Return the current optimize."""
    if params.optimizer == 'adam':
        return tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=params.adam_b1,
            beta2=params.adam_b2,
        )
    elif params.optimizer == 'rmsprop':
        return tf.train.RMSPropOptimizer(
            learning_rate=lr,
            decay=params.rms_decay,
            momentum=params.rms_momentum)
    else:
        raise Exception('Unknown optimizer.')


'''
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
    elif params.network_type == 'nasnet':
        if params.nasnet_freeze:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')
    elif params.network_type == 'resnet_152':
        if params.resnet_freeze:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')

    return var_list
'''


def parse_collections(collections):
    """Parse the lsit of collections given."""
    if collections is None:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    var_list = []
    for c in collections:
        var_list += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=c)
    return var_list


def get_var_list():
    """Fetch the varlist to use for training."""
    collections, activated = lf_adjuster.get_collection_list()
    if collections is None:
        var_lists = [parse_collections(collections)]  # Only one set of variable lists
        activated = [activated]
    else:
        var_lists = []
        for c in collections:
            var_lists.append(parse_collections(collections[c]))

    return var_lists, activated


def create_optimizer(total_loss, global_step):
    """Create an optimizer according to some rule."""
    optimizer = get()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        var_list, activated = get_var_list()
        # For each var_list, create a training operation
        train_ops = []
        for i_vars in var_list:
            train_ops.append(optimizer.minimize(total_loss,
                                                global_step=global_step,
                                                var_list=i_vars))
    return train_ops, activated
