import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
from nets import nets_factory

from .lib import resize_bil, upsample_resize_concat

CELLS_TO_USE = ['Cell_17', 'Cell_11', 'Cell_13']


def get_nasnet_header(x, params):
    """If a header is required, set it, also change start_filters."""

    # If coloring layer is desired, then it's added to the header
    if params.nasnet_add_coloring_layer:
        def very_leaky_relu(x):
            return tf.nn.leaky_relu(x, 1 / 5.5)

        # Creating a header infront of the VGG16 color network
        with tf.variable_scope('Coloring_layer'):
            x = tf.layers.Conv2D(10, (3, 3), padding='same', activation=very_leaky_relu)(x)
            x = tf.layers.Conv2D(3, (3, 3), padding='same', activation=very_leaky_relu)(x)
            tf.summary.image('colored_image', x)
    elif params.nasnet_stack_input_channels:
        x = tf.concat([x] * 3, axis=-1)

    return x


nas_net_for_training = nets_factory.get_network_fn('nasnet_large',
                                                   num_classes=None,
                                                   is_training=True)
nas_net_for_eval = nets_factory.get_network_fn('nasnet_large',
                                               num_classes=None,
                                               is_training=False)


def network(x, mode, params):

    # Get header
    x = get_nasnet_header(x, params)

    _, cells = nas_net_for_eval(x)

    # Get the first downsample operation
    name = 'cell_stem_1/Relu:0'
    first_resize = tf.get_default_graph().get_tensor_by_name(name)

    # Fetch the rest of the cells
    down_sample = []
    for iCell in CELLS_TO_USE:
        down_sample.append(cells[iCell])

    # Now, create the merge points, if Cells are 3, merge points are 2
    with tf.variable_scope('Decoder'):
        with tf.variable_scope('Merge'):
            merge_points = [first_resize]
            for i in range(len(down_sample) - 1):
                with tf.variable_scope('Merge_Point_%d' % i):
                    if i == 0:
                        maybe_up_sample = down_sample[i]
                    else:
                        maybe_up_sample = merge_points[-1]
                    merge_with = down_sample[i + 1]
                    merged = upsample_resize_concat(merge_with, maybe_up_sample, depth=2, filters=100)

                    # Collect
                    merge_points.append(merged)

        # Now, each merge point get resized to 101
        resized = []
        with tf.variable_scope('Resize_and_Concat'):
            for i, iPoint in enumerate(merge_points):
                with tf.variable_scope('Resize_%d' % i):
                    resized.append(resize_bil(iPoint, [101, 101]))

            # Then, concat
            concat = tf.concat(resized, axis=-1)

        # Finally, output result
        return tf.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='SAME', activation=None)(concat)
