"""Initialize input pipe."""
import tensorflow as tf
from src.lib.neptune import get_params
from .image_converter import get_augmenter
from .dataset import create_generator
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes


def get_input_fn(is_test=False):
    """Create a generator for datasets.

    This method will create input_fn that can be used to feed Tensorflow with
    images for training and inference. It uses the parameters specified in the
    ./config/neptune.yaml file. If this is run at test time, then train and validation
    input functions are not created, and vice-vesa

    Returns
    -------
    dict
        A dictionary which contains input_fn for the required task.
        keys:
            'train', 'valid', 'test'
    """
    # Experiment parameters
    params = get_params()

    # Get augmenter to apply boot strapping methods
    augmenter = get_augmenter()

    # Get generator methods
    generators, img_shape = create_generator(augmenter, is_test=is_test)

    def create_dataset(generator, augmenter=augmenter, params=params, is_train=True):
        types = (tf.uint8, tf.uint8)
        shapes = (tf.TensorShape(img_shape), tf.TensorShape(img_shape))

        # Initialize dataset
        dataset = tf.data.Dataset.from_generator(generator, types, shapes)

        # Manually set batch_size
        dataset._batch_size = ops.convert_to_tensor(params.batch_size, dtype=dtypes.int64, name="batch_size")

        # No need for this, generator does this
        dataset = dataset.prefetch(params.buffer_size)
        dataset = dataset.shuffle(params.buffer_size)
        dataset = dataset.batch(params.batch_size)

        # After batch has been collected, apply transformations
        if is_train:
            dataset = dataset.map(augmenter.apply, num_parallel_calls=params.num_parallel_calls)
        else:
            # The normalization is rather lightweight, no threads are allocated for it.
            dataset = dataset.map(augmenter.apply_normalization_tfunc)

        # Make into an iterator
        iterator = dataset.make_one_shot_iterator()
        image, mask = iterator.get_next()

        # Set shape manually since opencv is used in background, it returns unknown shape
        shape = [-1] + img_shape
        mask = tf.reshape(mask, shape)
        # When channels are augmented, the image becomes (N, W, H, 3) instead of (N, W, H, 1)
        if params.aug_channel_augmenter:
            shape[-1] = 3
        image = tf.reshape(image, shape)

        # Return two items
        features = {'img': image}
        labels = {'mask': mask}

        return features, labels

    input_fn = {}

    # If testing, then simply return a numpy generator
    if is_test:
        input_fn['test'] = generators
    else:
        def train_input_fn():
            return create_dataset(generators[0], is_train=True)

        def eval_input_fn():
            return create_dataset(generators[1], is_train=False)

        input_fn['train'] = train_input_fn
        input_fn['valid'] = eval_input_fn

    return input_fn
