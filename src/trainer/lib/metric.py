"""Contains different methods for metrics."""
import tensorflow as tf
from src.lib.neptune import get_params


def iou(binary_mask, prediction_mask, t=0.5):
    """Return the iou for the whole batch, given a threshold.

    Parameters
    ----------
    binary_mask : tf.Tensor
        The mask label
    prediction_mask : tf.Tensor
        The softmax probabilities
    T : float, optional
        Threshold that determine if a prediction is true

    Returns
    -------
    TYPE
        Description
    """
    params = get_params()

    if params.nbr_of_classes == 1:
        predicted_class = prediction_mask >= t
    else:
        raise NotImplemented
    actual_class = binary_mask > 0.5

    intersection = tf.cast(tf.logical_and(predicted_class, actual_class), tf.float32)
    union = tf.cast(tf.logical_or(predicted_class, actual_class), tf.float32)

    return tf.divide(tf.reduce_sum(intersection), tf.reduce_sum(union))


def probabilities_to_image(probabilities, threshold=None):
    """Convert mask probabilities to a summerizable image."""
    channels = probabilities.get_shape().as_list()[-1]
    if channels == 1:
        classes = probabilities
    else:
        classes = tf.expand_dims(tf.unstack(probabilities, axis=-1)[0], 3)

    if threshold:
        return tf.cast(classes > threshold, tf.uint8) * 255
    else:

        if classes.dtype == tf.int32:
            classes = tf.cast(classes, tf.uint8) * 255
        return classes


def dice_hard_coe(output, target, threshold=0.5, axis=(1, 2, 3), smooth=1e-5):
    """Non-differentiable Sørensen–Dice coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation i.e. labels are binary.
    The coefficient between 0 to 1, 1 if totally match.

    Parameters
    -----------
    output : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        The target distribution, format the same with `output`.
    threshold : float
        The threshold value to be true.
    axis : tuple of integer
        All dimensions are reduced, default ``(1,2,3)``.
    smooth : float
        This small value will be added to the numerator and denominator, see ``dice_coe``.

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    output = tf.cast(output > threshold, dtype=tf.float32)
    target = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(output, target), axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice


def iou_coe(output, target, threshold=0.5, axis=(1, 2, 3), smooth=1e-5):
    """Non-differentiable Intersection over Union (IoU) for comparing the
    similarity of two batch of data, usually be used for evaluating binary image segmentation.
    The coefficient between 0 to 1, and 1 means totally match.

    Parameters
    -----------
    output : tensor
        A batch of distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        The target distribution, format the same with `output`.
    threshold : float
        The threshold value to be true.
    axis : tuple of integer
        All dimensions are reduced, default ``(1,2,3)``.
    smooth : float
        This small value will be added to the numerator and denominator, see ``dice_coe``.

    Notes
    ------
    - IoU cannot be used as training loss, people usually use dice coefficient for training, IoU and hard-dice for evaluating.

    """
    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    batch_iou = (inse + smooth) / (union + smooth)
    iou = tf.reduce_mean(batch_iou)
    return iou