"""Contains different loss methods."""
import tensorflow as tf
from src.lib.neptune import get_params

params = get_params()


def get_cross_entropy(masks, logits, weighted=True):
    """Get the correct cross entropy loss, based on number of classes."""
    if params.nbr_of_classes == 1:
        loss_per_logit = tf.nn.sigmoid_cross_entropy_with_logits(labels=masks, logits=logits)
        if weighted and params.salt_vs_not_ratio:
            weight = masks * (1 - 2 * params.salt_vs_not_ratio) + params.salt_vs_not_ratio
            loss_per_logit = tf.multinomial(weight, loss_per_logit)

        return tf.reduce_mean(loss_per_logit)

    if params.nbr_of_classes == 2:
        raise NotImplemented


def discrete_iou(binary_mask, prediction_mask):
    """Discrete iou, can be used as additive loss."""
    if params.nbr_of_classes == 1:
        bin_cast = tf.cast(binary_mask, tf.float32)
        mx = tf.multiply(bin_cast, prediction_mask)
        den = bin_cast + prediction_mask - mx + 0.1  # For stability
        per_pixel = tf.divide(mx, den)
        return -tf.log(tf.reduce_mean(per_pixel) + 1e-8)  # For stability

    if params.nbr_of_classes == 2:
        raise NotImplemented


def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient.

    For comparing the similarity of two batch of data, usually be used for
    binary image segmentation i.e. labels are binary. The coefficient between
    0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background),
              dice = ```smooth/(small_value + smooth)``, then if smooth is very
              small, dice close to 0 (even the image values lower than the threshold),
              so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return 1 - dice
