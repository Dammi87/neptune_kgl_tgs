"""Model function."""
import src.trainer.arch as arch
import src.trainer.lib.loss as lib_loss
import src.trainer.lib.metric as lib_metric
import src.trainer.lib.optimizer as lib_optimizer
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes


def _model_fn(features, labels, mode, params):
    """Create the model function.

    This will handle all the different processes needed when using an Estimator.
    The estimator will change "modes" using the mode flag, and depending on that
    different outputs are provided.

    Parameters
    ----------
    features : Tensor
        4D Tensor where the first dimension is the batch size, then height, width
        and channels
    labels : Dict {'mask': Tensor}
    mode : tensorflow.python.estimator.model_fn.ModeKeys
        Class that contains the current mode
    params : class
        Contains all the hyper parameters that are available to the model. These can be different
        depending on which architecture (model type) is in use

    Returns
    -------
    tf.estimator.EstimatorSpec
        The requested estimator spec
    """
    feature_input = features['img']

    # Logits Layer
    logits = arch.get_network()(feature_input, mode, params)

    # If this is a prediction or evaluation mode, then we return
    # the class probabilities and the guessed pixel class
    if mode in (Modes.TRAIN, Modes.EVAL, Modes.PREDICT):
        probabilities = tf.nn.sigmoid(logits, name='softmax_logits')
        prediction_50 = tf.cast(probabilities > 0.5, tf.int32)

    # During training and evaluation, we calculate the loss
    if mode in (Modes.TRAIN, Modes.EVAL):
        global_step = tf.train.get_or_create_global_step()

        # Calculate cross entropy loss
        # cross_entropy_loss = lib_loss.get_cross_entropy(labels['mask'], logits)
        cross_entropy_loss = tf.losses.sigmoid_cross_entropy(labels['mask'], logits)

        # Add any regularization losses
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # Get dice loss
        dice_loss = lib_loss.dice_coe(probabilities, labels['mask'])

        # Sum of losses
        total_loss = 1.0 * cross_entropy_loss + 0.001 * reg_loss + 2.0 * dice_loss

        # tf.summary.scalar('iou_loss', iou_loss)
        tf.summary.image('img', feature_input)
        tf.summary.image('mask', lib_metric.probabilities_to_image(labels['mask']))
        tf.summary.image('pred_argmax', tf.cast(prediction_50 * 255, tf.uint8))
        tf.summary.image('pred_prob', probabilities)

    # When predicting (running inference only, during serving for example) we
    # need to return the output as a dictionary.
    if mode == Modes.PREDICT:
        predictions = {
            'probabilities': probabilities,
            'resized_probabilities': tf.image.resize_images(probabilities, [params.original_size] * 2)
        }
        export_outputs = {'prediction': tf.estimator.export.PredictOutput(predictions)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    # In training (not evaluation) we perform backprop
    if mode == Modes.TRAIN:
        train_op = lib_optimizer.create_optimizer(total_loss, global_step)
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)

    # If evaluating only, we perform evaluation operations
    if mode == Modes.EVAL:

        # Evaluation metrics
        iou_coe_50 = lib_metric.iou_coe(probabilities, labels['mask'])
        dice_hard_coe = lib_metric.iou(probabilities, labels['mask'])

        # Accuracy operations (Will be sent to neptune)
        eval_metric_ops = {
            'iou_coe_50': tf. metrics.mean(iou_coe_50),
            'dice_hard_coe': tf. metrics.mean(dice_hard_coe)
        }

        return tf.estimator.EstimatorSpec(
            mode,
            loss=total_loss,
            eval_metric_ops=eval_metric_ops)
