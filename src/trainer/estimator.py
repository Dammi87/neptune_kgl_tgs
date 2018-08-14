"""Contains all estimator methods."""
import os
import tensorflow as tf
from src.trainer.model import _model_fn
from src.input_pipe import get_input_fn
from src.lib.neptune import get_params, NeptuneCollector
from src.lib.tf_ops import EarlyStopping, GlobalStep

params = get_params()

# Global step object, to share the current global step between classes
global_step = GlobalStep()


def get_vgg16_variable_change():
    """VGG 16 variable conversions."""

    stack_1 = {
        'vgg_16/conv1/conv1_1/kernel': 'vgg_16/conv1/conv1_1/weights',
        'vgg_16/conv1/conv1_1/bias': 'vgg_16/conv1/conv1_1/biases',
        'vgg_16/conv1/conv1_2/kernel': 'vgg_16/conv1/conv1_2/weights',
        'vgg_16/conv1/conv1_2/bias': 'vgg_16/conv1/conv1_2/biases'
    }

    stack_2 = {
        'vgg_16/conv2/conv2_1/kernel': 'vgg_16/conv2/conv2_1/weights',
        'vgg_16/conv2/conv2_1/bias': 'vgg_16/conv2/conv2_1/biases',
        'vgg_16/conv2/conv2_2/kernel': 'vgg_16/conv2/conv2_2/weights',
        'vgg_16/conv2/conv2_2/bias': 'vgg_16/conv2/conv2_2/biases'
    }

    stack_3 = {
        'vgg_16/conv3/conv3_1/kernel': 'vgg_16/conv3/conv3_1/weights',
        'vgg_16/conv3/conv3_1/bias': 'vgg_16/conv3/conv3_1/biases',
        'vgg_16/conv3/conv3_2/kernel': 'vgg_16/conv3/conv3_2/weights',
        'vgg_16/conv3/conv3_2/bias': 'vgg_16/conv3/conv3_2/biases',
        'vgg_16/conv3/conv3_3/kernel': 'vgg_16/conv3/conv3_3/weights',
        'vgg_16/conv3/conv3_3/bias': 'vgg_16/conv3/conv3_3/biases'
    }

    stack_4 = {
        'vgg_16/conv4/conv4_1/kernel': 'vgg_16/conv4/conv4_1/weights',
        'vgg_16/conv4/conv4_1/bias': 'vgg_16/conv4/conv4_1/biases',
        'vgg_16/conv4/conv4_2/kernel': 'vgg_16/conv4/conv4_2/weights',
        'vgg_16/conv4/conv4_2/bias': 'vgg_16/conv4/conv4_2/biases',
        'vgg_16/conv4/conv4_3/kernel': 'vgg_16/conv4/conv4_3/weights',
        'vgg_16/conv4/conv4_3/bias': 'vgg_16/conv4/conv4_3/biases'
    }

    stack_5 = {
        'vgg_16/conv5/conv5_1/kernel': 'vgg_16/conv5/conv5_1/weights',
        'vgg_16/conv5/conv5_1/bias': 'vgg_16/conv5/conv5_1/biases',
        'vgg_16/conv5/conv5_2/kernel': 'vgg_16/conv5/conv5_2/weights',
        'vgg_16/conv5/conv5_2/bias': 'vgg_16/conv5/conv5_2/biases',
        'vgg_16/conv5/conv5_3/kernel': 'vgg_16/conv5/conv5_3/weights',
        'vgg_16/conv5/conv5_3/bias': 'vgg_16/conv5/conv5_3/biases'
    }
    stacks = [stack_1, stack_2, stack_3, stack_4, stack_5]

    return {k: v for d in stacks[:params.vgg_16_stack_depth] for k, v in d.items()}


def build_estimator(run_config):
    """Build the estimator using the desired model type.

    Parameters
    ----------
    run_config : RunConfig
        RunConfig
    warm_start_from : None, optional
        Checkpoint to warm start from

    Returns
    -------
    Estimator object
        A higher level API that simplifies training and evaluation of neural networks.
    """
    warm_start_from = None
    if params.warm_start:
        if params.vgg16_chkp is not None and params.network_type == 'vgg_16_unet':
            warm_start_from = tf.estimator.WarmStartSettings(
                ckpt_to_initialize_from=params.vgg16_chkp,
                var_name_to_prev_var_name=get_vgg16_variable_change(),
                vars_to_warm_start='vgg_16',
            )

        if params.nasnet_chkp is not None and params.network_type == 'nasnet':
            warm_start_from = tf.estimator.WarmStartSettings(
                ckpt_to_initialize_from=params.nasnet_chkp,
                vars_to_warm_start='^((?!Decoder).)*$'
            )

        if params.resnet_chkp is not None and params.network_type == 'resnet_152':
            warm_start_from = tf.estimator.WarmStartSettings(
                ckpt_to_initialize_from=params.resnet_chkp,
                vars_to_warm_start='^((?!Decoder).)*$'
            )

    return tf.estimator.Estimator(model_fn=_model_fn,
                                  model_dir=run_config.model_dir,
                                  config=run_config,
                                  params=params,
                                  warm_start_from=warm_start_from)


def serving_input_fn():
    """Input function to use when serving the model."""
    if params.aug_channel_augmenter:
        incoming = tf.placeholder(tf.float32, shape=(None, params.resize, params.resize, 3), name='input_image')
    else:
        incoming = tf.placeholder(tf.float32, shape=(None, params.resize, params.resize, 1), name='input_image')

    feature_input = {'img': incoming}

    return tf.estimator.export.ServingInputReceiver(feature_input, incoming)


def get_specs(params):
    """Get eval and train specs."""
    input_fn = get_input_fn()
    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn['train'],
        max_steps=params.train_steps
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_fn['valid'],
        steps=params.eval_steps,
        throttle_secs=params.eval_every_n_seconds,
        name='classifier_eval')

    return train_spec, eval_spec


def run(run_config, params):
    """Run a training and evaluate."""
    # Get the estimators an specs
    estimator = build_estimator(run_config)
    train_spec, eval_spec = get_specs(params)

    # And run the evaluation
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def run_manual(run_config, params):
    """Run a training in manual mode."""
    # Get the estimators an specs
    estimator = build_estimator(run_config)

    # Get the input functions
    input_fn = get_input_fn()

    # Neptune class for sending
    neptune_connection = NeptuneCollector()

    # Early stopping
    early_check = EarlyStopping(start_epoch=2, max_events=50, maximize=True)

    for i_epoch in range(params.train_epochs):
        estimator.train(input_fn=input_fn['train'])  # In the dataset loading, the input_fn only outputs one epoch

        if i_epoch < params.start_eval_at_epoch:
            continue

        # After each epoch, perform an evaluation step if required
        if i_epoch % params.eval_per_n_epoch == 0 or (i_epoch + 1) == params.train_epochs:
            metrics = estimator.evaluate(input_fn=input_fn['valid'])

            # Set the step
            global_step.current = metrics['global_step']

            # Send the metrics
            neptune_connection.send(metrics)

            # Add early stopping check
            early_check.add(metrics['iou_coe_50'])

            # Check if training should be stopped
            if early_check.should_stop():
                break

            # Check if this is a good model or not
            if early_check.is_better():
                number = os.path.basename(estimator.latest_checkpoint()).split('-')[-1]
                export_predict_model = os.path.join(run_config.model_dir, 'best_models', '%s' % number)
                estimator.export_savedmodel(export_predict_model, serving_input_fn)


def export(run_config, params, warm_start_from=None, checkpoint=None):
    """Export model."""
    # Export only if chief
    if run_config._is_chief:
        print("Master detected, exporting model..")
        if not checkpoint:
            export_predict_model = os.path.join(run_config.model_dir, 'saved_mdl')
        else:
            number = os.path.basename(checkpoint).split('-')[-1]
            export_predict_model = os.path.join(run_config.model_dir, 'saved_mdl_%s' % number)

        trained_estimator = build_estimator(run_config)
        trained_estimator.export_savedmodel(export_predict_model, serving_input_fn, checkpoint_path=checkpoint)
