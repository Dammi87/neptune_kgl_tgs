"""Executes training according to task."""
import os
import tensorflow as tf
from shutil import copyfile

from src.trainer import estimator
from src.lib.neptune import get_params, get_job_id


# Get current parameters
params = get_params()
job_id = get_job_id()

# IDEAS
# [X] Make a script to go through dataset, and show the ones with most error
# [ ] Find all masks with VERY straight lines in segmentation, make them not matter as much
# [X] Make a label smoother, i.e, make the labels exponentially decay around the edges.
# [ ] Make a scheduler class, allowing some criteria to control inputs, learning rate etc


def resume_training(model_dir):
    """Copy checkpoint to model_dir if it should be resumed from somewhere."""
    if not params.resume_training_from:
        return

    # Create directory
    os.makedirs(model_dir)

    # Files to be copied
    dir_path = os.path.dirname(params.resume_training_from)
    data_file = '%s.data-00000-of-00001' % params.resume_training_from
    index_file = '%s.index' % params.resume_training_from
    meta_file = '%s.meta' % params.resume_training_from
    chkp_file = os.path.basename(params.resume_training_from)

    # Text to put into checkpoint file
    text = 'model_checkpoint_path: "%s"\nall_model_checkpoint_paths: "%s"' % (chkp_file, chkp_file)

    # Copy files
    copyfile(data_file, data_file.replace(dir_path, model_dir))
    copyfile(index_file, index_file.replace(dir_path, model_dir))
    copyfile(meta_file, meta_file.replace(dir_path, model_dir))

    with open(os.path.join(model_dir, 'checkpoint'), "w") as text_file:
        text_file.write(text)


def get_run_config():
    """Create RunConfig."""
    # Attach job_id to model_directory
    model_dir = os.path.join(params.output_dir, params.network_type, job_id)

    # Potentially copy some checkpoint files
    resume_training(model_dir)

    # Create run configuration default
    config = tf.ConfigProto(allow_soft_placement=True)
    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(model_dir=model_dir)
    run_config = run_config.replace(save_summary_steps=params.save_summary_steps)
    run_config = run_config.replace(save_checkpoints_steps=params.save_checkpoints_steps)
    run_config = run_config.replace(save_checkpoints_secs=None)
    run_config = run_config.replace(session_config=config)
    run_config = run_config.replace(keep_checkpoint_max=params.keep_checkpoint_max)

    return run_config

# Make sure logging is on
tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    # Get run configuration
    run_config = get_run_config()

    # Run the training job, only if this is not just an export job
    if not params.export_from:
        estimator.run_manual(run_config, params)
    else:
        run_config = run_config.replace(model_dir=os.path.dirname(params.export_from))
        estimator.export(run_config, params, checkpoint=params.export_from)
