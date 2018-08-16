"""Send in a submission."""
import pandas as pd
import os
from src.input_pipe import get_input_fn
from src.lib.inference import RLenc, get_ensemble_predictors
from src.analysis.best_threshold import search_best


def get_latest_saved_model(model_dir):
    """Return the latest saved_model."""
    saved_models = os.path.join(model_dir, 'best_models')
    saved_chkp = sorted([int(mdl) for mdl in os.listdir(saved_models)])
    latest = saved_chkp[-1]
    path = os.path.join(saved_models, '%d' % latest)

    # Next, find the full path to the saved model
    mdl_time = os.listdir(path)

    # Return the final path
    return os.path.join(path, mdl_time[-1])


def create(path, saved_model, search_threshold=False):
    """Create submission, saved to csv file."""
    # Load the dataset
    print("Loading dataset")
    generator = get_input_fn(is_test=True)['test']()

    # Load the saved model
    print("Getting predictor")
    fcn = get_ensemble_predictors(saved_model)

    set_threshold = 0.5
    if search_threshold:
        set_threshold = search_best(saved_model)

    # Inference, and collect to dictionary
    pred_dict = {}
    n_samples = 18000
    report_every = int(float(n_samples) / 10)
    print("Running inference and RLE")
    for i, img_idx in enumerate(generator):
        img, idx = img_idx
        result = fcn(img.squeeze()) >= set_threshold
        pred_dict[idx] = RLenc(result)
        if i % report_every == 0:
            print("\t -> [%d/%d]" % (i, n_samples))

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(os.path.join(path, 'submission.csv'))


def submit(path, msg):
    """Submit a CSV to kaggle."""
    cmd = 'kaggle competitions submit'
    cmd = '%s %s' % (cmd, 'tgs-salt-identification-challenge')
    cmd = '%s -f %s' % (cmd, path)
    cmd = '%s -m "%s"' % (cmd, msg)
    os.system(cmd)


def submit_for_model(saved_models, tags=[], path='/hdd/datasets/TGS', search_threshold=True):
    """Auto tags the submission."""
    # Fetch the saved models id
    ids = [os.path.basename(mdl.split('/best_models')[0]) for mdl in saved_models]
    tags = tags + ids
    if search_threshold:
        tags.append('auto-threshold')
    tag_string = ', '.join(tags)
    create(path, saved_models, search_threshold=search_threshold)
    submit(os.path.join(path, 'submission.csv'), tag_string)


if __name__ == "__main__":
    model_dirs = ['/hdd/datasets/TGS/trained_models/network_types/vgg_16_unet/3254a838-ac23-4e80-bcf8-61f8b1b31f76']
    use_these = [get_latest_saved_model(mdl_dir) for mdl_dir in model_dirs]

    submit_for_model(
        use_these,
        tags=['128x128x1', 'VGG16_Unet', 'reduce_on_plateau', 'subpixel', 'textured_channels'],
        search_threshold=True)
