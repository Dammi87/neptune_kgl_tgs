"""Send in a submission."""
import pandas as pd
import os
from src.input_pipe import get_input_fn
from src.lib.inference import RLenc, get_ensemble_predictors
from src.analysis.best_threshold import search_best


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
        result = fcn(img) >= set_threshold
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
    submit_for_model(
        ['/hdd/datasets/TGS/trained_models/resnet_152/unet/0750bf9c-656c-4ef0-b622-ed3377704bfe/best_models/14040/1534166325'],
        tags=['128x128x1', 'Custom Unet', 'Smooth masks'],
        search_threshold=True)
