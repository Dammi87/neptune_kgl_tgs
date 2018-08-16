"""Method to perform inference, also to convert checkpoints to saved models."""
import tensorflow as tf
import numpy as np
from PIL import Image
from src.input_pipe.dataset import resize_img
from src.lib.neptune import get_params

params = get_params()


def get_predictor(saved_model_dir):
    """Create a predictor function from a saved model directory.

    Parameters
    ----------
    saved_model_dir : str
        Path to the folder that contains the saved model .pb file

    Returns
    -------
    Method
        Returns a method that will perform inference on a given input
    """
    predictor_fn = tf.contrib.predictor.from_saved_model(
        export_dir=saved_model_dir,
        signature_def_key="prediction"
    )
    return predictor_fn


def get_ensemble_predictors(saved_models):
    """Given list of saved models, return ensemble fcn."""
    predictors = []
    for imodel in saved_models:
        predictors.append(tf.contrib.predictor.from_saved_model(
            export_dir=imodel,
            signature_def_key="prediction"
        ))

    def ensemble(img):
        result = 0
        for fcn in predictors:
            result += fcn({'input': [img]})['probabilities']

        result = np.divide(result, len(predictors)).squeeze()

        # Downsample to correct size
        return np.array(resize_img(Image.fromarray(result), resize=params.original_size))

    return ensemble


def get_single_predictor(saved_model):
    """Given single saved_model, return results."""
    fcn = get_predictor(saved_model)

    def single(img):
        return resize_img(fcn({'input': [img]})['probabilities'], resize=params.original_size)

    return single


def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  # list of run lengths
    r = 0  # the current run length
    pos = 1  # count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs