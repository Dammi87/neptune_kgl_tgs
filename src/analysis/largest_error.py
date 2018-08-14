"""Some analysis scripts."""
import numpy as np
from PIL import Image
import os
from src.lib.neptune import get_params
from src.input_pipe.image_converter import get_augmenter
from src.input_pipe.dataset import get_dataset
from src.analysis.lib import iou_metric
from src.lib.inference import get_ensemble_predictors
from heapq import nlargest

params = get_params()

# Get the augmentor
augmenter = get_augmenter()

# Load in the dataset
dataset_loader = get_dataset(raw=True)

# Load a saved model
fcn = get_ensemble_predictors(['/hdd/datasets/TGS/trained_models/resnet_152/unet/0750bf9c-656c-4ef0-b622-ed3377704bfe/best_models/14040'])

# Threshold
T = 0.5

result_dict = {}
predictions = dataset_loader._df['images'].values
# Loop through the train AND validation set, calculate iou
for i, (idx, img, mask) in enumerate(zip(dataset_loader._df.index,
                                         dataset_loader._df['images'].values,
                                         dataset_loader._df['masks_org'].values)):

    # Apply normalization
    img, mask = augmenter.apply_normalization(img, mask)

    # Run through network
    pred = (fcn(img) > T).astype(np.int32)
    iou = iou_metric(mask.squeeze().astype(np.int32), pred.squeeze())

    # Collect predictions
    predictions[i] = pred.squeeze()

    result_dict[idx] = -iou

    print('%d/%d' % (i + 1, 4000))


dataset_loader._df['predictions'] = predictions

# Get the largets errors
result = nlargest(25, result_dict, key=result_dict.get)

# Create result images of them
for i, bad_img in enumerate(result):
    org_img = dataset_loader._df['images_org'][bad_img].squeeze()
    mask_img = dataset_loader._df['masks_org'][bad_img].squeeze()
    pred = (dataset_loader._df['predictions'][bad_img].squeeze() * 255).astype(np.uint8)

    # Concatenate them together
    stacked = np.concatenate([org_img, mask_img, pred], axis=1)

    # Image from pil
    img = Image.fromarray(stacked)

    # Nr
    nbr = str(i).zfill(3)

    # IOU
    iou = str(int(result_dict[bad_img] * 100)).zfill(3)

    prefix = '%s_%s' % (nbr, iou)

    # Save
    folder = os.path.join(params.path, 'results', '64156b6c-e098-4019-b96e-1f7755c0964e')
    if not os.path.exists(folder):
        os.makedirs(folder)
    img.save(os.path.join(folder, '%s_%s.png' % (prefix, bad_img)))
