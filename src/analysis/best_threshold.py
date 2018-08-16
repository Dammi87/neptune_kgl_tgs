"""Some analysis scripts."""
import numpy as np
from src.input_pipe.image_converter import get_augmenter
from src.input_pipe.dataset import get_dataset
from src.analysis.lib import iou_metric_batch
from src.lib.inference import get_ensemble_predictors


def search_best(saved_model):
    """Search for the best threshold of this model, according to validation set."""
    # Get the augmentor
    augmenter = get_augmenter()
    augmenter._setup()

    # Load in the dataset
    dataset_loader = get_dataset(raw=True)

    # Get train, validation split
    _, valid = dataset_loader.get_dataset()

    # Load a saved model
    fcn = get_ensemble_predictors(saved_model)

    # Loop through
    results = []
    masks = []

    for i, (img, mask, _id) in enumerate(zip(valid['x'], valid['y'], valid['id'])):

        # Apply normalization
        img, mask = augmenter.apply_normalization(img, mask)

        # Run through network
        pred = fcn(img.squeeze()).squeeze()

        # Collect all results
        results.append(pred)
        masks.append(dataset_loader._df['masks_org'][_id].squeeze() == 255)

    # Convert to arrays
    results = np.array(results)
    masks = np.array(masks).astype(np.int32)

    # Check threshold
    thresholds = np.linspace(0, 1, 50).tolist()
    ious = [iou_metric_batch(masks, np.int32(results > threshold)) for threshold in thresholds]

    best_iou = max(ious)
    best_t = thresholds[ious.index(best_iou)]
    print('Best threshold: %1.2f with iout of %1.2f' % (best_t, best_iou))

    return best_t
