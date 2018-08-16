"""Contains available augmentations, which are only applied to images."""
from imgaug import augmenters as iaa
import cv2
import numpy as np


intensity_seq = iaa.Sequential([
    # iaa.Invert(0.3),
    iaa.OneOf([
        iaa.Sometimes(0.3, iaa.ContrastNormalization((0.5, 1.5)))
    ]),
    iaa.OneOf([
        iaa.Noop(),
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.0, 1.0)),
            iaa.AverageBlur(k=(2, 2)),
            iaa.MedianBlur(k=(3, 3))
        ])
    ])
], random_order=False)


def _combination(images, random_state, parents, hooks):
    new_img = []
    for img in images:
        smoothness = np.absolute(cv2.Laplacian(img.copy(), cv2.CV_64F)).astype(np.uint8)
        edges = cv2.Canny(img.copy(), 150, 150).astype(np.uint8)
        new_img.append(np.stack([smoothness, img.squeeze(), edges], axis=-1))

    return np.stack(new_img)


channel_augmenter = iaa.Lambda(func_images=_combination, func_keypoints=None)


# Collect all into a dictionary
defined = {
    'intensity_seq': intensity_seq,
    'channel_augmenter': channel_augmenter}
