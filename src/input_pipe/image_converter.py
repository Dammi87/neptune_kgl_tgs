"""Classes and methods for dataset boostrapping and augmentations."""
from imgaug import augmenters as iaa
import numpy as np
import tensorflow as tf
from . import augmentations as aug
from src.lib.neptune import get_params


class AugmentImages:
    """A class that holds all augmentations required for each images."""

    def __init__(self):
        """Initialize class."""
        self._pre_process = []
        self._images_only = []
        self._masks_only = []
        self._mask_and_images = []
        self._is_setup = False

    def add_pre_process(self, iaa_method):
        """Add a iaa method to the preprocessing of the images."""
        self._pre_process.append(iaa_method)

    def add_images_only(self, iaa_method):
        """Add a iaa method that should be applied for image_converterping."""
        self._images_only.append(iaa_method)

    def add_masks_only(self, iaa_method):
        """Add a iaa method that should be applied for masks_converterping."""
        self._masks_only.append(iaa_method)

    def add_both(self, iaa_method):
        """Add a iaa method that should be applied on both masks and images."""
        self._mask_and_images.append(iaa_method)

    def _setup(self):
        """Setup the augmentation chain."""
        if self._is_setup:
            return

        # Only setup once
        self._is_setup = True

        methods = [
            '_pre_process',
            '_images_only',
            '_masks_only',
            '_mask_and_images'
        ]

        for method in methods:
            attr = getattr(self, method)
            if len(attr) == 0:
                new = iaa.Noop()
            elif len(attr) == 1:
                new = attr[0]
            else:
                new = iaa.Sequential(attr)

            if method == '_mask_and_images':
                new = new.to_deterministic()

            setattr(self, method, new)

    def apply_preprocess(self, img):
        """Apply image preprocessing steps."""
        self._setup()
        return self._pre_process.augment_images(img)

    def apply(self, img, mask):
        """Apply the augmentations to the image and mask.

        NOTE: Images and masks should NOT be normalized!!

        Parameters
        ----------
        img : TYPE
            Description
        mask : None, optional
            Description

        Returns
        -------
        img : np.ndarray
            The images with augmentations, normalized between 0 and 1
        mask : np.ndarray
            The mask with augmentations, binary 0 and 1
        """
        self._setup()

        def wrapper(img, mask):

            # Apply both
            img = self._mask_and_images.augment_images(img)
            mask = self._mask_and_images.augment_images(mask)

            # Apply image only step
            img = self._images_only.augment_images(img)

            # Apply mask only step
            mask = self._masks_only.augment_images(mask)

            return img.astype(np.float32) / 255, mask.astype(np.float32) / 255

        return tf.py_func(func=wrapper, inp=[img, mask], Tout=[tf.float32, tf.float32])

    def apply_image_normalization(self, img):
        """Apply desired normalization to the image."""
        return img.astype(np.float32) / 255

    def apply_mask_normalization(self, mask):
        """Apply desired normalization to the mask."""
        return (mask / 255).astype(np.float32)

    def apply_normalization(self, img, mask):
        """Normalize images."""
        return self.apply_image_normalization(img), self.apply_mask_normalization(mask)

    def apply_normalization_tfunc(self, img, mask):
        """Same as apply_normalization, but runs as a Tensorflow op."""
        def wrapper(img, mask):
            return self.apply_normalization(img, mask)

        return tf.py_func(func=wrapper, inp=[img, mask], Tout=[tf.float32, tf.float32])


def get_augmenter():
    """Load an augmenter class for training / inference.

    Returns
    -------
    AugmentImages
        A class that will apply the desired pre-processing  / augmentations of images
        Main methods are
            .apply(img, mask) # Mask is optional
            .apply_preprocess(img) # If preprocessing is needed, it's done here. This should
                                     also be done for the test set.

    """
    # Instantiate
    augmenter = AugmentImages()

    # Fetch what augmentations are requested. Augmentations start with aug_SOME_NAME
    params = get_params(as_dict=True)
    for param in params:
        if 'aug_' in param:

            # If false, dont go further
            if not params[param]:
                continue

            # Check if this augmentation exists and add it
            aug_name = param.replace('aug_', '')
            if aug_name in aug.pre_process and params[param]:
                augmenter.add_pre_process(aug.pre_process[aug_name])
            elif aug_name in aug.image:
                augmenter.add_images_only(aug.image[aug_name])
            elif aug_name in aug.mask:
                augmenter.add_masks_only(aug.mask[aug_name])
            elif aug_name in aug.image_and_masks:
                augmenter.add_both(aug.image_and_masks[aug_name])

            print("Applied %s" % param)

    return augmenter
