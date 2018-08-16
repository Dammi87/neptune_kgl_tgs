"""Contains normalization settings to use."""
import numpy as np

VGG_MEAN = np.array([123.68, 116.779, 103.939]).reshape(1, 1, 1, 3).astype(np.float32)


def normalize_01(img):
    """Normalize the input image to be between 0-1."""
    return img.astype(np.float32) / 255


def normalize_vgg(img):
    """Normalize the input image using the vgg16 mean.

    The method will also stack channels to be 3 if they aint already.
    """
    if img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)
    return img.astype(np.float32) - VGG_MEAN


defined = {
    'basic': {
        'img': normalize_01,
        'mask': normalize_01,
        'img_channels': 1
    },
    'vgg_16_unet': {
        'img': normalize_vgg,
        'mask': normalize_01,
        'img_channels': 3
    },
    'resnet_152': {
        'img': normalize_vgg,
        'mask': normalize_01,
        'img_channels': 3
    },
    'unet': {
        'img': normalize_01,
        'mask': normalize_01,
        'img_channels': 1
    }
}


def get_norm_method(network_type):
    """Fetch normalization methods."""
    if network_type not in defined:
        network_type = 'basic'

    return defined[network_type]
