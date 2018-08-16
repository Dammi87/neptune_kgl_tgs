"""Contains the class to load the dataset."""
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
from src.lib.neptune import get_params
from src.input_pipe.data import BAD_IDS

params = get_params()


class DatasetLoader():
    """Dataset loader."""

    def __init__(self, path, img_type='images', remove_bad=False, resize=128, split=0.1, post_df_methods=[], load_test_set=False):
        """Initialize the loader class.

        Parameters
        ----------
        path : str
            Path to the downloaded kaggle dataset
        resize : int, optional
            Specify the size to which the images should be resized to, None keeps
            the same size. Default is 128
        split : float, optional
            Validation ratio, default is 0.1
        post_df_methods : None, optional
            A list of methods to apply after dataframes have been loaded.
            The dataframes have the following columns
                images: Numpy arrays with the loaded images in the specified size, normalized
                masks: Numpy arrays with the loaded masks in the specified size, normalized
                coverage: Indicator of how much salts there is in each iamge
                z: The depthf of each image
        load_test_set : bool, optional
            Loads the test set instead of the training set
        """
        self._path = path
        self._img_type = img_type
        self._resize = resize
        self._post_df_methods = post_df_methods
        self._load_test_set = load_test_set
        self._split = split
        self._remove_bad = remove_bad and not load_test_set

        # Load the dataset with the properties specified
        self._load()

    def _load(self):
        train_df = pd.read_csv(os.path.join(self._path, "train.csv"), index_col="id", usecols=[0])
        depths_df = pd.read_csv(os.path.join(self._path, "depths.csv"), index_col="id")
        train_df = train_df.join(depths_df)

        # Remove bad id's
        if self._remove_bad:
            train_df = train_df[~train_df.index.isin(BAD_IDS)]

        bw = self._img_type == 'images'
        # Load test set
        if self._load_test_set:
            pd.options.mode.chained_assignment = None
            df = depths_df[~depths_df.index.isin(train_df.index)]

            img_paths = os.path.join(self._path, self._img_type)
            df["images"] = [load_img(os.path.join(img_paths, "{}.png".format(idx)), self._resize, bw) for idx in df.index]
        else:
            df = train_df
            img_paths = os.path.join(self._path, "train", self._img_type)
            mask_paths = os.path.join(self._path, "train", "masks")
            df["images"] = [load_img(os.path.join(img_paths, "{}.png".format(idx)), self._resize, bw) for idx in df.index]
            df["masks"] = [load_mask(os.path.join(mask_paths, "{}.png".format(idx)), self._resize) for idx in df.index]
            df["masks_org"] = [load_mask(os.path.join(mask_paths, "{}.png".format(idx)), None) for idx in df.index]
            df["images_org"] = [load_img(os.path.join(img_paths, "{}.png".format(idx)), None, bw) for idx in df.index]

            # Salt coverage
            img_size = df["images"][0].shape[1]
            df["coverage"] = df.masks.map(np.sum) / pow(img_size, 2)

        # Apply the post methods
        for method in self._post_df_methods:
            df = method(df)

        # Save
        self._df = df

    def _get_train_and_validation(self):
        """Split the dataframe into training and validation.

        Parameters
        ----------
        split : float, optional
            Ratio of the validation set

        Returns
        -------
        train: tuple
            A tuple containing two lists, where the first one is numpy arrays for images, and the
            second one is numpy array for masks
        validation: tuple
            A tuple containing two lists, where the first one is numpy arrays for images, and the
            second one is numpy array for masks
        """
        def cov_to_class(val):
            for i in range(0, 11):
                if (val / 255) / 10 <= i:
                    return i

        # Apply coverage binning for stratification
        self._df["coverage_class"] = self._df.coverage.map(cov_to_class)

        ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = \
            train_test_split(self._df.index.values,
                             np.array(self._df.images.tolist()),
                             np.array(self._df.masks.tolist()),
                             self._df.coverage.values,
                             self._df.z.values,
                             test_size=self._split, stratify=self._df.coverage_class, random_state=1337)

        return {'x': x_train, 'y': y_train, 'id': ids_train}, {'x': x_valid, 'y': y_valid, 'id': ids_valid}

    def _get_test(self):
        """Return the images from the test set."""
        return {'x': np.array(self._df.images.tolist()), 'y': None, 'id': self._df.index}, None

    def get_dataset(self):
        """Get the dataset."""
        if self._load_test_set:
            return self._get_test()
        else:
            return self._get_train_and_validation()


def load_img(img_path, resize=None, bw=True):
    """Load an image as a numpy array normalized, resize if specified.

    Parameters
    ----------
    img_path : TYPE
        Path to the image to be loaded
    resize : None, int
        Resize images to this width and height

    Returns
    -------
    np.ndarray, shape (width, height, 1) or (resize, resize, 1)
        A numpy array containing the pixel values for the image, normalized by 255
    """
    if bw:
        img = np.array(resize_img(Image.open(img_path).convert('L'), resize))
        return np.expand_dims(img, 2)
    else:
        return np.array(resize_img(Image.open(img_path), resize))


def load_mask(img_path, resize=None):
    """Load a mask as a numpy array, resize if specified.

    Parameters
    ----------
    img_path : TYPE
        Path to the image to be loaded
    resize : None, int
        Resize images to this width and height

    Returns
    -------
    np.ndarray, shape (width, height, 1) or (resize, resize, 1)
        A numpy array containing the pixel values for the mask, True value is salt
    """
    mask = np.array(resize_img(Image.open(img_path).convert('L'), resize, method=Image.NEAREST))
    return np.expand_dims(mask, 2)


def resize_img(img, resize, method=Image.BILINEAR):
    """Resize images using the specified method."""
    if resize:
        img = img.resize((resize, resize), method)
    return img


def get_dataset(raw=False, is_test=False):
    """Get the dataset class according to the neptune parameters."""
    params = get_params()
    loader = DatasetLoader(path=params.path,
                           img_type=params.img_type,
                           remove_bad=params.remove_bad_id,
                           resize=params.resize,
                           split=params.validation_split,
                           load_test_set=is_test)
    if raw:
        return loader
    return loader.get_dataset()


def create_generator(augmenter=None, is_test=False):
    """Create a generator for images.

    Parameters
    ----------
    augmenter : image_converter.AugmentImages
        Augmenter class

    No Longer Returned
    ------------------
    tuple
        ((generators), img_size)
    """

    # Load dataset
    datasets = get_dataset(is_test=is_test)

    # Create a augmenter dummy class
    class AugmentDummy:
        def apply(self, img, mask):
            return img, mask

        def apply_preprocess(self, img):
            return img

    if augmenter is None:
        augmenter = AugmentDummy()

    # Apply preprocessing, also capture the shape of the images for tensorflow
    for data in datasets:
        if data is not None:
            if augmenter is not None:
                data['x'] = augmenter.apply_preprocess(data['x'])
            img_shape = list(data['x'][0].shape)
            if not is_test:
                mask_shape = list(data['y'][0].shape)

    # Now, create the generators for each dataset. (During training, there are two datasets, thus two generators)
    def train_generator(dataset):
        n_samples = dataset['x'].shape[0]
        count = 0
        index = np.arange(n_samples)
        np.random.shuffle(index)
        while count < n_samples:
            idx = index[count]
            img = dataset['x'][idx, ::]
            mask = dataset['y'][idx, ::]
            count += 1

            yield img, mask

    def validation_generator(dataset):
        n_samples = dataset['x'].shape[0]
        count = 0
        while count < n_samples:
            img = dataset['x'].take(count, mode='wrap', axis=0)
            mask = dataset['y'].take(count, mode='wrap', axis=0)
            count += 1

            yield img, mask

    def test_generator(dataset):
        n_samples = dataset['x'].shape[0]
        count = 0
        while count < n_samples:
            img = dataset['x'][count, ::]
            _id = dataset['id'][count]
            count += 1

            yield augmenter.apply_image_normalization(img), _id

    if is_test:
        def test_gen():
            return test_generator(datasets[0])

        return test_gen, img_shape, None
    else:
        def train_gen():
            return train_generator(datasets[0])

        def valid_gen():
            return validation_generator(datasets[1])

        return (train_gen, valid_gen), img_shape, mask_shape


if __name__ == "__main__":
    """Testing if the generators work or not."""

    gen, img_shape = create_generator(AugmentDummy())

    if not get_params().is_test:
        train, valid = gen
        valid_generator = valid()
        for idx, item in enumerate(valid_generator):
            print(idx, item[0].shape)
        exit()
        train_generator = train()
        for idx, item in enumerate(train_generator):
            print(idx, item[0].shape)

            if idx > 8000:
                break

    else:
        for idx, item in enumerate(gen()):
            print(idx, item[0].shape)
