from pathlib import Path

import jax
import numpy as np
from jax.random import permutation, split

files_celeba = {
    "train_images": "celeba_train_images.npy",
    "train_labels": "celeba_train_labels.npy",
    "val_images": "celeba_val_images.npy",
    "val_labels": "celeba_val_labels.npy",
    "test_images": "celeba_test_images.npy",
    "test_labels": "celeba_test_labels.npy",
}

files = {
    "train_images": "train_images.npy",
    "train_labels": "train_labels.npy",
    "val_images": "val_images.npy",
    "val_labels": "val_labels.npy",
    "test_images": "test_images.npy",
    "test_labels": "test_labels.npy",
}


def get_arrays(root: str, files: dict = files):
    # check if the files exist
    for key, value in files.items():
        if not Path(root) / value:
            raise FileNotFoundError(
                f"{value} not found in {root}. Consider running the script to generate the files."
            )

    # load the files
    train_images = np.load(Path(root) / files["train_images"])
    # train_labels = np.load(Path(root) / files["train_labels"])
    val_images = np.load(Path(root) / files["val_images"])
    # val_labels = np.load(Path(root) / files["val_labels"])
    test_images = np.load(Path(root) / files["test_images"])
    # test_labels = np.load(Path(root) / files["test_labels"])

    return train_images, None, val_images, None, test_images, None


def to_batch(array, batch_size, key):
    """
    Convert an array to batches of a given size.
    """
    _, subkey = split(key)

    n_full_batches = array.shape[0] // batch_size
    permut = permutation(subkey, n_full_batches * batch_size)
    batches = array[permut].reshape(n_full_batches, batch_size, *array.shape[1:])
    return batches


def make_loader(batched_array):
    for batch in batched_array:
        yield {"image": batch, "label": batch}


def make_loader_svhn(batched_array):
    for batch in batched_array:
        # SVHN data is in the shape (batch_size, 32, 32, 3)
        # but we want to use it as (batch_size, 64,64, 3)
        batch = jax.image.resize(batch, (batch.shape[0], 64, 64, 3), method="nearest")
        yield {"image": batch, "label": batch}


def make_loader_latent_svhn(batched_array, encoding_func):
    for batch in batched_array:
        # SVHN data is in the shape (batch_size, 32, 32, 3)
        # but we want to use it as (batch_size, 64,64, 3)
        batch = jax.image.resize(batch, (batch.shape[0], 64, 64, 3), method="nearest")
        encoding = encoding_func(batch)
        yield {"image": encoding, "label": batch}


def make_loader_corrupted_latent(
    batched_array, encoding_func, corruption: str = "brightness"
):
    for batch in batched_array:
        if corruption == "contrast":
            batch = increase_contrast(batch, 1.5)
        elif corruption == "brightness":
            batch = change_brightness(batch, 0.2)
        elif corruption == "decrease_contrast":
            batch = decrease_contrast(batch, 0.5)
        elif corruption == "add_noise":
            batch = np.clip(batch + 0.5 * np.random.randn(*batch.shape), 0, 1).astype(
                np.float32
            )
        batch = encoding_func(batch)
        yield {"image": batch, "label": batch}


def increase_contrast(image, factor):
    """
    Increase the contrast of an image by a given factor.
    """
    mean = np.mean(image)
    image = (image - mean) * factor + mean
    return image


def decrease_contrast(image, factor):
    """
    Decrease the contrast of an image by a given factor.
    """
    mean = np.mean(image)
    image = (image - mean) / factor + mean
    return image


def change_brightness(image, factor):
    """
    Change the brightness of an image by a given factor.
    """
    image = np.clip(image + factor, 0, 1)
    return image


def loader_samples_from_prior(key, scale=1.0, num=1000):
    for i in range(num):
        key, subkey = split(key)
        yield {"image": jax.random.normal(subkey, (32, 64)) * scale, "label": None}
