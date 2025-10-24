import functools as ft
import math

import jax
import jax.numpy as jnp
import sklearn.datasets as datasets

# import tensorflow_datasets as tfds
from .mnist import load as load_mnist  # noqa: F401


def iter_indices(num_samples, batch_size, rng=None):
    indices = jnp.arange(num_samples)
    if rng is not None:
        indices = jax.random.permutation(rng, indices)
    for i in range(0, num_samples, batch_size):
        j = min(i + batch_size, num_samples)
        yield indices[i:j]


class BatchLoader:
    def __init__(self, dataset, batch_size=1, rng=None):
        self.batch_size = batch_size
        self.dataset = dataset
        self.rng = rng
        if isinstance(dataset, jax.Array):
            self.batch_fn = lambda indices: jnp.take(
                self.dataset, indices=indices, axis=0
            )
        else:
            # Assume dataset is a pytree
            self.batch_fn = lambda indices: jax.tree.map(
                ft.partial(jnp.take, indices=indices, axis=0), self.dataset
            )

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset
        self.num_samples = jax.tree.leaves(dataset)[0].shape[0]
        assert self.num_samples >= self.batch_size
        self._length = math.ceil(self.num_samples / self.batch_size)

    def _next_rng(self):
        if self.rng is None:
            rng = None
        else:
            self.rng, rng = jax.random.split(self.rng)
        return rng

    def __iter__(self):
        rng = self._next_rng()
        for indices in iter_indices(self.num_samples, self.batch_size, rng=rng):
            yield self.batch_fn(indices)

    def dyn_batch(self, i):
        """Returns the i-th mini-batch (according to `self.batch_size`).

        NOTE: This method will return an overlapping mini-batch if the
        last mini-batch requested is supposed to be smaller than the
        other mini-batches. Refer to the documentation of
        `jax.lax.dynamic_slice` for details.

        """
        dyn_slice = ft.partial(
            jax.lax.dynamic_slice_in_dim,
            start_index=i * self.batch_size,
            slice_size=self.batch_size,
            axis=0,
        )
        return jax.tree.map(dyn_slice, self.dataset)

    def __len__(self):
        return self._length


def normalize_data(x, offsets=0.0):
    x_min = jnp.amin(x, axis=0, keepdims=True) - offsets
    x_max = jnp.amax(x, axis=0, keepdims=True) + offsets
    x = (x - x_min) / (x_max - x_min)
    return x


def make_moons(num, random_state=42):
    x, y = datasets.make_moons(num, random_state=random_state, noise=0.05)
    x = normalize_data(x, offsets=0.5)
    return x, y


def make_blobs(num, num_classes, random_state=42):
    x, y = datasets.make_blobs(
        n_samples=num,
        n_features=2,
        centers=num_classes,
        random_state=random_state,
    )
    x = normalize_data(x, offsets=0.5)
    return x, y


# def load_mnist(split, is_training, batch_size=0, subset: int = None):
#     ds = tfds.load("mnist", split=split)
#     if is_training and subset is not None:
#         ds = ds.take(subset).cache()
#     if batch_size < 1:
#         batch_size = len(ds)
#     ds = ds.batch(batch_size)
#     return tfds.as_numpy(ds)


def make_wave(key, num: int):
    std = jnp.linspace(1e-3, 1e0, num)
    x = jnp.linspace(0.35, 0.65, num)
    y = 5 * jnp.sin(10 * x)

    z = jax.random.normal(key, shape=y.shape)
    return x, y + std * z
