import importlib
import os
import random
from typing import Any, Optional

import jax
import numpy as np
import torch


def compute_num_params(pytree):
    return sum(
        x.size if hasattr(x, "size") else 0 for x in jax.tree_util.tree_leaves(pytree)
    )


def random_split_like_tree(rng_key, target=None, treedef=None):
    # https://github.com/google/jax/discussions/9508
    if treedef is None:
        treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def tree_random_normal_like(rng_key, target, n_samples: Optional[int] = None):
    # https://github.com/google/jax/discussions/9508
    keys_tree = random_split_like_tree(rng_key, target)
    if n_samples is None:
        return jax.tree_util.tree_map(
            lambda l, k: jax.random.normal(k, l.shape, l.dtype),
            target,
            keys_tree,
        )
    else:
        return jax.tree_util.tree_map(
            lambda l, k: jax.random.normal(k, (n_samples,) + l.shape, l.dtype),
            target,
            keys_tree,
        )


def tree_random_uniform_like(
    rng_key, target, n_samples: Optional[int] = None, minval: int = 0, maxval: int = 1
):
    keys_tree = random_split_like_tree(rng_key, target)
    if n_samples is None:
        return jax.tree_util.tree_map(
            lambda l, k: jax.random.uniform(k, l.shape, l.dtype, minval, maxval),
            target,
            keys_tree,
        )
    else:
        return jax.tree_util.tree_map(
            lambda l, k: jax.random.uniform(
                k, (n_samples,) + l.shape, l.dtype, minval, maxval
            ),
            target,
            keys_tree,
        )


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def set_seed(seed: int = 666, precision: int = 10) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(precision=precision)
