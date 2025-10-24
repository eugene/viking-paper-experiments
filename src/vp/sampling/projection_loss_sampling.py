from typing import Callable

import jax
from jax import numpy as jnp
from tqdm import tqdm

from vp.helper import set_seed
from vp.sampling import (
    loss_kernel_gen_proj_vp,
    loss_kernel_proj_vp,
    precompute_loss_inv,
    precompute_loss_inv_gen,
)


def sample_loss_projections(
    model_fn: Callable,
    loss_fn: Callable,
    params_vec,
    eps,
    alpha: float,
    x_train_batched,
    y_train_batched,
    n_iterations: int,
    n_params,
    unflatten_fn: Callable,
    use_optimal_alpha: bool = False,
    acceleration: bool = False,
):
    # Eps is a Standard Random Normal Pytree
    prior_samples = eps
    batched_eigvecs, batched_inv_eigvals = precompute_loss_inv(
        model_fn, loss_fn, params_vec, x_train_batched, y_train_batched
    )
    proj_vp_fn = lambda v: loss_kernel_proj_vp(
        vec=v,
        model_fn=model_fn,
        loss_fn=loss_fn,
        params=params_vec,
        x_train_batched=x_train_batched,
        y_train_batched=y_train_batched,
        batched_eigvecs=batched_eigvecs,
        batched_inv_eigvals=batched_inv_eigvals,
        n_iterations=n_iterations,
        acceleration=acceleration,
    )
    projected_samples = jax.vmap(proj_vp_fn)(prior_samples)
    trace_proj = (
        jax.vmap(lambda e, x: jnp.dot(e, x), in_axes=(0, 0))(eps, projected_samples)
    ).mean()
    if use_optimal_alpha:
        print("Initial alpha:", alpha)
        alpha = jnp.dot(params_vec, params_vec) / (n_params - trace_proj)
        print("Optimal alpha:", alpha)
    posterior_samples = jax.vmap(
        lambda single_sample: unflatten_fn(
            params_vec + 1 / jnp.sqrt(alpha) * single_sample
        )
    )(projected_samples)
    metrics = {"kernel_dim": trace_proj, "alpha": alpha}
    return posterior_samples, metrics


def sample_loss_projections_dataloader(
    model_fn: Callable,
    loss_fn: Callable,
    params_vec,
    eps,
    alpha: float,
    train_loader,
    sample_batch_size: int,
    seed,
    n_iterations: int,
    x_val: jnp.ndarray,
    y_val: jnp.ndarray,
    n_params,
    unflatten_fn: Callable,
    vmap_dim: int = 5,
    use_optimal_alpha: bool = False,
    acceleration: bool = False,
):
    set_seed(seed)
    projected_samples = eps
    for i, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        x_data = jnp.asarray(batch["image"], dtype=float)
        y_data = jnp.asarray(batch["label"], dtype=float)
        N = x_data.shape[0]
        n_batches = N // sample_batch_size
        x_train_batched = x_data[: n_batches * sample_batch_size].reshape(
            (n_batches, -1) + x_data.shape[1:]
        )
        y_train_batched = y_data[: n_batches * sample_batch_size].reshape(
            (n_batches, -1) + y_data.shape[1:]
        )
        batched_eigvecs, batched_inv_eigvals = precompute_loss_inv(
            model_fn, loss_fn, params_vec, x_train_batched, y_train_batched
        )
        proj_vp_fn = lambda v: loss_kernel_proj_vp(
            vec=v,
            model_fn=model_fn,
            loss_fn=loss_fn,
            params=params_vec,
            x_train_batched=x_train_batched,
            y_train_batched=y_train_batched,
            batched_eigvecs=batched_eigvecs,
            batched_inv_eigvals=batched_inv_eigvals,
            n_iterations=n_iterations,
            acceleration=acceleration,
        )
        del (
            x_train_batched,
            x_data,
            y_train_batched,
            y_data,
            batched_eigvecs,
            batched_inv_eigvals,
            proj_vp_fn,
        )
    trace_proj = (
        jax.vmap(lambda e, x: jnp.dot(e, x), in_axes=(0, 0))(eps, projected_samples)
    ).mean()
    if use_optimal_alpha:
        print("Initial alpha:", alpha)
        alpha = jnp.dot(params_vec, params_vec) / (n_params - trace_proj)
        print("Optimal alpha:", alpha)
    posterior_samples = jax.vmap(
        lambda single_sample: unflatten_fn(
            params_vec + 1 / jnp.sqrt(alpha) * single_sample
        )
    )(projected_samples)
    metrics = {"kernel_dim": trace_proj, "alpha": alpha}
    return posterior_samples, metrics


def sample_loss_gen_projections_dataloader(
    loss_model_fn: Callable,
    params_vec,
    eps,
    train_loader,
    sample_batch_size: int,
    seed,
    n_iterations: int,
    x_val: jnp.ndarray,
    unflatten_fn: Callable,
    vmap_dim: int = 5,
    acceleration: bool = False,
):
    set_seed(seed)
    projected_samples = eps
    for i, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        x_data = jnp.asarray(batch["image"], dtype=float)
        N = x_data.shape[0]
        n_batches = N // sample_batch_size
        x_train_batched = x_data[: n_batches * sample_batch_size].reshape(
            (n_batches, -1) + x_data.shape[1:]
        )
        batched_eigvecs, batched_inv_eigvals = precompute_loss_inv_gen(
            loss_model_fn, params_vec, x_train_batched
        )
        proj_vp_fn = lambda v: loss_kernel_gen_proj_vp(
            vec=v,
            loss_model_fn=loss_model_fn,
            params=params_vec,
            x_train_batched=x_train_batched,
            batched_eigvecs=batched_eigvecs,
            batched_inv_eigvals=batched_inv_eigvals,
            n_iterations=n_iterations,
            acceleration=acceleration,
        )
        projected_samples = projected_samples.reshape(
            (-1, vmap_dim) + projected_samples.shape[1:]
        )
        projected_samples = jax.lax.map(
            lambda p: jax.vmap(proj_vp_fn)(p), projected_samples
        )
        projected_samples = projected_samples.reshape(
            (-1,) + projected_samples.shape[2:]
        )

        del x_train_batched, x_data, batched_eigvecs, batched_inv_eigvals, proj_vp_fn
    trace_proj = (
        jax.vmap(lambda e, x: jnp.dot(e, x), in_axes=(0, 0))(eps, projected_samples)
    ).mean()
    posterior_samples = jax.vmap(
        lambda single_sample: unflatten_fn(params_vec + single_sample)
    )(projected_samples)
    metrics = {"kernel_dim": trace_proj}
    return posterior_samples, metrics
