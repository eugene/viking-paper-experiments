from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import struct
from sklearn.metrics import roc_auc_score

from vp import viking
from vp.ood_functions.metrics import get_brier_score, get_calib


@struct.dataclass
class EvalState:
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Any
    batch_stats: Any
    key: Any
    opt_state: Any
    tx: Any = struct.field(pytree_node=False)


def prepare_from_checkpoint(model, checkpoint):
    params = checkpoint["params"]
    params["param_nn"], model_unflatten = jax.flatten_util.ravel_pytree(
        params["param_nn"]
    )
    state = EvalState(
        apply_fn=model.apply,
        params=params,
        batch_stats=checkpoint["batch_stats"],
        key=checkpoint["key"],
        opt_state=checkpoint.get("opt_state"),
        tx=None,
    )
    apply_fn = viking.apply_fn_from_state(state, train=False)

    def model_fn(p, x):
        return apply_fn(model_unflatten(p), x)

    return state, model_fn, model_unflatten


def prepare_from_checkpoint_generator(
    model, checkpoint_decoder=None, mle_checkpoint=None
):
    if mle_checkpoint is not None:
        print("Loading MLE checkpoint")
        params_mle = mle_checkpoint["params"]

    if checkpoint_decoder is not None:
        print("Loading decoder checkpoint")
        params_decoder = checkpoint_decoder["params"]

    if checkpoint_decoder is not None and mle_checkpoint is not None:
        print("Surgery")
        params_mle["param_nn"]["decoder"] = params_decoder["param_nn"]
        params_decoder["param_nn"] = params_mle["param_nn"]
        params = params_decoder
    elif checkpoint_decoder is not None and mle_checkpoint is None:
        print("Decoder only")
        params = params_decoder
    elif mle_checkpoint is not None and checkpoint_decoder is None:
        print("MLE only")
        params = params_mle
    else:
        raise ValueError("No vaild checkpoint provided")
    params["param_nn"], model_unflatten = jax.flatten_util.ravel_pytree(
        params["param_nn"]
    )

    if mle_checkpoint is not None:
        state = EvalState(
            apply_fn=model.apply,
            params=params,
            batch_stats=None,
            key=mle_checkpoint["key"],
            opt_state=mle_checkpoint.get("opt_state"),
            tx=None,
        )
    else:
        state = EvalState(
            apply_fn=model.apply,
            params=params,
            batch_stats=None,
            key=checkpoint_decoder["key"],
            opt_state=checkpoint_decoder.get("opt_state"),
            tx=None,
        )

    if mle_checkpoint is not None:
        apply_fn = viking.apply_fn_from_state(state, train=True)
    else:
        apply_fn = viking.apply_fn_from_state(state, train=False)

    def model_fn(p, x):
        return apply_fn(model_unflatten(p), x)

    return state, model_fn, model_unflatten


def auroc(scores_id, scores_ood):
    scores = np.concatenate((scores_id, scores_ood))
    labels = np.concatenate((np.zeros_like(scores_id), np.ones_like(scores_ood)))
    return roc_auc_score(labels, scores)


def evaluate(test_loader, posterior_samples, model_fns, num_classes):
    if isinstance(model_fns, Callable):
        # If model_fn is a single callable, use it
        #
        # This is used to check whether to vmap over posterior_samples
        # with a single model_fn or loop over model_fns, feeding each
        # a single row of posterior_samples
        model_fn = model_fns
        model_fns = None
    elif len(model_fns) != posterior_samples.shape[0]:
        raise ValueError("Mismatch in number of model_fn's and posterior samples")
    all_y_prob = []
    all_y_log_prob = []
    all_y_true = []
    all_y_var = []
    all_y_prob_var = []
    all_y_sample_probs = []
    for batch in test_loader:
        x_test = batch["image"]
        y_test = batch["label"]
        if len(y_test.shape) == 1:
            y_test = jax.nn.one_hot(y_test, num_classes=num_classes)

        if model_fns is None:
            predictive_samples, _ = jax.vmap(model_fn, in_axes=(0, None))(
                posterior_samples, x_test
            )
        else:
            predictive_samples = []
            for model_fn, posterior_sample in zip(model_fns, posterior_samples):
                out, _ = model_fn(posterior_sample, x_test)
                predictive_samples.append(out)
            predictive_samples = jnp.stack(predictive_samples, axis=0)

        # predictive_samples_mean = jnp.mean(predictive_samples, axis=0)
        # if eval_args["likelihood"] == "regression":
        # predictive_samples_std = jnp.std(predictive_samples, axis=0)
        # all_y_var.append(predictive_samples_std**2)

        y_prob = jnp.mean(jax.nn.softmax(predictive_samples, axis=-1), axis=0)
        y_log_prob = jnp.mean(jax.nn.log_softmax(predictive_samples, axis=-1), axis=0)

        # y_prob = jax.nn.softmax(predictive_samples_mean, axis=-1)
        # y_log_prob = jax.nn.log_softmax(predictive_samples_mean, axis=-1)

        if predictive_samples.shape[0] > 1:
            predictive_samples_std = jnp.std(predictive_samples, axis=0)
        else:
            predictive_samples_std = jnp.zeros_like(predictive_samples[0])
        all_y_var.append(predictive_samples_std**2)
        sample_probs = jax.nn.softmax(predictive_samples, axis=-1)
        if sample_probs.shape[0] > 1:
            y_prob_std = jnp.std(sample_probs, axis=0)
        else:
            y_prob_std = jnp.zeros_like(sample_probs[0])
        all_y_prob_var.append(y_prob_std**2)
        all_y_sample_probs.append(sample_probs)

        all_y_prob.append(y_prob)
        all_y_log_prob.append(y_log_prob)
        all_y_true.append(y_test)

    all_y_prob = jnp.concatenate(all_y_prob, axis=0)
    all_y_log_prob = jnp.concatenate(all_y_log_prob, axis=0)
    all_y_true = jnp.concatenate(all_y_true, axis=0)
    if all_y_true.shape[-1] != all_y_log_prob.shape[-1]:
        all_y_true = all_y_true[..., : all_y_log_prob.shape[-1]]

    # compute some metrics: mean confidence, accuracy and negative log-likelihood
    metrics = {}
    all_y_var = jnp.concatenate(all_y_var, axis=0)
    all_y_var = np.asarray(all_y_var)
    all_y_prob_var = jnp.concatenate(all_y_prob_var, axis=0)
    all_y_prob_var = np.asarray(all_y_prob_var)
    all_y_sample_probs = jnp.concatenate(all_y_sample_probs, axis=1)

    metrics["conf"] = (jnp.max(all_y_prob, axis=1)).mean().item()
    metrics["nll"] = (
        (-jnp.mean(jnp.sum(all_y_log_prob * all_y_true, axis=-1), axis=-1))
        .mean()
        .item()
    )
    metrics["acc"] = (
        (jnp.argmax(all_y_prob, axis=1) == jnp.argmax(all_y_true, axis=1)).mean().item()
    )
    all_y_prob = np.asarray(all_y_prob)
    all_y_true = np.asarray(all_y_true)
    metrics["brier"] = get_brier_score(all_y_prob, all_y_true)
    ece, mce = get_calib(all_y_prob, all_y_true)
    metrics["ece"], metrics["mce"] = ece.item(), mce.item()
    outputs = {
        "all_y_prob": all_y_prob,
        "all_y_true": all_y_true,
        "all_y_var": all_y_var,
        "all_y_prob_var": all_y_prob_var,
        "all_y_sample_probs": all_y_sample_probs,
    }
    return metrics, outputs


def evaluate_single(loader, params, model_fn, num_classes):
    return evaluate(loader, params[None, ...], model_fn, num_classes)


def reconstruct_and_plot(images, params, model_fn, save_path=None):
    ## for each image, generate a sample
    reconstructed_images = model_fn(params, images)[0][0]
    reconstructed_images = np.array(reconstructed_images)

    # plot the images in a grid
    plot_batch(reconstructed_images, save_path=save_path)
    return reconstructed_images


def reconstruct_and_plot_sampled_models(images, params, model_fn, save_path=None):
    reconstructed_images_list = [model_fn(param, images)[0] for param in params]
    reconstructed_images_stacked = jnp.stack(
        reconstructed_images_list, axis=0
    )  # (num_samples, batch_size, 64, 64, 3)
    # calciulate magnitude per pixel across samples and channels

    channel_norm = jnp.linalg.norm(
        reconstructed_images_stacked, axis=(4), ord=2
    )  # (num_samples, batch_size, 64, 64)
    channel_norm_means = jnp.mean(channel_norm, axis=0)  # (batch_size, 64, 64)
    channel_norm_std = jnp.std(channel_norm, axis=0)  # (batch_size, 64, 64)
    channel_norm_means = jnp.expand_dims(
        channel_norm_means, axis=-1
    )  # (batch_size, 64, 64, 1)
    channel_norm_std = jnp.expand_dims(
        channel_norm_std, axis=-1
    )  # (batch_size, 64, 64, 1)
    stds_rgb = jnp.std(reconstructed_images_stacked, axis=0)  # (batch_size, 64, 64, 3)
    max_values = jnp.max(stds_rgb, axis=(1, 2, 3))  # (batch_size,)

    rgb_stds_out = jax.vmap(lambda x, y: x / y)(stds_rgb, max_values)

    for i, batch in enumerate(reconstructed_images_list):
        plot_batch(batch, save_path=f"{save_path}_model_{i}.png")

    plot_batch(
        channel_norm_means,
        save_path=f"{save_path}_channel_norm_meaned_across_samples.png",
    )
    plot_batch(
        channel_norm_std, save_path=f"{save_path}_channel_norm_std_across_samples.png"
    )
    plot_batch(rgb_stds_out, save_path=f"{save_path}_stds_rgb.png")
    plot_batch(1 - rgb_stds_out, save_path=f"{save_path}_stds_rgb_inverse.png")

    return reconstructed_images_list


def get_scalar_deviation(images, params, model_fn):
    reconstructed_images_list = [model_fn(param, images)[0] for param in params]
    reconstructed_images_stacked = jnp.stack(
        reconstructed_images_list, axis=0
    )  # (num_samples, batch_size, 64, 64, 3)
    stds_rgb = jnp.std(reconstructed_images_stacked, axis=0)  # (batch_size, 64, 64, 3)
    max_values = jnp.max(stds_rgb, axis=(1, 2, 3))  # (batch_size,)

    return max_values, stds_rgb


def plot_batch(images, save_path=None):
    # images is a batch of images (batch_size, 64,64,3)
    # plot the images in a grid
    images = np.array(images)
    fig, ax = plt.subplots(4, 8, figsize=(12, 6))
    for i in range(4):
        for j in range(8):
            if i * 8 + j >= images.shape[0]:
                break
            if images[i * 8 + j].shape[-1] == 1:
                ax[i, j].imshow(images[i * 8 + j, ..., 0], cmap="gray")
            else:
                ax[i, j].imshow(
                    (jnp.clip(images[i * 8 + j], 0, 1) * 255).astype(np.uint8)
                )
            ax[i, j].axis("off")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()
