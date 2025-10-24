import argparse
import os
import pickle
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np

from vp import eval, utils, viking
from vp.data.arrays_data import (
    files,
    files_celeba,
    get_arrays,
    loader_samples_from_prior,
    make_loader_svhn,
    to_batch,
)
from vp.losses import PRCLoss
from vp.models.celeba_linen import VAE as celeba_vae
from vp.models.celeba_linen import CelebADecoder
from vp.models.fmnist_vae_linen import VAE as fmnist_vae
from vp.models.fmnist_vae_linen import FmnistDecoder
from vp.viking import apply_fn_from_state


def maybe_override(name, old_value, new_value):
    if new_value is None:
        return old_value
    print(
        "WARNING: Overriding {} (from: {}, to: {})".format(name, old_value, new_value)
    )
    return new_value


def main(args):
    key = jax.random.PRNGKey(seed=args.seed)
    info = utils.load_log(args.log)
    config = info["config"]

    # %%
    # Data loader(s) setup
    train_images, _, _, _, test_images, _ = get_arrays(
        args.dataset_root,
        (files_celeba if args.dataset == "celeba" else files),
    )

    _, _, _, _, test_images_celeba, _ = get_arrays(
        "/workspace/data/celeba",
        files_celeba,
    )
    test_images = jax.device_put(test_images)
    train_images = jax.device_put(train_images)

    test_images_celeba = jax.device_put(test_images_celeba)

    test_loader = make_loader_svhn(to_batch(test_images, args.batch_size, key))
    # train_loader = make_loader_svhn(to_batch(train_images, 16, key))

    batch = next(test_loader)
    model = celeba_vae(z_dim=64) if args.model == "celeba_vae" else fmnist_vae(z_dim=64)
    key, key2, reparam_key, init_key = jax.random.split(key, num=4)
    image_shape = batch["image"].shape[1:]
    print(f"Image shape: {image_shape}")
    loss = PRCLoss(image_shape=image_shape)

    original_checkpoint = deepcopy(info)
    # %%
    # Use MLE weights and eval
    state_mle, model_fn, model_unflatten = eval.prepare_from_checkpoint_generator(
        model, mle_checkpoint=info["checkpoints"]["mle"]
    )
    D = len(state_mle.params["param_nn"])
    print(f"Number of parameters: {D:,d}")
    model_fn = jax.jit(model_fn)

    state_mle_save = eval.EvalState(
        apply_fn=model.apply,
        params=model_unflatten(state_mle.params["param_nn"]),
        batch_stats=None,
        key=original_checkpoint["checkpoints"]["mle"].get("key"),
        opt_state=original_checkpoint["checkpoints"]["mle"].get("opt_state"),
        tx=None,
    )

    images_in = next(test_loader)["image"]
    os.makedirs(args.output_dir, exist_ok=True)

    print("*** Running evaluation on MLE")
    # images = eval.reconstruct_and_plot(
    #     images_in,
    #     state_mle.params["param_nn"],
    #     model_fn,
    #     save_path=os.path.join(args.output_dir, "mle_reconstructions.png"),
    # )

    # %%
    # Load ELBO-optimized weights and eval
    state, model_fn, model_unflatten = eval.prepare_from_checkpoint_generator(
        model,
        checkpoint_decoder=info["checkpoints"]["elbo"],
        mle_checkpoint=original_checkpoint["checkpoints"]["mle"],
    )
    model_fn = jax.jit(model_fn)

    print("*** Running evaluation on ELBO checkpoint")
    # images = eval.reconstruct_and_plot(
    #     images_in,
    #     state.params["param_nn"],
    #     model_fn,
    #     save_path=os.path.join(args.output_dir, "elbo_reconstructions.png"),
    # )

    # %%
    # Setup alternating projections

    encoding_apply = apply_fn_from_state(state=state_mle_save, train=False)

    @jax.jit
    def encoding_func(image):
        return encoding_apply(state_mle_save.params, image)[0][0]

    def make_latent_loader(batched_array):
        for batch in batched_array:
            encoding = encoding_func(batch)
            yield {"image": encoding, "label": batch}

    # Load ELBO-optimized weights and eval (decoder only)
    decoder_model = CelebADecoder() if args.model == "celeba_vae" else FmnistDecoder()
    state_decoder, model_fn_decoder, model_unflatten_decoder = (
        eval.prepare_from_checkpoint_generator(
            decoder_model,
            checkpoint_decoder=original_checkpoint["checkpoints"]["elbo"],
            mle_checkpoint=None,
        )
    )
    model_fn_decoder = jax.jit(model_fn_decoder)

    D = len(state_decoder.params["param_nn"])
    print(f"Number of parameters: {D:,d}")
    loss = PRCLoss(image_shape=image_shape, elbo=True)
    # projection = mrvn.projection_kernel_ggn(model_apply_fn, model_unflatten)

    if args.posterior_samples is not None:
        with open(args.posterior_samples, "rb") as f:
            posterior_samples = pickle.load(f)
        posterior_samples = jnp.array(posterior_samples)
    else:
        projection = viking.projection_state_kernel_param_to_loss(
            model_unflatten_decoder, loss, use_custom_vjp=config.custom_vjp
        )
        alt_projections = viking.make_state_alternating_projections_from_iterator(
            make_latent_loader(to_batch(train_images, 16, key2)), projection
        )
        print("*** Computing alternating projections...")
        log_scale_kernel = -state_decoder.params["log_precision"] / 2
        log_scale_image = state_decoder.params["log_scale_image"]
        key, iso_key = jax.random.split(key)
        iso_samples = jax.random.normal(iso_key, (args.num_mc_samples, D))
        kernel_samples = alt_projections(
            state_decoder, iso_samples, args.num_alt_proj_iter
        )
        image_samples = iso_samples - kernel_samples  # shape is (samples_mc, D)
        posterior_samples = (
            state_decoder.params["param_nn"][None, ...]
            + jnp.exp(log_scale_kernel) * kernel_samples
            + jnp.exp(log_scale_image) * image_samples
        )
        with open(os.path.join(args.output_dir, "posterior_samples.pkl"), "wb") as f:
            pickle.dump(posterior_samples, f)

        # posterior_samples_more_var = (
        #     state_decoder.params["param_nn"][None, ...]
        #     + 10 * jnp.exp(log_scale_kernel) * kernel_samples
        #     + jnp.exp(log_scale_image) * image_samples
        # )

    print(f"*** Running evaluation on {args.num_mc_samples} posterior samples")
    images_sampled_models_test = eval.reconstruct_and_plot_sampled_models(
        encoding_func(images_in),
        posterior_samples,
        model_fn_decoder,
        save_path=os.path.join(args.output_dir, "sampling_reconstructions.png"),
    )

    images_sampled_models_prior = eval.reconstruct_and_plot_sampled_models(
        jax.random.normal(reparam_key, (args.batch_size, 64)) * 1.5,
        posterior_samples,
        model_fn_decoder,
        save_path=os.path.join(args.output_dir, "sampling_generated.png"),
    )
    # save the posterior samples in pickle format
    with open(
        os.path.join(args.output_dir, "images_sampled_models_test.pkl"), "wb"
    ) as f:
        pickle.dump(images_sampled_models_test, f)
    # save the original checkpoint in pickle format
    with open(
        os.path.join(args.output_dir, "images_sampled_models_prior.pkl"), "wb"
    ) as f:
        pickle.dump(images_sampled_models_prior, f)

    # print(f"*** Running evaluation on {args.num_mc_samples} posterior samples with higher sigmas")

    # images_sampled_models_test = eval.reconstruct_and_plot_sampled_models(
    #     encoding_func(images_in),
    #     posterior_samples_more_var,
    #     model_fn_decoder,
    #     save_path=os.path.join(args.output_dir, "sampling_reconstructions_high_sigmas.png"),
    # )

    # images_sampled_models_prior = eval.reconstruct_and_plot_sampled_models(
    #     jax.random.normal(reparam_key, (args.batch_size, 64))*1.5,
    #     posterior_samples_more_var,
    #     model_fn_decoder,
    #     save_path=os.path.join(args.output_dir, "sampling_generated_high_sigmas.png"),
    # )
    #     # save the posterior samples in pickle format
    # with open(os.path.join(args.output_dir, "images_sampled_models_test_original.pkl"), "wb") as f:
    #     pickle.dump(images_sampled_models_test, f)
    # # save the original checkpoint in pickle format
    # with open(os.path.join(args.output_dir, "images_sampled_models_prior_higher_sigmas.pkl"), "wb") as f:
    #     pickle.dump(images_sampled_models_prior, f)

    deviations_corrupted = []
    deviations_vals_corrupted = []
    for batch in loader_samples_from_prior(key, scale=2.0, num=500):
        deviations_batch, deviations_vals_batch = eval.get_scalar_deviation(
            batch["image"], posterior_samples, model_fn_decoder
        )
        deviations_corrupted.append(deviations_batch)
        deviations_vals_corrupted.append(deviations_vals_batch)
    deviations_corrupted = jnp.concatenate(deviations_corrupted, axis=0)
    deviations_vals_corrupted = np.array(
        jnp.concatenate(deviations_vals_corrupted, axis=0)
    )
    with open(os.path.join(args.output_dir, "deviations_corrupted.pkl"), "wb") as f:
        pickle.dump(deviations_corrupted, f)
    with open(
        os.path.join(args.output_dir, "deviations_vals_corrupted.pkl"), "wb"
    ) as f:
        pickle.dump(deviations_vals_corrupted, f)

    deviations_in_distrib = []
    deviations_vals_in_distrib = []
    for batch in loader_samples_from_prior(key, scale=1.0, num=500):
        deviations_batch, deviations_vals_batch = eval.get_scalar_deviation(
            batch["image"], posterior_samples, model_fn_decoder
        )
        deviations_in_distrib.append(deviations_batch)
        deviations_vals_in_distrib.append(deviations_vals_batch)
    deviations_in_distrib = jnp.concatenate(deviations_in_distrib, axis=0)
    deviations_vals_in_distrib = np.array(
        jnp.concatenate(deviations_vals_in_distrib, axis=0)
    )
    with open(os.path.join(args.output_dir, "deviations_in_distrib.pkl"), "wb") as f:
        pickle.dump(deviations_in_distrib, f)
    with open(
        os.path.join(args.output_dir, "deviations_vals_in_distrib.pkl"), "wb"
    ) as f:
        pickle.dump(deviations_vals_in_distrib, f)

    # plot histograms of the deviations
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.title("Deviations of corrupted and in-distribution samples")
    plt.xlabel("Deviations")
    plt.ylabel("Frequency")
    plt.hist(
        np.array(deviations_corrupted),
        bins=200,
        alpha=0.5,
        label="Corrupted",
        density=True,
    )
    plt.hist(
        np.array(deviations_in_distrib),
        bins=200,
        alpha=0.5,
        label="In-distribution",
        density=True,
    )
    plt.legend()
    # xlim
    plt.xlim(0, 0.2)
    plt.savefig(os.path.join(args.output_dir, "deviations_histogram.png"))
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Seed for PRNGs")
    parser.add_argument(
        "log",
        help="Path to log directory",
    )
    parser.add_argument(
        "--num-mc-samples",
        default=100,
        type=int,
        help="Num. of MC samples for ELBO expectation (reconstruction) term",
    )
    parser.add_argument(
        "--num-alt-proj-iter",
        default=1,
        type=int,
        help="Num. of alternating projection iterations",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Override batch size",
    )
    parser.add_argument(
        "--log-precision", default=None, type=float, help="Override log-precision value"
    )
    parser.add_argument(
        "--log-scale-image",
        default=None,
        type=float,
        help="Override log-scale of image value",
    )
    parser.add_argument("--output-dir", default="outputs", type=str)
    parser.add_argument(
        "--dataset-root",
        default="data",
        type=str,
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--dataset",
        default="celeba",
        type=str,
        choices=["celeba", "fmnist", "svhn"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--model",
        default="celeba_vae",
        type=str,
        choices=["celeba_vae", "fmnist_vae"],
        help="Model to use",
    )
    parser.add_argument(
        "--posterior-samples",
        default=None,
        type=str,
        help="Path to posterior samples",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
