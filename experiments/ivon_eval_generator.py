import argparse
import os
import pickle
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np

from vp import eval, ivon, utils
from vp.data.arrays_data import (
    files,
    files_celeba,
    get_arrays,
    loader_samples_from_prior,
    make_loader_svhn,
    to_batch,
)
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
        "/workspace/svhn",
        files,
    )

    _, _, _, _, test_images_celeba, _ = get_arrays(
        "/workspace/data/celeba",
        files_celeba,
    )
    test_images_celeba = jax.device_put(test_images_celeba)
    test_images = jax.device_put(test_images)
    train_images = jax.device_put(train_images)
    test_loader = make_loader_svhn(to_batch(test_images, 32, key))
    # train_loader = make_loader_svhn(to_batch(train_images, 32, key))

    batch = next(test_loader)

    # %%
    # Model setup
    model = (
        celeba_vae(z_dim=64) if config.model == "celeba_vae" else fmnist_vae(z_dim=64)
    )
    key, key2, reparam_key, init_key = jax.random.split(key, num=4)
    image_shape = batch["image"].shape[1:]
    print(f"Image shape: {image_shape}")

    # %%
    # Use MLE weights and eval
    original_checkpoint = deepcopy(info)
    state, model_fn, model_unflatten = eval.prepare_from_checkpoint_generator(
        model, mle_checkpoint=info["checkpoints"]["mle"]
    )
    D = len(state.params["param_nn"])
    print(f"Number of parameters: {D:,d}")
    model_fn = jax.jit(model_fn)

    state_mle_save = eval.EvalState(
        apply_fn=model.apply,
        params=model_unflatten(state.params["param_nn"]),
        batch_stats=None,
        key=original_checkpoint["checkpoints"]["mle"].get("key"),
        opt_state=original_checkpoint["checkpoints"]["mle"].get("opt_state"),
        tx=None,
    )

    encoding_apply = apply_fn_from_state(state=state_mle_save, train=False)

    @jax.jit
    def encoding_func(image):
        return encoding_apply(state_mle_save.params, image)[0][0]

    def make_latent_loader(batched_array):
        for batch in batched_array:
            encoding = encoding_func(batch)
            yield {"image": encoding, "label": batch}

    images_in = next(test_loader)["image"]
    os.makedirs(args.output_dir, exist_ok=True)
    # model_fn_samples = deepcopy(model_fn)

    decoder_model = CelebADecoder() if config.model == "celeba_vae" else FmnistDecoder()

    state_decoder, model_fn_decoder, model_unflatten_decoder = (
        eval.prepare_from_checkpoint_generator(
            decoder_model,
            checkpoint_decoder=original_checkpoint["checkpoints"]["decoder"],
            mle_checkpoint=None,
        )
    )
    model_fn_decoder = jax.jit(model_fn_decoder)

    print("*** Running evaluation on MLE")
    # images = eval.reconstruct_and_plot(
    #     encoding_func(images_in),
    #     state_decoder.params["param_nn"],
    #     model_fn_decoder,
    #     save_path=os.path.join(args.output_dir, "mle_reconstructions.png"),
    # )
    # %%
    # Setup evaluation
    print(f"*** Sampling {args.num_mc_samples} posterior samples")

    def ivon_sample(s, key):
        params, s = ivon.sample_parameters(key, state_decoder.params, s)
        return s, params

    sampling_keys = jax.random.split(key, num=args.num_mc_samples + 1)
    opt_state, _ = ivon_sample(state_decoder.opt_state, sampling_keys[0])
    _, posterior_samples = jax.lax.scan(ivon_sample, opt_state, sampling_keys[1:])
    posterior_samples = posterior_samples["param_nn"]

    print(f"*** Running evaluation on {args.num_mc_samples} posterior samples")
    images_sampled_models_test = eval.reconstruct_and_plot_sampled_models(
        encoding_func(images_in),
        posterior_samples,
        model_fn_decoder,
        save_path=os.path.join(args.output_dir, "sampling_reconstructions.png"),
    )

    images_sampled_models_prior = eval.reconstruct_and_plot_sampled_models(
        jax.random.normal(reparam_key, (32, 64)),
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
    plt.hist(np.array(deviations_corrupted), bins=200, alpha=0.5, label="Corrupted")
    plt.hist(
        np.array(deviations_in_distrib), bins=200, alpha=0.5, label="In-distribution"
    )
    plt.legend()
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
        "--batch-size",
        default=None,
        type=int,
        help="Override batch size",
    )
    parser.add_argument("--output-dir", default="outputs", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
