import argparse
import os
from functools import partial
from typing import Any

import flax
import flax.training.train_state
import jax
import jax.numpy as jnp
import optax
import tqdm

from vp import eval, utils, viking
from vp.data.arrays_data import files, files_celeba, get_arrays, make_loader, to_batch
from vp.losses import PRCLoss
from vp.models.celeba_linen import VAE as celeba_vae
from vp.models.celeba_linen import CelebADecoder
from vp.models.fmnist_vae_linen import VAE as fmnist_vae
from vp.models.fmnist_vae_linen import FmnistDecoder
from vp.viking import apply_fn_from_state


class TrainState(flax.training.train_state.TrainState):
    batch_stats: Any
    key: Any

    def apply_updates(self, *, grads, updates):
        state = self.apply_gradients(grads=grads)
        if "batch_stats" in updates:
            state = state.replace(batch_stats=updates["batch_stats"])
        if self.key is not None:
            key = jax.random.fold_in(key=state.key, data=state.step)
            state = state.replace(key=key)
        return state

    def to_dict(self, model_unflatten):
        params = self.params.copy()
        params["param_nn"] = model_unflatten(params["param_nn"])
        return {
            "params": params,
            "batch_stats": self.batch_stats,
            "key": self.key,
        }


def main(args):
    print(f"Starting log for run {args.name}")
    if args.log is not None:
        print(f"Loading log from {args.log}")
        if not os.path.exists(args.log):
            print(f"{args.log}: Does not exist, aborting")
            exit(1)
        if args.output_dir is None:
            args.output_dir = os.path.join("outputs", args.dataset, args.model)
        log_path = os.path.join(args.output_dir, args.name)
        if os.path.exists(log_path):
            if args.exist_ok:
                print(f"WARNING: Overwriting existing log: {log_path}")
            else:
                print(f"{log_path}: Exists, aborting")
                exit(1)
    else:
        if args.output_dir is None:
            args.output_dir = os.path.join("outputs", args.dataset, args.model)
        log_path = os.path.join(args.output_dir, args.name)
        if os.path.exists(log_path):
            if args.exist_ok:
                print(f"WARNING: Overwriting existing log: {log_path}")
            else:
                print(f"{log_path}: Exists, aborting")
                exit(1)
    run = utils.RunLog(log_path, config=vars(args))
    key = jax.random.PRNGKey(seed=args.seed)

    # %%
    train_images, _, _, _, _, _ = get_arrays(
        args.dataset_root,
        (files_celeba if args.dataset == "celeba" else files),
    )

    train_images = jax.device_put(train_images)
    # val_images = jax.device_put(val_images)

    train_loader = make_loader(to_batch(train_images, args.batch_size, key))

    batch = next(train_loader)
    model = celeba_vae(z_dim=64) if args.model == "celeba_vae" else fmnist_vae(z_dim=64)

    # %%
    # Model initialization
    key, model_init_key, reparam_key, init_key = jax.random.split(key, num=4)
    image_shape = batch["image"].shape[1:]
    print(f"Image shape: {image_shape}")
    loss = PRCLoss(image_shape=image_shape)

    if args.log is None:
        variables = model.init(model_init_key, batch["image"], train=True)

        model_params = variables["params"]
        batch_stats = variables.get("batch_stats")
        param_nn, model_unflatten = jax.flatten_util.ravel_pytree(model_params)

        D = len(param_nn)
        print(f"Number of parameters: {D:,d}")

    if args.log is not None:
        info = utils.load_log(args.log)
        state, model_fn, model_unflatten = eval.prepare_from_checkpoint(
            model, info["checkpoints"]["mle"]
        )
        param_nn = state.params["param_nn"]
        batch_stats = state.batch_stats
        D = len(state.params["param_nn"])
        print(f"Number of parameters: {D:,d}")

    # %%
    # A few steps of MLE optimization
    loss_mle = viking.make_state_loss(model_unflatten, loss_single=loss)

    scheduler_mle = optax.warmup_exponential_decay_schedule(
        decay_rate=0.5,
        init_value=0.0,
        peak_value=args.lr_mle,
        transition_steps=100_000,
        warmup_steps=1_000,
    )
    optimizer_mle = optax.chain(optax.adamax(learning_rate=scheduler_mle))

    train_mle_step = viking.make_train_state_mle_step(loss_mle)
    train_mle_step = jax.jit(train_mle_step)
    state_mle = TrainState.create(
        apply_fn=model.apply,
        params={
            "param_nn": param_nn,
        },
        batch_stats=batch_stats,
        key=reparam_key,
        tx=optimizer_mle,
    )

    if args.log is None:
        with tqdm.trange(args.num_mle_epochs) as progressbar:
            for step_mle in progressbar:
                meters = utils.MeterCollection("loss")
                for batch in make_loader(to_batch(train_images, args.batch_size, key)):
                    state_mle, stats = train_mle_step(
                        state_mle, batch["image"], batch["label"]
                    )
                    meters.update(loss=stats["loss"].item())
                    progressbar.set_description(str(meters))
                progressbar.write(f"[{step_mle:05d}] {meters}")
                run.log(meters.summary_dict(), name="mle")
        run.checkpoint(state_mle.to_dict(model_unflatten), name="mle")

    def encoding_func(image, state):
        return apply_fn_from_state(state, False)(state.params["param_nn"], image)[0][0]

    encoding_func = partial(encoding_func, state=state_mle)

    def make_latent_loader(batched_array):
        for batch in batched_array:
            encoding = encoding_func(batch)
            yield {"image": encoding, "label": batch}

    decoder = CelebADecoder() if args.model == "celeba_vae" else FmnistDecoder()
    key1, key2, key3, key = jax.random.split(key, num=4)

    params = state_mle.params
    params["param_nn"] = model_unflatten(params["param_nn"])

    param_nn, model_unflatten = jax.flatten_util.ravel_pytree(
        params["param_nn"]["decoder"]
    )
    D = len(param_nn)
    print(f"Number of parameters: {D:,d}")
    loss = PRCLoss(image_shape=image_shape, elbo=True)

    # %%
    # Alternating projections setup. Used by sigmas optimisation and
    # full model optimisation.
    # projection = mrvn.projection_kernel_ggn(model_apply_fn, model_unflatten)
    projection = viking.projection_state_kernel_param_to_loss(
        model_unflatten, loss, use_custom_vjp=args.custom_vjp
    )
    if args.alt_proj:
        alt_projections = viking.make_state_alternating_projections_from_iterator(
            make_latent_loader(to_batch(train_images, 16, key1)), projection
        )
        # The following can be very expensive, it is better to call jax.jit()
        # somewhere inside the function that creates `alt_projections`:
        # alt_projections = jax.jit(alt_projections)

    # %%
    # Find best sigmas for the current maximum likelihood estimate
    optimizer_sigmas = optax.multi_transform(
        {
            "frozen": optax.set_to_zero(),
            "train": optax.adam(args.lr),
        },
        {
            "param_nn": "frozen",
            "log_precision": "train",
            "log_scale_image": "train",
        },
    )
    state_sigmas = TrainState.create(
        apply_fn=decoder.apply,
        params={
            "param_nn": param_nn,
            "log_precision": jnp.array(args.log_precision),
            "log_scale_image": jnp.array(args.log_scale_image),
        },
        batch_stats=batch_stats,
        key=reparam_key,
        tx=optimizer_sigmas,
    )
    if args.alt_proj:
        alt_projections = viking.make_state_alternating_projections_from_iterator(
            make_latent_loader(to_batch(train_images, 16, key2)), projection
        )
    expectation = viking.state_elbo_expectation(
        model_unflatten, loss_single=loss, is_linearized=args.linearized
    )
    loss_elbo_sigmas = viking.make_state_elbo(expectation=expectation)
    loss_elbo_sigmas_and_grad = jax.value_and_grad(
        loss_elbo_sigmas, argnums=0, has_aux=True
    )

    @jax.jit
    def train_step_sigmas(state, **kwargs):
        (loss_value, stats), loss_grads = loss_elbo_sigmas_and_grad(
            state.params,
            state,
            projection=projection,
            **kwargs,
        )
        updates = stats.pop("updates")
        state = state.apply_updates(grads=loss_grads, updates=updates)
        return state, stats

    print(f"-- Fitting optimal sigmas for {args.num_sigmas_epochs} epoch(s)")
    # Instead of one MC sample per epoch, here we take "number of
    # epochs" samples and use them across optimisation
    key, iso_key = jax.random.split(key)
    iso_samples = jax.random.normal(iso_key, (args.num_mc_samples, D))
    if args.alt_proj and args.num_sigmas_epochs > 0:
        print("Alternating projections...")
        iso_samples = alt_projections(state_sigmas, iso_samples, args.num_alt_proj_iter)
    with tqdm.trange(args.num_sigmas_epochs) as progressbar:
        for step_sigmas_elbo in progressbar:
            meters = utils.MeterCollection("rec", "kl", "R", "σ_ker", "σ_im", "dot")
            for batch in make_latent_loader(to_batch(train_images, 8, key)):
                key, eps_key = jax.random.split(key)
                eps_samples = jax.random.normal(eps_key, (args.num_sigmas_epochs, D))
                effective_samples = (
                    jnp.sqrt(args.gamma) * iso_samples
                    + jnp.sqrt(1.0 - args.gamma) * eps_samples
                )
                state_sigmas, stats = train_step_sigmas(
                    state_sigmas,
                    x=batch["image"],
                    y=batch["label"],
                    iso_samples=effective_samples,
                    beta=args.beta,
                )
                iso_samples = stats["kernel_samples"]
                meters.update(
                    rec=stats["E[]"].item(),
                    kl=stats["kl"].item(),
                    R=stats["R"].item() / D,
                    σ_ker=jnp.exp(-0.5 * state_sigmas.params["log_precision"]).item(),
                    σ_im=jnp.exp(state_sigmas.params["log_scale_image"]).item(),
                    dot=stats["dot"].item(),
                )
                progressbar.set_description(str(meters))
            progressbar.write(f"[{step_sigmas_elbo:05d}] {meters}")
    run.checkpoint(state_sigmas.to_dict(model_unflatten), name="sigmas")

    # %%
    # Setup for ELBO optimization

    state, model_fn, model_unflatten = eval.prepare_from_checkpoint_generator(
        model, checkpoint_decoder=info["checkpoints"]["elbo"], mle_checkpoint=None
    )

    scheduler_elbo = optax.warmup_exponential_decay_schedule(
        decay_rate=0.5,
        init_value=0.0,
        peak_value=args.lr,
        transition_steps=50_000,
        warmup_steps=1_000,
    )
    optimizer_elbo = optax.chain(optax.adamax(learning_rate=scheduler_elbo))
    if args.adaptive_grad_clip is not None:
        optimizer_elbo = optax.chain(
            optax.adaptive_grad_clip(args.adaptive_grad_clip),
            optimizer_elbo,
        )
    state_elbo = TrainState.create(
        apply_fn=decoder.apply,
        params=state.params,
        batch_stats=state.batch_stats,
        key=state.key,
        tx=optimizer_elbo,
    )
    loss_elbo = viking.make_state_elbo(expectation=expectation)
    train_elbo_step = viking.make_train_state_elbo_step(loss_elbo)
    train_elbo_step = jax.jit(train_elbo_step, static_argnums=0)

    print(f"-- ELBO optim. with log_precision={state_elbo.params['log_precision']:.2f}")
    with tqdm.trange(args.num_epochs) as progressbar:
        for step_elbo in progressbar:
            try:
                key, iso_key = jax.random.split(key)
                iso_samples = jax.random.normal(iso_key, (args.num_mc_samples, D))
                if args.alt_proj:
                    progressbar.set_description("(Alternating projections)")
                    iso_samples = alt_projections(
                        state_elbo, iso_samples, args.num_alt_proj_iter
                    )
                meters = utils.MeterCollection(
                    "rec",
                    "kl",
                    "R",
                    "σ_ker",
                    "σ_im",
                    "dot",
                    "res",
                    "pmin",
                    "pmax",
                )
                for batch in make_latent_loader(to_batch(train_images, 8, key)):
                    key, iso_key = jax.random.split(key)
                    eps_samples = jax.random.normal(iso_key, iso_samples.shape)
                    effective_samples = (
                        jnp.sqrt(args.gamma) * iso_samples
                        + jnp.sqrt(1.0 - args.gamma) * eps_samples
                    )
                    state_elbo, stats = train_elbo_step(
                        projection,
                        state_elbo,
                        batch["image"],
                        batch["label"],
                        effective_samples,
                        beta=args.beta,
                    )
                    if args.alt_proj:
                        iso_samples = stats["kernel_samples"]
                    meters.update(
                        rec=stats["E[]"].item(),
                        kl=stats["kl"].item(),
                        R=stats["R"].item() / D,
                        σ_ker=jnp.exp(-0.5 * state_elbo.params["log_precision"]).item(),
                        σ_im=jnp.exp(state_elbo.params["log_scale_image"]).item(),
                        dot=stats["dot"].item(),
                        res=stats["residuals"].item(),
                        pmin=stats["precond_min"].item(),
                        pmax=stats["precond_max"].item(),
                    )
                    progressbar.set_description(str(meters))
                if step_elbo % args.print_every == 0:
                    progressbar.write(f"[{step_elbo:05d}] {meters}")
                run.log(meters.summary_dict(), name="elbo")
            except KeyboardInterrupt:
                break
    checkpoint = state_elbo.to_dict(model_unflatten)
    run.checkpoint(checkpoint, name="elbo")


def parse_args():
    def type_maybe(type):
        def parse(value):
            if value.lower() == "none":
                return None
            return type(value)

        return parse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default=utils.make_random_name(),
        help="Name to use when logging results",
    )
    parser.add_argument("--seed", default=42, type=int, help="Seed for PRNGs")
    parser.add_argument(
        "--dataset",
        default="celeba",
        help="Dataset to use",
        choices=["celeba", "fmnist"],
    )
    parser.add_argument(
        "--dataset_root", default="data", type=str, help="Path to dataset root"
    )
    parser.add_argument(
        "--model", default="celeba_vae", choices=["celeba_vae", "fmnist_vae"]
    )
    parser.add_argument(
        "--beta",
        default=1e-2,
        type=float,
        help="This term rescales the ELBO expectation (reconstruction) term",
    )
    parser.add_argument(
        "--gamma",
        default=0.2,
        type=type_maybe(float),
        help="When set (between 0.0 and 1.0), how much noise to add to projected samples",
    )
    parser.add_argument(
        "--num-mle-epochs", default=5, type=int, help="Num. of MLE training epochs"
    )
    parser.add_argument(
        "--num-sigmas-epochs", default=5, type=int, help="Num. of sigmas warmup epochs"
    )
    parser.add_argument(
        "--num-epochs", default=50, type=int, help="Num. of training epochs"
    )
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument(
        "--lr-mle", default=1e-3, type=float, help="Learning rate for MLE training"
    )
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument(
        "--adaptive-grad-clip",
        default=None,
        type=type_maybe(float),
        help="Adaptive gradient clipping term",
    )
    parser.add_argument(
        "--num-mc-samples",
        default=1,
        type=int,
        help="Num. of MC samples for ELBO expectation (reconstruction) term",
    )
    parser.add_argument(
        "--val-mc-samples",
        default=20,
        type=int,
        help="Num. of MC samples for ELBO expectation term used in validation",
    )
    parser.add_argument(
        "--alt-proj",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use alternating projections?",
    )
    parser.add_argument(
        "--num-alt-proj-iter",
        default=1,
        type=int,
        help="Num. of alternating projection iterations (very expensive to change)",
    )
    parser.add_argument(
        "--custom-vjp",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use custom vjp on projection step?",
    )
    parser.add_argument(
        "--linearized",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use linearized predictive?",
    )
    parser.add_argument(
        "--log-precision",
        default=4.0,
        type=float,
        help="Initial log-precision (affects kernel space log-scale)",
    )
    parser.add_argument(
        "--log-scale-image",
        default=-2.0,
        type=float,
        help="Initial log-scale associated with the image space",
    )
    parser.add_argument(
        "--print-every",
        default=1,
        type=int,
        help="Print a log line every this many epochs",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--log",
        help="Path to log directory",
        default=None,
    )
    parser.add_argument(
        "--exist-ok",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Allow overwriting saved files?",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
