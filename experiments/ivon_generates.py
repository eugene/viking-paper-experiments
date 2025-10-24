import argparse
import os
from typing import Any

import flax
import flax.training.train_state
import jax
import tqdm

from vp import eval, ivon, utils, viking
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
            "opt_state": self.opt_state,
        }


def main(args):
    print(f"Starting log for run {args.name}")
    if args.output_dir is None:
        args.output_dir = os.path.join("outputs", "IVON", args.dataset, args.model)
    log_path = os.path.join(args.output_dir, args.name)
    if os.path.exists(log_path):
        if args.exist_ok:
            print(f"WARNING: Overwriting existing log: {log_path}")
        else:
            print(f"{log_path}: Exists, aborting")
            exit(1)
    run = utils.RunLog(log_path, config=vars(args))
    key = jax.random.PRNGKey(seed=args.seed)
    key_loader, key = jax.random.split(key, num=2)
    # %%
    # Data loader(s) setup
    train_images, _, _, _, _, _ = get_arrays(
        args.dataset_root,
        (files_celeba if args.dataset == "celeba" else files),
    )

    train_images = jax.device_put(train_images)

    train_loader = make_loader(to_batch(train_images, args.batch_size, key))
    len_train = sum(1 for _ in train_loader)
    train_loader = make_loader(to_batch(train_images, args.batch_size, key))

    batch = next(train_loader)
    model = celeba_vae(z_dim=64) if args.model == "celeba_vae" else fmnist_vae(z_dim=64)
    info = utils.load_log(args.log)
    state, model_fn, model_unflatten = eval.prepare_from_checkpoint(
        model, info["checkpoints"]["mle"]
    )
    param_nn = state.params["param_nn"]
    D = len(state.params["param_nn"])
    image_shape = batch["image"].shape[1:]
    print(f"Image shape: {image_shape}")

    encoder_apply = apply_fn_from_state(state, False)

    @jax.jit
    def encoding_func(image):
        return encoder_apply(state.params["param_nn"], image)[0][0]

    def make_latent_loader(batched_array):
        for batch in batched_array:
            encoding = encoding_func(batch)
            yield {"image": encoding, "label": batch}

    # %%
    decoder = CelebADecoder() if args.model == "celeba_vae" else FmnistDecoder()

    params = state.params
    params["param_nn"] = model_unflatten(params["param_nn"])

    param_nn, model_unflatten = jax.flatten_util.ravel_pytree(
        params["param_nn"]["decoder"]
    )
    D = len(param_nn)
    print(f"Number of parameters: {D:,d}")
    loss = PRCLoss(image_shape=image_shape, elbo=True)
    loss_mle = viking.make_state_loss(model_unflatten, loss_single=loss)

    optimizer = ivon.ivon(args.lr, ess=len_train * batch["image"].shape[0])
    train_step = viking.make_train_state_step_ivon(
        loss_mle, num_samples=args.num_mc_samples
    )
    train_step = jax.jit(train_step)
    state_ivon = TrainState.create(
        apply_fn=decoder.apply,
        params={"param_nn": param_nn},
        batch_stats=state.batch_stats,
        key=state.key,
        tx=optimizer,
    )
    print(f"-- IVON training for {args.num_epochs} epoch(s)")
    with tqdm.trange(args.num_epochs) as progressbar:
        for step in progressbar:
            meters = utils.MeterCollection("loss")
            for batch in make_latent_loader(
                to_batch(train_images, args.batch_size, key_loader)
            ):
                key, key_loader, subkey = jax.random.split(key, 3)
                state_ivon, stats, key = train_step(
                    state_ivon, batch["image"], batch["label"], subkey
                )
                meters.update(loss=stats["loss"].item())
                progressbar.set_description(str(meters))
            progressbar.write(f"[{step:05d}] {meters}")
            run.log(meters.summary_dict(), name="mle")
    checkpoint = state_ivon.to_dict(model_unflatten)
    run.checkpoint(checkpoint, name="decoder")


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
        "--dataset", default="MNIST", help="(classification) Dataset to train on"
    )
    parser.add_argument(
        "--model", default="celeba_vae", choices=["celeba_vae", "fmnist_vae"]
    )
    parser.add_argument(
        "--dataset_root", default="data", type=str, help="Path to dataset root"
    )
    parser.add_argument(
        "--num-epochs", default=50, type=int, help="Num. of training epochs"
    )
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument(
        "--num-mc-samples",
        default=5,
        type=int,
        help="Num. of MC samples for IVON gradients",
    )
    parser.add_argument(
        "--val-mc-samples",
        default=20,
        type=int,
        help="Num. of IVON samples from model used in validation",
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
        "--exist-ok",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Allow overwriting saved files?",
    )
    parser.add_argument(
        "--log",
        help="Path to log directory",
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
