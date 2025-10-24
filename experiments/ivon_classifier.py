import argparse
import json
import os
from typing import Any

import flax
import flax.training.train_state
import jax
import jax.numpy as jnp
import optax
import tqdm

from vp import eval, ivon, models, utils, viking
from vp.data import all_datasets as datasets
from vp.data.utils import get_output_dim
from vp.models.configs import get_model_hyperparams


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

    # %%
    # Data loader(s) setup
    train_loader, val_loader, _ = datasets.get_dataloaders(
        args.dataset,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        purp="train",
        seed=args.seed,
    )

    # %%
    # Model setup
    # model_init_fn, model_apply_fn = models.make_conv_classifier(
    #     num_classes=batch["label"].shape[-1]
    # )
    batch = next(iter(train_loader))
    num_classes = get_output_dim(args.dataset)
    model_hparams = get_model_hyperparams(num_classes, args.model)
    model_class = models.MODELS_DICT[args.model]
    model = model_class(**model_hparams)

    # %%
    # Model initialization
    key, model_init_key, dropout_key = jax.random.split(key, num=3)
    variables = model.init(model_init_key, batch["image"])
    model_params = variables["params"]
    batch_stats = variables.get("batch_stats")
    param_nn, model_unflatten = jax.flatten_util.ravel_pytree(model_params)
    loss = optax.losses.safe_softmax_cross_entropy
    D = len(param_nn)
    print(f"Number of parameters: {D:,d}")

    # %%
    # IVON optimization
    # FIXME: `ess` will be off by whatever amount of data the last
    # batch is missing (should be a small quantity)
    loss_mle = viking.make_state_loss(model_unflatten, loss_single=loss)
    optimizer = ivon.ivon(args.lr, ess=len(train_loader) * batch["label"].shape[0])
    train_step = viking.make_train_state_step_ivon(
        loss_mle, num_samples=args.num_mc_samples
    )
    train_step = jax.jit(train_step)
    state = TrainState.create(
        apply_fn=model.apply,
        params={
            "param_nn": param_nn,
        },
        batch_stats=batch_stats,
        key=dropout_key if args.model in models.MODELS_WITH_DROPOUT else None,
        tx=optimizer,
    )
    print(f"-- IVON training for {args.num_epochs} epoch(s)")
    with tqdm.trange(args.num_epochs) as progressbar:
        for step in progressbar:
            meters = utils.MeterCollection("loss", "acc")
            for batch in train_loader:
                key, subkey = jax.random.split(key)
                state, stats, key = train_step(
                    state, batch["image"], batch["label"], subkey
                )
                acc = jnp.mean(
                    jnp.argmax(stats["preds"], axis=-1)
                    == jnp.argmax(batch["label"], axis=-1)
                )
                meters.update(loss=stats["loss"].item(), acc=acc.item())
                progressbar.set_description(str(meters))
            progressbar.write(f"[{step:05d}] {meters}")
            run.log(meters.summary_dict(), name="mle")
    checkpoint = state.to_dict(model_unflatten)
    run.checkpoint(checkpoint, name="mle")
    state, model_fn, model_unflatten = eval.prepare_from_checkpoint(model, checkpoint)
    model_fn = jax.jit(model_fn)

    print(f"-- Sampling {args.val_mc_samples} posterior samples")

    def ivon_sample(s, key):
        params, s = ivon.sample_parameters(key, state.params, s)
        return s, params

    sampling_keys = jax.random.split(key, num=args.val_mc_samples + 1)
    opt_state, _ = ivon_sample(state.opt_state, sampling_keys[0])
    _, posterior_samples = jax.lax.scan(ivon_sample, opt_state, sampling_keys[1:])
    posterior_samples = posterior_samples["param_nn"]

    print(f"-- Running validation on {args.val_mc_samples} posterior samples")
    metrics, *_ = eval.evaluate(
        val_loader, posterior_samples, model_fn, num_classes=num_classes
    )
    with open(os.path.join(log_path, "validation.json"), "w") as f:
        json.dump(metrics, f)


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
        "--model", default="LeNet", choices=list(models.MODELS_DICT.keys())
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
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
