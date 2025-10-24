import argparse
import functools as ft
import json
import os
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
import tqdm
from flax import struct

import sgmcmc
from vp import eval, models, utils, viking
from vp.data import all_datasets as datasets
from vp.data.utils import get_output_dim
from vp.models.configs import get_model_hyperparams


@ft.partial(
    jax.tree_util.register_dataclass,
    meta_fields=["apply_fn", "update_fn"],
    data_fields=["step", "params", "opt_state", "batch_stats", "key"],
)
class TrainState(NamedTuple):
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    update_fn: Callable = struct.field(pytree_node=False)
    params: dict = struct.field(pytree_node=True)
    opt_state: Any = struct.field(pytree_node=True)
    batch_stats: Any = None
    key: Any = None

    @classmethod
    def create(cls, *, apply_fn, update_fn, params, opt_state, **kwargs):
        return cls(
            step=0,
            apply_fn=apply_fn,
            update_fn=update_fn,
            params=params,
            opt_state=opt_state,
            **kwargs,
        )

    def apply_updates(self, *, grads, updates, key, **kwargs):
        param_updates, opt_state = self.update_fn(grads, self.opt_state)
        params = optax.apply_updates(self.params, param_updates)
        self_key = None
        if self.key is not None:
            self_key = jax.random.fold_in(key=self.key, data=self.step)
        return self._replace(
            step=self.step + 1,
            params=params,
            opt_state=opt_state,
            batch_stats=updates.get("batch_stats"),
            key=self_key,
            **kwargs,
        )

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
        args.output_dir = os.path.join("outputs", "SGMC", args.dataset, args.model)
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
    D = len(param_nn)
    print(f"Number of parameters: {D:,d}")

    # %%
    # Optimizer setup
    final_dt = args.final_dt or args.dt
    burnin_steps = len(train_loader) * args.num_burnin_epochs
    lr_schedule = make_constant_lr_schedule_with_cosine_burnin(
        args.dt, final_dt, burnin_steps
    )
    preconditioner = None
    if args.preconditioner == "RMSprop":
        preconditioner = sgmcmc.get_rmsprop_preconditioner()
    optimizer = sgmcmc.sgld_gradient_update(
        lr_schedule,
        momentum_decay=args.momentum_decay,
        seed=args.seed + 1,
        preconditioner=preconditioner,
    )

    # %%
    # Optimization
    log_prior_fn = make_gaussian_log_prior(args.weight_decay)
    log_likelihood_fn = viking.make_state_loss(
        model_unflatten,
        loss_single=lambda logits, labels: labels * jax.nn.log_softmax(logits),
        reduction_fn=jnp.sum,
    )
    train_step = viking.make_train_state_step_sgmc(
        log_likelihood_fn, log_prior_fn, len(train_loader)
    )
    train_step = jax.jit(train_step)
    state = TrainState.create(
        apply_fn=model.apply,
        update_fn=optimizer.update,
        params={
            "param_nn": param_nn,
        },
        opt_state=optimizer.init({"param_nn": param_nn}),
        batch_stats=batch_stats,
        key=dropout_key if args.model in models.MODELS_WITH_DROPOUT else None,
    )
    print(f"-- SGMC training for {args.num_epochs} epoch(s)")
    posterior_samples = []
    with tqdm.trange(args.num_epochs) as progressbar:
        for step in progressbar:
            meters = utils.MeterCollection("log_likelihood", "log_prior", "acc")
            for batch in train_loader:
                key, subkey = jax.random.split(key)
                state, stats, key = train_step(
                    state, batch["image"], batch["label"], subkey
                )
                stats["acc"] = jnp.mean(
                    jnp.argmax(stats["preds"], axis=-1)
                    == jnp.argmax(batch["label"], axis=-1)
                )
                meters.update(
                    log_likelihood=stats["log_likelihood"].item(),
                    log_prior=stats["log_prior"].item(),
                    acc=stats["acc"].item(),
                )
                progressbar.set_description(str(meters))
            progressbar.write(f"[{step:05d}] {meters}")
            run.log(meters.summary_dict(), name="sgmcmc")

            if (
                step > args.num_burnin_epochs
                and (step - args.num_burnin_epochs + 1) % args.collect_every == 0
            ) or step + 1 == args.num_epochs:
                posterior_samples.append(state.params["param_nn"])
                sample_num = len(posterior_samples)
                name = f"pos-{sample_num}"
                checkpoint = state.to_dict(model_unflatten)
                run.checkpoint(checkpoint, name=name)

    _, model_fn, model_unflatten = eval.prepare_from_checkpoint(model, checkpoint)
    model_fn = jax.jit(model_fn)
    posterior_samples = jnp.stack(posterior_samples)
    print(f"-- Running validation on {len(posterior_samples)} posterior samples")
    metrics, *_ = eval.evaluate(
        val_loader, posterior_samples, model_fn, num_classes=num_classes
    )
    with open(os.path.join(log_path, "validation.json"), "w") as f:
        json.dump(metrics, f)


def make_constant_lr_schedule_with_cosine_burnin(init_lr, final_lr, burnin_steps):
    """Cosine LR schedule with burn-in for SG-MCMC."""

    def schedule(step):
        t = jnp.minimum(step / burnin_steps, 1.0)
        coef = (1 + jnp.cos(t * jnp.pi)) * 0.5
        return coef * init_lr + (1 - coef) * final_lr

    return schedule


def make_gaussian_log_prior(weight_decay):
    """Returns the Gaussian log-density given weight decay."""

    def log_prior(params):
        """Computes the Gaussian prior log-density."""
        num_params = sum([p.size for p in jax.tree.leaves(params)])
        params = params["param_nn"]
        log_prob = -(
            0.5 * jnp.dot(params, params) * weight_decay
            + 0.5 * num_params * jnp.log((2 * jnp.pi) / weight_decay)
        )
        return log_prob

    return log_prior


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
    parser.add_argument("--batch-size", default=80, type=int)
    parser.add_argument(
        "--num-epochs", default=300, type=int, help="Num. of SGMCMC epochs"
    )
    parser.add_argument(
        "--collect-every",
        default=50,
        type=int,
        help="Collect a posterior sample every this many epochs",
    )
    parser.add_argument(
        "--weight-decay",
        default=15.0,
        type=float,
        help="Weight decay, equivalent to setting prior std.",
    )
    parser.add_argument("--dt", default=1e-6, type=float, help="SGMCMC step size")
    parser.add_argument("--final-dt", default=None, type=float, help="SGMCMC step size")
    parser.add_argument(
        "--momentum-decay", default=0.9, type=float, help="Momentum decay for SGD"
    )
    parser.add_argument(
        "--preconditioner",
        default="None",
        type=str,
        choices=("None", "RMSprop"),
        help="Choice of preconditioner",
    )
    parser.add_argument(
        "--num-burnin-epochs",
        default=50,
        type=int,
        help="Number of epochs before final learning rate",
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
