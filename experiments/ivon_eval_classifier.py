import argparse
import json
import os

import jax

from vp import eval, ivon, models, utils
from vp.data import all_datasets as datasets
from vp.data.utils import get_output_dim
from vp.models.configs import get_model_hyperparams


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
    # NOTE: val_batch_size determines batch size for test loader as well
    batch_size = maybe_override("batch_size", config.batch_size, args.batch_size)
    train_loader, val_loader, test_loader = datasets.get_dataloaders(
        config.dataset,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        purp="sample",
        seed=args.seed,
    )

    # %%
    # Model setup
    num_classes = get_output_dim(config.dataset)
    model_hparams = get_model_hyperparams(num_classes, config.model)
    model_class = models.MODELS_DICT[config.model]
    model = model_class(**model_hparams)

    # %%
    # Use MLE weights and eval
    state, model_fn, _ = eval.prepare_from_checkpoint(model, info["checkpoints"]["mle"])
    D = len(state.params["param_nn"])
    print(f"Number of parameters: {D:,d}")
    model_fn = jax.jit(model_fn)
    print("*** Running evaluation on MLE")
    metrics, *_ = eval.evaluate_single(
        test_loader, state.params["param_nn"], model_fn, num_classes=num_classes
    )
    mle_metrics = metrics
    print("Metric\tValue")
    print("---")
    for name, value in metrics.items():
        print(f"{name}\t{value}")
    print()

    # %%
    # Setup evaluation
    print(f"*** Sampling {args.num_mc_samples} posterior samples")

    def ivon_sample(s, key):
        params, s = ivon.sample_parameters(key, state.params, s)
        return s, params

    sampling_keys = jax.random.split(key, num=args.num_mc_samples + 1)
    opt_state, _ = ivon_sample(state.opt_state, sampling_keys[0])
    _, posterior_samples = jax.lax.scan(ivon_sample, opt_state, sampling_keys[1:])
    posterior_samples = posterior_samples["param_nn"]

    print(f"*** Running evaluation on {args.num_mc_samples} posterior samples")
    metrics, *_ = eval.evaluate(
        test_loader, posterior_samples, model_fn, num_classes=num_classes
    )
    with open(os.path.join(args.log, "eval.json"), "w") as f:
        info = {
            "args": vars(args),
            "mle_metrics": mle_metrics,
            "metrics": metrics,
        }
        json.dump(info, f)
    print("Metric\tValue")
    print("---")
    for name, value in metrics.items():
        print(f"{name}\t{value}")
    print()


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
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
