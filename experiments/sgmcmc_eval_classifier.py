import argparse
import json
import os

import jax.numpy as jnp

from vp import eval, models, utils
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
    # Setup evaluation
    #
    # Here we ensure posterior samples are loaded in order, not
    # whatever order the dict iterator decides.
    ckpt_names = [f"pos-{i}" for i in range(1, len(info["checkpoints"]) + 1)]
    posterior_samples = []
    for name in ckpt_names:
        # Note that this implicitly leaves `model_fn` set for the last
        # model (using its state), which is what we intended. Tthis
        # keeps the setting the same as we do with other cheaper
        # posterior samplers -- ours, IVON, etc.
        state, model_fn, _ = eval.prepare_from_checkpoint(
            model, info["checkpoints"][name]
        )
        posterior_samples.append(state.params["param_nn"])
    posterior_samples = jnp.stack(posterior_samples)
    print(f"*** Running evaluation on {len(posterior_samples)} posterior samples")
    metrics, *_ = eval.evaluate(
        test_loader, posterior_samples, model_fn, num_classes=num_classes
    )
    with open(os.path.join(args.log, "eval.json"), "w") as f:
        info = {
            "args": vars(args),
            "mle_metrics": {},
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
        "--batch-size",
        default=None,
        type=int,
        help="Override batch size",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
