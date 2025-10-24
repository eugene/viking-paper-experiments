import argparse
import json
import os

import jax
import numpy as np

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
    # Load MLE weights
    state, model_fn, _ = eval.prepare_from_checkpoint(model, info["checkpoints"]["mle"])
    D = len(state.params["param_nn"])
    print(f"Number of parameters: {D:,d}")
    model_fn = jax.jit(model_fn)

    # %%
    # Compute posterior samples
    print(f"*** Sampling {args.num_mc_samples} posterior samples...")

    def ivon_sample(s, key):
        params, s = ivon.sample_parameters(key, state.params, s)
        return s, params

    sampling_keys = jax.random.split(key, num=args.num_mc_samples + 1)
    opt_state, _ = ivon_sample(state.opt_state, sampling_keys[0])
    _, posterior_samples = jax.lax.scan(ivon_sample, opt_state, sampling_keys[1:])
    posterior_samples = posterior_samples["param_nn"]

    # %%
    # In-distribution evaluation
    print(
        f"*** Running evaluation MLE model and {args.num_mc_samples} posterior samples"
    )
    mle_test_metrics, mle_outputs_id = eval.evaluate_single(
        test_loader, state.params["param_nn"], model_fn, num_classes=num_classes
    )
    test_metrics, outputs_id = eval.evaluate(
        test_loader, posterior_samples, model_fn, num_classes=num_classes
    )
    mle_score_id = 1.0 - np.max(mle_outputs_id["all_y_prob"], axis=-1)
    score_id = np.max(outputs_id["all_y_var"], axis=-1)
    score_prob_id = np.max(outputs_id["all_y_prob_var"], axis=-1)

    for ood_dataset in args.ood_datasets:
        _, _, ood_test_loader = datasets.get_dataloaders(
            ood_dataset,
            train_batch_size=batch_size,
            val_batch_size=batch_size,
            purp="sample",
            seed=args.seed,
        )

        print(
            f"*** {ood_dataset}: Running OOD evaluation MLE model and "
            f"{args.num_mc_samples} posterior samples"
        )
        mle_metrics_ood, mle_outputs_ood = eval.evaluate_single(
            ood_test_loader,
            state.params["param_nn"],
            model_fn,
            num_classes=num_classes,
        )
        metrics_ood, outputs_ood = eval.evaluate(
            ood_test_loader, posterior_samples, model_fn, num_classes=num_classes
        )
        mle_score_ood = 1.0 - np.max(mle_outputs_ood["all_y_prob"], axis=-1)
        score_ood = np.max(outputs_ood["all_y_var"], axis=-1)
        score_prob_ood = np.max(outputs_ood["all_y_prob_var"], axis=-1)

        mle_auroc = eval.auroc(mle_score_id, mle_score_ood)
        logit_auroc = eval.auroc(score_id, score_ood)
        prob_auroc = eval.auroc(score_prob_id, score_prob_ood)
        with open(os.path.join(args.log, f"ood-{ood_dataset}.json"), "w") as f:
            info = {
                "args": vars(args),
                "dataset": config.dataset,
                "ood_dataset": ood_dataset,
                "mle_auroc": mle_auroc,
                "logit_auroc": logit_auroc,
                "prob_auroc": prob_auroc,
                "metrics": metrics_ood,
            }
            json.dump(info, f)
        print("Metric\tValue")
        print("---")
        print(f"mle_auroc\t{mle_auroc}")
        print(f"logit_auroc\t{logit_auroc}")
        print(f"prob_auroc\t{prob_auroc}")
        print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Seed for PRNGs")
    parser.add_argument(
        "log",
        help="Path to log directory",
    )
    parser.add_argument(
        "--ood-datasets",
        default=[],
        nargs="+",
        help="Dataset(s) used for OOD detection benchmarking",
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
