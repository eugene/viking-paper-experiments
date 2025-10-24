"""Fit a sine-curve with IVON."""

import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm

from vp import datasets, ivon, models, viking


def main(
    seed=42,
    samples_data=10,
    lr_mse=1e-2,
    epochs_mse=5_000,
    hess_init=15.0,
    hidden=10,
    num_samples_ivon=5,
    num_plot_samples=100,
):
    # Create data
    key = jax.random.PRNGKey(seed=seed)
    key, subkey = jax.random.split(key)
    x, y = datasets.make_wave(subkey, num=samples_data + 10)
    x = jnp.concatenate((x[:5], x[-5:]))
    y = jnp.concatenate((y[:5], y[-5:]))

    # Initialise the NN params
    key, subkey = jax.random.split(key)
    model_init_fn, model_apply_fn = models.make_mlp(num_hidden=hidden)
    model_params = model_init_fn(num_inputs=1, key=key)
    param_nn, model_unflatten = jax.flatten_util.ravel_pytree(model_params)
    print(f"Number of parameters: {len(param_nn)}")

    # Optimise NN
    optimizer = ivon.ivon(learning_rate=lr_mse, ess=len(x), hess_init=hess_init)
    opt_state = optimizer.init(param_nn)
    loss_mle = make_loss(model_apply_fn, model_unflatten)
    train_step = viking.make_train_step_ivon(
        loss_mle, optimizer, num_samples=num_samples_ivon
    )
    train_step = jax.jit(train_step)

    loss_value, _ = loss_mle(param_nn, inputs=x, labels=y)

    progressbar_nn = tqdm.tqdm(range(epochs_mse), position=1, leave=True)
    progressbar_nn.set_description(f"NN: {loss_value:.1e}")
    for _ in progressbar_nn:
        key, subkey = jax.random.split(key, num=2)
        loss_value, stats, param_nn, opt_state, key = train_step(
            param_nn, opt_state, x, y, subkey
        )
        # progressbar_nn.set_description(f"NN: {loss_value:.1e}")
    print(f"NN: {loss_value:.1e}")

    # Plot the NN fit
    fig, ax = plt.subplots(nrows=2, sharex=True)

    model_params = model_unflatten(param_nn)
    x_eval = jnp.linspace(0, 1, 500)

    samples = []
    for _ in range(num_plot_samples):
        key, subkey = jax.random.split(key, num=2)
        param_sample, opt_state = ivon.sample_parameters(key, param_nn, opt_state)
        param_sample = model_unflatten(param_sample)
        y_eval = jax.vmap(model_apply_fn, in_axes=(None, 0))(param_sample, x_eval)
        samples.append(y_eval)
        ax[0].plot(x_eval, y_eval, color="C0", alpha=0.1, linewidth=1)

    y_eval = jax.vmap(model_apply_fn, in_axes=(None, 0))(model_params, x_eval)
    ax[0].plot(x_eval, y_eval, color="C3")

    ax[0].set_xlim((x_eval[0], x_eval[-1]))
    ax[0].plot(x, y, marker="P", linestyle="None", color="black")

    samples = jnp.stack(samples)
    marginal_std = jnp.std(samples, axis=0)

    # ax[0].set_ylabel("Prediction")
    ax[0].set_xticks([])
    ax[1].plot(x_eval, marginal_std, color="C0", linewidth=1)
    ax[1].set_ylabel("Standard deviation")
    ax[1].set_ylim(0, jnp.max(marginal_std))
    # ax[1].set_ylim((0., 2.0))
    ax[1].grid(which="major", axis="y", alpha=0.75, ls=":")
    for a in ax[0], ax[1]:
        for s in "top", "right", "bottom":
            a.spines[s].set_visible(False)
    plt.savefig("outputs/sinusoid_ivon.pdf", bbox_inches="tight")
    plt.show()


def make_loss(model_apply_fn, model_unflatten):
    def loss_mle(param_nn, inputs, labels):
        model_params = model_unflatten(param_nn)
        preds = jax.vmap(model_apply_fn, in_axes=(None, 0))(model_params, inputs)
        log_likelihood = jnp.sum(mle_single(labels, preds))
        return log_likelihood, {}

    @jax.vmap
    def mle_single(u, pred):
        return jnp.dot(u - pred, u - pred)

    return loss_mle


if __name__ == "__main__":
    main()
