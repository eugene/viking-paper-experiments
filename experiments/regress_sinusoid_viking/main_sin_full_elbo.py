"""Fit a sine-curve with the VIKING algorithm."""

import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tqdm

from vp import datasets, models, viking


def main(
    seed=42,
    samples_data=10,
    num_hidden=10,
    lr_elbo=1e-2,
    beta=1.0,
    epochs_elbo=6_000,
    samples_mc=100,
    num_plot_samples=100,
    use_custom_vjp=True,
    is_linearized=True,
    adaptive_grad_clip=None,  # Use None for no clipping
):
    # Initialise the hyperparams and VI params
    param_hyper = {"log_precision": 0.0, "log_scale_noise": jnp.log(1e-2)}
    param_vi = {"log_scale_image": -2.0}

    # Create data
    key = jax.random.PRNGKey(seed=seed)
    key, subkey = jax.random.split(key)
    x, y = datasets.make_wave(subkey, num=samples_data + 10)
    x = jnp.concatenate((x[:5], x[-5:]))
    y = jnp.concatenate((y[:5], y[-5:]))

    # Initialise the NN params
    key, subkey = jax.random.split(key)
    model_init_fn, model_apply_fn = models.make_mlp(num_hidden=num_hidden)
    vmapped_apply_fn = jax.vmap(model_apply_fn, in_axes=(None, 0))
    model_params = model_init_fn(num_inputs=1, key=subkey)
    param_nn, model_unflatten = jax.flatten_util.ravel_pytree(model_params)

    def loss_single(pred, u):
        return -jax.scipy.stats.norm.logpdf(
            u, loc=pred, scale=jnp.exp(param_hyper["log_scale_noise"])
        )

    projection_ggn = viking.projection_kernel_ggn(
        vmapped_apply_fn,
        model_unflatten,
        viking.solve_normaleq_cg_fixed_step_reortho(maxiter=10),
        use_custom_vjp=use_custom_vjp,
    )
    projection_loss = viking.projection_kernel_param_to_loss(
        vmapped_apply_fn,
        model_unflatten,
        loss_single,
        use_custom_vjp=use_custom_vjp,
    )
    projection = projection_ggn
    D = len(param_nn)
    print(f"Number of parameters: {D}")

    expectation = viking.elbo_expectation(
        vmapped_apply_fn,
        model_unflatten,
        loss_single=loss_single,
        is_linearized=is_linearized,
    )
    loss_elbo = viking.make_elbo(expectation=expectation)
    optimizer_elbo = optax.adam(lr_elbo)
    if adaptive_grad_clip is not None:
        optimizer_elbo = optax.chain(
            optax.adaptive_grad_clip(adaptive_grad_clip),
            optimizer_elbo,
        )
    train_elbo_step = viking.make_train_elbo_step(loss_elbo, optimizer=optimizer_elbo)
    train_elbo_step = jax.jit(train_elbo_step, static_argnums=0)

    ##################################################################
    # Optimise ELBO
    ##################################################################
    # Initialise an optimiser
    opt_state_elbo = optimizer_elbo.init((param_hyper, param_vi, param_nn))

    # Prepare the progress display

    key, subkey = jax.random.split(key)
    iso_samples = jax.random.normal(subkey, (samples_mc, D))
    loss_value, stats = loss_elbo(
        param_nn,
        param_hyper,
        param_vi,
        x=x,
        y=y,
        projection=projection,
        iso_samples=iso_samples,
        beta=beta,
    )
    progressbar_elbo = tqdm.trange(epochs_elbo)
    for step_elbo in progressbar_elbo:
        key, subkey = jax.random.split(key)
        iso_samples = jax.random.normal(subkey, (samples_mc, D))
        try:
            loss_value, stats, param_hyper, param_vi, param_nn, opt_state_elbo = (
                train_elbo_step(
                    projection,
                    param_hyper,
                    param_vi,
                    param_nn,
                    opt_state_elbo,
                    x,
                    y,
                    iso_samples,
                    beta=1.0,
                )
            )
            progressbar_elbo.set_description(
                make_description(
                    loss_value,
                    stats,
                    param_hyper,
                    param_vi,
                )
            )
            if step_elbo % 1000 == 0:
                progressbar_elbo.write(
                    f"[{step_elbo:05d}] {
                        make_description(
                            loss_value,
                            stats,
                            param_hyper,
                            param_vi,
                        )
                    }"
                )
        except KeyboardInterrupt:
            break
    print(
        make_description(
            loss_value,
            stats,
            param_hyper,
            param_vi,
        )
    )

    key, subkey = jax.random.split(key)
    iso_samples = jax.random.normal(subkey, (num_plot_samples, D))
    UUt_kernel, D, projection_stats = projection(param_nn, x, y)
    kernel_samples, _ = jax.vmap(UUt_kernel)(iso_samples)
    image_samples = iso_samples - kernel_samples  # shape is (samples_mc, D)

    model_params = model_unflatten(param_nn)
    x_eval = jnp.linspace(0, 1, 500)
    y_eval = jax.vmap(model_apply_fn, in_axes=(None, 0))(model_params, x_eval)

    sample_posterior = viking.make_sampler(
        vmapped_apply_fn, model_unflatten, is_linearized=is_linearized
    )
    samples = sample_posterior(
        param_nn,
        log_scale_kernel=-param_hyper["log_precision"] / 2,
        log_scale_image=param_vi["log_scale_image"],
        x_eval=x,
        image_samples=image_samples[0:1] * 0.0,
        kernel_samples=kernel_samples[0:1],
    )[0]
    lin_apply_fn = viking.prepare_apply_fn(
        vmapped_apply_fn,
        model_unflatten,
        is_linearized=is_linearized,
    )
    y_pred = lin_apply_fn(param_nn, param_nn, x)
    if projection is projection_loss:
        losses_sample = jax.vmap(loss_single)(samples, y)
        losses_pred = jax.vmap(loss_single)(y_pred, y)
        jax.debug.print("{}", losses_sample - losses_pred)
    if projection is projection_ggn:
        jax.debug.print("{}", samples - y_pred)

    def plot(log_scale_kernel, log_scale_image, ker_fac=1.0, im_fac=1.0):
        fig, ax = plt.subplots(
            nrows=2,
            sharex=True,
            height_ratios=(1.8, 1.2),
        )
        # ax[0].set_title(
        #     r"$\sigma_{{ker}}={:.3f}$ and $\sigma_{{im}}={:.3f}$".format(
        #         jnp.exp(-0.5 * param_hyper["log_precision"]),
        #         jnp.exp(param_vi["log_scale_image"]),
        #     )
        # )

        # Plot samples from the variational approximation
        samples = sample_posterior(
            param_nn,
            log_scale_kernel=log_scale_kernel,
            log_scale_image=log_scale_image,
            x_eval=x_eval,
            image_samples=im_fac * image_samples,
            kernel_samples=ker_fac * kernel_samples,
        )
        for sample in samples:
            ax[0].plot(x_eval, sample, alpha=0.1, color="C0", linewidth=1)

        ax[0].plot(x_eval, y_eval, color="C3")

        ax[0].set_xlim((x_eval[0], x_eval[-1]))
        ax[0].plot(x, y, marker="P", linestyle="None", color="black")

        marginal_std = jnp.std(jnp.asarray(samples), axis=0)
        ax[1].plot(x_eval, marginal_std, color="C0", linewidth=1)
        ax[1].set_ylabel("Standard deviation")
        ax[1].set_ylim(0, jnp.max(marginal_std))
        ax[0].set_xticks([])
        # ax[1].set_ylim((0., 2.0))
        ax[1].grid(which="major", axis="y", alpha=0.75, ls=":")
        for a in ax[0], ax[1]:
            for s in "top", "right", "bottom":
                a.spines[s].set_visible(False)
        plt.savefig("outputs/sinusoid_viking.pdf", bbox_inches="tight")
        plt.show()

    plot(
        log_scale_kernel=-0.5 * param_hyper["log_precision"],
        log_scale_image=param_vi["log_scale_image"],
    )

    def plot_ax(ax, log_scale_kernel, log_scale_image, ker_fac=1.0, im_fac=1.0):
        # Plot samples from the variational approximation
        samples = sample_posterior(
            param_nn,
            log_scale_kernel=log_scale_kernel,
            log_scale_image=log_scale_image,
            x_eval=x_eval,
            image_samples=im_fac * image_samples,
            kernel_samples=ker_fac * kernel_samples,
        )
        for sample in samples:
            ax.plot(x_eval, sample, alpha=0.1, color="C0", linewidth=1)
        ax.plot(x_eval, y_eval, color="C3")
        ax.set_xlim((x_eval[0], x_eval[-1]))
        ax.plot(x, y, marker="P", linestyle="None", color="black")
        ax.set_xticks([])
        for s in "top", "right", "bottom":
            ax.spines[s].set_visible(False)
        ax.yaxis.set_ticks([-5.0, 0.0, 5, 10.0, 15.0])

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(4, 2))
    plot_ax(
        ax,
        log_scale_kernel=-1.0,
        log_scale_image=-3.5,
        im_fac=0.0,
    )
    fig.savefig("outputs/sinusoid_ker.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 2))
    plot_ax(
        ax,
        log_scale_kernel=-1.0,
        log_scale_image=-3.5,
        ker_fac=0.0,
    )
    fig.savefig("outputs/sinusoid_im.pdf", bbox_inches="tight")
    plt.close(fig)


def make_description(loss_value, stats, param_hyper, param_vi):
    R = stats["R"]
    log_precision = param_hyper["log_precision"]
    log_scale_image = param_vi["log_scale_image"]
    items = [
        f"E[]: {stats['E[]']:.1e}",
        f"kl: {stats['kl']:.1e}",
        f"R={R:.0f}",
        f"log_precision={log_precision:.2f}",
        f"log_scale_image={log_scale_image:.2f}",
        f"σ_ker={jnp.exp(-0.5 * log_precision):.2f}",
        f"σ_im={jnp.exp(log_scale_image):.2f}",
        f"dot={stats['dot']:.2f}",
        f"residuals={stats['residuals']:.2f}",
        f"precond_min={stats['precond_min']:.2f}",
        f"precond_max={stats['precond_max']:.2f}",
    ]
    description = ", ".join(items)
    return description


if __name__ == "__main__":
    main()
