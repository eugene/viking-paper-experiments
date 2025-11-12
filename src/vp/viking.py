"""A library of algorithms, utilities, and glue code for VIKING and baselines."""

import functools as ft
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
import scipy as sp
from tqdm import tqdm

from vp import cg, datasets, ivon


def calc_trace(
    params, key, inputs, model, solve_normaleq, num_samples, use_custom_vjp=False
):
    thetas_mean = params
    thetas_mean_flat, unflatten = jax.flatten_util.ravel_pytree(thetas_mean)

    D = sum(x.size for x in jax.tree_util.tree_leaves(thetas_mean))
    num_rows = inputs.shape[0] * 10
    num_cols = D

    def model_fun_vec(p):
        return model.apply(unflatten(p), inputs).reshape(-1)

    def matvec(_param, v):
        return jax.jvp(model_fun_vec, (thetas_mean_flat,), (v,))[1]

    _, vjp_fun = jax.vjp(model_fun_vec, thetas_mean_flat)

    def vecmat(param, v):
        return vjp_fun(v)[0]

    projfun_vjp = projection_matfree(
        matvec,
        vecmat,
        num_rows=num_rows,
        num_cols=num_cols,
        solve_normaleq=solve_normaleq,
        use_custom_vjp=use_custom_vjp,
    )
    vs = jax.random.normal(key, (num_samples, D))
    trace = jax.vmap(projfun_vjp, in_axes=(None, 0))(thetas_mean_flat, vs)
    trace = trace[:, :D]

    return trace, vs


def prod_tr(mat_a, mat_b):
    """Computes tr(mat_a, mat_b) without computing all entries of mat_a @ mat_b."""
    tr = jnp.sum(jnp.sum(mat_a * mat_b.T, axis=-1), axis=-1)
    return tr


def nystroem_pp(A_samples, iso_samples):
    """The Nyström++ trace estimator.

    Args:
      A_samples: A @ iso_samples
      iso_samples: i.i.d. samples from a standard Gaussian

    Reference:
    Persson, David, Alice Cortinovis, and Daniel Kressner.
    "Improved variants of the Hutch++ algorithm for trace estimation."
    SIAM Journal on Matrix Analysis and Applications 43.3 (2022):
    1162-1185.
    """
    num_samples = iso_samples.shape[0]
    Om, Psi = jnp.split(iso_samples.T, 2, axis=-1)
    X, Y = jnp.split(A_samples.T, 2, axis=-1)

    # Rewrites X @ pinv(Om.T @ X) @ X.T as X @ Q @ R^-1 @ X.T
    Q, R = jnp.linalg.qr(X.T @ Om)
    R_inv_XT = jsp.linalg.solve_triangular(R, X.T, trans="T")
    X_pinv_XT = X @ Q @ R_inv_XT
    fst_tr = jnp.linalg.trace(X_pinv_XT)
    snd_tr = prod_tr(Psi.T, Y)
    trd_tr = prod_tr(X_pinv_XT, Psi @ Psi.T)
    tr = fst_tr + 2 / num_samples * (snd_tr - trd_tr)
    # jax.debug.print("nyström: {:8.4f} {:8.4f} {:8.4f} ({:8.4f})", fst_tr, snd_tr, trd_tr, tr)
    return tr


def projection_dense(use_custom_vjp=False):
    """Construct a function that projects a vector onto the kernel of a (dense) matrix."""

    def _projsolve(A, c):
        _, m = A.shape
        c_1, c_2 = jnp.split(c, [m])
        AA_t = A @ A.T

        proj_c_1 = A @ c_1
        proj_c_1 = jnp.linalg.solve(AA_t, proj_c_1)
        inv_c_2 = jnp.linalg.solve(AA_t, c_2)

        x_2 = proj_c_1 - inv_c_2  # image
        proj_c_1 = c_1 - A.T @ proj_c_1
        inv_c_2 = A.T @ inv_c_2
        x_1 = proj_c_1 + inv_c_2  # kernel
        stats = {"residuals": jnp.nan}
        return jnp.concatenate((x_1, x_2)), stats

    def projfun(A, v):
        c = jnp.concatenate((v, jnp.zeros(A.shape[0])))
        return _projsolve(A, c)

    if not use_custom_vjp:
        return projfun

    projfun_vjp = jax.custom_vjp(projfun)

    def projfun_fwd(A, v):
        out, stats = projfun(A, v)
        return (out, stats), {"A": A, "v": v, "z": out}

    def projfun_rev(cache, vjp_incoming):
        A, v, z = cache["A"], cache["v"], cache["z"]
        xi, _stats = _projsolve(A, vjp_incoming[0])

        full_grad_B = -jnp.outer(xi, z)
        full_grad_c = xi

        grad_A = (full_grad_B + full_grad_B.T)[-A.shape[0] :, : A.shape[1]]
        grad_v = full_grad_c[: v.shape[0]]

        return grad_A, grad_v

    projfun_vjp.defvjp(projfun_fwd, projfun_rev)

    return projfun_vjp


def projection_matfree(
    matvec, vecmat, *, num_rows, num_cols, solve_normaleq, use_custom_vjp=False
):
    """Construct a function like projection_dense but without materializing the matrix."""

    def _projsolve(param, c):
        # cols/rows according to A's cols/rows
        c_cols, c_rows = jnp.split(c, [num_cols])

        Ac_cols = matvec(param, c_cols)
        rhs = Ac_cols - c_rows
        lagrange_multiplier, stats = solve_normaleq(matvec, vecmat, param, rhs)

        # If test breaks, reevaluate sign in front of inv_c_rows
        c_projection = c_cols - vecmat(param, lagrange_multiplier)
        return jnp.concatenate((c_projection, lagrange_multiplier)), stats

    def projfun(param, v):
        c = jnp.concatenate((v, jnp.zeros(num_rows)))
        return _projsolve(param, c)

    if not use_custom_vjp:
        return projfun

    projfun_vjp = jax.custom_vjp(projfun)

    def projfun_fwd(param, v):
        out, stats = projfun(param, v)
        return (out, stats), {"param": param, "v": v, "z": out}

    def projfun_rev(cache, vjp_incoming):
        param, _v, z = cache["param"], cache["v"], cache["z"]
        xi, _stats = _projsolve(param, vjp_incoming[0])

        # P_perp splices -num_row: and P_par splices :num_cols,
        # Also the sign is tricky - outer(xi, z)
        def fn_1(p):
            v_1 = xi[-num_rows:]
            v_2 = z[:num_cols]
            Av_2 = matvec(p, v_2)
            return -jnp.dot(v_1, Av_2)

        def fn_2(p):
            v_1 = z[-num_rows:]
            v_2 = xi[:num_cols]
            Av_2 = matvec(p, v_2)
            return -jnp.dot(v_1, Av_2)

        grad_1 = jax.grad(fn_1)(param)
        grad_2 = jax.grad(fn_2)(param)

        grad_v = xi[:num_cols]
        grad_param = grad_1 + grad_2
        return grad_param, grad_v

    projfun_vjp.defvjp(projfun_fwd, projfun_rev)

    return projfun_vjp


def projection_matfree_gd(
    matvec, vecmat, *, num_rows, num_cols, lr=1e-3, maxiter=10, use_custom_vjp=True
):
    def projsolve(param, init_solution):
        optimizer = optax.polyak_sgd(max_learning_rate=lr)

        v, init_λ = jnp.split(init_solution, [num_cols])

        def loss(solution):
            x, λ = solution
            sq_norm_grad_x = jnp.sum(jnp.square(x - v + vecmat(param, λ)))
            sq_norm_grad_λ = jnp.sum(jnp.square(matvec(param, x)))
            loss_val = sq_norm_grad_x + sq_norm_grad_λ
            return loss_val

        def opt_step(loop_state, _):
            solution, opt_state = loop_state
            value, grad = jax.value_and_grad(loss)(solution)
            updates, opt_state = optimizer.update(
                grad, opt_state, solution, value=value
            )
            solution = optax.apply_updates(solution, updates)
            return (solution, opt_state), _

        solution = (v, init_λ)
        opt_state = optimizer.init(solution)
        (solution, opt_state), _ = jax.lax.scan(
            opt_step, (solution, opt_state), length=maxiter
        )
        stats = {"residuals": jnp.linalg.norm(matvec(param, solution[0]))}
        return jnp.concatenate(solution), stats

    def projfun(param, v):
        return projsolve(
            param,
            jnp.concatenate(
                (
                    v,
                    jnp.zeros(
                        num_rows,
                    ),
                )
            ),
        )

    if not use_custom_vjp:
        return projfun

    projfun_vjp = jax.custom_vjp(projfun)

    def projfun_fwd(param, v):
        out, stats = projfun(param, v)
        return (out, stats), {"param": param, "v": v, "z": out}

    def projfun_rev(cache, vjp_incoming):
        param, _v, z = cache["param"], cache["v"], cache["z"]
        xi, _stats = projsolve(param, vjp_incoming[0])

        # P_perp splices -num_row: and P_par splices :num_cols,
        # Also the sign is tricky - outer(xi, z)
        def fn_1(p):
            v_1 = xi[-num_rows:]
            v_2 = z[:num_cols]
            Av_2 = matvec(p, v_2)
            return -jnp.dot(v_1, Av_2)

        def fn_2(p):
            v_1 = z[-num_rows:]
            v_2 = xi[:num_cols]
            Av_2 = matvec(p, v_2)
            return -jnp.dot(v_1, Av_2)

        grad_1 = jax.grad(fn_1)(param)
        grad_2 = jax.grad(fn_2)(param)

        grad_v = xi[:num_cols]
        grad_param = grad_1 + grad_2
        return grad_param, grad_v

    projfun_vjp.defvjp(projfun_fwd, projfun_rev)
    return projfun_vjp


def solve_normaleq_qr_of_jac():
    def solve(matvec, vecmat, param, rhs):
        del matvec  # unused argument

        Jt = jax.jacfwd(lambda v: vecmat(param, v))(rhs)
        Lt = jnp.linalg.qr(Jt, mode="r")

        cho_factor = (Lt, False)
        solution = jsp.linalg.cho_solve(cho_factor, rhs)
        stats = {"residuals": jnp.linalg.norm(rhs - Jt.T @ Jt @ solution)}
        return solution, stats

    return solve


def solve_normaleq_materialize(solver_fn):
    """Solve the normal equation by calling a dense solver."""

    def solve(matvec, vecmat, param, rhs):
        def AAt_mv(s):
            Atv = vecmat(param, s)
            return matvec(param, Atv)

        # Since AAt is linear, the Jacobian at _any_
        # value will materialize the matrix.
        # For shape reasons, we mimic the RHS vector.
        # But any vector of the correct shape would do.
        irrelevant_value = jnp.zeros_like(rhs)
        AA_t = jax.jacfwd(AAt_mv)(irrelevant_value)
        # M_inv = 1.0 / jnp.diag(AA_t)
        # solution = solver_fn(M_inv[:, None] * AA_t, M_inv * rhs)
        solution = solver_fn(AA_t, rhs)
        residuals = jnp.linalg.norm(rhs - AAt_mv(solution))
        stats = {"residuals": residuals / jnp.linalg.norm(rhs)}
        return solution, stats

    return solve


def solve_normaleq_cg(tol=1e-5, atol=1e-12, maxiter=10):
    """Solve the normal equation by conjugate gradients."""

    def solve(matvec, vecmat, param, rhs):
        def AAt_mv(s):
            Atv = vecmat(param, s)
            return matvec(param, Atv)

        solution, _ = jsp.sparse.linalg.cg(
            AAt_mv, rhs, tol=tol, atol=atol, maxiter=maxiter
        )
        residuals = jnp.linalg.norm(rhs - AAt_mv(solution))
        stats = {"residuals": residuals / jnp.linalg.norm(rhs)}
        return solution, stats

    return solve


def solve_normaleq_cg_fixed_step_reortho(maxiter=10):
    """Solve the normal equation by conjugate gradients."""

    cg_solve = cg.cg_fixed_step_reortho(maxiter)

    def solve(matvec, vecmat, param, rhs):
        def AAt_mv(s):
            Atv = vecmat(param, s)
            return matvec(param, Atv)

        solution, _ = cg_solve(AAt_mv, rhs)
        residuals = jnp.linalg.norm(rhs - AAt_mv(solution))
        stats = {"residuals": residuals / jnp.linalg.norm(rhs)}
        return solution, stats

    return solve


def solver_fn_cholesky(symmetrize, eps):
    def solver_fn_cho(mat, rhs):
        D = mat.shape[-1]
        noise = eps * jnp.eye(D)
        mat = mat + noise

        if symmetrize:
            mat = (mat + mat.T) / 2

        cho_factor = jsp.linalg.cho_factor(mat)
        return jsp.linalg.cho_solve(cho_factor, rhs)

    return solver_fn_cho


def solver_fn_lu():
    return jnp.linalg.solve


def solver_fn_eig(eps):
    def solve(mat, rhs):
        eigvals, eigvecs = jnp.linalg.eigh(mat)

        eigvals = jnp.where(eigvals < eps, -1.0, eigvals)
        out = eigvecs.T @ rhs
        inv_eigvals = jnp.reciprocal(eigvals)
        inv_eigvals = jnp.where(inv_eigvals < eps, 0.0, inv_eigvals)
        out = jnp.diag(inv_eigvals) @ out
        return eigvecs @ out

    return solve


def normal_spherical_sample(key, num_samples, num_dims, end=1.0):
    """This function samples and normalizes two standard Gaussian
    points, then samples along the great circle defined by them on the
    hypersphere.
    """
    # NOTE: end=1.0 means "loop" all the way
    endpoints = jax.random.normal(key, shape=(2, num_dims))
    end_a = endpoints[0]
    end_b = endpoints[1]
    # This ensures that end_b is orthogonal to end_a, meaning we can
    # control interpolation a bit better (and enforce that end=1.0
    # means a full loop), see below
    end_a, end_b = _gram_schmidt(end_a, end_b)
    # Expected norm is proportional to sqrt(num_dims)
    expected_norm = jnp.sqrt(num_dims)
    end_a /= jnp.linalg.norm(end_a)
    end_b /= jnp.linalg.norm(end_b)
    omega = jnp.arccos(jnp.inner(end_a, end_b))
    end_a *= expected_norm
    end_b *= expected_norm
    t = jnp.linspace(0.0, end, num_samples, endpoint=True)[:, None]
    samples = (
        # Since end_a and end_b are orthogonal, they are "a quarter"
        # of the way apart vs. a full loop. Hence, we multiply t by 4
        # when parameterising the interpolation to ensure that t=1
        # results in a full loop.
        jnp.sin((1.0 - 4 * t) * omega) * end_a[None, ...]
        + jnp.sin(4 * t * omega) * end_b[None, ...]
    ) / jnp.sin(omega)
    return samples


def normal_noisy_spherical_sample(key, num_samples, num_dims, end=1.0):
    """Similar to `normal_spherical_sample`, but this function makes
    the great circle noisy along the surface of the sphere, with a (cubic)
    spline passing along the sampled points.
    """
    key, key_s, key_n = jax.random.split(key, num=3)
    num_clean_samples = max(num_samples // 5, 2)
    clean_samples = normal_spherical_sample(
        key_s, num_samples=num_clean_samples, num_dims=num_dims, end=end
    )
    # Noise should be orthogonal to the direction the clean samples
    # are, meaning we don't make samples in that direction
    # closer/further away in their natural trajectory along the loop.
    # Thus, our noise vectors only wiggle the clean samples "left and
    # right", if following that trajectory of samples in a 3d sense.
    _, noise = jax.vmap(_gram_schmidt)(
        clean_samples[1:] - clean_samples[0:-1],
        jax.random.normal(key_n, shape=clean_samples.shape)[:-1],
    )
    clean_samples = clean_samples[:-1] + 0.2 * noise
    # Adds first point as last for the periodic spline
    y_train = jnp.concatenate((clean_samples, clean_samples[0:1]), axis=0)
    y_train = y_train / jnp.linalg.norm(y_train, axis=-1, keepdims=True)
    y_train = _from_hypersphere(y_train)
    t = jnp.linspace(0, 1, num_clean_samples)
    spline = sp.interpolate.CubicSpline(t, y_train, bc_type="periodic")
    t = jnp.linspace(0, 1, num_samples)
    expected_norm = jnp.sqrt(num_dims)
    samples = _to_hypersphere(spline(t)) * expected_norm
    return samples


def _gram_schmidt(v1, v2):
    u1 = v1
    u2 = v2 - _project_onto(u1, v2)
    return u1, u2


def _project_onto(u, v):
    """Projects `v` onto `u`."""
    return jnp.inner(v, u) / jnp.inner(u, u) * u


def _to_hypersphere(x):
    d_norm = 1 + jnp.sum(x * x, axis=-1, keepdims=True)
    z = jnp.concatenate((2 * x / d_norm, 1 - 2 / d_norm), axis=-1)
    return z


def _from_hypersphere(z):
    x = z[:, :-1] / (1.0 - z[:, -1]).reshape((-1, 1))
    return x


def prepare_apply_fn(
    model_apply_fn: Callable, model_unflatten: Callable, is_linearized: bool
):
    """Wraps `model_apply_fn` (which is assumed to work on batches) in
    a calling convention used for both linearised predictions or
    regular predictions.
    """

    # API supposed to match linearized_batch_apply
    def batch_apply(_param_mean, p, x_eval):
        p = model_unflatten(p)
        return model_apply_fn(p, x_eval)

    def linearized_batch_apply(param_mean, p, x_eval):
        def preds_at_param(p):
            return batch_apply(None, p, x_eval)

        map_preds = preds_at_param(param_mean)
        _, lin_preds = jax.jvp(preds_at_param, (param_mean,), (p - param_mean,))
        return map_preds + lin_preds

    if is_linearized:
        return linearized_batch_apply
    return batch_apply


def apply_fn_from_state(state, train=True):
    has_updates = False
    collections = {"params": None}
    extra_kwargs = {}
    if state.batch_stats is not None:
        collections["batch_stats"] = state.batch_stats
        extra_kwargs["train"] = train
        extra_kwargs["mutable"] = ["batch_stats"] if train else False
        has_updates = train
    if state.key is not None:
        extra_kwargs["train"] = train
        extra_kwargs["rngs"] = {"reparam_key": state.key}

    def apply_fn(p, x):
        collections["params"] = p
        return state.apply_fn(
            collections,
            x=x,
            **extra_kwargs,
        )

    if has_updates:
        return apply_fn

    def apply_fn_with_empty_updates(p, x):
        return apply_fn(p, x), {}

    return apply_fn_with_empty_updates


def make_sampler(model_apply_fn, model_unflatten, is_linearized=False):
    apply_fn = prepare_apply_fn(model_apply_fn, model_unflatten, is_linearized)

    def sample_posterior(
        param_mean,
        log_scale_kernel,
        log_scale_image,
        x_eval,
        image_samples,
        kernel_samples,
    ):
        def sample_posterior_single(image_sample, kernel_sample):
            sample_p = (
                param_mean
                + jnp.exp(log_scale_kernel) * kernel_sample
                + jnp.exp(log_scale_image) * image_sample
            )
            preds = apply_fn(param_mean, sample_p, x_eval)
            return preds

        predictions = jax.vmap(sample_posterior_single)(image_samples, kernel_samples)
        return predictions

    return sample_posterior


def make_state_sampler(model_unflatten, is_linearized=False):
    def sample_posterior(
        params,
        state,
        x_eval,
        image_samples,
        kernel_samples,
    ):
        log_scale_kernel = -0.5 * params["log_precision"]
        log_scale_image = params["log_scale_image"]
        param_mean = params["param_nn"]
        apply_fn = prepare_apply_fn(
            apply_fn_from_state(state), model_unflatten, is_linearized
        )

        def sample_posterior_single(image_sample, kernel_sample):
            sample_p = (
                param_mean
                + jnp.exp(log_scale_kernel) * kernel_sample
                + jnp.exp(log_scale_image) * image_sample
            )
            preds, updates = apply_fn(param_mean, sample_p, x_eval)
            return preds, updates

        predictions, updates = jax.vmap(sample_posterior_single)(
            image_samples, kernel_samples
        )
        return predictions, updates

    return sample_posterior


def projection_kernel_ggn(
    model_apply_fn, model_unflatten, solve_normaleq, use_custom_vjp=True
):
    def computes_kernel_basis(param_nn, x, y_unused=None):
        def model_p(p):
            params = model_unflatten(p)
            return model_apply_fn(params, x).reshape(-1)

        def matvec(p, s):
            _, jvp_x = jax.jvp(model_p, (p,), (s,))
            return jvp_x

        def vecmat(p, s):
            _, vjp = jax.vjp(model_p, p)
            return vjp(s)[0]

        # This determines the number of output samples and their dimensions
        y_dummy = jax.eval_shape(model_p, param_nn)
        projfun = projection_matfree(
            matvec,
            vecmat,
            num_rows=y_dummy.size,
            num_cols=len(param_nn),
            solve_normaleq=solve_normaleq,
            use_custom_vjp=use_custom_vjp,
        )
        D = len(param_nn)

        def kernel_projection(vec):
            fx, stats = projfun(param_nn, vec)
            stats["precond"] = 0.0
            return fx[:D], stats

        return kernel_projection, D, {}

    return computes_kernel_basis


def projection_kernel_param_to_loss(
    model_apply_fn, model_unflatten, loss_fn, use_custom_vjp=True, precond_min_norm=1.0
):
    """Similar to `projection_kernel_ggn`, but uses a loss function instead
    of the model outputs.

    This scales better for models with large output dimensions at the
    cost of shifting the requirement of maintaining model outputs at
    training data to maintaining the loss values at those data points.
    For more details, refer to [1; Sec. 4.1].

    [1] https://arxiv.org/abs/2410.16901

    """
    # model_apply_fn = jax.vmap(model_apply_fn, in_axes=(None, 0))

    def computes_kernel_basis(param_nn, x, y):
        def model_p(p):
            params = model_unflatten(p)
            preds = model_apply_fn(params, x)
            return jax.vmap(loss_fn)(preds, y)

        def model_single(p, s, t):
            params = model_unflatten(p)
            return loss_fn(model_apply_fn(params, s), t)

        def precond_at(p):
            def grad_norm_fn(s, t):
                return jnp.linalg.norm(jax.grad(model_single)(p, s, t))

            precond = jax.vmap(grad_norm_fn)(x, y)
            precond = jnp.clip(precond, min=precond_min_norm)
            return precond

        def matvec(p, s):
            _, jvp_x = jax.jvp(model_p, (p,), (s,))
            return jvp_x

        def matvec_precond(p, s):
            row_norms = precond_at(p)
            _, jvp_x = jax.jvp(model_p, (p,), (s,))
            jvp_x = jvp_x / row_norms
            return jvp_x

        def vecmat(p, s):
            _, vjp = jax.vjp(model_p, p)
            return vjp(s)[0]

        def vecmat_precond(p, s):
            row_norms = precond_at(p)
            _, vjp = jax.vjp(model_p, p)
            return vjp(s / row_norms)[0]

        projfun = projection_matfree(
            matvec,
            vecmat,
            num_rows=len(x),
            num_cols=len(param_nn),
            # solve_normaleq=solve_normaleq_qr_of_jac(),
            # solve_normaleq=solve_normaleq_materialize(solver_fn_eig(eps=1e-6)),
            # solve_normaleq=solve_normaleq_cg(tol=1e-5, atol=1e-12, maxiter=10),
            solve_normaleq=solve_normaleq_cg_fixed_step_reortho(maxiter=10),
            use_custom_vjp=use_custom_vjp,
        )
        D = len(param_nn)

        def kernel_projection(vec):
            fx, stats = projfun(param_nn, vec)
            # stats["precond"] = precond_at(param_nn)
            stats["precond"] = 0.0
            return fx[:D], stats

        return kernel_projection, D, {}

    return computes_kernel_basis


def projection_state_kernel_param_to_loss(
    model_unflatten, loss_fn, use_custom_vjp=True, precond_min_norm=1.0
):
    """Similar to `projection_kernel_ggn`, but uses a loss function instead
    of the model outputs.

    This scales better for models with large output dimensions at the
    cost of shifting the requirement of maintaining model outputs at
    training data to maintaining the loss values at those data points.
    For more details, refer to [1; Sec. 4.1].

    [1] https://arxiv.org/abs/2410.16901

    """

    def computes_kernel_basis(params, state, x, y):
        model_apply_fn = apply_fn_from_state(state, train=False)
        param_nn = params["param_nn"]

        def model_p(p):
            params = model_unflatten(p)
            preds, _ = model_apply_fn(params, x)
            return jax.vmap(loss_fn)(preds, y)

        def model_single(p, s, t):
            params = model_unflatten(p)
            return loss_fn(model_apply_fn(params, s), t)

        def precond_at(p):
            def grad_norm_fn(s, t):
                return jnp.linalg.norm(jax.grad(model_single)(p, s, t))

            precond = jax.vmap(grad_norm_fn)(x, y)
            precond = jnp.clip(precond, min=precond_min_norm)
            return precond

        def matvec(p, s):
            _, jvp_x = jax.jvp(model_p, (p,), (s,))
            return jvp_x

        def matvec_precond(p, s):
            row_norms = precond_at(p)
            _, jvp_x = jax.jvp(model_p, (p,), (s,))
            jvp_x = jvp_x / row_norms
            return jvp_x

        def vecmat(p, s):
            _, vjp = jax.vjp(model_p, p)
            return vjp(s)[0]

        def vecmat_precond(p, s):
            row_norms = precond_at(p)
            _, vjp = jax.vjp(model_p, p)
            return vjp(s / row_norms)[0]

        projfun = projection_matfree(
            matvec,
            vecmat,
            num_rows=len(x),
            num_cols=len(param_nn),
            # solve_normaleq=solve_normaleq_qr_of_jac(),
            # solve_normaleq=solve_normaleq_materialize(solver_fn_eig(eps=1e-6)),
            # solve_normaleq=solve_normaleq_cg(tol=1e-5, atol=1e-12, maxiter=10),
            solve_normaleq=solve_normaleq_cg_fixed_step_reortho(maxiter=10),
            use_custom_vjp=use_custom_vjp,
        )
        D = len(param_nn)

        def kernel_projection(vec):
            fx, stats = projfun(param_nn, vec)
            # stats["precond"] = precond_at(param_nn)
            stats["precond"] = 0.0
            return fx[:D], stats

        return kernel_projection, D, {}

    return computes_kernel_basis


# XXX: Unused
def projection_kernel_ggn_dense(model):
    def computes_kernel_basis(param_nn, x):
        # Shape of Jacobian matrix: num.data x num.outputs x num_param
        jacobian = jax.jacfwd(model, argnums=0)
        jacobian_batch = jax.vmap(jacobian, in_axes=(None, 0))
        jacobian_matrix = jacobian_batch(param_nn, x)
        ggn = jacobian_matrix.T @ jacobian_matrix

        # Linear algebra
        eigvals, Q = jnp.linalg.eigh(ggn)
        small_value = jnp.finfo(eigvals.dtype).eps
        UUt_kernel = Q @ jnp.diag(eigvals < small_value) @ Q.T
        D = jnp.shape(UUt_kernel)[0]

        distance_matrix = (eigvals[:, None] - eigvals[None, :]) ** 2
        dist = jnp.sqrt(jnp.amin(distance_matrix + jnp.eye(len(eigvals))))
        stats = {"eigvals": eigvals, "small_value": small_value, "distance": dist}
        return UUt_kernel, D, stats

    return computes_kernel_basis


def make_alternating_projections(loader: datasets.BatchLoader, projection: Callable):
    """Returns a function that calls the `projection` function
    (returned from `projection_kernel_ggn` or
    `projection_kernel_param_to_loss`) successively over mini-batches
    from `loader`, starting from random samples and ending up with an
    approximation of the projection on the full data [1].

    [1] https://arxiv.org/abs/2410.16901

    """

    def project_batch(i, batch_carry):
        param_nn, carry_kernel_samples = batch_carry
        batch_x, batch_y = loader.dyn_batch(i)
        est_UUt_kernel, *_ = projection(param_nn, batch_x, batch_y)
        batch_kernel_samples, _ = jax.vmap(est_UUt_kernel)(carry_kernel_samples)
        return param_nn, batch_kernel_samples

    def projection_iter(_, iter_carry):
        return jax.lax.fori_loop(0, len(loader), project_batch, iter_carry)

    def alt_projections(param_nn, iso_samples, num_iter):
        _, batch_kernel_samples = jax.lax.fori_loop(
            0, num_iter, projection_iter, (param_nn, iso_samples)
        )
        return batch_kernel_samples

    return alt_projections


def make_alternating_projections_from_iterator(loader, projection: Callable):
    """Same as `make_alternating_projections`, but assumes loader is a
    regular iterable.

    """

    # This will be VERY slow without jax.jit()
    @jax.jit
    def projection_iter(param_nn, samples, batch_x, batch_y):
        est_UUt_kernel, *_ = projection(param_nn, batch_x, batch_y)
        samples, _ = jax.vmap(est_UUt_kernel)(samples)
        return samples

    def alt_projections(param_nn, iso_samples, num_iter):
        kernel_samples = iso_samples
        for _ in range(num_iter):
            for batch in loader:
                kernel_samples = projection_iter(
                    param_nn, kernel_samples, batch["image"], batch["label"]
                )
        return kernel_samples

    return alt_projections


def make_state_alternating_projections_from_iterator(loader, projection: Callable):
    """Same as `make_alternating_projections`, but assumes loader is a
    regular iterable and uses a Flax-like model state.

    """

    # This will be VERY slow without jax.jit()
    @jax.jit
    def projection_iter(state, samples, batch_x, batch_y):
        est_UUt_kernel, *_ = projection(state.params, state, batch_x, batch_y)
        samples, _ = jax.vmap(est_UUt_kernel)(samples)
        return samples

    def alt_projections(state, iso_samples, num_iter):
        kernel_samples = iso_samples
        for _ in range(num_iter):
            for batch in tqdm(loader, desc="Projecting"):
                kernel_samples = projection_iter(
                    state, kernel_samples, batch["image"], batch["label"]
                )
        return kernel_samples

    return alt_projections


def make_loss(model_apply_fn, model_unflatten, loss_single):
    # model_apply_fn = jax.vmap(model_apply_fn, in_axes=(None, 0))
    def loss(param_nn, inputs, labels):
        p = model_unflatten(param_nn)
        outputs = model_apply_fn(p, inputs)

        mean_loss = jnp.mean(jax.vmap(loss_single)(outputs, labels))
        stats = {"loss": mean_loss}
        return mean_loss, stats

    return loss


def make_state_loss(model_unflatten, loss_single, reduction_fn=jnp.mean):
    def loss(params, state, inputs, labels):
        model_apply_fn = apply_fn_from_state(state)
        p = model_unflatten(params["param_nn"])
        outputs, updates = model_apply_fn(p, inputs)

        mean_loss = reduction_fn(jax.vmap(loss_single)(outputs, labels))
        stats = {
            "loss": mean_loss,
            "preds": outputs,
            "updates": updates,
        }
        return mean_loss, stats

    return loss


def make_loss_map(model_apply_fn, model_unflatten, loss_single):
    @ft.partial(jax.vmap, in_axes=(0, None))
    def logpdf_prior(u, log_precision):
        return -jax.scipy.stats.norm.logpdf(
            u, loc=0.0, scale=jnp.exp(-0.5 * log_precision)
        )

    @jax.vmap
    def logpdf_like(u, pred):
        return loss_single(pred, u)

    def loss_map(param_nn, param_hyper, inputs, labels):
        model_params = model_unflatten(param_nn)
        preds = jax.vmap(model_apply_fn, in_axes=(None, 0))(model_params, inputs)
        log_prior = jnp.sum(logpdf_prior(param_nn, param_hyper["log_precision"]))

        log_likelihood = jnp.sum(logpdf_like(labels, preds))
        stats = {"log_prior": log_prior, "log_likelihood": log_likelihood}
        return log_prior + log_likelihood, stats

    return loss_map


def make_train_map_step(loss_map, optimizer):
    def train_map_step(param_nn, param_hyper, opt_state, inputs, labels):
        (loss_value, stats), loss_grad = jax.value_and_grad(loss_map, has_aux=True)(
            param_nn, param_hyper, inputs, labels
        )
        updates, opt_state = optimizer.update(loss_grad, opt_state)
        param_nn = optax.apply_updates(param_nn, updates)
        return loss_value, stats, param_nn, opt_state

    return train_map_step


def make_train_mle_step(loss_mle, optimizer):
    def train_mle_step(param_nn, opt_state, inputs, labels):
        (loss_value, stats), loss_grad = jax.value_and_grad(loss_mle, has_aux=True)(
            param_nn, inputs, labels
        )
        updates, opt_state = optimizer.update(loss_grad, opt_state)
        param_nn = optax.apply_updates(param_nn, updates)
        return loss_value, stats, param_nn, opt_state

    return train_mle_step


def make_train_state_mle_step(loss_mle):
    def train_mle_step(state, inputs, labels):
        (loss_value, stats), loss_grad = jax.value_and_grad(loss_mle, has_aux=True)(
            state.params, state, inputs, labels
        )
        stats["loss"] = loss_value
        updates = stats.pop("updates")
        state = state.apply_updates(grads=loss_grad, updates=updates)
        return state, stats

    return train_mle_step


def make_train_calibration_elbo_step(loss_elbo, optimizer):
    """This function uses the ELBO to calibrate `param_hyper` and
    `param_vi`, but NOT the model parameters (`param_nn`).
    """

    def train_elbo_step(
        param_hyper,
        param_vi,
        param_nn,
        opt_state,
        inputs,
        labels,
        image_samples,
        kernel_samples,
    ):
        (loss_value, stats), loss_grad = jax.value_and_grad(
            loss_elbo, argnums=(1, 2), has_aux=True
        )(
            param_nn,
            param_hyper,
            param_vi,
            x=inputs,
            y=labels,
            image_samples=image_samples,
            kernel_samples=kernel_samples,
        )
        updates, opt_state = optimizer.update(loss_grad, opt_state)
        (param_hyper, param_vi) = optax.apply_updates(
            (param_hyper, param_vi),
            updates,
        )
        return loss_value, stats, param_hyper, param_vi, opt_state

    return train_elbo_step


def make_train_elbo_step(loss_elbo, optimizer):
    """This function is for fully optimizing the model (including
    `param_hyper` and `param_vi`) using the ELBO loss."""

    def train_elbo_step(
        projection,
        param_hyper,
        param_vi,
        param_nn,
        opt_state,
        inputs,
        labels,
        iso_samples,
        beta,
    ):
        (loss_value, stats), loss_grad = jax.value_and_grad(
            loss_elbo, argnums=(1, 2, 0), has_aux=True
        )(
            param_nn,
            param_hyper,
            param_vi,
            projection=projection,
            x=inputs,
            y=labels,
            iso_samples=iso_samples,
            beta=beta,
        )
        updates, opt_state = optimizer.update(
            loss_grad, opt_state, (param_hyper, param_vi, param_nn)
        )
        (param_hyper, param_vi, param_nn) = optax.apply_updates(
            (param_hyper, param_vi, param_nn),
            updates,
        )
        return loss_value, stats, param_hyper, param_vi, param_nn, opt_state

    return train_elbo_step


def make_train_state_elbo_step(loss_elbo):
    """This function is for fully optimizing the model (including
    `log_precision` and `log_scale_image`) using the ELBO loss."""

    def train_elbo_step(
        projection,
        state,
        inputs,
        labels,
        iso_samples,
        beta,
    ):
        (loss_value, stats), loss_grad = jax.value_and_grad(
            loss_elbo, argnums=0, has_aux=True
        )(
            state.params,
            state,
            projection=projection,
            x=inputs,
            y=labels,
            iso_samples=iso_samples,
            beta=beta,
        )
        stats["loss"] = loss_value
        updates = stats.pop("updates")
        state = state.apply_updates(grads=loss_grad, updates=updates)
        return state, stats

    return train_elbo_step


def make_train_step_ivon(loss_mle, optimizer, num_samples):
    assert num_samples > 0

    def train_step(param_nn, opt_state, inputs, labels, key):
        sample_grads = ft.partial(ivon_sample_grads, param_nn, inputs, labels)

        def body_fun(_, state):
            key, opt_state = state
            key, grads, opt_state = sample_grads(key, opt_state)
            opt_state = ivon.accumulate_gradients(grads, opt_state)
            return key, opt_state

        if num_samples > 1:
            # Call once to determine the correct input shapes
            init_val = body_fun(0, (key, opt_state))

            # Run a bunch of MC samples
            key, opt_state = jax.lax.fori_loop(
                0, num_samples - 2, body_fun, init_val=init_val
            )

        # Do the final step + update instead of step + accumulate
        key, grads, opt_state = sample_grads(key, opt_state)
        updates, opt_state = optimizer.update(grads, opt_state, param_nn)
        param_nn = optax.apply_updates(param_nn, updates)

        # Evaluate loss and go
        loss_value, stats = loss_mle(param_nn, inputs, labels)
        return loss_value, stats, param_nn, opt_state, key

    def ivon_sample_grads(param_nn, inputs, labels, key, opt_state):
        key, subkey = jax.random.split(key, num=2)
        param_sample, opt_state = ivon.sample_parameters(subkey, param_nn, opt_state)
        grads, _ = jax.grad(loss_mle, has_aux=True)(param_sample, inputs, labels)
        return key, grads, opt_state

    return train_step


def make_train_state_step_ivon(loss_mle, num_samples):
    assert num_samples > 0

    def train_step(state, inputs, labels, key):
        sample_grads = ft.partial(ivon_sample_grads, state, inputs, labels)

        def body_fun(_, loop_state):
            key, opt_state = loop_state
            key, grads, opt_state = sample_grads(key, opt_state)
            opt_state = ivon.accumulate_gradients(grads, opt_state)
            return key, opt_state

        if num_samples > 1:
            # Call once to determine the correct input shapes
            init_val = body_fun(0, (key, state.opt_state))

            # Run a bunch of MC samples
            key, opt_state = jax.lax.fori_loop(
                0, num_samples - 2, body_fun, init_val=init_val
            )
        else:
            opt_state = state.opt_state

        # Do the final step + update instead of step + accumulate
        key, grads, opt_state = sample_grads(key, opt_state)
        state = state.replace(opt_state=opt_state)

        # Evaluate loss and go
        loss_value, stats = loss_mle(state.params, state, inputs, labels)
        stats["loss"] = loss_value
        updates = stats.pop("updates")
        state = state.apply_updates(grads=grads, updates=updates)
        return state, stats, key

    def ivon_sample_grads(state, inputs, labels, key, opt_state):
        key, subkey = jax.random.split(key, num=2)
        param_sample, opt_state = ivon.sample_parameters(
            subkey, state.params, opt_state
        )
        grads, _ = jax.grad(loss_mle, has_aux=True)(param_sample, state, inputs, labels)
        return key, grads, opt_state

    return train_step


def make_train_state_step_sgmc(log_likelihood_fn, log_prior_fn, num_batches):
    def log_prob_fn(params, state, inputs, labels):
        log_likelihood, stats = log_likelihood_fn(params, state, inputs, labels)
        log_prior = log_prior_fn(params)
        stats["log_likelihood"] = log_likelihood
        stats["log_prior"] = log_prior
        return log_prior + log_likelihood * num_batches, stats

    value_and_grad_fn = jax.value_and_grad(log_prob_fn, has_aux=True)

    def train_step(state, inputs, labels, key):
        (log_posterior, stats), grads = value_and_grad_fn(
            state.params, state, inputs, labels
        )
        updates = stats.pop("updates")
        state = state.apply_updates(grads=grads, updates=updates, key=key)
        return state, stats, key

    return train_step


def make_elbo(expectation):
    def elbo(
        param_nn,
        param_hyper: dict,
        param_vi: dict,
        *,
        projection,
        x,
        y,
        iso_samples,
        beta,
    ):
        UUt_kernel, D, projection_stats = projection(param_nn, x, y)
        kernel_samples, kernel_stats = jax.vmap(UUt_kernel)(iso_samples)
        # kernel_samples = jax.lax.stop_gradient(kernel_samples)
        image_samples = iso_samples - kernel_samples  # shape is (samples_mc, D)
        # image_samples = jax.lax.stop_gradient(image_samples)

        # Stochastic trace estimation
        # Alternatives:
        # - Mean sq. norms of kernel samples:
        # kernel_norms = jnp.sum(jnp.square(kernel_samples), axis=-1)
        # R = jnp.mean(kernel_norms, axis=0)
        # - Nyström++:
        # R = nystroem_pp(kernel_samples, iso_samples)
        # - Hutchinson:
        R = jnp.mean(jax.vmap(jnp.dot)(iso_samples, kernel_samples), axis=-1)
        R = jnp.clip(R, max=D - 1)
        R = jax.lax.stop_gradient(R)

        # Compute loss terms
        exp, exp_stats = expectation(
            param_nn,
            param_hyper,
            param_vi,
            x=x,
            y=y,
            image_samples=image_samples,
            kernel_samples=kernel_samples,
        )
        kl = elbo_kldiv(
            param_nn,
            param_hyper,
            param_vi,
            R=R,
            D=D,
        )
        stats = {
            "R": R,
            "D": D,
            "E[]": exp,
            "kl": kl * beta,
            "dot": jnp.mean(jax.vmap(jnp.dot)(kernel_samples, image_samples)),
            "kernel_samples": kernel_samples,
            "preds": exp_stats["preds"],
            "precond_min": jnp.min(kernel_stats["precond"]),
            "precond_max": jnp.max(kernel_stats["precond"]),
            "residuals": jnp.mean(kernel_stats["residuals"]),
        }
        return exp + beta * kl, stats

    return elbo


def make_state_elbo(expectation, precomputed_projections=False):
    def elbo_from_precomputed_samples(
        params,
        state,
        *,
        x,
        y,
        kernel_samples,
        iso_samples,
        beta,
    ):
        image_samples = iso_samples - kernel_samples  # shape is (samples_mc, D)
        # image_samples = jax.lax.stop_gradient(image_samples)

        # Stochastic trace estimation
        # Alternatives:
        # - Mean sq. norms of kernel samples:
        # kernel_norms = jnp.sum(jnp.square(kernel_samples), axis=-1)
        # R = jnp.mean(kernel_norms, axis=0)
        # - Nyström++:
        # R = nystroem_pp(kernel_samples, iso_samples)
        # - Hutchinson:
        D = iso_samples.shape[-1]
        R = jnp.mean(jax.vmap(jnp.dot)(iso_samples, kernel_samples), axis=-1)
        R = jnp.clip(R, max=D - 1)
        R = jax.lax.stop_gradient(R)

        # Compute loss terms
        exp, exp_stats = expectation(
            params,
            state,
            x=x,
            y=y,
            image_samples=image_samples,
            kernel_samples=kernel_samples,
        )
        kl = state_elbo_kldiv(params, R=R, D=D)
        stats = {
            "R": R,
            "D": D,
            "E[]": exp,
            "kl": beta * kl,
            "dot": jnp.mean(jax.vmap(jnp.dot)(kernel_samples, image_samples)),
            "kernel_samples": kernel_samples,
            "preds": exp_stats["preds"],
            "updates": exp_stats["updates"],
        }
        return exp + beta * kl, stats

    def elbo(
        params,
        state,
        *,
        projection,
        x,
        y,
        iso_samples,
        beta,
    ):
        UUt_kernel, D, projection_stats = projection(params, state, x, y)
        kernel_samples, kernel_stats = jax.vmap(UUt_kernel)(iso_samples)
        # kernel_samples = jax.lax.stop_gradient(kernel_samples)
        loss, stats = elbo_from_precomputed_samples(
            params,
            state,
            x=x,
            y=y,
            kernel_samples=kernel_samples,
            iso_samples=iso_samples,
            beta=beta,
        )
        stats["precond_min"] = jnp.min(kernel_stats["precond"])
        stats["precond_max"] = jnp.max(kernel_stats["precond"])
        stats["residuals"] = jnp.mean(kernel_stats["residuals"])
        return loss, stats

    if precomputed_projections:
        return elbo_from_precomputed_samples
    return elbo


def make_calibration_elbo(expectation):
    """This should be used in tandem with `make_train_calibration_elbo_step`."""

    def elbo(
        param_nn,
        param_hyper: dict,
        param_vi: dict,
        *,
        x,
        y,
        image_samples,
        kernel_samples,
    ):
        # Stochastic trace estimation
        D = len(param_nn)
        kernel_norms = jnp.sum(jnp.square(kernel_samples), axis=-1)
        R = jnp.clip(jnp.mean(kernel_norms, axis=0), max=D)

        # Compute loss terms
        exp = expectation(
            param_nn,
            param_hyper,
            param_vi,
            x=x,
            y=y,
            image_samples=image_samples,
            kernel_samples=kernel_samples,
        )
        kl = elbo_kldiv(
            param_nn,
            param_hyper,
            param_vi,
            R=R,
            D=D,
        )
        stats = {"R": R, "D": D, "E[]": exp, "kl": kl}
        return exp + kl, stats

    return elbo


def elbo_expectation(model_apply_fn, model_unflatten, loss_single, is_linearized=False):
    # loss: typically, either optax.losses.sigmoid_binary_cross_entropy
    sample_posterior = make_sampler(model_apply_fn, model_unflatten, is_linearized)

    def expectation(
        param_nn, param_hyper, param_vi, *, x, y, image_samples, kernel_samples
    ):
        def logpdf(pred):
            return loss_single(pred, y)

        # preds will be of shape (kernel_samples.shape[0], x.shape[0])
        preds = sample_posterior(
            param_nn,
            log_scale_kernel=-0.5 * param_hyper["log_precision"],
            log_scale_image=param_vi["log_scale_image"],
            x_eval=x,
            image_samples=image_samples,
            kernel_samples=kernel_samples,
        )
        logpdfs = jnp.sum(jax.vmap(logpdf)(preds), axis=-1)
        stats = {"preds": preds}
        return jnp.mean(logpdfs, axis=0), stats

    return expectation


def state_elbo_expectation(model_unflatten, loss_single, is_linearized=False):
    # loss: typically, either optax.losses.sigmoid_binary_cross_entropy
    sample_posterior = make_state_sampler(model_unflatten, is_linearized)

    def expectation(params, state, *, x, y, image_samples, kernel_samples):
        def logpdf(pred):
            return loss_single(pred, y)

        # preds will be of shape (kernel_samples.shape[0], x.shape[0])
        preds, updates = sample_posterior(
            params,
            state,
            x_eval=x,
            image_samples=image_samples,
            kernel_samples=kernel_samples,
        )
        logpdfs = jnp.sum(jax.vmap(logpdf)(preds), axis=-1)
        stats = {
            "preds": preds,
            # Make sure we average updates over every "model" we sampled from
            "updates": jax.tree.map(ft.partial(jnp.mean, axis=0), updates),
        }
        return jnp.mean(logpdfs, axis=0), stats

    return expectation


def _elbo_kldiv(param_nn: jax.Array, log_precision, log_scale_image, *, R, D):
    alpha_tr_sigma = (
        R + jnp.exp(log_precision) * (D - R) * jnp.exp(log_scale_image) ** 2
    )
    logdet_sigma = -R * log_precision + 2 * (D - R) * log_scale_image
    kl = 0.5 * (
        alpha_tr_sigma
        - D
        + jnp.exp(log_precision) * jnp.sum(jnp.square(param_nn))
        - D * log_precision
        - logdet_sigma
    )
    return kl


def state_elbo_kldiv(params, *, R, D):
    return _elbo_kldiv(
        params["param_nn"],
        log_precision=params["log_precision"],
        log_scale_image=params["log_scale_image"],
        R=R,
        D=D,
    )


def elbo_kldiv(param_nn: jax.Array, param_hyper, param_vi, *, R, D):
    return _elbo_kldiv(
        param_nn,
        log_precision=param_hyper["log_precision"],
        log_scale_image=param_vi["log_scale_image"],
        R=R,
        D=D,
    )
