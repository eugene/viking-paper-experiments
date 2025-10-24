from functools import partial
from typing import Callable

import jax
from jax import numpy as jnp


@partial(
    jax.jit, static_argnames=("model_fn", "output_dims", "n_iterations", "acceleration")
)
def kernel_proj_vp(
    vec,
    model_fn: Callable,
    params,
    x_train_batched: jnp.ndarray,
    batched_eigvecs: jnp.ndarray,
    batched_inv_eigvals: jnp.ndarray,
    output_dims: int,
    n_iterations: int,
    acceleration: bool = False,
):
    def orth_proj_vp(v, x, eigvecs, inv_eigvals):
        lmbd = lambda p: model_fn(p, x)
        _, Jv = jax.jvp(lmbd, (params,), (v,))
        JJt_inv_Jv = eigvecs.T @ Jv.reshape(-1)
        JJt_inv_Jv = eigvecs @ (inv_eigvals * JJt_inv_Jv)
        _, jtv_fn = jax.vjp(lmbd, params)
        JJt_inv_Jv = JJt_inv_Jv.reshape((x.shape[0], output_dims))
        Jt_JJt_inv_Jv = jtv_fn(JJt_inv_Jv)[0]
        return v - Jt_JJt_inv_Jv

    def proj_through_data(iter, v):
        def body_fun(carry, batch):
            x, eigvecs, inv_eigvals = batch
            pv = carry
            out = orth_proj_vp(pv, x, eigvecs, inv_eigvals)
            return out, None

        init_carry = v
        Qv, _ = jax.lax.scan(
            body_fun,
            init_carry,
            (x_train_batched, batched_eigvecs, batched_inv_eigvals),
        )  # memory error?
        if acceleration:
            t_k = v @ (v - Qv) / ((v - Qv) @ (v - Qv))
            x_k = t_k * Qv + (1 - t_k) * v
            return x_k
        else:
            return Qv

    Pv = jax.lax.fori_loop(0, n_iterations, proj_through_data, vec)
    return Pv
