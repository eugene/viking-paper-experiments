import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv


def k_colors(k, s=1.0, v=1.0):
    hues = np.linspace(0, 1, k, endpoint=False) + 2 / (3 * k)
    s = np.full(k, s)
    v = np.full(k, v)
    colors = np.stack((hues, s, v), axis=0)
    colors = colors.T
    colors = hsv_to_rgb(colors)
    return colors


def plot_class_simplex(ax, probs, labels=None, cmap=None, random_state=0):
    """Creates a class simplex to plot the class probabilities `probs`
    on, using `labels` as the ground-truth coloring.

    The function will also accept `probs` of shape `(num_samples,)`
    and will assume a binary classification scenario.

    Args:
      ax: an `Axis` instance from `matplotlib`
      probs: float array of shape `(num_points, num_classes)` with class probabilities
      labels: integer array of shape `(num_points,)` with class labels

    """
    # If we get only one probability per sample, then make our lives
    # easier by stacking the opposite probabilities as a second class
    should_make_two_classes = len(probs.shape) == 1 or probs.shape[1] == 1
    if len(probs.shape) > 1 and probs.shape[1] == 1:
        probs = probs[:, 0]
    if should_make_two_classes:
        probs = jnp.stack((probs, 1.0 - probs), axis=-1)
    num_classes = probs.shape[1]

    # Plotting on a line (2-vertex simplex) is boring, make sure we
    # add some jitter to the points to make the plot more interesting
    add_jitter = num_classes == 2

    rp = matplotlib.patches.RegularPolygon(
        (0, 0),
        num_classes,
        radius=1,
        orientation=(num_classes == 2) * (-0.5) * np.pi,
        fill=False,
        ec="darkgray",
    )
    ax.add_patch(rp)
    plot_basis = rp.get_path().vertices[:-1]
    # This ensures our basis also follows transformations such as the
    # `orientation` argument in the constructor of `RegularPolygon`
    plot_basis = rp.get_patch_transform().transform(plot_basis)
    xy = np.dot(probs, plot_basis)
    if add_jitter:
        key = jax.random.PRNGKey(random_state)
        xy[:, 1] = xy[:, 1] + jax.random.normal(key, xy[:, 1].shape) * 0.05
    if labels is not None and cmap is not None:
        color_kwargs = {"c": labels, "cmap": cmap}
    else:
        color_kwargs = {"c": "k"}
    ax.scatter(xy[:, 0], xy[:, 1], s=2, alpha=0.2, **color_kwargs)
    # for i in range(2):
    #     ax.plot([], [], marker="o", color=colors[i], label=str(i))


def plot_class_simplex_ext(ax, probs, labels=None, cmap=None, random_state=0):
    """Creates a class simplex to plot the class probabilities `probs`
    on, using `labels` as the ground-truth coloring.

    Args:
      ax: an `Axis` instance from `matplotlib`
      probs: float array of shape `(num_samples, num_data_points, num_classes)`
             with class probabilities
      labels: integer array of shape `(num_samples,)` with class labels

    """
    num_samples, num_data_points, num_classes = probs.shape
    rp = matplotlib.patches.RegularPolygon(
        (0, 0),
        num_classes,
        radius=1,
        fill=False,
        ec="darkgray",
    )
    ax.add_patch(rp)
    plot_basis = rp.get_path().vertices[:-1]
    # This ensures our basis also follows transformations such as the
    # `orientation` argument in the constructor of `RegularPolygon`
    plot_basis = rp.get_patch_transform().transform(plot_basis)
    optimal_permutation = jnp.array([0, 3, 2, 1, 4])
    # optimal_permutation = jnp.array([4, 1, 3, 0, 2]) # using logit means
    # optimal_permutation = jnp.array([0, 2, 4, 1, 3]) # using all logits
    # optimal_permutation = jnp.array([3, 4, 1, 0, 2]) # using softmax
    xy = np.einsum("ijk,kl->ijl", probs[:, :, optimal_permutation], plot_basis)
    if cmap is None:
        cmap = lambda _: "k"
    for i in range(num_data_points):
        ax.plot(
            xy[:, i, 0],
            xy[:, i, 1],
            linestyle="-",
            color=cmap(labels[i]),
            alpha=0.05,
        )
    # for i in range(2):
    #     ax.plot([], [], marker="o", color=colors[i], label=str(i))
