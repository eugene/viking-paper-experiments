import equinox as eqx
import jax
import jax.numpy as jnp


def make_mlp(num_hidden):
    def init_fn(num_inputs=2, *, key):
        k_w1, k_b1, k_w2, k_b2, k_w3, k_b3 = jax.random.split(key, num=6)
        return dict(
            w1=jax.random.normal(k_w1, shape=(num_hidden, num_inputs)),
            b1=jax.random.normal(k_b1, shape=(num_hidden,)),
            w2=jax.random.normal(k_w2, shape=(num_hidden, num_hidden)),
            b2=jax.random.normal(k_b2, shape=(num_hidden,)),
            w3=jax.random.normal(k_w3, shape=(num_hidden,)),
            b3=jax.random.normal(k_b3, shape=()),
        )

    def apply_fn(params, x_single):
        # Prevents shape errors with matmul (@) by ensuring ndim > 0
        x = x_single + jnp.zeros((1,))
        x = params["w1"] @ x + params["b1"]
        x = jnp.tanh(x)
        x = params["w2"] @ x + params["b2"]
        x = jnp.tanh(x)
        x = params["w3"] @ x + params["b3"]
        return x

    return init_fn, apply_fn


def make_mlp_classifier(num_hidden, num_outputs):
    def init_fn(num_inputs=2, *, key):
        k_w1, k_b1, k_w2, k_b2, k_w3, k_b3 = jax.random.split(key, num=6)
        return dict(
            w1=jax.random.normal(k_w1, shape=(num_hidden, num_inputs)),
            b1=jax.random.normal(k_b1, shape=(num_hidden,)) * 0.01,
            w2=jax.random.normal(k_w2, shape=(num_hidden, num_hidden)),
            b2=jax.random.normal(k_b2, shape=(num_hidden,)) * 0.01,
            w3=jax.random.normal(k_w3, shape=(num_outputs, num_hidden)),
            b3=jax.random.normal(k_b3, shape=(num_outputs,)) * 0.01,
        )

    def apply_fn(params, x_single):
        # Prevents shape errors with matmul (@) by ensuring ndim > 0
        x = x_single + jnp.zeros((1,))
        x = params["w1"] @ x + params["b1"]
        x = jnp.tanh(x)
        x = params["w2"] @ x + params["b2"]
        x = jnp.tanh(x)
        x = params["w3"] @ x + params["b3"]
        return x

    return init_fn, apply_fn


def conv2d(params, x, **kwargs):
    kernel, bias = params
    x = jax.lax.conv(
        jnp.expand_dims(x, axis=0),  # lhs = NCHW image tensor
        kernel,  # rhs = OIHW conv kernel tensor
        padding="SAME",
        **kwargs,
    )
    x = jnp.squeeze(x, axis=0)
    return x + bias


def make_conv_classifier__(num_hidden_channels, num_classes):
    init_w = jax.nn.initializers.glorot_normal()

    def conv_weights(
        key,
        num_out: int,
        num_in: int,
        kernel_size: tuple,
    ):
        return (
            # kernel
            init_w(key, (num_out, num_in, *kernel_size)),
            # bias
            jnp.zeros((num_out, 1, 1)),
        )

    def init_fn(num_input_channels, key):
        key_conv1, key_conv2, key_conv3, key_conv4, key_conv5, key_linear = (
            jax.random.split(key, num=6)
        )
        linear_input_size = 2 * num_hidden_channels * 4 * 4
        return {
            "conv1": conv_weights(
                key_conv1,
                num_hidden_channels,
                num_input_channels,
                kernel_size=(3, 3),
            ),
            "conv2": conv_weights(
                key_conv2,
                num_hidden_channels,
                num_hidden_channels,
                kernel_size=(3, 3),
            ),
            "conv3": conv_weights(
                key_conv3,
                2 * num_hidden_channels,
                num_hidden_channels,
                kernel_size=(3, 3),
            ),
            "conv4": conv_weights(
                key_conv4,
                2 * num_hidden_channels,
                2 * num_hidden_channels,
                kernel_size=(3, 3),
            ),
            "conv5": conv_weights(
                key_conv5,
                2 * num_hidden_channels,
                2 * num_hidden_channels,
                kernel_size=(3, 3),
            ),
            "linear": (
                init_w(key_linear, (num_classes, linear_input_size)),
                jnp.zeros((num_classes,)),
            ),
        }

    def apply_fn(params, x):
        # 28x28 => 14x14
        x = conv2d(params["conv1"], x, window_strides=(2, 2))
        x = jax.nn.gelu(x)
        x = conv2d(params["conv2"], x, window_strides=(1, 1))
        x = jax.nn.gelu(x)
        # 14x14 => 7x7
        x = conv2d(params["conv3"], x, window_strides=(2, 2))
        x = jax.nn.gelu(x)
        x = conv2d(params["conv4"], x, window_strides=(1, 1))
        x = jax.nn.gelu(x)
        # 7x7 => 4x4
        x = conv2d(params["conv5"], x, window_strides=(2, 2))
        x = jax.nn.gelu(x)
        x = jnp.ravel(x)
        w, b = params["linear"]
        x = w @ x + b
        return x

    return init_fn, apply_fn


class ConvClassifier(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    conv4: eqx.nn.Conv2d
    conv5: eqx.nn.Conv2d
    linear: eqx.nn.Linear

    def __init__(self, num_input_channels, num_hidden_channels, num_classes, *, key):
        key_conv1, key_conv2, key_conv3, key_conv4, key_conv5, key_linear = (
            jax.random.split(key, num=6)
        )
        self.conv1 = eqx.nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=num_hidden_channels,
            padding=1,
            kernel_size=3,
            stride=2,
            padding_mode="REPLICATE",
            key=key_conv1,
        )
        self.conv2 = eqx.nn.Conv2d(
            in_channels=num_hidden_channels,
            out_channels=num_hidden_channels,
            padding=1,
            kernel_size=3,
            padding_mode="REPLICATE",
            key=key_conv2,
        )
        self.conv3 = eqx.nn.Conv2d(
            in_channels=num_hidden_channels,
            out_channels=2 * num_hidden_channels,
            padding=1,
            kernel_size=3,
            stride=2,
            padding_mode="REPLICATE",
            key=key_conv3,
        )
        self.conv4 = eqx.nn.Conv2d(
            in_channels=2 * num_hidden_channels,
            out_channels=2 * num_hidden_channels,
            padding=1,
            kernel_size=3,
            padding_mode="REPLICATE",
            key=key_conv4,
        )
        self.conv5 = eqx.nn.Conv2d(
            in_channels=2 * num_hidden_channels,
            out_channels=2 * num_hidden_channels,
            padding=1,
            kernel_size=3,
            stride=2,
            padding_mode="REPLICATE",
            key=key_conv5,
        )
        in_features = 2 * num_hidden_channels * 4 * 4
        self.linear = eqx.nn.Linear(
            in_features=in_features, out_features=num_classes, key=key_linear
        )

    def __call__(self, x):
        x = self.conv1(x)
        x = jax.nn.gelu(x)
        x = self.conv2(x)
        x = jax.nn.gelu(x)
        x = self.conv3(x)
        x = jax.nn.gelu(x)
        x = self.conv4(x)
        x = jax.nn.gelu(x)
        x = self.conv5(x)
        x = jax.nn.gelu(x)
        x = jnp.ravel(x)
        x = self.linear(x)
        return x


class LeNet(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    linear_out: eqx.nn.Linear

    def __init__(self, num_input_channels, num_classes=10, *, key):
        key, subkey = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding_mode="REPLICATE",
            key=subkey,
        )

        key, subkey = jax.random.split(key)
        self.conv2 = eqx.nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding_mode="REPLICATE",
            key=subkey,
        )

        key, subkey = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(
            in_features=256,
            out_features=120,
            key=subkey,
        )

        key, subkey = jax.random.split(key)
        self.linear2 = eqx.nn.Linear(
            in_features=120,
            out_features=84,
            key=subkey,
        )

        key, subkey = jax.random.split(key)
        self.linear_out = eqx.nn.Linear(
            in_features=84,
            out_features=num_classes,
            key=subkey,
        )

    def __call__(self, x):
        pool = eqx.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=0,
        )

        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = pool(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        x = pool(x)
        # x = jnp.transpose(x, (2, 0, 1))
        x = jnp.ravel(x)
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        x = jax.nn.relu(x)
        x = self.linear_out(x)
        return x


def make_conv_classifier(num_classes):
    def init_fn(*args, **kwargs):
        return LeNet(
            *args,
            num_classes=num_classes,
            **kwargs,
        )

    def apply_fn(params, x):
        return params(x)

    return init_fn, apply_fn
