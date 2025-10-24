import pickle

import jax
from flax import linen as nn
from jax.numpy import exp


class ResizeAndConv(nn.Module):
    features: int
    kernel_size: tuple
    strides: tuple

    def setup(self):
        self.conv = nn.Conv(
            features=self.features, kernel_size=self.kernel_size, strides=(1, 1)
        )

    def __call__(self, x):
        if self.strides != (1, 1):
            x = jax.image.resize(
                x,
                (
                    x.shape[0],
                    x.shape[1] * self.strides[0],
                    x.shape[2] * self.strides[1],
                    x.shape[3],
                ),
                method="nearest",
            )

        x = self.conv(x)
        return x


class FmnistDecoder(nn.Module):
    @nn.compact
    def __call__(self, x, train=True):
        x_dec = nn.Dense(256 * 4)(x)
        x_dec = nn.elu(x_dec)
        x_dec = x_dec.reshape((x.shape[0], 2, 2, 256))

        x_dec = ResizeAndConv(128, kernel_size=(3, 3), strides=(2, 2))(x_dec)
        x_dec = nn.elu(x_dec)
        x_dec = ResizeAndConv(64, kernel_size=(3, 3), strides=(1, 1))(x_dec)
        x_dec = nn.elu(x_dec)
        x_dec = ResizeAndConv(32, kernel_size=(3, 3), strides=(2, 2))(x_dec)
        x_dec = nn.elu(x_dec)
        x_dec = ResizeAndConv(16, kernel_size=(3, 3), strides=(2, 2))(x_dec)
        x_dec = nn.elu(x_dec)
        x_dec = ResizeAndConv(1, kernel_size=(3, 3), strides=(2, 2))(x_dec)
        x_dec = nn.elu(x_dec)

        x_dec = jax.image.resize(
            x_dec,
            (x_dec.shape[0], 28, 28, 1),
            method="nearest",
        )
        return x_dec


class VAE(nn.Module):
    z_dim: int = 64
    rng_name: str = "reparam_key"

    def setup(self):
        self.encoder = nn.Sequential(
            [
                nn.Conv(
                    16, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
                ),  # 28 → 14
                nn.elu,
                nn.Conv(
                    32, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
                ),  # 14 → 7
                nn.elu,
                nn.Conv(
                    64, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
                ),  # 7 → 4
                nn.elu,
                nn.Conv(
                    128, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
                ),  # 4 → 2
                nn.elu,
                nn.Conv(
                    256, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
                ),  # 2 → 1
                nn.elu,
            ]
        )
        self.enc_mu = nn.Dense(self.z_dim)
        self.enc_logvar = nn.Dense(self.z_dim)

        self.decoder = FmnistDecoder()

    def encode(self, x, reparam: bool, rng):
        x = self.encoder(x)
        x = x.reshape((x.shape[0], -1))
        z_mu = self.enc_mu(x)
        z_logvar = self.enc_logvar(x)
        if not reparam:
            return z_mu, z_mu, z_logvar
        z = self.reparametrize(z_mu, z_logvar, rng)
        return z, z_mu, z_logvar

    def reparametrize(self, mu, logvar, rng):
        std = jax.random.normal(rng, mu.shape)
        return mu + exp(0.5 * logvar) * std

    def decode(self, z):
        return self.decoder(z)

    def __call__(self, x, train=True):
        if not train:
            return self.encode(x, train, self.make_rng(self.rng_name))
        z, z_mu, z_logvar = self.encode(x, train, self.make_rng(self.rng_name))
        x_dec = self.decode(z)
        return x_dec, z_mu, z_logvar

    def dump(self, path):
        with open(path, "wb") as file:
            pickle.dump({"state": self.state_dict()}, file)

    def load(self, path):
        with open(path, "rb") as file:
            data = pickle.load(file)
        return data["variables"]
