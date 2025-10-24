import pickle

import flax.linen as nn
import jax
import jax.numpy as jnp


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


class CelebADecoder(nn.Module):
    def setup(self):
        self.fc_dec = nn.Sequential(
            [
                nn.Dense(256 * 8 * 8),
                nn.elu,
            ]
        )
        self.convs = nn.Sequential(
            [
                ResizeAndConv(128, kernel_size=(4, 4), strides=(1, 1)),
                nn.elu,
                ResizeAndConv(64, kernel_size=(4, 4), strides=(2, 2)),
                nn.elu,
                ResizeAndConv(32, kernel_size=(4, 4), strides=(2, 2)),
                nn.elu,
                ResizeAndConv(16, kernel_size=(4, 4), strides=(2, 2)),
                nn.elu,
                ResizeAndConv(3, kernel_size=(4, 4), strides=(1, 1)),
            ]
        )

    def __call__(self, x, train=True):
        x_dec = self.fc_dec(x)
        x_dec = x_dec.reshape(-1, 8, 8, 256)
        x_dec = self.convs(x_dec)
        return x_dec


class VAE(nn.Module):
    z_dim: int = 64
    rng_name: str = "reparam_key"

    def setup(self):
        self.encoder = nn.Sequential(
            [
                nn.Conv(16, kernel_size=(4, 4), strides=(1, 1)),
                nn.elu,
                nn.Conv(32, kernel_size=(4, 4), strides=(2, 2)),
                nn.elu,
                nn.Conv(64, kernel_size=(4, 4), strides=(2, 2)),
                nn.elu,
                nn.Conv(128, kernel_size=(4, 4), strides=(2, 2)),
                nn.elu,
                nn.Conv(256, kernel_size=(4, 4), strides=(1, 1)),
                nn.elu,
            ]
        )
        self.enc = nn.Sequential(
            [
                nn.Dense(256),
                nn.elu,
            ]
        )
        self.enc_mu = nn.Dense(self.z_dim)
        self.enc_logvar = nn.Dense(self.z_dim)

        self.decoder = CelebADecoder()

    def encode(self, x, reparam=False, reparam_key=None):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = self.enc(x)
        z_mu = self.enc_mu(x)
        z_logvar = self.enc_logvar(x)
        if not reparam:
            return z_mu, z_mu, z_logvar
        z = self.reparametrize(z_mu, z_logvar, reparam_key) if reparam else z_mu
        return z, z_mu, z_logvar

    def reparametrize(self, mu, logvar, key):
        std = jax.random.normal(key, (mu.shape[0], mu.shape[1]))
        return mu + jnp.exp(0.5 * logvar) * std

    def __call__(self, x, train):
        if not train:
            return self.encode(x, True, self.make_rng(self.rng_name))
        z, z_mu, z_logvar = self.encode(x, True, self.make_rng(self.rng_name))
        x_dec = self.decode(z)
        return x_dec, z_mu, z_logvar

    def decode(self, z):
        return self.decoder(z)

    def dump(self, path):
        with open(path, "wb") as file:
            pickle.dump({"state": self.state_dict()}, file)

    def load(self, path):
        with open(path, "rb") as file:
            data = pickle.load(file)
