import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Embedding import Embedding
from Recovery import Recovery
from Generator import Generator
from Discriminator import Discriminator
from Supervisor import Supervisor


class TimeGAN(nn.Module):
    def __init__(self, cfg):
        super(TimeGAN, self).__init__()
        self.device = cfg['system']['device']

        # Architecture
        self.emb = Embedding(cfg)
        self.rec = Recovery(cfg)
        self.g = Generator(cfg)
        self.d = Discriminator(cfg)
        self.sup = Supervisor(cfg)

    def _recovery_forward(self, x, t):
        # Forward pass
        h = self.emb(x, t)
        _x = self.rec(h, t)

        # Joint training
        _h_sup = self.sup(h, t)

        # Teacher forcing next step
        g_loss_sup = F.mse_loss(
            _h_sup[:, :-1, :],
            h[:, 1:, :]
        )

        # Compute reconstruction loss
        # Reconstruction Loss
        e_loss_t0 = F.mse_loss(_x, x)
        e_loss0 = 10 * torch.sqrt(e_loss_t0)
        e_loss = e_loss0 + 0.1 * g_loss_sup
        return e_loss, e_loss0, e_loss_t0

    def _supervisor_forward(self, x, t):
        # Supervisor forward pass
        h = self.emb(x, t)
        _h_sup = self.sup(h, t)

        # Supervised loss

        # Teacher forcing next output
        s_loss = F.mse_loss(
            _h_sup[:, :-1, :],
            h[:, 1:, :]
        )

        return s_loss

    def _discriminator_forward(self, x, t, z, gamma=1.0):
        # Discriminator forward pass and adversarial loss

        h = self.emb(x, t).detach()
        _h = self.sup(h, t).detach()
        _e = self.g(z, t).detach()

        # Forward Pass
        y_real = self.d(h, t)  # Encoded original data
        y_fake = self.d(_h, t)  # Output of supervisor
        y_fake_e = self.d(_e, t)  # Output of generator

        d_loss_real = F.binary_cross_entropy_with_logits(y_real, torch.ones_like(y_real))
        d_loss_fake = F.binary_cross_entropy_with_logits(y_fake, torch.zeros_like(y_fake))
        d_loss_fake_e = F.binary_cross_entropy_with_logits(y_fake_e, torch.zeros_like(y_fake_e))

        d_loss = d_loss_fake + d_loss_real + gamma * d_loss_fake_e

        return d_loss

    def _generator_forward(self, x, t, z, gamma=1.0):
        # Supervised Forward Pass
        h = self.emb(x, t)
        _h_sup = self.sup(h, t)
        _x = self.rec(h, t)

        # Generator Forward Pass
        _e = self.g(z, t)
        _h = self.sup(_e, t)

        # Synthetic generated data
        _x = self.rec(_h, t)  # recovered data

        # Generator Loss

        # 1. Adversarial loss
        y_fake = self.d(_h, t)  # Output of supervisor
        y_fake_e = self.d(_e, t)  # Output of generator

        g_loss_u = F.binary_cross_entropy_with_logits(y_fake, torch.ones_like(y_fake))
        g_loss_u_e = F.binary_cross_entropy_with_logits(y_fake_e, torch.ones_like(y_fake_e))

        # 2. Supervised loss
        g_loss_s = torch.nn.functional.mse_loss(_h_sup[:, :-1, :], h[:, 1:, :])  # Teacher forcing next output

        # 3. Two Moments
        g_loss_v1 = torch.mean(torch.abs(
            torch.sqrt(_x.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(x.var(dim=0, unbiased=False) + 1e-6)))
        g_loss_v2 = torch.mean(torch.abs((_x.mean(dim=0)) - (x.mean(dim=0))))

        g_loss_v = g_loss_v1 + g_loss_v2

        # 4. Sum
        g_loss = g_loss_u + gamma * g_loss_u_e + 100 * torch.sqrt(g_loss_s) + 100 * g_loss_v

        return g_loss

    def _inference(self, z, t):
        # Generate synthetic data

        # Generator Forward Pass
        _e = self.g(z, t)
        _h = self.sup(_e, t)

        # Synthetic generated data (reconstructed)
        _x = self.rec(_h, t)
        return _x

    def forward(self, x, t, z, stage, gamma=1.0):
        if stage != 'inference':
            if x is None:
                raise ValueError("x is not given")
            x = torch.FloatTensor(x)
            x = x.to(self.device)

        if z is not None:
            z = torch.FloatTensor(z)
            z = z.to(self.device)

        if stage == 'embedding':
            # Embedding & Recovery
            loss = self._recovery_forward(x, t)
            return loss

        elif stage == 'supervisor':
            # Supervisor
            loss = self._supervisor_forward(x, t)
            return loss

        elif stage == 'generator':
            if z is None:
                raise ValueError("z is not given")

            # Generator
            loss = self._generator_forward(x, t, z, gamma)
            return loss

        elif stage == 'discriminator':
            if z is None:
                raise ValueError("z is not given")

            # Discriminator
            loss = self._discriminator_forward(x, t, z, gamma)
            return loss

        elif stage == 'inference':
            _x = self._inference(z, t)
            _x = _x.cpu().detach()
            return _x

        else:
            raise ValueError('stage should be either [embedding, supervisor, generator, discriminator, inference]')

    @property
    def device(self):
        return next(self.parameters()).device

    @device.setter
    def device(self, value):
        self._device = value


def run_time_gan_test():
    cfg = {
        "emb": {
            "dim_features": 5,  # feature dimension
            "dim_hidden": 100,  # latent space dimension
            "num_layers": 50  # number of layers in GRU
        },
        "g": {
            "dim_latent": 64,  # Z (input latent space dimension) size (eq. 128) [ INPUT ]
            "dim_hidden": 100,  # representation latent space dimension [ ENCODING ]
            "num_layers": 50  # number of layers in GRU
        },
        "d": {
            "dim_hidden": 100,  # representation latent space dimension (H)
            "num_layers": 50  # number of layers in GRU
        },
        "rec": {
            "dim_output": 5,  # output feature dimension
            "dim_hidden": 100,  # latent space dimension
            "num_layers": 50  # number of layers in GRU
        },
        "sup": {
            "dim_features": 5,  # feature dimension (unused - middleware network -)
            "dim_hidden": 100,  # latent space dimension (H)
            "num_layers": 50  # number of layers in GRU
        },
        "system": {
            "seq_len": 150,
            "padding_value": 0.0,  # default on 0.0
            "device": "cuda:0"
        }
    }

    tgan = TimeGAN(cfg)

    z = torch.randn(size=(10, 150, 64))
    x = torch.randn(size=(10, 150, 5))
    t = torch.ones(size=(10,))

    embedding_stage_loss = tgan(x, t, z, "embedding")

    supervisor_stage_loss = tgan(x, t, z, "supervisor")

    discriminator_stage_loss = tgan(x, t, z, "discriminator")

    generator_stage_loss = tgan(x, t, z, "generator")

    inference_stage_loss = tgan(x, t, z, "inference")


def time_gan_trainer(model: TimeGAN, data: np.ndarray, time: np.ndarray, cfg) -> None:
    # Init dataset
    pass


if __name__ == '__main__':
    run_time_gan_test()
