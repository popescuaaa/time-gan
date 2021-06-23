import torch
from torch import Tensor
from typing import Tuple, Any
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from torch.utils.data import DataLoader, Dataset
from data import Energy, SineWave, Stock
import yaml
import wandb
from metrics import visualisation
from torch.optim import Optimizer, Adam
import numpy as np
import os


class Embedding(nn.Module):
    """

        ENCODER

    """

    def __init__(self, cfg: Dict):
        super(Embedding, self).__init__()
        self.dim_features = int(cfg['emb']['dim_features'])
        self.dim_hidden = int(cfg['emb']['dim_hidden'])
        self.num_layers = int(cfg['emb']['num_layers'])
        self.seq_len = int(cfg['system']['seq_len'])

        # Dynamic RNN input
        self.padding_value = int(cfg['system']['padding_value'])

        # Architecture
        self.emb_rnn = nn.GRU(input_size=self.dim_hidden,
                              hidden_size=self.dim_hidden,
                              num_layers=self.num_layers,
                              batch_first=True)
        self.emb_linear = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.norm = nn.LayerNorm(self.dim_hidden)

        self.mlp = nn.Sequential(
            nn.Linear(self.dim_features, self.dim_hidden),
            nn.LayerNorm(self.dim_hidden),
            nn.ReLU(),
            nn.Linear(self.dim_hidden, self.dim_hidden)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
            :param x: time series batch * sequence_len * features
            :param t: temporal information batch * 1
            :return: (H) latent space embeddings batch * sequence_len * H
        """
        x = self.mlp(x)
        x_packed = nn.utils.rnn.pack_padded_sequence(input=x,
                                                     lengths=t,
                                                     batch_first=True,
                                                     enforce_sorted=True)

        h_0, _ = self.emb_rnn(x_packed)
        h_0, _ = nn.utils.rnn.pad_packed_sequence(sequence=h_0,
                                                  batch_first=True,
                                                  padding_value=self.padding_value,
                                                  total_length=self.seq_len)

        h_0 = self.norm(h_0)
        embedded_x = self.emb_linear(h_0)

        return embedded_x

    @property
    def device(self):
        return next(self.parameters()).device


def run_embedding_test() -> None:
    cfg = {
        "emb": {
            "dim_features": 5,  # feature dimension
            "dim_hidden": 100,  # latent space dimension
            "num_layers": 50  # number of layers in GRU
        },
        "system": {
            "seq_len": 150,
            "padding_value": 0.0  # default on 0.0
        }
    }

    emb = Embedding(cfg)
    x = torch.randn(size=(10, 150, 5))
    t = torch.ones(size=(10,))
    result = emb(x, t)
    assert result.shape == torch.Size((10, 150, 100)), 'Embedding failed to encode input data'


class Recovery(nn.Module):
    """

        DECODER

    """

    def __init__(self, cfg: Dict):
        super(Recovery, self).__init__()
        self.dim_output = int(cfg['rec']['dim_output'])
        self.dim_hidden = int(cfg['rec']['dim_hidden'])
        self.num_layers = int(cfg['rec']['num_layers'])
        self.seq_len = int(cfg['system']['seq_len'])

        # Dynamic RNN input
        self.padding_value = int(cfg['system']['padding_value'])

        # Architecture
        self.rec_rnn = nn.GRU(input_size=self.dim_hidden,
                              hidden_size=self.dim_hidden,
                              num_layers=self.num_layers,
                              batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(self.dim_hidden, self.dim_hidden),
            nn.LayerNorm(self.dim_hidden),
            nn.ReLU(),
            nn.Linear(self.dim_hidden, self.dim_output)
        )

        self.norm = nn.LayerNorm(self.dim_hidden)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
            :param h: latent representation batch * seq_len * H (from embedding)
            :param t: temporal information batch * 1
            :return: (~x) recovered data batch * seq_len * features
        """
        h_packed = nn.utils.rnn.pack_padded_sequence(input=h,
                                                     lengths=t,
                                                     batch_first=True,
                                                     enforce_sorted=True)
        h_0, _ = self.rec_rnn(h_packed)
        h_0, _ = nn.utils.rnn.pad_packed_sequence(sequence=h_0,
                                                  batch_first=True,
                                                  padding_value=self.padding_value,
                                                  total_length=self.seq_len)

        h_0 = self.norm(h_0)

        # recovered_x = self.rec_linear(h_0)
        recovered_x = self.mlp(h_0)
        return recovered_x

    @property
    def device(self):
        return next(self.parameters()).device


def run_recovery_test() -> None:
    cfg = {
        "rec": {
            "dim_output": 5,  # output feature dimension
            "dim_hidden": 100,  # latent space dimension
            "num_layers": 50  # number of layers in GRU
        },
        "system": {
            "seq_len": 150,
            "padding_value": 0.0  # default on 0.0
        }
    }

    rec = Recovery(cfg)
    h = torch.randn(size=(10, 150, 100))
    t = torch.ones(size=(10,))
    result = rec(h, t)
    assert result.shape == torch.Size((10, 150, 5)), 'Recovery failed to decode input data'


class Supervisor(nn.Module):
    """

        DECODER, for predicting next step data
        - middleware network -

    """

    def __init__(self, cfg: Dict):
        super(Supervisor, self).__init__()
        self.dim_hidden = int(cfg['sup']['dim_hidden'])
        self.num_layers = int(cfg['sup']['num_layers'])
        self.seq_len = int(cfg['system']['seq_len'])

        # Dynamic RNN input
        self.padding_value = int(cfg['system']['padding_value'])

        # Architecture
        self.sup_rnn = nn.GRU(input_size=self.dim_hidden,
                              hidden_size=self.dim_hidden,
                              num_layers=self.num_layers,
                              batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(self.dim_hidden, self.dim_hidden),
            nn.LayerNorm(self.dim_hidden),
            nn.ReLU(),
            nn.Linear(self.dim_hidden, self.dim_hidden)
        )

        self.norm = nn.LayerNorm(self.dim_hidden)
        # self.sup_linear = nn.Linear(self.dim_hidden, self.dim_hidden)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
            :param h: latent representation batch * sequence_len * H
            :param t: temporal information batch * 1
            :return: (_H) predicted next step data (latent form) batch * sequence_len * H
        """
        # h = self.mlp(h)
        h_packed = nn.utils.rnn.pack_padded_sequence(input=h,
                                                     lengths=t,
                                                     batch_first=True,
                                                     enforce_sorted=True)
        h_0, _ = self.sup_rnn(h_packed)
        h_0, _ = nn.utils.rnn.pad_packed_sequence(sequence=h_0,
                                                  batch_first=True,
                                                  padding_value=self.padding_value,
                                                  total_length=self.seq_len)

        h_0 = self.norm(h_0)
        supervised_h = self.mlp(h_0)
        return supervised_h

    @property
    def device(self):
        return next(self.parameters()).device


def run_supervisor_test() -> None:
    cfg = {
        "sup": {
            "dim_hidden": 100,  # latent space dimension (H)
            "num_layers": 50  # number of layers in GRU
        },
        "system": {
            "seq_len": 150,
            "padding_value": 0.0  # default on 0.0
        }
    }

    sup = Supervisor(cfg)
    h = torch.randn(size=(10, 150, 100))
    t = torch.ones(size=(10,))
    result = sup(h, t)
    assert result.shape == torch.Size((10, 150, 100)), 'Supervisor failed to encode input data'


class Generator(nn.Module):
    """

        ENCODER

    """

    def __init__(self, cfg: Dict):
        super(Generator, self).__init__()
        self.dim_latent = int(cfg['g']['dim_latent'])
        self.dim_hidden = int(cfg['g']['dim_hidden'])
        self.num_layers = int(cfg['g']['num_layers'])
        self.seq_len = int(cfg['system']['seq_len'])

        # Dynamic RNN input
        self.padding_value = int(cfg['system']['padding_value'])

        # Architecture
        self.g_rnn = nn.GRU(input_size=self.dim_hidden,
                            hidden_size=self.dim_hidden,
                            num_layers=self.num_layers,
                            batch_first=True)

        self.g_linear = nn.Linear(self.dim_hidden, self.dim_hidden)

        self.mlp = nn.Sequential(
            nn.Linear(self.dim_latent, self.dim_hidden * 2),
            nn.LayerNorm(self.dim_hidden * 2),
            nn.ReLU(),
            nn.Linear(self.dim_hidden * 2, self.dim_hidden)
        )

        self.norm = nn.LayerNorm(self.dim_hidden)

    def forward(self, z: torch.Tensor, t: torch.Tensor):
        """
            :param z: random noise batch * sequence_len * dim_latent
            :param t: temporal information batch * 1
            :return: (H) latent space embeddings batch * sequence_len * H
        """
        z = self.mlp(z)
        x_packed = nn.utils.rnn.pack_padded_sequence(input=z,
                                                     lengths=t,
                                                     batch_first=True,
                                                     enforce_sorted=True)
        h_0, _ = self.g_rnn(x_packed)
        h_0, _ = nn.utils.rnn.pad_packed_sequence(sequence=h_0,
                                                  batch_first=True,
                                                  padding_value=self.padding_value,
                                                  total_length=self.seq_len)

        h_0 = self.norm(h_0)
        h = self.g_linear(h_0)
        return h

    @property
    def device(self):
        return next(self.parameters()).device


def run_generator_test() -> None:
    cfg = {
        "g": {
            "dim_latent": 64,  # Z (input latent space dimension) size (eq. 128) [ INPUT ]
            "dim_hidden": 100,  # representation latent space dimension [ ENCODING ]
            "num_layers": 50  # number of layers in GRU
        },
        "system": {
            "seq_len": 150,
            "padding_value": 0.0  # default on 0.0
        }
    }

    g = Generator(cfg)
    z = torch.randn(size=(10, 150, 64))
    t = torch.ones(size=(10,))
    result = g(z, t)
    assert result.shape == torch.Size((10, 150, 100)), 'Generator failed to generate correct shape data'


class Discriminator(nn.Module):
    """

        DECODER

    """

    def __init__(self, cfg: Dict):
        super(Discriminator, self).__init__()
        self.dim_hidden = int(cfg['d']['dim_hidden'])
        self.num_layers = int(cfg['d']['num_layers'])
        self.seq_len = int(cfg['system']['seq_len'])

        # Dynamic RNN input
        self.padding_value = int(cfg['system']['padding_value'])

        # Architecture
        self.d_rnn = nn.GRU(input_size=self.dim_hidden,
                            hidden_size=self.dim_hidden,
                            num_layers=self.num_layers,
                            batch_first=True)

        # self.d_linear = nn.Linear(self.dim_hidden, 1)
        self.norm = nn.LayerNorm(self.dim_hidden)
        self.mlp = nn.Sequential(
            # nn.Linear(self.dim_hidden, self.dim_hidden),
            # nn.LayerNorm(self.dim_hidden),
            # nn.ReLU(),
            nn.Linear(self.dim_hidden, 1)
        )

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
            :param h: latent representation batch * seq_len * H (from embedding)
            :param t: temporal information batch * 1
            :return: (logits) predicted data batch * seq_len * 1
        """
        h_packed = nn.utils.rnn.pack_padded_sequence(input=h,
                                                     lengths=t,
                                                     batch_first=True,
                                                     enforce_sorted=True)

        h_0, _ = self.d_rnn(h_packed)
        h_0, _ = nn.utils.rnn.pad_packed_sequence(sequence=h_0,
                                                  batch_first=True,
                                                  padding_value=self.padding_value,
                                                  total_length=self.seq_len)

        h_0 = self.norm(h_0)
        h_0 = self.mlp(h_0)
        return h_0

    @property
    def device(self):
        return next(self.parameters()).device


def run_discriminator_test() -> None:
    cfg = {
        "d": {
            "dim_hidden": 100,  # representation latent space dimension (H)
            "num_layers": 50  # number of layers in GRU
        },
        "system": {
            "seq_len": 150,
            "padding_value": 0.0  # default on 0.0
        }
    }

    d = Discriminator(cfg)
    h = torch.randn(size=(10, 150, 100))
    t = torch.ones(size=(10,))
    result = d(h, t)
    assert result.shape == torch.Size((10, 150)), 'Discriminator failed to decode data'


def _embedding_forward_side(emb: Embedding,
                            rec: Recovery,
                            x: Tensor,
                            t: Tensor) -> Tuple[Tensor, Tensor]:
    assert x.device == emb.device, 'x and EMB are not on the same device'

    # Forward pass
    h = emb(x, t)
    _x = rec(h, t)

    e_loss_t0 = F.mse_loss(_x, x)
    e_loss0 = 10 * torch.sqrt(e_loss_t0)

    return e_loss0, _x


def _embedding_forward_main(emb: Embedding,
                            sup: Supervisor,
                            rec: Recovery,
                            x: Tensor,
                            t: Tensor) -> Tuple[Any, Any, Tensor]:
    assert x.device == emb.device, 'x and EMB are not on the same device'

    # Forward pass
    h = emb(x, t)
    _x = rec(h, t)

    # Joint training
    _h_sup = sup(h, t)

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


def _supervisor_forward(emb: Embedding,
                        sup: Supervisor,
                        x: Tensor,
                        t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    assert x.device == emb.device, 'x and EMB are not on the same device'

    # Supervisor forward pass
    h = emb(x, t)
    _h_sup = sup(h, t)

    # Supervised loss

    # Teacher forcing next output
    s_loss = F.mse_loss(
        _h_sup[:, :-1, :],
        h[:, 1:, :]
    )

    return s_loss, h, _h_sup


def _discriminator_forward(emb: Embedding,
                           sup: Supervisor,
                           g: Generator,
                           d: Discriminator,
                           x: Tensor,
                           t: Tensor,
                           z: Tensor,
                           gamma=1.0) -> Tensor:
    assert x.device == emb.device, 'x and EMB are not on the same device'
    assert z.device == g.device, 'z and G are not on the same device'

    # Discriminator forward pass and adversarial loss

    h = emb(x, t).detach()
    _h = sup(h, t).detach()
    _e = g(z, t).detach()

    # Forward Pass
    y_real = d(h, t)  # Encoded original data
    y_fake = d(_h, t)  # Output of supervisor
    y_fake_e = d(_e, t)  # Output of generator

    d_loss_real = F.binary_cross_entropy_with_logits(y_real, torch.ones_like(y_real))
    d_loss_fake = F.binary_cross_entropy_with_logits(y_fake, torch.zeros_like(y_fake))
    d_loss_fake_e = F.binary_cross_entropy_with_logits(y_fake_e, torch.zeros_like(y_fake_e))

    d_loss = d_loss_fake + d_loss_real + gamma * d_loss_fake_e

    return d_loss


def _generator_forward(emb: Embedding,
                       sup: Supervisor,
                       g: Generator,
                       d: Discriminator,
                       rec: Recovery,
                       x: Tensor,
                       t: Tensor,
                       z: Tensor,
                       gamma=1.0) -> Tensor:
    assert x.device == emb.device, 'x and EMB are not on the same device'
    assert z.device == g.device, 'z and G are not on the same device'

    # Supervised Forward Pass
    h = emb(x, t)
    _h_sup = sup(h, t)
    _x = rec(h, t)

    # Generator Forward Pass
    _e = g(z, t)
    _h = sup(_e, t)

    # Synthetic generated data
    _x = rec(_h, t)  # recovered data

    # Generator Loss

    # 1. Adversarial loss
    y_fake = d(_h, t)  # Output of supervisor
    y_fake_e = d(_e, t)  # Output of generator

    g_loss_u = F.binary_cross_entropy_with_logits(y_fake, torch.ones_like(y_fake))
    g_loss_u_e = F.binary_cross_entropy_with_logits(y_fake_e, torch.ones_like(y_fake_e))

    # 2. Supervised loss
    g_loss_s = torch.nn.functional.mse_loss(_h_sup[:, :-1, :], h[:, 1:, :])  # Teacher forcing next output

    # 3. Two Moments
    g_loss_v1 = torch.mean(torch.abs((_x.std(dim=0, unbiased=False) + 1e-6) -
                                     x.std(dim=0, unbiased=False) + 1e-6))

    g_loss_v2 = torch.mean(torch.abs((_x.mean(dim=0)) - (x.mean(dim=0))))

    g_loss_v = g_loss_v1 + g_loss_v2

    # 4. Sum
    g_loss = g_loss_u + gamma * g_loss_u_e + 100 * torch.sqrt(g_loss_s) + 1000 * g_loss_v

    return g_loss


def _inference(sup: Supervisor,
               g: Generator,
               rec: Recovery,
               z: Tensor,
               t: Tensor) -> Tensor:
    # Generate synthetic data
    assert z.device == g.device, 'Z and Generator are not on the same device'

    # Generator Forward Pass
    _e = g(z, t)
    _h = sup(_e, t)

    # Synthetic generated data (reconstructed)
    _x = rec(_h, t)
    return _x


def embedding_trainer(emb: Embedding,
                      sup: Supervisor,
                      rec: Recovery,
                      emb_opt: Optimizer,
                      rec_opt: Optimizer,
                      dl: DataLoader,
                      cfg: Dict,
                      real_samples: np.ndarray) -> None:
    num_epochs = int(cfg['emb']['num_epochs'])
    device = torch.device(cfg['system']['device'])
    batch_size = int(cfg['system']['batch_size'])

    for epoch in range(num_epochs):
        for idx, real_data in enumerate(dl):
            x = real_data
            t, _ = Energy.extract_time(real_data)

            x = x.float()
            x = x.view(*x.shape)
            x = x.to(device)

            # Reset gradients
            emb.zero_grad()
            rec.zero_grad()
            sup.zero_grad()

            # Forward Pass
            e_loss0, _x = _embedding_forward_side(emb=emb, rec=rec, x=x, t=t)
            loss = np.sqrt(e_loss0.item())

            # Backward Pass
            e_loss0.backward()

            # Update model parameters
            emb_opt.step()
            rec_opt.step()

            if idx % 10 == 0:
                y1 = x.detach().cpu().numpy()[0, :, 0].tolist()
                y2 = _x.detach().cpu().numpy()[0, :, 0].tolist()
                x = list(range(len(y1)))

                real_samples_tensor = torch.from_numpy(np.array(real_samples[:1000]))
                # real_samples_tensor = real_samples_tensor.view(real_samples_tensor.shape[0],
                #                                                real_samples_tensor.shape[1] * \
                #                                                real_samples_tensor.shape[2])

                generated_samples = []
                with torch.no_grad():
                    for e in real_samples[:1000]:
                        e_tensor = torch.from_numpy(e).repeat(batch_size, 1, 1).float()
                        e_tensor = e_tensor.to(device)
                        e_tensor = e_tensor.float()
                        _t, _ = Energy.extract_time(e_tensor)
                        _, sample = _embedding_forward_side(emb=emb, rec=rec, x=e_tensor, t=_t)
                        generated_samples.append(sample.detach().cpu().numpy()[0, :, :])

                generated_samples_tensor = torch.from_numpy(np.array(generated_samples))
                # generated_samples_tensor = generated_samples_tensor.view(generated_samples_tensor.shape[0],
                #                                                          generated_samples_tensor.shape[1] * \
                #                                                          generated_samples_tensor.shape[2])

                fig = visualisation.visualize(real_data=real_samples_tensor.numpy(),
                                              generated_data=generated_samples_tensor.numpy(),
                                              perplexity=40,
                                              legend=['Embedded sequence', 'Recovered sequence'])

                wandb.log({"Reconstructed data plot": wandb.plot.line_series(xs=x,
                                                                             ys=[y1, y2],
                                                                             keys=['Original', 'Reconstructed'],
                                                                             xname='time',
                                                                             title="Reconstructed data plot")},
                          step=epoch * len(dl) + idx)

                wandb.log({"Population": fig}, step=epoch * len(dl) + idx)
                wandb.log({'Emb loss': e_loss0}, step=epoch * len(dl) + idx)


def supervisor_trainer(emb: Embedding,
                       sup: Supervisor,
                       rec: Recovery,
                       sup_opt: Optimizer,
                       dl: DataLoader,
                       cfg: Dict,
                       real_samples: np.ndarray) -> None:
    num_epochs = int(cfg['sup']['num_epochs'])
    device = torch.device(cfg['system']['device'])
    batch_size = int(cfg['system']['batch_size'])

    for epoch in range(num_epochs):
        for idx, real_data in enumerate(dl):
            x = real_data
            t, _ = Energy.extract_time(real_data)

            x = x.float()
            x = x.view(*x.shape)
            x = x.to(device)

            # Reset gradients
            emb.zero_grad()
            sup.zero_grad()

            # Forward Pass
            sup_loss, h, _h_sup = _supervisor_forward(emb=emb, sup=sup, x=x, t=t)

            # Backward Pass
            sup_loss.backward()
            loss = np.sqrt(sup_loss.item())

            # Update model parameters
            sup_opt.step()

            if idx % 10 == 0:
                y1 = h.detach().cpu().numpy()[0, :, 0].tolist()
                y2 = _h_sup.detach().cpu().numpy()[0, :, 0].tolist()
                x = list(range(len(y1)))

                embedding_samples = []
                supervised_samples = []

                with torch.no_grad():
                    for e in real_samples[:1000]:
                        e_tensor = torch.from_numpy(e).repeat(batch_size, 1, 1).float()
                        e_tensor = e_tensor.to(device)
                        e_tensor = e_tensor.float()
                        _t, _ = Energy.extract_time(e_tensor)
                        _, sample = _embedding_forward_side(emb=emb, rec=rec, x=e_tensor, t=_t)
                        _, h, _h_sup = _supervisor_forward(emb=emb, sup=sup, x=e_tensor, t=_t)
                        supervised_samples.append(_h_sup.detach().cpu().numpy()[0, :, :])
                        embedding_samples.append(h.detach().cpu().numpy()[0, :, :])

                embedding_samples_tensor = torch.from_numpy(np.array(embedding_samples))
                # embedding_samples_tensor = embedding_samples_tensor.view(embedding_samples_tensor.shape[0],
                #                                                          embedding_samples_tensor.shape[1] * \
                #                                                          embedding_samples_tensor.shape[2])

                supervised_samples_tensor = torch.from_numpy(np.array(supervised_samples))
                # supervised_samples_tensor = supervised_samples_tensor.view(supervised_samples_tensor.shape[0],
                #                                                            supervised_samples_tensor.shape[1] * \
                #                                                            supervised_samples_tensor.shape[2])

                fig = visualisation.visualize(real_data=embedding_samples_tensor.numpy(),
                                              generated_data=supervised_samples_tensor.numpy(),
                                              perplexity=40,
                                              legend=['Embedded data', 'Supervised data'])

                wandb.log({"Reconstructed data plot": wandb.plot.line_series(xs=x,
                                                                             ys=[y1, y2],
                                                                             keys=['Embedding', 'Supervised'],
                                                                             xname='time',
                                                                             title="Supervised data plot")},
                          step=epoch * len(dl) + idx)

                wandb.log({"Population": fig}, step=epoch * len(dl) + idx)
                wandb.log({'Supervisor loss': loss}, step=epoch * len(dl) + idx)

        print('Current epoch: {}'.format(epoch))


def joint_trainer(emb: Embedding,
                  sup: Supervisor,
                  g: Generator,
                  d: Discriminator,
                  rec: Recovery,
                  g_opt: Optimizer,
                  d_opt: Optimizer,
                  sup_opt: Optimizer,
                  rec_opt: Optimizer,
                  emb_opt: Optimizer,
                  dl: DataLoader,
                  real_samples: np.ndarray,
                  cfg: Dict) -> None:
    num_epochs = int(cfg['system']['jointly_num_epochs'])
    d_threshold = float(cfg['d']['threshold'])
    device = torch.device(cfg['system']['device'])
    batch_size = int(cfg['system']['batch_size'])

    for epoch in range(num_epochs):
        for idx, real_data in enumerate(dl):
            x = real_data
            t, _ = Energy.extract_time(real_data)

            x = x.float()
            x = x.view(*x.shape)
            x = x.to(device)

            # Generator Training
            for _ in range(10):
                # Random sequence
                z = torch.rand_like(x)

                # Forward Pass (Generator)
                emb.zero_grad()
                rec.zero_grad()
                sup.zero_grad()
                g.zero_grad()
                d.zero_grad()

                g_loss = _generator_forward(emb=emb, sup=sup, rec=rec, g=g, d=d, x=x, t=t, z=z)
                g_loss.backward()
                g_loss = np.sqrt(g_loss.item())

                # Update model parameters
                g_opt.step()
                sup_opt.step()

                # Forward Pass (Embedding)
                emb.zero_grad()
                rec.zero_grad()
                sup.zero_grad()

                e_loss, _, e_loss_t0 = _embedding_forward_main(emb=emb, rec=rec, sup=sup, x=x, t=t)
                e_loss.backward()
                e_loss = np.sqrt(e_loss.item())

                # Update model parameters
                emb_opt.step()
                rec_opt.step()

            # Random sequence
            z = torch.rand_like(x)

            # Discriminator Training
            emb.zero_grad()
            sup.zero_grad()
            g.zero_grad()
            d.zero_grad()

            # Forward Pass
            d_loss = _discriminator_forward(emb=emb, sup=sup, g=g, d=d, x=x, t=t, z=z)

            # Check Discriminator loss
            if d_loss > d_threshold:
                # Backward Pass
                d_loss.backward()

                # Update model parameters
                d_opt.step()

            d_loss = d_loss.item()

            if idx % 10 == 0:
                # Generate sample
                sample = _inference(sup=sup, g=g, rec=rec, z=z, t=t)
                fake_sample = sample.detach().cpu().numpy()[0, :, 0]
                real_sample = x.detach().cpu().numpy()[0, :, 0]

                y1 = fake_sample.tolist()
                y2 = real_sample.tolist()
                x = list(range(len(y1)))

                generated_samples = []
                comp_real_samples = []

                with torch.no_grad():
                    # rs_tensor = torch.from_numpy(np.array(real_samples[:1000])).to(device).float()
                    # _t, _ = Energy.extract_time(rs_tensor)
                    # z = torch.rand_like(rs_tensor)
                    # gs = _inference(sup=sup, g=g, z=z, t=_t, rec=rec)

                    for e in real_samples[:1000]:
                        e_tensor = torch.from_numpy(e).repeat(batch_size, 1, 1).float()
                        e_tensor = e_tensor.to(device)
                        e_tensor = e_tensor.float()
                        _t, _ = Energy.extract_time(e_tensor)
                        z = torch.rand_like(e_tensor)
                        gs = _inference(sup=sup, g=g, z=z, t=_t, rec=rec)
                        comp_real_samples.append(e_tensor.detach().cpu().numpy()[0, :, :])
                        generated_samples.append(gs.detach().cpu().numpy()[0, :, :])

                generated_samples_tensor = torch.from_numpy(np.array(generated_samples))
                # generated_samples_tensor = generated_samples_tensor.view(generated_samples_tensor.shape[0],
                #                                                          generated_samples_tensor.shape[1] * \
                #                                                          generated_samples_tensor.shape[2])

                comp_real_samples_tensor = torch.from_numpy(np.array(real_samples[:1000]))
                # comp_real_samples_tensor = comp_real_samples_tensor.view(comp_real_samples_tensor.shape[0],
                #                                                          comp_real_samples_tensor.shape[1] * \
                #                                                          comp_real_samples_tensor.shape[2])

                fig = visualisation.visualize(real_data=comp_real_samples_tensor.numpy(),
                                              generated_data=generated_samples_tensor.detach().cpu().numpy(),
                                              perplexity=40,
                                              legend=['Generated data', 'Real data'])

                wandb.log({"Generated data plot": wandb.plot.line_series(xs=x,
                                                                         ys=[y1, y2],
                                                                         keys=['Generated', 'Real'],
                                                                         xname='time',
                                                                         title="Generated data plot")},
                          step=epoch * len(dl) + idx)

                wandb.log({"Population": fig}, step=epoch * len(dl) + idx)
                wandb.log({'G loss': g_loss}, step=epoch * len(dl) + idx)
                wandb.log({'D loss': d_loss}, step=epoch * len(dl) + idx)
                wandb.log({'E loss': e_loss}, step=epoch * len(dl) + idx)

    print(f"[JOINT] Epoch: {epoch}, E_loss: {e_loss:.4f}, G_loss: {g_loss:.4f}, D_loss: {d_loss:.4f}")


def get_dataset(name: str) -> Dataset:
    if name == 'energy':
        return Energy.Energy(seq_len=24, path='./data/energy.csv')
    elif name == 'sine':
        return SineWave.SineWave(samples_number=24 * 1000, seq_len=24, features_dim=28)
    elif name == 'stock':
        return Stock.Stock(seq_len=24, path='./data/stock.csv')
    else:
        raise ValueError('The dataset does not exist')


def time_gan_trainer(cfg: Dict, step: str) -> None:
    # Init all parameters and models
    seq_len = int(cfg['system']['seq_len'])
    batch_size = int(cfg['system']['batch_size'])
    device = torch.device(cfg['system']['device'])

    print('Current device', device)

    lr = float(cfg['system']['lr'])

    ds = get_dataset(cfg['system']['dataset'])
    dl = DataLoader(ds, num_workers=10, batch_size=batch_size, shuffle=True)

    # TimeGAN elements
    emb = Embedding(cfg=cfg).to(device)
    rec = Recovery(cfg=cfg).to(device)
    sup = Supervisor(cfg=cfg).to(device)
    g = Generator(cfg=cfg).to(device)
    d = Discriminator(cfg=cfg).to(device)

    # Optimizers
    emb_opt_side = Adam(emb.parameters(), lr=lr)
    emb_opt_main = Adam(emb.parameters(), lr=lr)
    rec_opt = Adam(rec.parameters(), lr=lr)
    sup_opt = Adam(sup.parameters(), lr=lr)
    g_opt = Adam(g.parameters(), lr=lr)
    d_opt = Adam(d.parameters(), lr=lr)

    if step == "embedding":
        print(f"[EMB] Start Embedding network training")
        embedding_trainer(emb=emb,
                          rec=rec,
                          sup=sup,
                          emb_opt=emb_opt_side,
                          rec_opt=rec_opt,
                          dl=dl,
                          cfg=cfg,
                          real_samples=ds.get_distribution())

        emb = emb.to('cpu')
        rec = rec.to('cpu')
        torch.save(emb.state_dict(), './trained_models/emb_{}.pt'.format(config['system']['dataset']))
        torch.save(rec.state_dict(), './trained_models/rec_{}.pt'.format(config['system']['dataset']))
    elif step == "supervisor":
        emb = Embedding(cfg=cfg)
        emb.load_state_dict(torch.load('./trained_models/emb_{}.pt'.format(config['system']['dataset'])))
        emb = emb.to(device)
        emb.train()

        rec = Recovery(cfg=cfg)
        rec.load_state_dict(torch.load('./trained_models/rec_{}.pt'.format(config['system']['dataset'])))
        rec = rec.to(device)
        rec.train()

        print(f"[SUP] Start Supervisor network training")
        supervisor_trainer(emb=emb,
                           sup=sup,
                           rec=rec,
                           sup_opt=sup_opt,
                           dl=dl,
                           cfg=cfg,
                           real_samples=ds.get_distribution())

        sup = sup.to('cpu')
        torch.save(sup.state_dict(), './trained_models/sup_{}.pt'.format(config['system']['dataset']))
    elif step == 'joint':
        print(f"[JOINT] Start joint training")
        emb = Embedding(cfg=cfg)
        emb.load_state_dict(torch.load('./trained_models/emb_{}.pt'.format(config['system']['dataset'])))
        emb = emb.to(device)

        rec = Recovery(cfg=cfg)
        rec.load_state_dict(torch.load('./trained_models/rec_{}.pt'.format(config['system']['dataset'])))
        rec = rec.to(device)

        sup = Supervisor(cfg=cfg)
        sup.load_state_dict(torch.load('./trained_models/sup_{}.pt'.format(config['system']['dataset'])))
        sup = sup.to(device)

        joint_trainer(emb=emb,
                      rec=rec,
                      sup=sup,
                      g=g,
                      d=d,
                      emb_opt=emb_opt_main,
                      rec_opt=rec_opt,
                      sup_opt=sup_opt,
                      g_opt=g_opt,
                      d_opt=d_opt,
                      dl=dl,
                      cfg=cfg,
                      real_samples=ds.get_distribution())

        g = g.to('cpu')
        torch.save(g.state_dict(), './trained_models/g_{}.pt'.format(config['system']['dataset']))

        d = d.to('cpu')
        torch.save(d.state_dict(), './trained_models/d_{}.pt'.format(config['system']['dataset']))

    else:
        raise ValueError('The step should be: embedding, supervisor or joint')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--perplexity', type=int, required=True)
    # args = parser.parse_args()

    torch.random.manual_seed(42)
    with open('config/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    step = 'joint'

    config['system']['dataset'] = 'stock'
    config['system']['device'] = 'cuda:0'

    if config['system']['dataset'] == 'stock':
        config['g']['dim_latent'] = 6
        config['emb']['dim_features'] = 6
        config['rec']['dim_output'] = 6

    run_name = config['system']['run_name'] + ' ' + config['system']['dataset'] + ' ' + step
    wandb.init(config=config, project='thesis', name=run_name)

    time_gan_trainer(cfg=config, step=step)
