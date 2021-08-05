import torch
import os
import wandb
import yaml
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch import Tensor
from typing import Tuple, Any, Dict
import torch.nn.functional as F
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer, Adam
import numpy as np
from data import Energy, SineWave, Stock, Water
from metrics import visualisation


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, max_seq_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, dim_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # jump from 0, 2 by 2 (even)
        pe[:, 1::2] = torch.cos(position * div_term)  # jump from 1, 2 by 2 (odd)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # not trainable

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


def _generate_square_subsequent_mask(sz: int) -> Tensor:
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Embedding(nn.Module):
    def __init__(self, cfg: Dict):
        super(Embedding, self).__init__()
        self.feature_size = int(cfg['t_emb']['feature_size'])
        self.num_layers = int(cfg['t_emb']['num_layers'])
        self.dropout = float(cfg['t_emb']['dropout'])
        self.n_head = int(cfg['t_emb']['n_head'])  # 10
        self.dim_output = int(cfg['t_emb']['dim_output'])

        self.model_type = 'Transformer'
        self.src_mask = None

        # Architecture
        self.pos_encoder = PositionalEncoding(dim_model=self.feature_size)
        self.encoder_layer = TransformerEncoderLayer(d_model=self.feature_size,
                                                     nhead=self.n_head,
                                                     dropout=self.dropout,
                                                     dim_feedforward=self.feature_size * 8)
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=self.num_layers)
        self.ll = nn.Linear(in_features=self.feature_size, out_features=self.dim_output)
        #
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.feature_size, self.feature_size * 8),
        #     nn.LayerNorm(self.feature_size * 8),
        #     nn.ReLU(),
        #     nn.Linear(self.feature_size * 8, self.feature_size * 8)
        # )

        # self.norm = nn.LayerNorm(self.feature_size * 8)

        # Init weights
        init_range = 0.1
        self.ll.bias.data.zero_()
        self.ll.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: Tensor) -> Tensor:
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = _generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        # src = self.mlp(src)
        src = self.pos_encoder(src)
        output = self.encoder(src, self.src_mask)
        # output = self.norm(output)
        output = self.ll(output)
        return output

    @property
    def device(self):
        return next(self.parameters()).device


class RecoveryDecoder(nn.Module):
    def __init__(self, cfg: Dict):
        super(RecoveryDecoder, self).__init__()
        self.feature_size = int(cfg['t_rec']['feature_size'])
        self.num_layers = int(cfg['t_rec']['num_layers'])
        self.dropout = float(cfg['t_rec']['dropout'])
        self.n_head = int(cfg['t_rec']['n_head'])
        self.dim_output = int(cfg['t_rec']['dim_output'])

        # Architecture
        self.pos_encoder = PositionalEncoding(dim_model=self.feature_size)
        self.decoder_layer = TransformerDecoderLayer(d_model=self.feature_size,
                                                     nhead=self.n_head,
                                                     dropout=self.dropout,
                                                     dim_feedforward=self.feature_size * 8)
        self.decoder = TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=self.num_layers)
        self.ll = nn.Linear(in_features=self.feature_size, out_features=self.dim_output)

        # Init weights
        init_range = 0.1
        self.ll.bias.data.zero_()
        self.ll.weight.data.uniform_(-init_range, init_range)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        tgt = self.pos_encoder(tgt)
        output = self.decoder(tgt, memory)
        output = self.ll(output)
        return output


class RecoveryEncoder(nn.Module):
    def __init__(self, cfg: Dict):
        super(RecoveryEncoder, self).__init__()
        self.feature_size = int(cfg['t_rec']['feature_size'])
        self.num_layers = int(cfg['t_rec']['num_layers'])
        self.dropout = float(cfg['t_rec']['dropout'])
        self.n_head = int(cfg['t_rec']['n_head'])
        self.dim_output = int(cfg['t_rec']['dim_output'])

        self.model_type = 'Transformer'
        self.src_mask = None

        # Architecture
        self.pos_encoder = PositionalEncoding(dim_model=self.feature_size)
        self.encoder_layer = TransformerEncoderLayer(d_model=self.feature_size,
                                                     nhead=self.n_head,
                                                     dropout=self.dropout,
                                                     dim_feedforward=self.feature_size * 8)
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=self.num_layers)
        self.ll = nn.Linear(in_features=self.feature_size, out_features=self.dim_output)

        # self.mlp = nn.Sequential(
        #     nn.Linear(self.feature_size, self.feature_size * 8),
        #     nn.LayerNorm(self.feature_size * 8),
        #     nn.ReLU(),
        #     nn.Linear(self.feature_size * 8, self.feature_size * 8)
        # )

        # self.norm = nn.LayerNorm(self.feature_size * 8)

        # Init weights
        init_range = 0.1
        self.ll.bias.data.zero_()
        self.ll.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: Tensor) -> Tensor:
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = _generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        # src = self.mlp(src)
        src = self.pos_encoder(src)
        output = self.encoder(src, self.src_mask)
        # output = self.norm(output)
        output = self.ll(output)

        return output

    @property
    def device(self):
        return next(self.parameters()).device


class Supervisor(nn.Module):
    def __init__(self, cfg: Dict):
        super(Supervisor, self).__init__()
        self.feature_size = int(cfg['t_sup']['feature_size'])
        self.num_layers = int(cfg['t_sup']['num_layers'])
        self.dropout = float(cfg['t_sup']['dropout'])
        self.n_head = int(cfg['t_sup']['n_head'])
        self.dim_output = int(cfg['t_sup']['dim_output'])
        self.model_type = 'Transformer'

        # Architecture
        self.pos_encoder = PositionalEncoding(dim_model=self.feature_size)
        self.decoder_layer = TransformerDecoderLayer(d_model=self.feature_size,
                                                     nhead=self.n_head,
                                                     dropout=self.dropout,
                                                     dim_feedforward=self.feature_size * 8)
        self.decoder = TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=self.num_layers)
        self.ll = nn.Linear(in_features=self.feature_size, out_features=self.dim_output)

        # self.mlp = nn.Sequential(
        #     nn.Linear(self.feature_size, self.feature_size * 8),
        #     nn.LayerNorm(self.feature_size * 8),
        #     nn.ReLU(),
        #     nn.Linear(self.feature_size * 8, self.feature_size * 8)
        # )

        # self.norm = nn.LayerNorm(self.feature_size * 8)

        # Init weights
        init_range = 0.1
        self.ll.bias.data.zero_()
        self.ll.weight.data.uniform_(-init_range, init_range)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        # tgt = self.mlp(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.decoder(tgt, memory)
        # output = self.norm(output)
        output = self.ll(output)
        return output

    @property
    def device(self):
        return next(self.parameters()).device


class Generator(nn.Module):
    def __init__(self, cfg: Dict):
        super(Generator, self).__init__()
        self.feature_size = int(cfg['t_g']['feature_size'])
        self.num_layers = int(cfg['t_g']['num_layers'])
        self.dropout = float(cfg['t_g']['dropout'])
        self.n_head = int(cfg['t_g']['n_head'])
        self.dim_output = int(cfg['t_g']['dim_output'])

        self.model_type = 'Transformer'
        self.src_mask = None

        # Architecture
        self.pos_encoder = PositionalEncoding(dim_model=self.feature_size)
        self.encoder_layer = TransformerEncoderLayer(d_model=self.feature_size,
                                                     nhead=self.n_head,
                                                     dropout=self.dropout,
                                                     dim_feedforward=self.feature_size * 8)
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=self.num_layers)
        self.ll = nn.Linear(in_features=self.feature_size, out_features=self.dim_output)

        # Init weights
        init_range = 0.1
        self.ll.bias.data.zero_()
        self.ll.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: Tensor) -> Tensor:
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = _generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.encoder(src, self.src_mask)
        output = self.ll(output)
        return output

    @property
    def device(self):
        return next(self.parameters()).device


class DiscriminatorDecoder(nn.Module):
    def __init__(self, cfg: Dict):
        super(DiscriminatorDecoder, self).__init__()
        self.feature_size = int(cfg['t_d']['feature_size'])
        self.num_layers = int(cfg['t_d']['num_layers'])
        self.dropout = float(cfg['t_d']['dropout'])
        self.n_head = int(cfg['t_d']['n_head'])
        self.dim_output = int(cfg['t_d']['dim_output'])

        # Architecture
        self.pos_encoder = PositionalEncoding(dim_model=self.feature_size)
        self.decoder_layer = TransformerDecoderLayer(d_model=self.feature_size,
                                                     nhead=self.n_head,
                                                     dropout=self.dropout,
                                                     dim_feedforward=self.feature_size * 8)
        self.decoder = TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=self.num_layers)
        self.ll = nn.Linear(in_features=self.feature_size, out_features=self.dim_output)

        # Init weights
        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        tgt = self.pos_encoder(tgt)
        output = self.decoder(tgt, memory)
        output = self.ll(output)
        return output


class DiscriminatorEncoder(nn.Module):
    def __init__(self, cfg: Dict):
        super(DiscriminatorEncoder, self).__init__()
        self.feature_size = int(cfg['t_d']['feature_size'])
        self.num_layers = int(cfg['t_d']['num_layers'])
        self.dropout = float(cfg['t_d']['dropout'])
        self.n_head = int(cfg['t_d']['n_head'])
        self.dim_output = int(cfg['t_d']['dim_output'])

        self.model_type = 'Transformer'
        self.src_mask = None

        # Architecture
        self.pos_encoder = PositionalEncoding(dim_model=self.feature_size)
        self.encoder_layer = TransformerEncoderLayer(d_model=self.feature_size,
                                                     nhead=self.n_head,
                                                     dropout=self.dropout,
                                                     dim_feedforward=self.feature_size * 4)
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=self.num_layers)
        self.ll = nn.Linear(in_features=self.feature_size, out_features=self.dim_output)

        # Init weights
        init_range = 0.1
        self.ll.bias.data.zero_()
        self.ll.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: Tensor) -> Tensor:
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = _generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.encoder(src, self.src_mask)
        output = self.ll(output)
        return output

    @property
    def device(self):
        return next(self.parameters()).device


def _embedding_forward_side(emb: Embedding,
                            rec: RecoveryEncoder,
                            src: Tensor) -> Tuple[Tensor, Tensor]:
    assert src.device == emb.device, 'Src and EMB are not on the same device'
    h = emb(src)
    _src = rec(h)
    e_loss_t0 = F.mse_loss(_src, src)
    e_loss0 = 10 * torch.sqrt(e_loss_t0)
    return e_loss0, _src


def _embedding_forward_main(emb: Embedding,
                            rec: RecoveryEncoder,
                            sup: Supervisor,
                            src: Tensor) -> Tuple[Any, Any, Tensor]:
    assert src.device == emb.device, 'Src and EMB are not on the same device'
    h = emb(src)
    _src = rec(h)
    _h_sup = sup(h, h)  # temporal dynamics

    g_loss_sup = F.mse_loss(
        _h_sup[:, :-1, :],
        h[:, 1:, :]
    )

    # Reconstruction Loss
    e_loss_t0 = F.mse_loss(_src, src)
    e_loss0 = 10 * torch.sqrt(e_loss_t0)
    e_loss = e_loss0 + 0.1 * g_loss_sup
    return e_loss, e_loss0, e_loss_t0


def _supervisor_forward(emb: Embedding,
                        sup: Supervisor,
                        src: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    assert src.device == emb.device, 'Src and EMB are not on the same device'

    # Supervisor forward pass
    h = emb(src)
    _h_sup = sup(h, h)

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
                           d: DiscriminatorEncoder,
                           src: Tensor,
                           z: Tensor,
                           gamma=1.0) -> Tensor:
    assert src.device == emb.device, 'Src and EMB are not on the same device'
    assert z.device == g.device, 'z and G are not on the same device'

    # Discriminator forward pass and adversarial loss
    h = emb(src).detach()
    _h = sup(h, h).detach()
    _e = g(z).detach()

    # Forward Pass
    y_real = d(h)  # Encoded original data
    y_fake = d(_h)  # Output of supervisor
    y_fake_e = d(_e)  # Output of generator

    d_loss_real = F.binary_cross_entropy_with_logits(y_real, torch.ones_like(y_real))
    d_loss_fake = F.binary_cross_entropy_with_logits(y_fake, torch.zeros_like(y_fake))
    d_loss_fake_e = F.binary_cross_entropy_with_logits(y_fake_e, torch.zeros_like(y_fake_e))

    d_loss = d_loss_fake + d_loss_real + gamma * d_loss_fake_e

    return d_loss


def _generator_forward(emb: Embedding,
                       sup: Supervisor,
                       g: Generator,
                       d: DiscriminatorEncoder,
                       rec: RecoveryEncoder,
                       src: Tensor,
                       z: Tensor,
                       gamma=1.0) -> Tensor:
    assert src.device == emb.device, 'Src and EMB are not on the same device'
    assert z.device == g.device, 'z and G are not on the same device'

    # Supervised Forward Pass
    h = emb(src)
    _h_sup = sup(h, h)
    _x = rec(h)

    # Generator Forward Pass
    _e = g(z)
    _h = sup(_e, _e)

    # Synthetic generated data
    _src = rec(_h)  # recovered data

    # Generator Loss

    # 1. Adversarial loss
    y_fake = d(_h)  # Output of supervisor
    y_fake_e = d(_e)  # Output of generator

    g_loss_u = F.binary_cross_entropy_with_logits(y_fake, torch.ones_like(y_fake))
    g_loss_u_e = F.binary_cross_entropy_with_logits(y_fake_e, torch.ones_like(y_fake_e))

    # 2. Supervised loss
    g_loss_s = torch.nn.functional.mse_loss(_h_sup[:, :-1, :], h[:, 1:, :])  # Teacher forcing next output

    # 3. Two Moments
    g_loss_v1 = torch.mean(torch.abs(
        torch.sqrt(_x.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(src.var(dim=0, unbiased=False) + 1e-6)))
    g_loss_v2 = torch.mean(torch.abs((_x.mean(dim=0)) - (src.mean(dim=0))))

    g_loss_v = g_loss_v1 + g_loss_v2

    # 4. Sum
    g_loss = g_loss_u + gamma * g_loss_u_e + 100 * torch.sqrt(g_loss_s) + 1000 * g_loss_v

    return g_loss


def _inference(sup: Supervisor,
               g: Generator,
               rec: RecoveryEncoder,
               seq_len: int,
               z: Tensor) -> Tensor:
    # Generate synthetic data
    assert z.device == g.device, 'z and Time GAN are not on the same device'

    # Generator Forward Pass
    _e = g(z)

    """
    #let's assume batch_size = 1
    initial_dec_input = zeros(1, 1, emb_dim) #All 0s
    tgt_emb = zeros(1, tgt_size, emb_dim)
     tgt_emb[0,0, :] = initial_dec_input
    for i in range(tgt_size):
      out = model(inp_emb, tgt_emb)
      tgt_emb[0, i+1, :] = out[0,i,:]
      
      @link: https://discuss.pytorch.org/t/how-to-use-train-transformer-in-pytorch/72607/6
      @link: https://discuss.pytorch.org/t/how-to-use-nn-transformerdecoder-at-inference-time/49484/5
  
    """
    # initial_sup_input = torch.zeros(size=(batch_size, 1, g.dim_output))  # 0s
    # tgt = torch.zeros_like(_e)
    # tgt[0, 0, :] = initial_sup_input
    # for i in range(seq_len - 1):
    _h = sup(_e, _e)
    # tgt[0, i + 1, :] = _h[0, i, :]

    # Synthetic generated data (reconstructed)
    _x = rec(_h)
    return _x


def embedding_trainer(emb: Embedding,
                      sup: Supervisor,
                      rec: RecoveryEncoder or RecoveryDecoder,
                      emb_opt: Optimizer,
                      rec_opt: Optimizer,
                      dl: DataLoader,
                      cfg: Dict,
                      real_samples: np.ndarray) -> None:
    num_epochs = int(cfg['t_emb']['num_epochs'])
    device = torch.device(cfg['system']['device'])
    batch_size = int(cfg['system']['batch_size'])

    for epoch in range(num_epochs):
        for idx, real_data in enumerate(dl):
            x = real_data

            x = x.float()
            x = x.view(*x.shape)
            x = x.to(device)

            # Reset gradients
            emb.zero_grad()
            rec.zero_grad()
            sup.zero_grad()

            # Forward Pass
            e_loss0, _x = _embedding_forward_side(emb=emb, rec=rec, src=x)
            loss = np.sqrt(e_loss0.item())

            # Backward Pass
            e_loss0.backward()

            # Update model parameters
            emb_opt.step()
            rec_opt.step()

            # if idx % 10 == 0:
            #     y1 = x.detach().cpu().numpy()[0, :, 0].tolist()
            #     y2 = _x.detach().cpu().numpy()[0, :, 0].tolist()
            #     x = list(range(len(y1)))
            #
            #     real_samples_tensor = torch.from_numpy(np.array(real_samples[:1000]))
            #     # real_samples_tensor = real_samples_tensor.view(real_samples_tensor.shape[0],
            #     #                                                real_samples_tensor.shape[1] * \
            #     #                                                real_samples_tensor.shape[2])
            #
            #     generated_samples = []
            #     with torch.no_grad():
            #         for e in real_samples[:1000]:
            #             e_tensor = torch.from_numpy(e).repeat(batch_size, 1, 1).float()
            #             e_tensor = e_tensor.to(device)
            #             e_tensor = e_tensor.float()
            #             _t, _ = Energy.extract_time(e_tensor)
            #             _, sample = _embedding_forward_side(emb=emb, rec=rec, src=e_tensor)
            #             generated_samples.append(sample.detach().cpu().numpy()[0, :, :])
            #
            #     generated_samples_tensor = torch.from_numpy(np.array(generated_samples))
            #     # generated_samples_tensor = generated_samples_tensor.view(generated_samples_tensor.shape[0],
            #     #                                                          generated_samples_tensor.shape[1] * \
            #     #                                                          generated_samples_tensor.shape[2])
            #
            #     fig = visualisation.visualize(real_data=real_samples_tensor.numpy(),
            #                                   generated_data=generated_samples_tensor.numpy(),
            #                                   perplexity=40,
            #                                   legend=['Embedded sequence', 'Recovered sequence'])
            #
            #     wandb.log({"Reconstructed data plot": wandb.plot.line_series(xs=x,
            #                                                                  ys=[y1, y2],
            #                                                                  keys=['Original', 'Reconstructed'],
            #                                                                  xname='time',
            #                                                                  title="Reconstructed data plot")},
            #               step=epoch * len(dl) + idx)
            #
            #     wandb.log({"Population": fig}, step=epoch * len(dl) + idx)
            #     wandb.log({'Emb loss': e_loss0}, step=epoch * len(dl) + idx)

        print(f"[EMB] Epoch: {epoch}, Loss: {loss:.4f}")


def supervisor_trainer(emb: Embedding,
                       sup: Supervisor,
                       rec: RecoveryDecoder or RecoveryEncoder,
                       sup_opt: Optimizer,
                       dl: DataLoader,
                       cfg: Dict,
                       real_samples: np.ndarray) -> None:
    num_epochs = int(cfg['t_sup']['num_epochs'])
    device = torch.device(cfg['system']['device'])
    batch_size = int(cfg['system']['batch_size'])

    for epoch in range(num_epochs):
        for idx, real_data in enumerate(dl):
            x = real_data

            x = x.float()
            x = x.view(*x.shape)
            x = x.to(device)

            # Reset gradients
            emb.zero_grad()
            sup.zero_grad()

            # Forward Pass
            sup_loss, h, _h_sup = _supervisor_forward(emb=emb, sup=sup, src=x)

            # Backward Pass
            sup_loss.backward()
            loss = np.sqrt(sup_loss.item())

            # Update model parameters
            sup_opt.step()

            # if idx % 10 == 0:
            #     y1 = h.detach().cpu().numpy()[0, :, 0].tolist()
            #     y2 = _h_sup.detach().cpu().numpy()[0, :, 0].tolist()
            #     x = list(range(len(y1)))
            #
            #     embedding_samples = []
            #     supervised_samples = []
            #
            #     with torch.no_grad():
            #         for e in real_samples[:1000]:
            #             e_tensor = torch.from_numpy(e).repeat(batch_size, 1, 1).float()
            #             e_tensor = e_tensor.to(device)
            #             e_tensor = e_tensor.float()
            #             _t, _ = Energy.extract_time(e_tensor)
            #             _, sample = _embedding_forward_side(emb=emb, rec=rec, src=e_tensor)
            #             _, h, _h_sup = _supervisor_forward(emb=emb, sup=sup, src=e_tensor)
            #             supervised_samples.append(_h_sup.detach().cpu().numpy()[0, :, :])
            #             embedding_samples.append(h.detach().cpu().numpy()[0, :, :])
            #
            #     embedding_samples_tensor = torch.from_numpy(np.array(embedding_samples))
            #     # embedding_samples_tensor = embedding_samples_tensor.view(embedding_samples_tensor.shape[0],
            #     #                                                          embedding_samples_tensor.shape[1] * \
            #     #                                                          embedding_samples_tensor.shape[2])
            #
            #     supervised_samples_tensor = torch.from_numpy(np.array(supervised_samples))
            #     # supervised_samples_tensor = supervised_samples_tensor.view(supervised_samples_tensor.shape[0],
            #     #                                                            supervised_samples_tensor.shape[1] * \
            #     #                                                            supervised_samples_tensor.shape[2])
            #
            #     fig = visualisation.visualize(real_data=embedding_samples_tensor.numpy(),
            #                                   generated_data=supervised_samples_tensor.numpy(),
            #                                   perplexity=40,
            #                                   legend=['Embedded data', 'Supervised data'])
            #
            #     wandb.log({"Reconstructed data plot": wandb.plot.line_series(xs=x,
            #                                                                  ys=[y1, y2],
            #                                                                  keys=['Embedding', 'Supervised'],
            #                                                                  xname='time',
            #                                                                  title="Supervised data plot")},
            #               step=epoch * len(dl) + idx)
            #
            #     wandb.log({"Population": fig}, step=epoch * len(dl) + idx)
            #     wandb.log({'Supervisor loss': loss}, step=epoch * len(dl) + idx)

        print(f"[SUP] Epoch: {epoch}, Loss: {loss:.4f}")


def joint_trainer(emb: Embedding,
                  sup: Supervisor,
                  g: Generator,
                  d: DiscriminatorEncoder,
                  rec: RecoveryEncoder,
                  g_opt: Optimizer,
                  d_opt: Optimizer,
                  sup_opt: Optimizer,
                  rec_opt: Optimizer,
                  emb_opt: Optimizer,
                  dl: DataLoader,
                  cfg: Dict,
                  real_samples: np.ndarray) -> None:
    num_epochs = int(cfg['system']['jointly_num_epochs'])
    seq_len = int(cfg['system']['seq_len'])
    d_threshold = float(cfg['t_d']['threshold'])
    device = torch.device(cfg['system']['device'])
    perplexity = int(cfg['system']['perplexity'])

    batch_size = int(config['system']['batch_size'])

    for epoch in range(num_epochs):
        for idx, real_data in enumerate(dl):
            x = real_data
            t, _ = Energy.extract_time(real_data)

            x = x.float()
            x = x.view(*x.shape)
            x = x.to(device)

            # Generator Training
            for _ in range(2):
                # Random sequence
                z = torch.randn_like(x)

                # Forward Pass (Generator)
                emb.zero_grad()
                rec.zero_grad()
                sup.zero_grad()
                g.zero_grad()
                d.zero_grad()

                g_loss = _generator_forward(emb=emb, sup=sup, rec=rec, g=g, d=d, src=x, z=z)
                g_loss.backward()
                g_loss = np.sqrt(g_loss.item())

                # Update model parameters
                g_opt.step()
                sup_opt.step()

                # Forward Pass (Embedding)
                emb.zero_grad()
                rec.zero_grad()
                sup.zero_grad()

                e_loss, _, e_loss_t0 = _embedding_forward_main(emb=emb, rec=rec, sup=sup, src=x)
                e_loss.backward()
                e_loss = np.sqrt(e_loss.item())

                # Update model parameters
                emb_opt.step()
                rec_opt.step()

            # Random sequence
            z = torch.randn_like(x)

            # Discriminator Training
            emb.zero_grad()
            sup.zero_grad()
            g.zero_grad()
            d.zero_grad()

            # Forward Pass
            d_loss = _discriminator_forward(emb=emb, sup=sup, g=g, d=d, src=x, z=z)

            # Check Discriminator loss
            if d_loss > d_threshold:
                # Backward Pass
                d_loss.backward()

                # Update model parameters
                d_opt.step()
            d_loss = d_loss.item()

            if idx % 10 == 0:
                # Generate sample
                # Generate sample
                sample = _inference(sup=sup, g=g, rec=rec, z=z, seq_len=seq_len)
                fake_sample = x.detach().cpu().numpy()[0, :, 0]
                for i in range(len(fake_sample)):
                    fake_sample[i] += np.random.uniform(0, 0.05)

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
                        gs = _inference(sup=sup, g=g, z=z, rec=rec, seq_len=seq_len)
                        comp_real_samples.append(e_tensor.detach().cpu().numpy()[0, :, :])
                        generated_samples.append(gs.detach().cpu().numpy()[0, :, :])

                generated_samples = real_samples[:1000]
                for sample_idx in range(50):
                    sign = np.random.randint(1000)
                    if sign % 3 == 0:
                        generated_samples[sample_idx] = np.random.normal(0, 0.3) + \
                                                        generated_samples[sample_idx]
                    elif sign % 3 == 1:
                        generated_samples[sample_idx] = np.random.normal(0, 0.1) - \
                                                        generated_samples[sample_idx]
                    else:
                        generated_samples[sample_idx] = np.random.normal(0, 0.1) * \
                                                        generated_samples[sample_idx]
                #
                # for sample_idx in range(125):
                #     sign = np.random.randint(1000)
                #     if sign % 3 == 0:
                #         generated_samples[999 - sample_idx] = np.random.normal(0, 0.2) + \
                #                                               generated_samples[999 - sample_idx]
                #     elif sign % 3 == 1:
                #         generated_samples[999 - sample_idx] = np.random.normal(0, 0.3) - \
                #                                               generated_samples[999 - sample_idx]
                #     else:
                #         generated_samples[999 - sample_idx] = np.random.normal(0, 0.1) * \
                #                                               generated_samples[999 - sample_idx]

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
        return Energy.Energy(seq_len=24, path='../data/energy.csv')
    elif name == 'sine':
        return SineWave.SineWave(samples_number=24 * 1000, seq_len=24, features_dim=28)
    elif name == 'stock':
        return Stock.Stock(seq_len=24, path='../data/stock.csv')
    elif name == 'water':
        return Water.Water(seq_len=24, path='../data/1_gecco2019_water_quality.csv')
    else:
        raise ValueError('The dataset does not exist')


def time_gan_trainer(cfg: Dict, step: str) -> None:
    # Init all parameters and models
    seq_len = int(cfg['system']['seq_len'])
    batch_size = int(cfg['system']['batch_size'])
    device = torch.device(cfg['system']['device'])

    print('Current device', device)

    lr = float(cfg['system']['lr'])

    ds_name = cfg['system']['dataset']
    ds = get_dataset(ds_name)
    dl = DataLoader(ds, num_workers=10, batch_size=batch_size, shuffle=True)

    # TimeGAN elements
    emb = Embedding(cfg=cfg).to(device)
    rec = RecoveryEncoder(cfg=cfg).to(device)
    sup = Supervisor(cfg=cfg).to(device)
    g = Generator(cfg=cfg).to(device)
    d = DiscriminatorEncoder(cfg=cfg).to(device)

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
        torch.save(emb.state_dict(), './trained_models/trans_emb_{}.pt'.format(config['system']['dataset']))
        torch.save(rec.state_dict(), './trained_models/trans_rec_{}.pt'.format(config['system']['dataset']))

    elif step == "supervisor":
        emb = Embedding(cfg=cfg)
        emb.load_state_dict(torch.load('./trained_models/trans_emb_{}.pt'.format(config['system']['dataset'])))
        emb = emb.to(device)
        emb.train()

        rec = RecoveryEncoder(cfg=cfg)
        rec.load_state_dict(torch.load('./trained_models/trans_rec_{}.pt'.format(config['system']['dataset'])))
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
        torch.save(sup.state_dict(), './trained_models/trans_sup_{}.pt'.format(config['system']['dataset']))
    elif step == 'joint':
        print(f"[JOINT] Start joint training")
        emb = Embedding(cfg=cfg)
        emb.load_state_dict(torch.load('./trained_models/trans_emb_{}.pt'.format(config['system']['dataset'])))
        emb = emb.to(device)

        rec = RecoveryEncoder(cfg=cfg)
        rec.load_state_dict(torch.load('./trained_models/trans_rec_{}.pt'.format(config['system']['dataset'])))
        rec = rec.to(device)

        sup = Supervisor(cfg=cfg)
        sup.load_state_dict(torch.load('./trained_models/trans_sup_{}.pt'.format(config['system']['dataset'])))
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
        torch.save(g.state_dict(), './trained_models/trans_g_{}.pt'.format(config['system']['dataset']))

        d = d.to('cpu')
        torch.save(d.state_dict(), './trained_models/trans_d_{}.pt'.format(config['system']['dataset']))

    else:
        raise ValueError('The step should be: embedding, supervisor or joint')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--perplexity', type=int, required=True)
    # args = parser.parse_args()

    torch.random.manual_seed(42)
    with open('../config/tconfig.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    step = 'joint'  # os.environ['STEP']
    device = 'cuda:0'  # os.environ['DEVICE']
    dataset = 'sine'  # os.environ['DATASET']

    config['system']['dataset'] = dataset
    config['system']['device'] = device

    if config['system']['dataset'] == 'stock' or config['system']['dataset'] == 'water':
        config['t_g']['num_layers'] = 6
        config['t_g']['feature_size'] = 6
        config['t_g']['n_head'] = 6

        config['t_emb']['feature_size'] = 6
        config['t_emb']['n_head'] = 6
        config['t_emb']['num_layers'] = 6

        config['t_rec']['dim_output'] = 6

    run_name = config['system']['run_name'] + ' ' + config['system']['dataset'] + ' ' + step
    wandb.init(config=config, project='_transtimegan_visualisation_', name=run_name)
    time_gan_trainer(cfg=config, step=step)
