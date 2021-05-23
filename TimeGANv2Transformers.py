import torch
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
from data import Energy


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
        self.feature_size = int(cfg['emb']['feature_size'])
        self.num_layers = int(cfg['emb']['num_layers'])
        self.dropout = float(cfg['emb']['dropout'])
        self.n_head = int(cfg['emb']['n_head'])  # 10
        self.dim_output = int(cfg['emb']['dim_output'])

        self.model_type = 'Transformer'
        self.src_mask = None

        # Architecture
        self.pos_encoder = PositionalEncoding(dim_model=self.feature_size)
        self.encoder_layer = TransformerEncoderLayer(d_model=self.feature_size,
                                                     nhead=self.n_head,
                                                     dropout=self.dropout,
                                                     dim_feedforward=self.feature_size * 4)
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=self.num_layers)
        self.decoder = nn.Linear(in_features=self.feature_size, out_features=self.dim_output)

        # Init weights
        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: Tensor) -> Tensor:
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = _generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    @property
    def device(self):
        return next(self.parameters()).device


class RecoveryDecoder(nn.Module):
    def __init__(self, cfg: Dict):
        super(RecoveryDecoder, self).__init__()
        self.feature_size = int(cfg['rec']['feature_size'])
        self.num_layers = int(cfg['rec']['num_layers'])
        self.dropout = float(cfg['rec']['dropout'])
        self.n_head = int(cfg['rec']['n_head'])
        self.dim_output = int(cfg['rec']['dim_output'])

        # Architecture
        self.pos_encoder = PositionalEncoding(dim_model=self.feature_size)
        self.decoder_layer = TransformerDecoderLayer(d_model=self.feature_size,
                                                     nhead=self.n_head,
                                                     dropout=self.dropout,
                                                     dim_feedforward=self.feature_size * 4)
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


class RecoveryEncoder(nn.Module):
    def __init__(self, cfg: Dict):
        super(RecoveryEncoder, self).__init__()
        self.feature_size = int(cfg['rec']['feature_size'])
        self.num_layers = int(cfg['rec']['num_layers'])
        self.dropout = float(cfg['rec']['dropout'])
        self.n_head = int(cfg['rec']['n_head'])
        self.dim_output = int(cfg['rec']['dim_output'])

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


class Supervisor(nn.Module):
    def __init__(self, cfg: Dict):
        super(Supervisor, self).__init__()
        self.feature_size = int(cfg['sup']['feature_size'])
        self.num_layers = int(cfg['sup']['num_layers'])
        self.dropout = float(cfg['sup']['dropout'])
        self.n_head = int(cfg['sup']['n_head'])
        self.dim_output = int(cfg['sup']['dim_output'])
        self.model_type = 'Transformer'

        # Architecture
        self.pos_encoder = PositionalEncoding(dim_model=self.feature_size)
        self.decoder_layer = TransformerDecoderLayer(d_model=self.feature_size,
                                                     nhead=self.n_head,
                                                     dropout=self.dropout,
                                                     dim_feedforward=self.feature_size * 4)
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

    @property
    def device(self):
        return next(self.parameters()).device


class Generator(nn.Module):
    def __init__(self, cfg: Dict):
        super(Generator, self).__init__()
        self.feature_size = int(cfg['g']['feature_size'])
        self.num_layers = int(cfg['g']['num_layers'])
        self.dropout = float(cfg['g']['dropout'])
        self.n_head = int(cfg['g']['n_head'])
        self.dim_output = int(cfg['g']['dim_output'])

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


class DiscriminatorDecoder(nn.Module):
    def __init__(self, cfg: Dict):
        super(DiscriminatorDecoder, self).__init__()
        self.feature_size = int(cfg['d']['feature_size'])
        self.num_layers = int(cfg['d']['num_layers'])
        self.dropout = float(cfg['d']['dropout'])
        self.n_head = int(cfg['d']['n_head'])
        self.dim_output = int(cfg['d']['dim_output'])

        # Architecture
        self.pos_encoder = PositionalEncoding(dim_model=self.feature_size)
        self.decoder_layer = TransformerDecoderLayer(d_model=self.feature_size,
                                                     nhead=self.n_head,
                                                     dropout=self.dropout,
                                                     dim_feedforward=self.feature_size * 4)
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
        self.feature_size = int(cfg['d']['feature_size'])
        self.num_layers = int(cfg['d']['num_layers'])
        self.dropout = float(cfg['d']['dropout'])
        self.n_head = int(cfg['d']['n_head'])
        self.dim_output = int(cfg['d']['dim_output'])

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
    _h_sup = sup(h)  # temporal dynamics

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
    _h_sup = sup(h)

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
    _h = sup(h).detach()
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
    _h_sup = sup(h)
    _x = rec(h)

    # Generator Forward Pass
    _e = g(z)
    _h = sup(_e)

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
    g_loss = g_loss_u + gamma * g_loss_u_e + 100 * torch.sqrt(g_loss_s) + 100 * g_loss_v

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
    _h = sup(_e)
    # tgt[0, i + 1, :] = _h[0, i, :]

    # Synthetic generated data (reconstructed)
    _x = rec(_h)
    return _x


def embedding_trainer(emb: Embedding,
                      sup: Supervisor,
                      rec: RecoveryEncoder,
                      emb_opt: Optimizer,
                      rec_opt: Optimizer,
                      dl: DataLoader,
                      cfg: Dict) -> None:
    num_epochs = int(cfg['emb']['num_epochs'])
    device = torch.device(cfg['system']['device'])

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

            if idx == len(dl) - 1:
                pass
        print(f"[EMB] Epoch: {epoch}, Loss: {loss:.4f}")


def supervisor_trainer(emb: Embedding,
                       sup: Supervisor,
                       sup_opt: Optimizer,
                       dl: DataLoader,
                       cfg: Dict) -> None:
    num_epochs = int(cfg['t_sup']['num_epochs'])
    device = torch.device(cfg['system']['device'])

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

            if idx == len(dl) - 1:
                pass
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
                  cfg: Dict) -> None:

    num_epochs = int(cfg['system']['jointly_num_epochs'])
    seq_len = int(cfg['system']['seq_len'])
    d_threshold = float(cfg['t_d']['threshold'])
    device = torch.device(cfg['system']['device'])
    perplexity = int(cfg['system']['perplexity'])

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

            if idx == len(dl) - 1:
                pass
        print(f"[JOINT] Epoch: {epoch}, E_loss: {e_loss:.4f}, G_loss: {g_loss:.4f}, D_loss: {d_loss:.4f}")


def time_gan_trainer(cfg: Dict) -> None:
    # Init all parameters and models
    seq_len = int(cfg['system']['seq_len'])
    batch_size = int(cfg['system']['batch_size'])
    device = torch.device(cfg['system']['device'])

    lr = float(cfg['system']['lr'])
    # ds_generator = GeneralDataset.GeneralDataset(seq_len, dataset_name, model_name)
    # ds = ds_generator.get_dataset()

    ds = Energy.Energy(seq_len)
    dl = DataLoader(ds, num_workers=10, batch_size=batch_size, shuffle=True)

    pass


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--perplexity', type=int, required=True)
    # args = parser.parse_args()

    torch.random.manual_seed(42)
    with open('config/tconfig.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    run_name = config['system']['run_name'] + ' ' + config['system']['dataset']
    wandb.init(config=config, project='_transtimegan_visualisation_', name=run_name)
    time_gan_trainer(cfg=config)
