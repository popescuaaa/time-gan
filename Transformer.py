import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from typing import Dict, Tuple
import wandb
from data import Energy
from torch.utils.data import DataLoader
from utils import plot_two_time_series


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


class Encoder(nn.Module):
    def __init__(self, cfg: Dict):
        super(Encoder, self).__init__()
        self.feature_size = int(cfg['encoder']['feature_size'])
        self.num_layers = int(cfg['encoder']['num_layers'])
        self.dropout = float(cfg['encoder']['dropout'])
        self.n_head = int(cfg['encoder']['n_head'])  # 10
        self.dim_output = int(cfg['encoder']['dim_output'])

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


class Decoder(nn.Module):
    def __init__(self, cfg: Dict):
        super(Decoder, self).__init__()
        self.feature_size = int(cfg['decoder']['feature_size'])
        self.num_layers = int(cfg['decoder']['num_layers'])
        self.dropout = float(cfg['decoder']['dropout'])
        self.n_head = int(cfg['decoder']['n_head'])
        self.dim_output = int(cfg['decoder']['dim_output'])
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


def run_transformer_test():
    # S is the source sequence length
    # T is the target sequence length
    # N is the batch size
    # E is the feature number

    # src = torch.rand((10, 32, 512)) # (S,N,E)
    # tgt = torch.rand((20, 32, 512)) # (T,N,E)
    # out = transformer_model(src, tgt)

    cfg = {
        'encoder': {
            'feature_size': 28,
            'num_layers': 1,
            'dropout': 0.1,
            'n_head': 7,
            'dim_output': 1
        }
    }
    t = Encoder(cfg=cfg)
    src = torch.randn(size=(128, 24, 28))
    out = t(src)
    assert out.shape == torch.Size((128, 24, 1)), 'Transformer failed to produce correct output shape'


# Training step
def _embedding_forward_step(emb: Encoder,
                            rec: Decoder,
                            src: Tensor,
                            batch_size: int,
                            seq_len: int) -> Tuple[Tensor, Tensor]:
    assert src.device == emb.device, 'Src and model are not on the same device'
    h = emb(src)

    # initial_sup_input = torch.zeros(size=(batch_size, 1, emb.dim_output))  # 0s
    tgt = torch.zeros_like(h)
    # print(initial_sup_input[batch_size - 1, 1, :].shape)
    # tgt[batch_size - 1, 0, :] = initial_sup_input

    for index in range(seq_len - 1):
        _src = rec(tgt, h)
        # print(_src.shape)
        # print(tgt[0, index + 1, :].shape)
        # print(_src[0, index, :].shape)
        tgt[0, index + 1, :] = _src[0, index, :emb.dim_output]

    emb_loss = F.mse_loss(_src, src)
    return emb_loss, _src


if __name__ == '__main__':
    run_transformer_test()
    ds = Energy.Energy(seq_len=24)
    dl = DataLoader(ds, num_workers=2, batch_size=128, shuffle=True)
    cfg = {
        'encoder': {
            'feature_size': 28,
            'num_layers': 10,
            'dropout': 0.1,
            'n_head': 7,
            'dim_output': 14
        },
        'decoder': {
            'feature_size': 14,
            'num_layers': 10,
            'dropout': 0.1,
            'n_head': 7,
            'dim_output': 28
        }
    }
    emb = Encoder(cfg=cfg)
    emb_opt = torch.optim.Adam(emb.parameters(), lr=1e-4)
    rec = Decoder(cfg=cfg)
    rec_opt = torch.optim.Adam(rec.parameters(), lr=1e-4)
    device = torch.device('cuda:0')

    emb = emb.to(device)
    rec = rec.to(device)

    # Logging
    # wandb.init(config=cfg, project='_transformer_test_', name='Transformer test [ energy ]')

    for epoch in range(100):
        for i, e in enumerate(dl):
            rd = e.float()
            rd = rd.to(device)
            emb.zero_grad()
            rec.zero_grad()

            loss, _src = _embedding_forward_step(emb=emb, rec=rec, src=rd, batch_size=128, seq_len=24)

            loss.backward()
            rec_opt.step()
            emb_opt.step()

            rd = rd.detach().cpu().numpy()[0, :, 0]
            _src = _src.detach().cpu().numpy()[0, :, 0]

            fig = plot_two_time_series(real=rd,
                                       real_data_description='Real data',
                                       reconstructed=_src,
                                       reconstructed_data_description='Reconstructed data')
            # if i == len(dl) - 1:
            #     wandb.log({
            #         'epoch': epoch,
            #         'loss': loss,
            #         'Embedding result': fig
            #     })

        print(f"[T_EMB] Epoch: {epoch}, Loss: {loss:.4f}")
