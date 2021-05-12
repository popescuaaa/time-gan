import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from typing import Dict
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


class Transformer(nn.Module):
    def __init__(self, cfg: Dict):
        super(Transformer, self).__init__()
        self.feature_size = int(cfg['transformer']['feature_size'])
        self.num_layers = int(cfg['transformer']['num_layers'])
        self.dropout = float(cfg['transformer']['dropout'])
        self.n_head = int(cfg['transformer']['n_head'])  # 10
        self.dim_output = int(cfg['transformer']['dim_output'])

        self.model_type = 'Transformer'
        self.src_mask = None

        # Architecture
        self.pos_encoder = PositionalEncoding(dim_model=self.feature_size)
        self.encoder_layer = TransformerEncoderLayer(d_model=self.feature_size, nhead=self.n_head, dropout=self.dropout)
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


def run_transformer_test():
    # S is the source sequence length
    # T is the target sequence length
    # N is the batch size
    # E is the feature number

    # src = torch.rand((10, 32, 512)) # (S,N,E)
    # tgt = torch.rand((20, 32, 512)) # (T,N,E)
    # out = transformer_model(src, tgt)

    cfg = {
        'transformer': {
            'feature_size': 28,
            'num_layers': 1,
            'dropout': 0.1,
            'n_head': 7,
            'dim_output': 1
        }
    }
    t = Transformer(cfg=cfg)
    src = torch.randn(size=(128, 24, 28))
    out = t(src)
    assert out.shape == torch.Size((128, 24, 1)), 'Transformer failed to produce correct output shape'


# Training step
def _embedding_forward_step(emb: Transformer,
                            rec: Transformer,
                            src: Tensor):
    assert src.device == emb.device, 'Src and model are not on the same device'
    h = emb(src)
    _src = rec(h)
    emb_loss = F.mse_loss(_src, src)
    return emb_loss, _src


if __name__ == '__main__':
    run_transformer_test()
    ds = Energy.Energy(seq_len=24)
    dl = DataLoader(ds, num_workers=2, batch_size=128, shuffle=True)
    cfg = {
        'transformer': {
            'feature_size': 28,
            'num_layers': 1,
            'dropout': 0.1,
            'n_head': 7,
            'dim_output': 14
        }
    }
    emb = Transformer(cfg=cfg)
    emb_opt = torch.optim.Adam(emb.parameters(), lr=1e-4)

    cfg = {
        'transformer': {
            'feature_size': 14,
            'num_layers': 1,
            'dropout': 0.1,
            'n_head': 7,
            'dim_output': 28
        }
    }
    rec = Transformer(cfg=cfg)
    rec_opt = torch.optim.Adam(rec.parameters(), lr=1e-4)
    device = torch.device('cuda:0')

    emb = emb.to(device)
    rec = rec.to(device)


    # Logging
    wandb.init(config=cfg, project='_transformer_test_', name='Transformer test [ energy ]')

    for epoch in range(100):
        for i, e in enumerate(dl):
            rd = e.float()
            rd = rd.to(device)
            emb.zero_grad()
            rec.zero_grad()

            loss, _src = _embedding_forward_step(emb=emb, rec=rec, src=rd)

            loss.backward()
            rec_opt.step()
            emb_opt.step()

            rd = rd.detach().cpu().numpy()[0, :, 0]
            _src = _src.detach().cpu().numpy()[0, :, 0]

            fig = plot_two_time_series(real=rd,
                                       real_data_description='Real data',
                                       reconstructed=_src,
                                       reconstructed_data_description='Reconstructed data')
            if i == len(dl) - 1:
                wandb.log({
                    'epoch': epoch,
                    'loss': loss,
                    'Embedding result': fig
                })

        print(f"[T_EMB] Epoch: {epoch}, Loss: {loss:.4f}")




