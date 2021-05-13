from Transformer import PositionalEncoding, _generate_square_subsequent_mask
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import torch.nn as nn
from torch import Tensor
from typing import Dict


class TDiscriminatorDecoder(nn.Module):
    """
    Discriminator implemented with decoder model
    """

    def __init__(self, cfg: Dict):
        super(TDiscriminatorDecoder, self).__init__()
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


class TDiscriminatorEncoder(nn.Module):
    """
    Recovery implemented with decoder model
    """

    def __init__(self, cfg: Dict):
        super(TDiscriminatorEncoder, self).__init__()
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