import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict


def _get_rnn_module(model_type: str) -> nn.RNN or nn.LSTM or nn.GRU:
    if model_type == "rnn":
        return nn.RNN
    elif model_type == "lstm":
        return nn.LSTM
    elif model_type == "gru":
        return nn.GRU


class GeneralRNN(nn.Module):
    def __init__(self, cfg: Dict):
        super(GeneralRNN, self).__init__()
        self.type = cfg['rnn']['type']
        self.dim_input = int(cfg['rnn']['dim_input'])
        self.dim_output = int(cfg['rnn']['dim_output'])
        self.dim_hidden = int(cfg['rnn']['dim_hidden'])
        self.num_layers = int(cfg['rnn']['num_layers'])
        self.dropout = float(cfg['rnn']['dropout'])
        self.bidirectional = bool(cfg['rnn']['bidirectional'])
        self.padding_value = float(cfg['rnn']['padding_value'])
        self.max_seq_len = int(cfg['rnn']['max_seq_len'])

        self.module = _get_rnn_module(self.type)

        self.rnn_layer = self.module(
            input_size=self.dim_input,
            hidden_size=self.dim_hidden,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )

        self.linear_layer = nn.Linear(
            in_features=self.dim_hidden,
            out_features=self.dim_output
        )

    def forward(self, x: Tensor, t: Tensor):
        x_packed = nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=t,
            batch_first=True,
            enforce_sorted=False
        )

        h_0, h_t = self.rnn_layer(x_packed)

        h_0, _ = nn.utils.rnn.pad_packed_sequence(
            sequence=h_0,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        lgs = self.linear_layer(h_0)
        return lgs


def run_general_rnn_test() -> None:
    cfg = {
        'rnn': {
            'type': 'lstm',
            'dim_input': 150,
            'dim_output': 1,
            'dim_hidden': 64,
            'num_layers': 1,
            'dropout': 0.3,
            'bidirectional': 'false',
            'padding_value': 0.0,
            'max_seq_len': 150
        }
    }

    gen_rnn = GeneralRNN(cfg=cfg)
    x = torch.randn(size=(10, 150, 1))
    t = torch.randn(size=(150,))

