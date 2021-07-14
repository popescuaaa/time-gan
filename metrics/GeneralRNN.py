import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict
from data import Energy
from torch.utils.data import DataLoader


def _get_rnn_module(model_type: str) -> nn.RNN or nn.LSTM or nn.GRU:
    if model_type == "rnn":
        return nn.RNN
    elif model_type == "lstm":
        return nn.LSTM
    elif model_type == "gru":
        return nn.GRU


class Logger(nn.Module):
    def __init__(self):
        super(Logger, self).__init__()

    def forward(self, x: Tensor):
        print('Logger: {}'.format(x[0].shape))
        return x[0]


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

        self.module = _get_rnn_module(self.type)

        self.net = nn.Sequential(
            nn.LSTM(input_size=self.dim_input,
                    hidden_size=self.dim_hidden,
                    num_layers=self.num_layers,
                    dropout=self.dropout),
            Logger(),
            nn.Linear(
                in_features=self.dim_hidden,
                out_features=self.dim_output
            ),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor):
        logs = self.net(x)
        return logs


def run_general_rnn_test() -> None:
    cfg = {
        'rnn': {
            'type': 'lstm',
            'dim_input': 28,
            'dim_output': 1,
            'dim_hidden': 256,
            'num_layers': 1,
            'dropout': 0.3,
            'bidirectional': 'false',
            'padding_value': 0.0,
            'max_seq_len': 24
        }
    }

    gen_rnn = GeneralRNN(cfg=cfg)
    x = torch.randn(size=(1, 24, 28))
    out = gen_rnn(x)
    print(out.shape)


if __name__ == '__main__':
    run_general_rnn_test()
    ds = Energy.Energy(seq_len=24, path='../data/energy.csv')
    dl = DataLoader(ds, batch_size=100, num_workers=2, shuffle=True)

    for epoch in range(100):
        for idx, e in enumerate(dl):
            pass
