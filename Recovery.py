import torch
import torch.nn as nn
from typing import Dict


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
        self.rec_linear = nn.Linear(self.dim_hidden, self.dim_output)

        with torch.no_grad():
            for name, param in self.rec_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.rec_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

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

        recovered_x = self.rec_linear(h_0)
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


if __name__ == '__main__':
    run_recovery_test()
