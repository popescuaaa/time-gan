import torch
import torch.nn as nn
from typing import Dict


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
        self.d_linear = nn.Linear(self.dim_hidden, 1)

        with torch.no_grad():
            for name, param in self.d_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.d_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

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

        logits = self.d_linear(h_0).squeeze(-1)
        return logits

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


if __name__ == '__main__':
    run_discriminator_test()
