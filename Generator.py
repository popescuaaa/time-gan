import torch
import torch.nn as nn
from typing import Dict


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
        self.g_rnn = nn.GRU(input_size=self.dim_latent,
                            hidden_size=self.dim_hidden,
                            num_layers=self.num_layers,
                            batch_first=True)

        self.g_linear = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.g_sigmoid = nn.Sigmoid()

        with torch.no_grad():
            for name, param in self.g_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.g_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, z: torch.Tensor, t: torch.Tensor):
        """
            :param z: random noise batch * sequence_len * dim_latent
            :param t: temporal information batch * 1
            :return: (H) latent space embeddings batch * sequence_len * H
        """
        x_packed = nn.utils.rnn.pack_padded_sequence(input=z,
                                                     lengths=t,
                                                     batch_first=True,
                                                     enforce_sorted=True)
        h_0, _ = self.g_rnn(x_packed)
        h_0, _ = nn.utils.rnn.pad_packed_sequence(sequence=h_0,
                                                  batch_first=True,
                                                  padding_value=self.padding_value,
                                                  total_length=self.seq_len)

        logits = self.g_linear(h_0)
        h = self.g_sigmoid(logits)
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


if __name__ == '__main__':
    run_generator_test()
