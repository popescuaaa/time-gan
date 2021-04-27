import torch
import torch.nn as nn
from typing import Dict


class Embedding(nn.Module):
    """

        ENCODER

    """

    def __init__(self, cfg: Dict):
        super(Embedding, self).__init__()
        self.dim_features = int(cfg['emb']['dim_features'])
        self.dim_hidden = int(cfg['emb']['dim_hidden'])
        self.num_layers = int(cfg['emb']['num_layers'])
        self.seq_len = int(cfg['system']['seq_len'])

        # Dynamic RNN input
        self.padding_value = int(cfg['system']['padding_value'])

        # Architecture
        self.emb_rnn = nn.GRU(input_size=self.dim_features,
                              hidden_size=self.dim_hidden,
                              num_layers=self.num_layers,
                              batch_first=True)
        self.emb_linear = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.emb_sigmoid = nn.Sigmoid()

        with torch.no_grad():
            for name, param in self.emb_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.emb_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
            :param x: time series batch * sequence_len * features
            :param t: temporal information batch * 1
            :return: (H) latent space embeddings batch * sequence_len * H
        """
        x_packed = nn.utils.rnn.pack_padded_sequence(input=x,
                                                     lengths=t,
                                                     batch_first=True,
                                                     enforce_sorted=True)

        h_0, _ = self.emb_rnn(x_packed)
        h_0, _ = nn.utils.rnn.pad_packed_sequence(sequence=h_0,
                                                  batch_first=True,
                                                  padding_value=self.padding_value,
                                                  total_length=self.seq_len)

        logits = self.emb_linear(h_0)
        h = self.emb_sigmoid(logits)

        return h

    @property
    def device(self):
        return next(self.parameters()).device


def run_embedding_test() -> None:
    cfg = {
        "emb": {
            "dim_features": 5,  # feature dimension
            "dim_hidden": 100,  # latent space dimension
            "num_layers": 50  # number of layers in GRU
        },
        "system": {
            "seq_len": 150,
            "padding_value": 0.0  # default on 0.0
        }
    }

    emb = Embedding(cfg)
    x = torch.randn(size=(10, 150, 5))
    t = torch.ones(size=(10,))
    result = emb(x, t)
    assert result.shape == torch.Size((10, 150, 100)), 'Embedding failed to encode input data'


if __name__ == '__main__':
    run_embedding_test()
