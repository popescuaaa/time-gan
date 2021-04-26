import torch
import torch.nn as nn


class Generator(nn.Module):
    """

        ENCODER

    """

    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.dim_latent = int(cfg['g']['dim_latent'])
        self.dim_hidden = int(cfg['g']['dim_hidden'])
        self.num_layers = int(cfg['e']['num_layers'])
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

    def forward(self, z, t):
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
