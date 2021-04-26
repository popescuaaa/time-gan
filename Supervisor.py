import torch
import torch.nn as nn


class Supervisor(nn.Module):
    """

        DECODER, for predicting next step data

    """

    def __init__(self, cfg):
        super(Supervisor, self).__init__()
        self.dim_hidden = int(cfg['sup']['dim_hidden'])
        self.num_layers = int(cfg['sup']['num_layers'])
        self.seq_len = int(cfg['system']['seq_len'])

        # Dynamic RNN input
        self.padding_value = int(cfg['system']['padding_value'])

        # Architecture
        self.sup_rnn = nn.GRU(input_size=self.dim_hidden,
                              hidden_size=self.dim_hidden,
                              num_layers=self.num_layers,
                              batch_first=True)
        self.sup_linear = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.sup_sigmoid = nn.Sigmoid()

        with torch.no_grad():
            for name, param in self.sup_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.sup_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, h, t):
        """
            :param h: latent representation batch * sequence_len * H
            :param t: temporal information batch * 1
            :return: (_H) predicted next step data (latent form) batch * sequence_len * H
        """
        h_packed = nn.utils.rnn.pack_padded_sequence(input=h,
                                                     lengths=t,
                                                     batch_first=True,
                                                     enforce_sorted=True)
        h_0, _ = self.emb_rnn(h_packed)
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
