from TimeGANv1 import _inference, Embedding, Generator, Supervisor, Recovery, get_dataset
import torch
import yaml
from data import Energy
from torch.utils.data import DataLoader
import numpy as np
from metrics import GeneralRNN

def generate():
    samples = []
    device = torch.device('cpu')

    with open('config/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cfg = config

    emb = Embedding(cfg=cfg)
    emb.load_state_dict(torch.load('./trained_models/emb_{}.pt'.format(config['system']['dataset'])))
    emb = emb.to(device)

    rec = Recovery(cfg=cfg)
    rec.load_state_dict(torch.load('./trained_models/rec_{}.pt'.format(config['system']['dataset'])))
    rec = rec.to(device)

    sup = Supervisor(cfg=cfg)
    sup.load_state_dict(torch.load('./trained_models/sup_{}.pt'.format(config['system']['dataset'])))
    sup = sup.to(device)

    g = Generator(cfg=cfg)
    g.load_state_dict(torch.load('./trained_models/g_{}.pt'.format(config['system']['dataset'])))
    g = g.to(device)

    ds = get_dataset(config['system']['dataset']) # Energy
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    count = 0
    for idx, e in enumerate(dl):
        count += 1
        if count > 1000:
            break
        x = e
        x = x.float()
        t, _ = Energy.extract_time(x)
        z = torch.rand_like(x)
        sample = _inference(sup=sup, g=g, rec=rec, z=z, t=t)
        samples.append(sample.detach().numpy()[0, :, :])

    return np.array(samples)


if __name__ == '__main__':
    samples = generate()
    print(samples.shape)
    ds = get_dataset('energy')
    real_samples = np.array(ds[:1000])
    print(real_samples.shape)

    rnn_cfg = {
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

    gen_rnn = GeneralRNN.GeneralRNN(cfg=rnn_cfg)
    x = samples[0]
    out = gen_rnn(x)
    print(out.shape)

