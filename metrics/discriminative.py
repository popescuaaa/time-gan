from GeneralRNN import GeneralRNN
from data import SineWave, Energy
from TimeGANv1 import Generator, Recovery, Supervisor, _inference
import torch
import yaml
import numpy as np

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


if __name__ == '__main__':
    ds = SineWave.SineWave(samples_number=1000, seq_len=24, features_dim=28)
    real_samples = ds.get_distribution()

    # Load generator
    with open('../config/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device(config['system']['device'])

    rec = Recovery(cfg=config)
    rec.load_state_dict(torch.load('../trained_models/rec_energy.pt'))
    rec = rec.to(device)
    rec.eval()

    g = Generator(cfg=config)
    g.load_state_dict(torch.load('../trained_models/g_energy.pt'))
    g = g.to(device)
    g.eval()

    sup = Supervisor(cfg=config)
    sup.load_state_dict(torch.load('../trained_models/sup_energy.pt'))
    sup = sup.to(device)
    sup.eval()

    rs_tensor = torch.from_numpy(np.array(real_samples[:1000])).to(device).float()
    _t, _ = Energy.extract_time(rs_tensor)
    z = torch.rand_like(rs_tensor)
    gs = _inference(sup=sup, g=g, z=z, t=_t, rec=rec)

    g = GeneralRNN(cfg=cfg)

    synth_data = gs.detach().cpu().numpy()

    # Train test split
    rd = np.asarray(real_samples)
    sd = synth_data[:len(real_samples)]
    n_events = len(rd)

    # Split data on train and test
    idx = np.arange(n_events)
    n_train = int(.75 * n_events)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    seq_len = int(config['system']['seq_len'])
    # Define the X for synthetic and real data
    X_stock_train = rd[train_idx, :seq_len, :]
    X_synth_train = sd[train_idx, :seq_len, :]

    X_stock_test = rd[test_idx, :seq_len, :]
    y_stock_test = rd[test_idx, -1, :]

    # Define the y for synthetic and real datasets
    y_stock_train = rd[train_idx, -1, :]
    y_synth_train = sd[train_idx, -1, :]

    print('Synthetic X train: {}'.format(X_synth_train.shape))
    print('Real X train: {}'.format(X_stock_train.shape))

    print('Synthetic y train: {}'.format(y_synth_train.shape))
    print('Real y train: {}'.format(y_stock_train.shape))

    print('Real X test: {}'.format(X_stock_test.shape))
    print('Real y test: {}'.format(y_stock_test.shape))
