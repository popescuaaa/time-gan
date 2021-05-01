import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import torch


class NIPSDataSet(Dataset):
    MODELS = ['timegan', 'rcgan']

    def __init__(self, seq_len, data: np.ndarray, model: str, offset: int):
        self.seq_len = seq_len
        self.raw_data = data
        self.model = model
        if self.model == 'timegan':
            self.data = [torch.from_numpy(np.array(self.raw_data[i:i + self.seq_len]))
                         for i in range(0, len(self.raw_data) - self.seq_len)]

            # Compute ∆t (deltas)
            self.data = self.data[:(len(self.data) - offset)]
            self.dt_data = [torch.tensor([self.seq_len], dtype=torch.int64) for _ in range(len(self.data))]

            self.full_data = [(self.data[i], self.dt_data[i]) for i in range(min(len(self.data), len(self.dt_data)))]

        elif self.model == 'rcgan':
            self.data = [torch.from_numpy(np.array(self.raw_data[i:i + self.seq_len]))
                         for i in range(0, len(self.raw_data) - self.seq_len)]

            # Compute ∆t (deltas)
            self.dt_data = [torch.from_numpy(np.concatenate([np.array([0]),
                                                             self.data[i][1:].numpy() -
                                                             self.data[i][:-1].numpy()]))
                            for i in range(len(self.data))]

            # Filter for small size chunks
            self.data = list(filter(lambda t: t.shape[0] == seq_len, self.data))
            self.dt_data = list(filter(lambda t: t.shape[0] == seq_len, self.dt_data))

            self.dt_data = self.dt_data[:(len(self.dt_data) - offset)]
            self.full_data = [(self.data[i], self.dt_data[i]) for i in range(min(len(self.data), len(self.dt_data)))]

        else:
            raise ValueError('Model should be either rcgan or timegan!')

    def __len__(self):
        return len(self.full_data)

    def __getitem__(self, item):
        return self.full_data[item]

    def mean_reshape(self, arr: np.array):
        mean = np.mean(arr)
        return np.repeat(mean, self.seq_len)

    def get_real_distribution(self):
        real_distribution = np.array(list(map(lambda t: t.numpy(), self.data)))
        return np.array(list(map(self.mean_reshape, real_distribution)))


class GeneralDataset:
    PATHS = {
        'electricity': './data/electricity_nips/train/data.json',
        'solar': './data/solar_nips/train/train.json',
        'traffic': './data/traffic_nips/train/data.json',
        'exchange': './data/exchange_rate_nips/train/train.json',
        'taxi': 'taxi_30min/train/train.json'
    }

    OFFSETS = {
        'electricity': 3,
        'solar': 9,
        'traffic': 1,
        'exchange': 2,
        'taxi': 8
    }

    def __init__(self, seq_len: int, ds_type: str, model: str):
        self.seq_len = seq_len
        self.ds_type = ds_type
        self.model = model
        self.offset = self.OFFSETS[ds_type]
        self.json_data = []
        self.path = self.PATHS[self.ds_type]

        with open(self.path) as f:
            for item in f:
                data = json.loads(item)
                self.json_data.append(data)

        self.data = pd.DataFrame(self.json_data)
        self.data = self.data.sort_values(by='start')
        self.timestamps = self.data['start']
        self.values = self.data['target'].values

    def get_dataset(self):
        return NIPSDataSet(seq_len=self.seq_len, data=self.values[0], model=self.model, offset=self.offset)


if __name__ == '__main__':
    ds_generator = GeneralDataset(150, 'electricity', 'timegan')
    ds = ds_generator.get_dataset()
    dl = DataLoader(ds, num_workers=10, batch_size=10, shuffle=True)
    for idx, e in enumerate(dl):
        data, dt = e
        print(data)
        print(dt)
        break

