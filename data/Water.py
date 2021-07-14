from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


def MinMaxScaler(data: np.ndarray) -> np.ndarray:
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def extract_time(data: np.ndarray):
    time = []
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len


class Water(Dataset):
    def __init__(self, seq_len: int, path: str):
        super(Water, self).__init__()
        self.original_data = pd.read_csv(path)
        self.data = self.original_data.drop(
            [self.original_data.columns[0], self.original_data.columns[1], self.original_data.columns[-1]], axis=1)
        self.data = self.data.to_numpy()
        self.processed_data = []
        for i in range(0, len(self.data) - seq_len):
            _x = self.data[i:i + seq_len]
            self.processed_data.append(_x)
        print('done')

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, item):
        return self.processed_data[item]

    def get_distribution(self):
        return self.processed_data


if __name__ == '__main__':
    ds = Water(seq_len=120, path='./1_gecco2019_water_quality.csv')
    dl = DataLoader(ds, batch_size=100, num_workers=2, shuffle=False)
    for i, e in enumerate(dl):
        print(e)