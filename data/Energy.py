from torch.utils.data import Dataset, DataLoader
import numpy as np


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


class Energy(Dataset):
    def __init__(self, seq_len: int, path: str):
        super(Energy, self).__init__()
        original_data = np.loadtxt(path, delimiter=",", skiprows=1)
        original_data = original_data[::-1]
        original_data = MinMaxScaler(original_data)
        self.data = []
        for i in range(0, len(original_data) - seq_len):
            _x = original_data[i:i + seq_len]
            self.data.append(_x)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def get_distribution(self):
        return self.data


if __name__ == '__main__':
    ds = Energy(24)
    dl = DataLoader(ds, num_workers=2, batch_size=128)
    for i, e in enumerate(dl):
        t, msl = extract_time(e)
        print(len(t), e.shape)
