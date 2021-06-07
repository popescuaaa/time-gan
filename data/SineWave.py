from torch.utils.data import Dataset
import numpy as np


class SineWave(Dataset):
    def __init__(self, samples_number: int, seq_len: int, features_dim: int):
        self.data = []
        # Generate sine data
        for i in range(samples_number):
            # Initialize each time-series
            temp = []
            # For each feature
            for k in range(features_dim):
                # Randomly drawn frequency and phase
                freq = np.random.uniform(0, 0.1)
                phase = np.random.uniform(0, 0.1)

                # Generate sine signal based on the drawn frequency and phase
                temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
                temp.append(temp_data)

            # Align row/column
            temp = np.transpose(np.asarray(temp))
            # Normalize to [0,1]
            temp = (temp + 1) * 0.5
            # Stack the generated data
            self.data.append(temp)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def get_distribution(self):
        return self.data
