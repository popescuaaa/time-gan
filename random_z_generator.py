import numpy
import numpy as np
import torch

def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    """Random vector generation.

    Args:
      - batch_size: size of the random vector
      - z_dim: dimension of random vector
      - T_mb: time information for the random vector
      - max_seq_len: maximum sequence length

    Returns:
      - Z_mb: generated random vector
    """
    Z_mb = list()
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
        temp[:T_mb[i], :] = temp_Z
        Z_mb.append(temp_Z)
    return Z_mb


if __name__ == '__main__':
    z = random_generator(batch_size=100, z_dim=28, T_mb=100 * [24], max_seq_len=24)
    tz = torch.from_numpy(numpy.array(z))
    print(tz.shape)
    print(z)
    print(torch.rand_like(tz))
