import random
from scipy.special import lambertw
import numpy as np
import torch
import torch.utils.data

def compute_constants(reg, nz, R=1, num_random_samples=100, seed=49):
    q = (1 / 2) + (R ** 2) / reg
    y = R ** 2 / (reg * nz)
    q = np.real((1 / 2) * np.exp(lambertw(y)))

    C = (2 * q) ** (nz / 4)

    np.random.seed(seed)
    var = (q * reg) / 4
    U = np.random.multivariate_normal(
        np.zeros(nz), var * np.eye(nz), num_random_samples
    )
    U = torch.from_numpy(U)

    U_init = U
    C_init = torch.DoubleTensor([C])
    q_init = torch.DoubleTensor([q])

    return q_init, C_init, U_init