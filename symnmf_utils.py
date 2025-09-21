import numpy as np

def init_H(k, n, W, seed=1234):
    np.random.seed(seed)
    scale = 2 * np.sqrt(W / k)
    return np.random.uniform(0, scale, size=(n, k))
