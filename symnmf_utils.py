import numpy as np
import sys

def init_H(k, n, W, seed=1234):
    np.random.seed(seed)
    m = np.mean(W)  # scalar
    scale = 2 * np.sqrt(m / k)
    return np.random.uniform(0, scale, size=(n, k))

def error():
    print("An error has occurred", file=sys.stderr)
    sys.exit(1) 
