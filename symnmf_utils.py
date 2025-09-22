import numpy as np
import sys

def error():
    print("An error has occurred", file=sys.stderr)
    sys.exit(1) 

def init_H(k, n, W, seed=1234):
    np.random.seed(seed)
    m = np.mean(W, dtype=np.float64)
    scale = 2 * np.sqrt(m / k)
    return np.random.uniform(0, scale, size=(n, k))
