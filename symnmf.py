import sys
import numpy as np
import symnmf as symnmf_c

# Define functions for Symmetric Non-negative Matrix Factorization (SymNMF)
def error():
    print("An Error Has Occurred")
    sys.exit(1)

# Function to load data from a text file
def load_data(path):
    try:
        return np.loadtxt(path, delimiter=',', ndmin=2).tolist()
    except Exception:
        error()

# Function to print matrix with 4 decimal places
def print_matrix(M):
    for row in M:
        print(",".join(f"{x:.4f}" for x in row))

# Function to initialize H matrix
# WARNING: If you need to import init_H for testing, import it from symnmf_utils.py, not from this file.
# This file is shadowed by the C extension when imported as 'symnmf'.
def init_H(k, n, W, seed=1234):
    np.random.seed(seed)
    scale = 2 * np.sqrt(W / k)
    return np.random.uniform(0, scale, size=(n, k))

def main():
    if len(sys.argv) != 4: error()
    try:
        k = int(sys.argv[1])
        goal = sys.argv[2]
        file_path = sys.argv[3]
    except (ValueError, IndexError): error()
    if goal not in {"symnmf", "sym", "ddg", "norm"}: error()
    data = load_data(file_path)
    n = len(data)
    dim = len(data[0])

    try:
        A = symnmf_c.sym(data, dim)
        if goal == "sym": print_matrix(A); return
        
        D = symnmf_c.ddg(A)
        if goal == "ddg": print_matrix(D); return

        W = symnmf_c.norm(A, D)
        if goal == "norm": print_matrix(W); return

        if not (0 < k < n): error()
        H0 = init_H(k, n, np.mean(W))           
        H = symnmf_c.symnmf(H0, W, k)
        print_matrix(H)
    except Exception:
        error()

if __name__ == "__main__":
    main()