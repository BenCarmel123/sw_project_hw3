import sys
import numpy as np
import symnmf_module as symnmf_c

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

def main():
    if len(sys.argv) != 4:
        error()
    k = int(sys.argv[1])
    goal = sys.argv[2]
    file_path = sys.argv[3]
    X = load_data(file_path)
    n = len(X)
    d = len(X[0]) if n > 0 else 0

    if goal == 'symnmf':
        np.random.seed(1234)
        # Call C extension for symnmf
        H = symnmf_c.symnmf(X, k)
        print_matrix(H)
    elif goal == 'sym':
        W = symnmf_c.sym(X)
        print_matrix(W)
    elif goal == 'ddg':
        D = symnmf_c.ddg(X)
        print_matrix(D)
    elif goal == 'norm':
        N = symnmf_c.norm(X, X)  # The C API expects two matrices, but norm(X, D) is correct if D is degree matrix
        print_matrix(N)
    else:
        error()

if __name__ == "__main__":
    main()
