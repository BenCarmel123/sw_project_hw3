import sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)
np.random.seed(1234)

# Define functions for Symmetric Non-negative Matrix Factorization (SymNMF)
sym = lambda X: np.exp(-0.5 * np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)**2) # Calculate the similarity matrix
ddg = lambda A: np.diag(np.sum(A, axis=1)) # Calculate the degree diagonal matrix
D_inv_sqrt = lambda D: np.diag(1.0 / np.sqrt(np.diag(D))) # Inverse square root of the degree matrix
norm = lambda A, D: D_inv_sqrt(D) @ A @ D_inv_sqrt(D)  # Normalized graph Laplacian
forbenius_norm = lambda X: np.linalg.norm(X, 'fro') # Frobenius norm
def optimize(W, n, k): # Symmetric Non-negative Matrix Factorization
    H, m = np.zeros((n, k)), np.mean(W)
    for i in range(n):
        for j in range(k):
            H[i, j] = np.random.uniform(0, 2 * np.sqrt(m / k))
    for it in range(300):
        H_old = H.copy()
        H_new = H * (0.5 + 0.5 * (W @ H_old) / (H_old @ H_old.T @ H_old))
        if (forbenius_norm(W - H_new @ H_new.T)**2) < 1e-4:
            break
    return H_new
symnmf = lambda X, n, k: optimize(norm(sym(X), ddg(sym(X))), n, k) # Main function

# Main execution block
if __name__ == "__main__":
    try:
        k, goal, file_path = int(sys.argv[1]), sys.argv[2], sys.argv[3] # Read command line arguments
        X = np.loadtxt(file_path, delimiter=',') # Load data from CSV file
        n = X.shape[0]
        funcs = { "sym": lambda: sym(X), "ddg": lambda: ddg(sym(X)), "norm": lambda: norm(sym(X, n), ddg(sym(X, n), n)), "symnmf": lambda: symnmf(X, n, k, 100, 1e-5) } # Map goals to functions
        M = funcs[goal]() # Execute the selected function
        for row in M: 
            print(','.join(f"{x:.4f}" for x in row)) # Print the resulting matrix with 4 decimal places
    except Exception: 
        print("An Error Has Occurred")
        sys.exit()

