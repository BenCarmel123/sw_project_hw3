import sys
import numpy as np
from sklearn.metrics import silhouette_score
from symnmf import symnmf_c
from kmeans import initCentr, assignClusters, updateCentr
from symnmf import error

def silhouette_symnmf(X, k): # SymNMF silhouette score calculation
    n = X.shape[0]
    H = symnmf_c.symnmf(X, n, k) # Perform SymNMF
    labels = np.argmax(H, axis=1).tolist()  # Assign clusters based on max value in each row
    if k == 1 or k == n:
        return 0.0
    return silhouette_score(X, labels)

def kmeans_silhouette(X, k): # KMeans silhouette score calculation
    vectors, labels = X.tolist(), []
    centroids = initCentr(vectors, k) # Initialize centroids
    for _ in range(300): 
        clusters = assignClusters(vectors, centroids, k) # Assign clusters
        centroids, converged = updateCentr(vectors, centroids, clusters) # Update centroids
        if not converged:
            break
    for vector in vectors: # Assign labels based on closest centroid
        distances = [np.linalg.norm(np.array(vector) - np.array(centroid)) for centroid in centroids]
        labels.append(np.argmin(distances))
    if k == 1 or k == len(vectors):
        return 0.0
    return silhouette_score(X, labels) # Calculate silhouette score

if __name__ == "__main__":

    try:
        k, file_name = int(sys.argv[1]), sys.argv[2]
        X = np.loadtxt(file_name, delimiter=',') # Load data
        sym_sklearn, kmeans_sklearn = silhouette_symnmf(X, k), kmeans_silhouette(X, k)
        print(f"nmf: {sym_sklearn:.4f}\nKmeans: {kmeans_sklearn:.4f}")

    except Exception as e:
        error()