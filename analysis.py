import sys
import numpy as np
from sklearn.metrics import silhouette_score
import symnmf as symnmf_c
from kmeans import initCentr, assignClusters, updateCentr
from symnmf_utils import error, init_H

#SymNMF silhouette score calculation - 
#runs the SymNMF clustering algorithm and gets its silhouette score. 
def symnmf_silhouette(X, k): 
 
    n = X.shape[0]
    A = symnmf_c.sym(X.tolist())
    if A is None: error()
    D = symnmf_c.ddg(A)
    if D is None: error()
    W = symnmf_c.norm(A, D)
    if W is None: error()

    # Initialize H0 using the init_H function 
    H0 = init_H(k, n, W).tolist()
    if H0 is None: error()
    
    H = symnmf_c.symnmf(H0, W)
    if H is None: error()
    
    # Assign a cluster based on the highest value in each row of H
    nmf_labels = np.argmax(np.array(H), axis=1)
    return silhouette_score(X, nmf_labels)

# KMeans silhouette score calculation
def kmeans_silhouette(X, k): 
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
        sym_sklearn, kmeans_sklearn = symnmf_silhouette(X, k), kmeans_silhouette(X, k)
        print(f"nmf: {sym_sklearn:.4f}", file=sys.stderr)
        print(f"kmeans: {kmeans_sklearn:.4f}", file=sys.stderr)

    except Exception as e:
        error()