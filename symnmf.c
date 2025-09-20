#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
 
// Global variable for memory allocation check
#define CHECK_ALLOC(ptr) \
    if (!(ptr)) { \
        printf("Memory allocation failed.\n"); \
        exit(1); \
    }

// Error handling function
void* error(void) {
    fprintf(stderr, "An Error Has Occurred\n");
    return NULL;
}


// Function to print matrix according to the specified format
void print_matrix(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.4f", matrix[i][j]);
            if (j < cols - 1) printf(",");
        }
        printf("\n");
    }
}

// Compute L2 norm between two vectors
double l2_norm(double* a, double* b, int dim) { 
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Free allocated memory for a 2D matrix
void free_matrix(double** matrix, int rows) { 
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Free multiple 2D matrices
void free_matrices(double*** matrices, int count, int rows) { 
    for (int i = 0; i < count; i++) {
        free_matrix(matrices[i], rows);
    }
    free(matrices);
}

// Multiply two matrices A (n x m) and B (m x p)
double** matrix_multiply(double** A, double** B, int n, int m, int p) { 
    double** C = malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        C[i] = malloc(p * sizeof(double));
        for (int j = 0; j < p; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < m; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    free_matrix(A, n);
    free_matrix(B, m);
    return C;
}

 // Subtract matrix B from A (A - B)
double** matrix_subtract(double** A, double** B, int n, int m) {
    double** C = malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        C[i] = malloc(m * sizeof(double));
        for (int j = 0; j < m; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    free_matrix(A, n);
    free_matrix(B, n);
    return C;
}

// Compute Frobenius norm of a matrix
double forbenius_norm(double** A, int n) { 
    double norm = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            norm += A[i][j] * A[i][j];
        }
    }
    return sqrt(norm);
}
// Compute similarity matrix using Gaussian kernel
double** sym (double** X, int n) { 
    double** A = malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        A[i] = malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            A[i][j] = exp(-0.5 * pow(l2_norm(X[i], X[j], n), 2));
        }
    }
    return A;
}

// Compute degree diagonal matrix
double** ddg(double** A, int n) { 
    double** D = malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        D[i] = malloc(n * sizeof(double));
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i][j];
        }
        for (int j = 0; j < n; j++) {
            D[i][j] = (i == j) ? sum : 0.0;
        }
    }
    return D;
}

// Compute normalized graph Laplacian
double** norm(double** A, double** D, int n) { 
    double** D_inv_sqrt = malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        D_inv_sqrt[i] = malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) 
            D_inv_sqrt[i][j] = (i == j) ? 1.0 / sqrt(D[i][j]) : 0.0;
    }

    double** temp = malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        temp[i] = malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            temp[i][j] = 0.0;
            for (int k = 0; k < n; k++) 
                temp[i][j] += D_inv_sqrt[i][k] * A[k][j];
        }
    }

    double** N = malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        N[i] = malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            N[i][j] = 0.0;
            for (int k = 0; k < n; k++)
                 N[i][j] += temp[i][k] * D_inv_sqrt[k][j];
        }
    }

    // Free intermediate matrices
    free_matrices((double**[]){D_inv_sqrt, temp, A, D}, 4, n);
    return N;
}

double** optimize(double** W, int n, int k) {

    // Initialize H with random values
    double** H = malloc(n * sizeof(double*));
    double m = 0.0;
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) m += W[i][j];
    m /= (n * n);
    for (int i = 0; i < n; i++) {
        H[i] = malloc(k * sizeof(double));
        for (int j = 0; j < k; j++) 
            H[i][j] = ((double)rand() / RAND_MAX) * 2 * sqrt(m / k); 
    }
    // Iterative optimization
    for (int it = 0; it < 300; it++) {
        double** H_old = malloc(n * sizeof(double*)); 
        for (int i = 0; i < n; i++) {
            H_old[i] = malloc(k * sizeof(double));
            memcpy(H_old[i], H[i], k * sizeof(double)); // Copy H to H_old
        }
        // Matrix operations and updates
        double** WH = matrix_multiply(W, H_old, n, n, k); // WH = W * H_old
        double** H_T = malloc(k * sizeof(double*));
        for (int i = 0; i < k; i++) {
            H_T[i] = malloc(n * sizeof(double));
            for (int j = 0; j < n; j++) H_T[i][j] = H_old[j][i];
        }
        double** HHT = matrix_multiply(H_T, H_old, k, n, k); // HHT = H_old^T * H_old
        double** HHTH = matrix_multiply(H_old, HHT, n, k, k); // HHTH = H * HHT
        for (int i = 0; i < n; i++) 
            for (int j = 0; j < k; j++) 
                H[i][j] = H[i][j] * (0.5 + 0.5 * (WH[i][j] / (HHTH[i][j] + 1e-10)));
        // Convergence check
        double** HHT2 = matrix_multiply(H, H_T, n, k, n); // HHT = H * H^T
        double norm_diff = pow(forbenius_norm(matrix_subtract(W, HHT2, n, n), n), 2); // ||W - HHT||
        free_matrices((double**[]){H_old, WH, H_T, HHT, HHTH, HHT2}, 6, n); // Free intermediate matrices
        if (norm_diff < 1e-4) break;
    }

    return H;
}

double** symnmf(double** X, int n, int k) {
    double** W = sym(X, n); // Compute similarity matrix
    double** D = ddg(W, n); // Compute degree diagonal matrix
    double** N = norm(W, D, n); // Compute normalized graph Laplacian
    double** H = optimize(N, n, k); // Perform SymNMF optimization
    return H;
}

/* Read info from Terminal */
double** readData(int argc, char* argv[], int* k, int* numVectors, int* dim) {

    double **vectors = malloc(100 * sizeof(double*)), *vector;
    int vectorCount = 0, capacity = 100, i;
    char line[1024], *token, *p;
    *k = atoi(argv[1]);
    vectors = malloc(capacity * sizeof(double*)); /* Allocate memory to store all vectors*/
    // Read from stdin line by line
    while (fgets(line, sizeof(line), stdin)) {
        if (line[0] == '\n' || line[0] == '\0') continue; /* skip empty lines */
        if (vectorCount==0) {
            *dim =1;
            for (p = line; *p; p++) if (*p == ',') (*dim)++;
        }
        // Allocate memory for each vector
        vector = malloc(*dim * sizeof(double));
        CHECK_ALLOC(vector);
        i = 0;
        token = strtok(line, ","); /* Split line by commas */
        while (token) { /* Convert each line to a Vector */
            vector[i++] = atof(token);
            token = strtok(NULL, ",");
        }
        *dim = i; 
        // Store the vector and manage memory
        if (vectorCount == capacity) { /* Check if we need to reallocate memory */
            capacity *= 2;
            vectors = realloc(vectors, capacity * sizeof(double*));
            CHECK_ALLOC(vectors);
        }
        vectors[vectorCount++] = vector;
    }
    // Validate number of clusters
    if (vectorCount <= *k) { 
        printf("Incorrect number of clusters!\n"); /*Error 1 - Stop Program*/
        exit(1);
    }
    // Resize vectors array to actual number of vectors
    vectors = realloc(vectors, vectorCount * sizeof(double*)); /* Resize to actual number }*/
    *numVectors = vectorCount;
    return vectors;
}
int main(int argc, char *argv[]) {
    int k, n, dim;
    char* goal;
    double **X = NULL, **W = NULL, **D = NULL, **N = NULL;
    goal = argv[2];

    // Read input data
    X = readData(argc, argv, &k, &n, &dim);
    // Compute similarity matrix
    W = sym(X, n);
    if (strcmp(goal, "sym") == 0) {
        // Print similarity matrix
        print_matrix(W, n, n);
        free_matrix(W, n);
    } else {
        // Compute degree diagonal matrix
        D = ddg(W, n);
        if (strcmp(goal, "ddg") == 0) {
            // Print degree diagonal matrix
            print_matrix(D, n, n);
            free_matrices((double**[]){W, D}, 2, n);
        } else if (strcmp(goal, "norm") == 0) {
            // Compute normalized graph Laplacian
            N = norm(W, D, n);
            // Print normalized Laplacian
            print_matrix(N, n, n);
            free_matrices((double**[]){W, D, N}, 3, n);

        } else {
            // Invalid goal argument
            printf("Invalid goal!\n");
            free_matrices((double**[]){X, W, D}, 3, n);
            return 1;
        }
    }
    // Free input data
    free_matrix(X, n);
    return 0;
}
