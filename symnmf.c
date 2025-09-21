#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


/* Error handling function */
void* error(void) {
    fprintf(stderr, "An Error Has Occurred\n");
    exit(1);
    return NULL;
}

/* Global variable for memory allocation check */
#define CHECK_ALLOC(ptr) \
    if (!(ptr)) { \
        printf("Memory allocation failed.\n"); \
        return error(); \
    }

/* Function to print matrix according to the specified format */
void print_matrix(double** matrix, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%.4f", matrix[i][j]);
            if (j < cols - 1) printf(",");
        }
        printf("\n");
    }
}

/* Compute L2 norm between two vectors */
double l2_norm(double* a, double* b, int dim) { 
    double sum = 0.0;
    int i;
    for (i = 0; i < dim; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

/* Free allocated memory for a 2D matrix */
void free_matrix(double** matrix, int rows) { 
    int i;
    for (i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

/* Multiply two matrices A (n x m) and B (m x p) */
double** matrix_multiply(double** A, double** B, int n, int m, int p) { 
    double** C = malloc(n * sizeof(double*));
    int i, j, k;
    for (i = 0; i < n; i++) {
        C[i] = malloc(p * sizeof(double));
        for (j = 0; j < p; j++) {
            C[i][j] = 0.0;
            for (k = 0; k < m; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    free_matrix(A, n);
    free_matrix(B, m);
    return C;
}

/* Subtract matrix B from A (A - B) */
double** matrix_subtract(double** A, double** B, int n, int m) {
    double** C = malloc(n * sizeof(double*));
    int i, j;
    for (i = 0; i < n; i++) {
        C[i] = malloc(m * sizeof(double));
        for (j = 0; j < m; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    free_matrix(A, n);
    free_matrix(B, n);
    return C;
}

/* Compute Frobenius norm of a matrix */
double forbenius_norm(double** A, int n) { 
    double norm = 0.0;
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            norm += A[i][j] * A[i][j];
        }
    }
    return sqrt(norm);
}
/* Compute similarity matrix using Gaussian kernel */
double** sym (double** X, int n, int dim) { 
    double** A = malloc(n * sizeof(double*));
    int i, j;
    for (i = 0; i < n; i++) {
        A[i] = malloc(n * sizeof(double));
        for (j = 0; j < n; j++) {
            if (i == j) {
                A[i][j] = 0.0;
            } else {
                double dist = l2_norm(X[i], X[j], dim);
                A[i][j] = exp(-0.5 * dist * dist);
            }
        }
    }
    return A;
}


/* Compute degree diagonal matrix */
double** ddg(double** A, int n) { 
    double** D = malloc(n * sizeof(double*));
    int i, j;
    double sum;
    for (i = 0; i < n; i++) {
        D[i] = calloc(n, sizeof(double)); 
        sum = 0.0;
        for (j = 0; j < n; j++) {
            sum += A[i][j];
        }
        D[i][i] = sum;
    }
    return D;
}

/* Compute normalized graph Laplacian */
double** norm(double** A, double** D, int n) { 
    double** D_inv_sqrt;
    double** temp;
    double** N;
    double **matrices4[4];
    int i, j, k;
    D_inv_sqrt = malloc(n * sizeof(double*));
    temp = NULL;
    N = NULL;
    for (i = 0; i < n; i++) {
        D_inv_sqrt[i] = malloc(n * sizeof(double));
        for (j = 0; j < n; j++) 
            D_inv_sqrt[i][j] = (i == j) ? 1.0 / sqrt(D[i][j]) : 0.0;
    }

    temp = malloc(n * sizeof(double*));
    for (i = 0; i < n; i++) {
        temp[i] = malloc(n * sizeof(double));
        for (j = 0; j < n; j++) {
            temp[i][j] = 0.0;
            for (k = 0; k < n; k++) 
                temp[i][j] += D_inv_sqrt[i][k] * A[k][j];
        }
    }

    N = malloc(n * sizeof(double*));
    for (i = 0; i < n; i++) {
        N[i] = malloc(n * sizeof(double));
        for (j = 0; j < n; j++) {
            N[i][j] = 0.0;
            for (k = 0; k < n; k++)
                 N[i][j] += temp[i][k] * D_inv_sqrt[k][j];
        }
    }

    /* Free intermediate matrices */
    matrices4[0] = D_inv_sqrt;
    matrices4[1] = temp;
    matrices4[2] = A;
    matrices4[3] = D;
    free_matrix(D_inv_sqrt, n);
    free_matrix(temp, n);
    free_matrix(A, n);
    free_matrix(D, n);
    return N;
}

double** optimize(double** W, int n, int k) {
    double** H;
    double m = 0.0;
    int i, j, it;
    double **H_old;
    double **WH;
    double **H_T;
    double **HHT;
    double **HHTH;
    double **HHT2;
    double **matrices6[6];
    double norm_diff;
    double** tmp;
    H = malloc(n * sizeof(double*));
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) m += W[i][j];
    m /= (n * n);
    for (i = 0; i < n; i++) {
        H[i] = malloc(k * sizeof(double));
        for (j = 0; j < k; j++) 
            H[i][j] = ((double)rand() / RAND_MAX) * 2 * sqrt(m / k); 
    }
    /* Iterative optimization */
    for (it = 0; it < 300; it++) {
        H_old = malloc(n * sizeof(double*)); 
        for (i = 0; i < n; i++) {
            H_old[i] = malloc(k * sizeof(double));
            memcpy(H_old[i], H[i], k * sizeof(double)); /* Copy H to H_old */
        }
        /* Matrix operations and updates */
        WH = matrix_multiply(W, H_old, n, n, k); /* WH = W * H_old */
        H_T = malloc(k * sizeof(double*));
        for (i = 0; i < k; i++) {
            H_T[i] = malloc(n * sizeof(double));
            for (j = 0; j < n; j++) H_T[i][j] = H_old[j][i];
        }
        HHT = matrix_multiply(H_T, H_old, k, n, k); /* HHT = H_old^T * H_old */
        HHTH = matrix_multiply(H_old, HHT, n, k, k); /* HHTH = H * HHT */
        for (i = 0; i < n; i++) 
            for (j = 0; j < k; j++) 
                H[i][j] = H[i][j] * (0.5 + 0.5 * (WH[i][j] / (HHTH[i][j] + 1e-10)));
        /* Convergence check */
        HHT2 = matrix_multiply(H, H_T, n, k, n); /* HHT = H * H^T */
        tmp = matrix_subtract(W, HHT2, n, n);
        norm_diff = pow(forbenius_norm(tmp, n), 2); /* ||W - HHT|| */
        free_matrix(tmp, n);
        matrices6[0] = H_old;
        matrices6[1] = WH;
        matrices6[2] = H_T;
        matrices6[3] = HHT;
        matrices6[4] = HHTH;
        matrices6[5] = HHT2;
        free_matrix(H_old, n);
        free_matrix(WH, n);
        free_matrix(H_T, k);
        free_matrix(HHT, k);
        free_matrix(HHTH, n);
        free_matrix(HHT2, n);
        if (norm_diff < 1e-4) break;
    }
    return H;
}

double** symnmf(double** X, int n, int k) {
    double** W = sym(X, n, k); /* Compute similarity matrix */
    double** D = ddg(W, n); /* Compute degree diagonal matrix */
    double** N = norm(W, D, n); /* Compute normalized graph Laplacian */
    double** H = optimize(N, n, k); /* Perform SymNMF optimization */
    return H;
}

/* Read info from Terminal */
double** readData(int argc, char* argv[], int* k, int* numVectors, int* dim) {
    double **vectors;
    double *vector;
    int vectorCount;
    int capacity;
    int i;
    char line[1024];
    char *token;
    char *p;
    FILE *fp;
    (void)argc;
    vectorCount = 0;
    capacity = 100;
    fp = fopen(argv[2], "r"); /* Open file for reading */
    if (!fp) {
        return error();
    }
    vectors = malloc(capacity * sizeof(double*)); /* Allocate memory to store all vectors*/
    /* Read from file line by line */
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '\n' || line[0] == '\0') continue; /* skip empty lines */
        if (vectorCount==0) {
            *dim =1;
            for (p = line; *p; p++) if (*p == ',') (*dim)++;
        }
        /* Allocate memory for each vector */
        vector = malloc(*dim * sizeof(double));
        CHECK_ALLOC(vector);
        i = 0;
        token = strtok(line, ","); /* Split line by commas */
        while (token) { /* Convert each line to a Vector */
            vector[i++] = atof(token);
            token = strtok(NULL, ",");
        }
        *dim = i; 
        /* Store the vector and manage memory */
        if (vectorCount == capacity) { /* Check if we need to reallocate memory */
            capacity *= 2;
            vectors = realloc(vectors, capacity * sizeof(double*));
            CHECK_ALLOC(vectors);
        }
        vectors[vectorCount++] = vector;
    }
    fclose(fp);
    /* Validate number of clusters */
    if (vectorCount <= *k) { 
        printf("Incorrect number of clusters!\n"); /*Error 1 - Stop Program*/
        return error();
    }
    /* Resize vectors array to actual number of vectors */
    vectors = realloc(vectors, vectorCount * sizeof(double*)); /* Resize to actual number }*/
    *numVectors = vectorCount;
    return vectors;
}
int main(int argc, char *argv[]) {
    int k, n, dim;
    char* goal;
    double **X = NULL, **W = NULL, **D = NULL, **N = NULL, **H = NULL;
    double **matrices3[3];
    double **matrices2[2];
    if (argc != 3) {
        printf("An Error Has Occurred\n");
        return 1;
    }
    goal = argv[1];

    /* Read input data */
    X = readData(argc, argv, &k, &n, &dim);
    /* Compute similarity matrix */
    W = sym(X, n, dim);
    if (strcmp(goal, "sym") == 0) {
        /* Print similarity matrix */
        print_matrix(W, n, n);
        free_matrix(W, n);
    } else {
        /* Compute degree diagonal matrix */
        if (strcmp(goal, "ddg") == 0) {
            D = ddg(W, n);
            /* Print degree diagonal matrix */
            print_matrix(D, n, n);
            matrices2[0] = W;
            matrices2[1] = D;
            free_matrix(W, n);
            free_matrix(D, n);
        } else if (strcmp(goal, "norm") == 0) {
            D = ddg(W, n);
            /* Compute normalized graph Laplacian */
            N = norm(W, D, n);
            /* Print normalized Laplacian */
            print_matrix(N, n, n);
            matrices3[0] = W;
            matrices3[1] = D;
            matrices3[2] = N;
            free_matrix(X, n);
            free_matrix(W, n);
            free_matrix(D, n);
        } else if (strcmp(goal, "symnmf") == 0) {
            D = ddg(W, n);
            /* Perform Symmetric Non-negative Matrix Factorization */
            N = norm(W, D, n);
            H = optimize(N, n, k);
            /* Print factorized matrix H */
            print_matrix(H, n, k);
            free_matrix(H, n);
        } else {
            printf("Invalid goal!\n");
            matrices3[0] = X;
            matrices3[1] = W;
            matrices3[2] = D;
            free_matrix(X, n);
            free_matrix(W, n);
            free_matrix(D, n);
            return 1;
        }
    }
    /* Free input data */
    free_matrix(X, n);
    return 0;
}
