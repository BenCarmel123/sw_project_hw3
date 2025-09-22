#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

#define BETA 0.5
#define MAX_ITER 300
#define EPSILON 0.0001
#define DELIMITER ','
#define SYM "sym"
#define DDG "ddg"
#define NORM "norm"

const char *ERR_MSG = "An Error Has Occurred\n";

/* Error handling function */
void* error(void) {
    fprintf(stderr, "%s", ERR_MSG);
    exit(1);
    return NULL;
}

/* Function to print matrix according to the specified format */
void print_matrix(double **M, int n, int m) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            if (j == m - 1) {
                printf("%.4f", M[i][j]);   
            } else {
                printf("%.4f,", M[i][j]);  
            }
        }
        printf("\n"); 
    }
}

/* Function to create a 2D matrix */
double** create_matrix(int rows, int cols) {
    double** matrix = malloc(rows * sizeof(double*));
    int i;
    for (i = 0; i < rows; i++) {
        matrix[i] = malloc(cols * sizeof(double));
        CHECK_ALLOC(matrix[i]);
    }
    return matrix;
}

/* Compute L2 norm between two vectors */
double l2_norm(double* a, double* b, int dim) { 
    long double sum = 0.0;
    int i;
    for (i = 0; i < dim; i++) {
        long double diff = (long double)a[i] - (long double)b[i];
        sum += diff * diff;
    }
    return sqrt((double)sum);
}

/* Free allocated memory for a 2D matrix */
void free_matrix(double** matrix, int rows) { 
    int i;
    for (i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

/* Multiply two matrices 
A (n x m) and B (m x p) to produce C (n x p), 
assuming C is pre-allocated 
*/ 
void matrix_multiply(double** A, double** B, double** C, int n, int m, int p) {
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            C[i][j] = 0.0;
            for (k = 0; k < m; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

/* Subtract matrix B from A 
-> (A - B) -> to produce matrix C, 
assuming C is pre-allocated
*/
double** matrix_subtract(double** A, double** B, double** C, int n, int m) { 
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

/* Compute Frobenius norm of a matrix */
double forbenius_norm(double** A, int n, int m) { 
    long double norm = 0.0;
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            norm += (long double)A[i][j] * (long double)A[i][j];
        }
    }
    return sqrt((double)norm);
}
/* Compute similarity matrix using Gaussian kernel */
double** sym(double** X, int n, int dim) { 
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
    long double sum;
    for (i = 0; i < n; i++) {
        D[i] = calloc(n, sizeof(double)); 
        sum = 0.0;
        for (j = 0; j < n; j++) {
            sum += (long double)A[i][j];
        }
        D[i][i] = (double)sum;
    }
    return D;
}

/* Compute normalized graph Laplacian */
double** norm(double** A, double** D, int n) { 
    double** D_inv_sqrt;
    double** temp;
    double** W;
    double **matrices4[4];
    int i, j, k;
    D_inv_sqrt = malloc(n * sizeof(double*));
    temp = NULL;
    W = NULL;
    for (i = 0; i < n; i++) {
        D_inv_sqrt[i] = malloc(n * sizeof(double));
        for (j = 0; j < n; j++) 
           D_inv_sqrt[i][j] = (i == j && D[i][i] > 0.0) ? 1.0 / sqrt(D[i][i]) : 0.0;
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

    W = malloc(n * sizeof(double*));
    for (i = 0; i < n; i++) {
        W[i] = malloc(n * sizeof(double));
        for (j = 0; j < n; j++) {
            W[i][j] = 0.0;
            for (k = 0; k < n; k++)
                 W[i][j] += temp[i][k] * D_inv_sqrt[k][j];
        }
    }
    matrices4[2] = D_inv_sqrt;
    matrices4[3] = temp;
    free_matrix(D_inv_sqrt, n);
    free_matrix(temp, n);
    return W;
}


/* Perform one optimization step for H, return ||H - H_old||_F^2 */
double optimize(double*** optimization_matrices, double** W, int n, int k) {
    /* Unpack matrices */
    double **H, **H_old, **WH, **H_T, **HHT, **HHTH, **tmp;
    double norm_diff, beta;
    int i, j;

    H     = optimization_matrices[0];
    H_old = optimization_matrices[1];
    WH    = optimization_matrices[2];
    H_T   = optimization_matrices[3];
    HHT   = optimization_matrices[4];
    HHTH  = optimization_matrices[5];
    tmp   = optimization_matrices[6];

    /* Store previous H into H_old */
    for (i = 0; i < n; i++) 
        memcpy(H_old[i], H[i], k * sizeof(double));

    /* Compute WH = W * H_old */
    matrix_multiply(W, H_old, WH, n, n, k);

    /* H_T = H_old^T */
    for (j = 0; j < k; j++) {
        for (i = 0; i < n; i++) 
            H_T[j][i] = H_old[i][j];
    }

    /* HHT = H_T * H_old = k x k */
    matrix_multiply(H_T, H_old, HHT, k, n, k);

    /* HHTH = H_old * HHT = n x k */
    matrix_multiply(H_old, HHT, HHTH, n, k, k);

    /* Update H */
    beta = BETA;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            double numer = WH[i][j];
            double denom = HHTH[i][j];
            double frac = (denom != 0.0) ? numer / denom : 0.0;
            H[i][j] = H[i][j] * (1.0 - beta + beta * frac);
            if (H[i][j] < 1e-15) /* Avoid division by zero */
                H[i][j] = 1e-15;
        }
    }

    /* Compute norm_diff = ||H - H_old||_F^2 */
    matrix_subtract(H, H_old, tmp, n, k);
    norm_diff = pow(forbenius_norm(tmp, n, k), 2);

    return norm_diff;
}


/* Perform Symmetric Non-negative Matrix Factorization */
double** symnmf(double** H0, double** W, int n, int k) {

    double **H_old, **WH, **H_T, **HHT, **HHTH, **tmp;
    double*** optimization_matrices;
    int it;
    double norm_diff;
    /* Allocate matrices */
    optimization_matrices = malloc(7 * sizeof(double**));
    optimization_matrices[0] = H0;                
    H_old = create_matrix(n, k);
    optimization_matrices[1] = H_old;
    WH = create_matrix(n, k);
    optimization_matrices[2] = WH;
    H_T = create_matrix(k, n);
    optimization_matrices[3] = H_T;
    HHT = create_matrix(k, k);
    optimization_matrices[4] = HHT;
    HHTH = create_matrix(n, k);
    optimization_matrices[5] = HHTH;
    tmp = create_matrix(n, k);  
    optimization_matrices[6] = tmp;

    /* Iterative optimization */
    for (it = 0; it < MAX_ITER; it++) {
        norm_diff = optimize(optimization_matrices, W, n, k);
        if (norm_diff < EPSILON) break;
    }
    /* Free allocated memory (but return H0) */
    free_matrix(H_old, n);
    free_matrix(WH, n);
    free_matrix(H_T, k);
    free_matrix(HHT, k);
    free_matrix(HHTH, n);
    free_matrix(tmp, n);
    free(optimization_matrices);
    return H0;
}

/* Read info from Terminal */
double** readData(const char *path, int *outRows, int *outCols) {
    double** data;
    int i, j, rows, cols;
    char ch;
    double val;
    FILE *file = fopen(path, "r");
    if (!file) {
        return error();
    }
    rows = 0;
    cols = 0;
    while ((ch = fgetc(file)) != EOF) {
        if (ch == '\n') rows++;
        if (rows == 0 && ch == DELIMITER) cols++;
    }
    cols++; 
    fseek(file, 0, SEEK_SET);
    data = create_matrix(rows, cols);
    CHECK_ALLOC(data);
    i = 0;
    j = 0;
    while (fscanf(file, "%lf", &val) != EOF) {
        data[i][j++] = val;
        ch = fgetc(file);
        if (ch == DELIMITER) continue;
        if (ch == '\n') {
            i++;
            j = 0;
        }
    }
    fclose(file);
    *outRows = rows;
    *outCols = cols;
    return data;
}


int main(int argc, char *argv[]) {

    int n, dim;
    char* goal;
    double **X = NULL, **A = NULL, **D = NULL, **W = NULL;
    goal = argv[1];

    if ((strcmp(goal, SYM) != 0 && strcmp(goal, DDG) != 0 && strcmp(goal, NORM) != 0) 
    || argc != 3)   /* Only allow sym, ddg, norm and 3 arguments in CMD */
        error();

    X = readData(argv[2], &n, &dim);
    A = sym(X, n, dim);

    if (strcmp(goal, SYM) == 0) 
    {
        print_matrix(A, n, n);
        free_matrix(A, n);
    } 
    else {
        D = ddg(A, n);
        if (strcmp(goal, DDG) == 0) 
        {
            print_matrix(D, n, n);
            free_matrix(A, n);
            free_matrix(D, n);
        }
        else {
            W = norm(A, D, n);
            print_matrix(W, n, n);
            free_matrix(A, n);
            free_matrix(D, n);
            free_matrix(W, n);
        }
    }
        free_matrix(X, n);
        return 0;
}
