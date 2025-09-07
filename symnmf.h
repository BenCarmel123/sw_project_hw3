#ifndef SYMNMF_H
#define SYMNMF_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK_ALLOC(ptr) \
    if (!(ptr)) { \
        printf("Memory allocation failed.\n"); \
        exit(1); \
    }

double l2_norm(double* a, double* b, int dim);
void free_matrix(double** matrix, int rows);
void free_matrices(double*** matrices, int count, int rows);
double** matrix_multiply(double** A, double** B, int n, int m, int p);
double** matrix_subtract(double** A, double** B, int n, int m);
double forbenius_norm(double** A, int n);
double** sym(double** X, int n, int dim);
double** ddg(double** A, int n);
double** norm(double** A, double** D, int n);
double** optimize(double** W, int n, int k);
double** symnmf(double** X, int n, int k, int dim);
double** readData(int argc, char* argv[], int* k, int* numVectors, int* dim);

#endif // SYMNMF_H
