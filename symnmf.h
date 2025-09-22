#ifndef SYMNMF_H
#define SYMNMF_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK_ALLOC(ptr) \
    if (!(ptr)) { \
        fprintf(stderr, "Memory allocation failed.\n"); \
        exit(1); \
    }

double l2_norm(double* a, double* b, int dim);
void free_matrix(double** matrix, int rows);
double** create_matrix(int rows, int cols);
void print_matrix(double **M, int n, int m);

void matrix_multiply(double** A, double** B, double** C, int n, int m, int p);
double** matrix_subtract(double** A, double** B, double** C, int n, int m);
double forbenius_norm(double** A, int n, int m);
double** sym(double** X, int n, int dim);
double** ddg(double** A, int n);
double** norm(double** A, double** D, int n);
double** initialize_H(double **W, int n, int k);

double** symnmf(double** H0, double** W, int n, int k);
double** readData(const char *path, int *outRows, int *outCols);

#endif /* SYMNMF_H */
