// python c api wrapper for symnmf
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

// ---- Utility functions to handle errors and convert between Python lists and C matrices ---- //

static PyObject* py_error(void) {
    PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
    return NULL;
}

static int py_to_c_matrix(PyObject *py_matrix, double ***c_matrix_ptr, int *rows_ptr, int *cols_ptr) {
    Py_ssize_t i, j, rows, cols;
    double **c_matrix;
    PyObject *row_obj, *cell_obj;
    if (!PyList_Check(py_matrix) || (rows = PyList_Size(py_matrix)) == 0) return 0;
    row_obj = PyList_GetItem(py_matrix, 0);
    if (!PyList_Check(row_obj) || (cols = PyList_Size(row_obj)) == 0) return 0;
    c_matrix = (double**)malloc(rows * sizeof(double*));
    if (!c_matrix) return 0;
    for (i = 0; i < rows; i++) {
        row_obj = PyList_GetItem(py_matrix, i);
        if (!PyList_Check(row_obj) || PyList_Size(row_obj) != cols) {
            free_matrix(c_matrix, i); return 0;
        }
        c_matrix[i] = (double*)malloc(cols * sizeof(double));
        if (!c_matrix[i]) { free_matrix(c_matrix, i); return 0; }
        for (j = 0; j < cols; j++) {
            cell_obj = PyList_GetItem(row_obj, j);
            c_matrix[i][j] = PyFloat_AsDouble(cell_obj);
        }
    }
    *c_matrix_ptr = c_matrix;
    *rows_ptr = rows;
    *cols_ptr = cols;
    return 1;
}

static PyObject* c_to_py_matrix(double **c_matrix, int rows, int cols) {
    Py_ssize_t i, j;
    PyObject *py_list = PyList_New(rows);
    if (!py_list) return NULL;
    for (i = 0; i < rows; i++) {
        PyObject *py_row = PyList_New(cols);
        if (!py_row) { Py_DECREF(py_list); return NULL; }
        for (j = 0; j < cols; j++) {
            PyList_SET_ITEM(py_row, j, PyFloat_FromDouble(c_matrix[i][j]));
        }
        PyList_SET_ITEM(py_list, i, py_row);
    }
    return py_list;
}

// ---- Python wrapper functions ---- //
static PyObject* py_norm(PyObject *self, PyObject *args) {
    PyObject *py_A, *py_D;
    double **A, **D, **result;
    int rows_A, cols_A, rows_D, cols_D;
    
    if (!PyArg_ParseTuple(args, "OO", &py_A, &py_D)) {
        return py_error();
    }
    
    if (!py_to_c_matrix(py_A, &A, &rows_A, &cols_A)) return py_error();
    if (!py_to_c_matrix(py_D, &D, &rows_D, &cols_D)) {
        free_matrix(A, rows_A);
        return py_error();
    }
    
    result = norm(A, D, rows_A);  // Call external C function
    
    PyObject* py_result = c_to_py_matrix(result, rows_A, rows_A);
    
    free_matrix(A, rows_A);
    free_matrix(D, rows_D);
    free_matrix(result, rows_A);
    
    return py_result;
}

static PyObject* py_ddg(PyObject *self, PyObject *args) {
    PyObject *py_A;
    double **A, **result;
    int rows_A, cols_A;
    
    if (!PyArg_ParseTuple(args, "O", &py_A)) {
        return py_error();
    }
    
    if (!py_to_c_matrix(py_A, &A, &rows_A, &cols_A)) return py_error();
    
    result = ddg(A, rows_A);  // Call external C function
    
    PyObject* py_result = c_to_py_matrix(result, rows_A, rows_A);
    
    free_matrix(A, rows_A);
    free_matrix(result, rows_A);
    
    return py_result;
}

static PyObject* py_sym(PyObject *self, PyObject *args) {
    PyObject *py_X;
    int dim;
    double **X, **result;
    int rows_X, cols_X;
    
    if (!PyArg_ParseTuple(args, "O", &py_X)) {
        return py_error();
    }
    
    if (!py_to_c_matrix(py_X, &X, &rows_X, &cols_X)) return py_error();
    
    dim = cols_X;
    result = sym(X, rows_X, dim);  
    
    PyObject* py_result = c_to_py_matrix(result, rows_X, rows_X);
    
    free_matrix(X, rows_X);
    free_matrix(result, rows_X);
    
    return py_result;
}

static PyObject* py_symnmf(PyObject *self, PyObject *args) {
    PyObject *py_H0 = NULL, *py_W = NULL, *py_res = NULL;
    double **W = NULL, **H0 = NULL, **H = NULL;
    int n = 0, k = 0, nW_r = 0, nW_c = 0, nH_r = 0, nH_c = 0;

    if (!PyArg_ParseTuple(args, "OO", &py_H0, &py_W)) {
        fprintf(stderr, "your message 1\n");
        fflush(stderr);
        return py_error();
    }

    if (!py_to_c_matrix(py_H0, &H0, &nH_r, &nH_c))  {
        fprintf(stderr, "your message 2\n");
        fflush(stderr);
        return py_error();
    }
    if (!py_to_c_matrix(py_W, &W, &nW_r, &nW_c) || nW_r != nW_c) {
        free_matrix(H0, nH_r);
        if (W) free_matrix(W, nW_r);
        fprintf(stderr, "your message 3\n");
        fflush(stderr);
        return py_error();
    }
    n = nH_r;
    k = nH_c;
    if (n != nW_r) {
        free_matrix(H0, nH_r);
        free_matrix(W, nW_r);
        fprintf(stderr, "your message 4\n");
        fflush(stderr);
        return py_error();
    }

    H = symnmf(H0, W, n, k); // in-place; H == H0
    free_matrix(W, nW_r);
    if (!H) { 
        free_matrix(H0, n); 
        fprintf(stderr, "your message 5\n");
        fflush(stderr);
        return py_error();
     }

    py_res = c_to_py_matrix(H, n, k); // copy while H is alive
    free_matrix(H, n);                // free H/H0 after copying
    if (!py_res) {
            fprintf(stderr, "your message 6\n");
            fflush(stderr);
            return py_error();
    }
    return py_res;
}

static PyMethodDef SymNMFMethods[] = {
    {"norm", py_norm, METH_VARARGS, "Compute normalized graph Laplacian."},
    {"ddg", py_ddg, METH_VARARGS, "Compute degree diagonal matrix."},
    {"sym", py_sym, METH_VARARGS, "Compute similarity matrix."},
    {"symnmf", py_symnmf, METH_VARARGS, "Perform Symmetric Non-negative Matrix Factorization."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",   // name of module
    NULL, // module documentation, may be NULL
    -1,       // size of per-interpreter state of the module,
              // or -1 if the module keeps state in global variables.
    SymNMFMethods
};

PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}






