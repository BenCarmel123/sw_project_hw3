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
    
    PyObject* py_result = c_to_py_matrix(result, rows_A, cols_A);
    
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
    
    PyObject* py_result = c_to_py_matrix(result, rows_A, cols_A);
    
    free_matrix(A, rows_A);
    free_matrix(result, rows_A);
    
    return py_result;
}

static PyObject* py_sym(PyObject *self, PyObject *args) {
    PyObject *py_X;
    int dim;
    double **X, **result;
    int rows_X, cols_X;
    
    if (!PyArg_ParseTuple(args, "Oi", &py_X, &dim)) {
        return py_error();
    }
    
    if (!py_to_c_matrix(py_X, &X, &rows_X, &cols_X)) return py_error();
    
    result = sym(X, rows_X, dim);  // Call external C function with dim
    
    PyObject* py_result = c_to_py_matrix(result, rows_X, rows_X);
    
    free_matrix(X, rows_X);
    free_matrix(result, rows_X);
    
    return py_result;
}

static PyObject* py_symnmf(PyObject *self, PyObject *args) {
    PyObject *py_W;
    int k;
    double **W, **result;
    int rows_W, cols_W;
    
    if (!PyArg_ParseTuple(args, "Oi", &py_W, &k)) {
        return py_error();
    }
    
    if (!py_to_c_matrix(py_W, &W, &rows_W, &cols_W)) return py_error();
    
    result = symnmf(W, rows_W, k);  // Call C function with correct signature
    
    PyObject* py_result = c_to_py_matrix(result, rows_W, k);
    
    free_matrix(W, rows_W);
    free_matrix(result, rows_W);
    
    return py_result;
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






