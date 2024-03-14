#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
// #include <numpy/arrayscalars.h>
#include <stdlib.h>

static PyObject *TrainTestSplit(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *X_obj, *y_obj;
    PyArrayObject *X_array, *y_array;

    double test_size = 0.25;
    long random_state = 42;
    int shuffle = 1;

    static char *kwlist[] = {"X", "y", "test_size", "random_state", "shuffle", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|dli", kwlist, &X_obj, &y_obj, &test_size, &random_state, &shuffle)) {
        return NULL;
    }

    X_array = (PyArrayObject *)PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    y_array = (PyArrayObject *)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (X_array == NULL || y_array == NULL) {
        Py_XDECREF(X_array);
        Py_XDECREF(y_array);
        return NULL;
    }

    int n_samples = PyArray_DIM(X_array, 0);
    int n_features = PyArray_DIM(X_array, 1);
    int n_train, n_test;

    n_train = (int)((1.0 - test_size) * n_samples);
    n_test = n_samples - n_train;

    int* indices = (int*)malloc(n_samples * sizeof(int));
    for (int i = 0; i < n_samples; i++) {
        indices[i] = i;
    }

    if (shuffle) {
        srand(random_state);
        for (int i = n_samples - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
    }

    npy_intp train_dims[2] = {n_train, n_features};
    npy_intp test_dims[2] = {n_test, n_features};
    npy_intp train_target_dims[1] = {n_train};
    npy_intp test_target_dims[1] = {n_test};

    PyArrayObject* X_train = (PyArrayObject*)PyArray_SimpleNew(2, train_dims, NPY_DOUBLE);
    PyArrayObject* X_test = (PyArrayObject*)PyArray_SimpleNew(2, test_dims, NPY_DOUBLE);
    PyArrayObject* y_train = (PyArrayObject*)PyArray_SimpleNew(1, train_target_dims, NPY_DOUBLE);
    PyArrayObject* y_test = (PyArrayObject*)PyArray_SimpleNew(1, test_target_dims, NPY_DOUBLE);


    for (int i = 0; i < n_train; i++) {
        for (int j = 0; j < n_features; j++) {
            *(double*)PyArray_GETPTR2(X_train, i, j) = *(double*)PyArray_GETPTR2(X_array, indices[i], j);
        }
        *(double*)PyArray_GETPTR1(y_train, i) = *(double*)PyArray_GETPTR1(y_array, indices[i]);
    }

    for (int i = 0; i < n_test; i++) {
        for (int j = 0; j < n_features; j++) {
            *(double*)PyArray_GETPTR2(X_test, i, j) = *(double*)PyArray_GETPTR2(X_array, indices[n_train + i], j);
        }
        *(double*)PyArray_GETPTR1(y_test, i) = *(double*)PyArray_GETPTR1(y_array, indices[n_train + i]);
    }

    free(indices);

    PyObject* result = PyTuple_Pack(4, X_train, X_test, y_train, y_test);

    Py_XDECREF(X_array);
    Py_XDECREF(y_array);

    return result;
}
