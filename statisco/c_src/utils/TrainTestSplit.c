#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>

static PyObject *TrainTestSplit(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyArrayObject *X_in, *y_in;
    PyArrayObject *X_train, *X_test, *y_train, *y_test;
    double test_size = 0.33;
    int random_state = 42;

    static char *kwlist[] = {"X", "y", "test_size", "random_state", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!|di", kwlist,
                                     &PyArray_Type, &X_in,
                                     &PyArray_Type, &y_in,
                                     &test_size, &random_state)) {
        return NULL;
    }

    if (test_size <= 0.0 || test_size >= 1.0) {
        PyErr_SetString(PyExc_ValueError, "test_size must be between 0.0 and 1.0 exclusive");
        return NULL;
    }

    if (PyArray_NDIM(X_in) != 2) {
        PyErr_SetString(PyExc_ValueError, "X must be a 2-dimensional array");
        return NULL;
    }

    if (PyArray_NDIM(y_in) != 1) {
        PyErr_SetString(PyExc_ValueError, "y must be a 1-dimensional array");
        return NULL;
    }

    npy_intp *dims_X = PyArray_DIMS(X_in);
    npy_intp *dims_y = PyArray_DIMS(y_in);
    npy_intp num_samples = dims_X[0];
    npy_intp num_features = dims_X[1];
    npy_intp num_labels = dims_y[0];

    if (num_samples != num_labels) {
        PyErr_SetString(PyExc_ValueError, "Number of samples in X and y must be equal");
        return NULL;
    }

    npy_intp num_test = (npy_intp)(test_size * num_samples);
    npy_intp num_train = num_samples - num_test;

    // Allocate memory for X_train, X_test, y_train, y_test
    npy_intp dims_train[2] = {num_train, num_features};
    npy_intp dims_test[2] = {num_test, num_features};
    npy_intp dims_y_train[1] = {num_train};
    npy_intp dims_y_test[1] = {num_test};

    X_train = (PyArrayObject*)PyArray_SimpleNew(2, dims_train, PyArray_TYPE(X_in));
    X_test = (PyArrayObject*)PyArray_SimpleNew(2, dims_test, PyArray_TYPE(X_in));
    y_train = (PyArrayObject*)PyArray_SimpleNew(1, dims_y_train, PyArray_TYPE(y_in));
    y_test = (PyArrayObject*)PyArray_SimpleNew(1, dims_y_test, PyArray_TYPE(y_in));

    // Initialize random number generator
    srand(random_state);

    // Allocate memory for shuffled indices
    size_t *indices = malloc(num_samples * sizeof(size_t));
    if (indices == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    // Shuffle indices
    for (size_t i = 0; i < num_samples; ++i) {
        indices[i] = i;
    }
    for (size_t i = 0; i < num_samples; ++i) {
        size_t j = rand() % (i + 1);
        size_t temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    // Copy data
    for (npy_intp i = 0; i < num_test; ++i) {
        size_t index = indices[i];
        memcpy(PyArray_GETPTR2(X_test, i, 0), PyArray_GETPTR2(X_in, index, 0), num_features * sizeof(double));
        *((double*)PyArray_GETPTR1(y_test, i)) = *((double*)PyArray_GETPTR1(y_in, index));
    }

    for (npy_intp i = num_test; i < num_samples; ++i) {
        size_t index = indices[i];
        memcpy(PyArray_GETPTR2(X_train, i - num_test, 0), PyArray_GETPTR2(X_in, index, 0), num_features * sizeof(double));
        *((double*)PyArray_GETPTR1(y_train, i - num_test)) = *((double*)PyArray_GETPTR1(y_in, index));
    }

    // Free memory for shuffled indices
    free(indices);

    return Py_BuildValue("(OOOO)", X_train, X_test, y_train, y_test);
}
