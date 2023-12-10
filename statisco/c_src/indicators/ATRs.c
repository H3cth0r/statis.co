#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>



// Applies SMA
PyObject *ATR(PyObject *self, PyObject *args) {
    PyObject *Close_t_obj, *High_t_obj, *Low_t_obj;
    npy_int32 window_t;

    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "OOOi", &Close_t_obj, &High_t_obj, &Low_t_obj, &window_t)) {
        PyErr_SetString(PyExc_TypeError, "Invalid input arguments");
        return NULL;
    }

    // Convert Python objects to NumPy arrays
    PyArrayObject *Close_t = (PyArrayObject *)PyArray_FROM_OTF(Close_t_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *High_t = (PyArrayObject *)PyArray_FROM_OTF(High_t_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *Low_t = (PyArrayObject *)PyArray_FROM_OTF(Low_t_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (Close_t == NULL || High_t == NULL || Low_t == NULL) {
        PyErr_SetString(PyExc_TypeError, "Invalid input arrays");
        Py_XDECREF(Close_t);
        Py_XDECREF(High_t);
        Py_XDECREF(Low_t);
        return NULL;
    }

    int len = PyArray_SIZE(Close_t);

    if (len != PyArray_SIZE(High_t) || len != PyArray_SIZE(Low_t)) {
        PyErr_SetString(PyExc_ValueError, "Close_t, High_t, and Low_t should have the same length");
        Py_XDECREF(Close_t);
        Py_XDECREF(High_t);
        Py_XDECREF(Low_t);
        return NULL;
    }

    double *Close_t_data = (double *)PyArray_DATA(Close_t);
    double *High_t_data = (double *)PyArray_DATA(High_t);
    double *Low_t_data = (double *)PyArray_DATA(Low_t);

    PyArrayObject *atr = (PyArrayObject *)PyArray_SimpleNew(1, PyArray_DIMS(Close_t), NPY_DOUBLE);
    double *atr_data = (double *)PyArray_DATA(atr);

    for (int i = 0; i < window_t - 1; i++) {
        atr_data[i] = 0.0;
    }

    for (int i = window_t - 1; i < len; i++) {
        double sum_true_range = 0.0;

        for (int j = i - window_t + 1; j <= i; j++) {
            double high_low = High_t_data[j] - Low_t_data[j];
            double high_close = fabs(High_t_data[j] - Close_t_data[j - 1]);
            double low_close = fabs(Low_t_data[j] - Close_t_data[j - 1]);
            double true_range = fmax(high_low, fmax(high_close, low_close));
            sum_true_range += true_range;
        }

        atr_data[i] = sum_true_range / window_t;
    }

    Py_XDECREF(Close_t);
    Py_XDECREF(High_t);
    Py_XDECREF(Low_t);
    return PyArray_Return(atr);
}


PyObject *ATRwma(PyObject *self, PyObject *args) {
    PyObject *Close_t_obj, *High_t_obj, *Low_t_obj;
    npy_int32 window_t;

    if (!PyArg_ParseTuple(args, "OOOi", &Close_t_obj, &High_t_obj, &Low_t_obj, &window_t)) {
        PyErr_SetString(PyExc_TypeError, "Invalid input arguments");
        return NULL;
    }

    PyArrayObject *Close_t  = (PyArrayObject *)PyArray_FROM_OTF(Close_t_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *High_t   = (PyArrayObject *)PyArray_FROM_OTF(High_t_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *Low_t    = (PyArrayObject *)PyArray_FROM_OTF(Low_t_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (Close_t == NULL || High_t == NULL || Low_t == NULL) {
        PyErr_SetString(PyExc_TypeError, "Invalid input arrays");
        Py_XDECREF(Close_t);
        Py_XDECREF(High_t);
        Py_XDECREF(Low_t);
        return NULL;
    }

    int len = PyArray_SIZE(Close_t);

    if (len != PyArray_SIZE(High_t) || len != PyArray_SIZE(Low_t)) {
        PyErr_SetString(PyExc_ValueError, "Close_t, High_t, and Low_t should have the same length");
        Py_XDECREF(Close_t);
        Py_XDECREF(High_t);
        Py_XDECREF(Low_t);
        return NULL;
    }

    double *Close_t_data = (double *)PyArray_DATA(Close_t);
    double *High_t_data = (double *)PyArray_DATA(High_t);
    double *Low_t_data = (double *)PyArray_DATA(Low_t);

    PyArrayObject *atr = (PyArrayObject *)PyArray_SimpleNew(1, PyArray_DIMS(Close_t), NPY_DOUBLE);
    double *atr_data = (double *)PyArray_DATA(atr);

    for (int i = 0; i < window_t - 1; i++) {
        atr_data[i] = 0.0;
    }

    for (int i = window_t - 1; i < len; i++) {
        double sum_true_range = 0.0;
        double weight_sum = 0.0;

        for (int j = i - window_t + 1, k = 0; j <= i; j++, k++) {
            double high_low = High_t_data[j] - Low_t_data[j];
            double high_close = fabs(High_t_data[j] - Close_t_data[j - 1]);
            double low_close = fabs(Low_t_data[j] - Close_t_data[j - 1]);
            double true_range = fmax(high_low, fmax(high_close, low_close));

            // Calculate WMA weights
            double weight = (k + 1.0) / ((window_t * (window_t + 1)) / 2.0);

            sum_true_range += true_range * weight;
            weight_sum += weight;
        }

        atr_data[i] = sum_true_range / weight_sum;
    }

    Py_XDECREF(Close_t);
    Py_XDECREF(High_t);
    Py_XDECREF(Low_t);
    return PyArray_Return(atr);
}



PyObject *ATRema(PyObject *self, PyObject *args) {
    PyObject *Close_t_obj, *High_t_obj, *Low_t_obj;
    npy_int32 window_t;

    if (!PyArg_ParseTuple(args, "OOOi", &Close_t_obj, &High_t_obj, &Low_t_obj, &window_t)) {
        PyErr_SetString(PyExc_TypeError, "Invalid input arguments");
        return NULL;
    }

    PyArrayObject *Close_t = (PyArrayObject *)PyArray_FROM_OTF(Close_t_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *High_t = (PyArrayObject *)PyArray_FROM_OTF(High_t_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *Low_t = (PyArrayObject *)PyArray_FROM_OTF(Low_t_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (Close_t == NULL || High_t == NULL || Low_t == NULL) {
        PyErr_SetString(PyExc_TypeError, "Invalid input arrays");
        Py_XDECREF(Close_t);
        Py_XDECREF(High_t);
        Py_XDECREF(Low_t);
        return NULL;
    }

    int len = PyArray_SIZE(Close_t);

    if (len != PyArray_SIZE(High_t) || len != PyArray_SIZE(Low_t)) {
        PyErr_SetString(PyExc_ValueError, "Close_t, High_t, and Low_t should have the same length");
        Py_XDECREF(Close_t);
        Py_XDECREF(High_t);
        Py_XDECREF(Low_t);
        return NULL;
    }

    double *Close_t_data = (double *)PyArray_DATA(Close_t);
    double *High_t_data = (double *)PyArray_DATA(High_t);
    double *Low_t_data = (double *)PyArray_DATA(Low_t);

    PyArrayObject *atr = (PyArrayObject *)PyArray_SimpleNew(1, PyArray_DIMS(Close_t), NPY_DOUBLE);
    double *atr_data = (double *)PyArray_DATA(atr);

    double smoothing_factor = 2.0 / (window_t + 1.0);
    double ema = 0.0;

    for (int i = 0; i < len; i++) {
        double high_low = High_t_data[i] - Low_t_data[i];
        double high_close = fabs(High_t_data[i] - Close_t_data[i - 1]);
        double low_close = fabs(Low_t_data[i] - Close_t_data[i - 1]);
        double true_range = fmax(high_low, fmax(high_close, low_close));

        if (i == 0) {
            ema = true_range;
        } else {
            ema = (true_range - ema) * smoothing_factor + ema;
        }

        atr_data[i] = ema;
    }

    Py_XDECREF(Close_t);
    Py_XDECREF(High_t);
    Py_XDECREF(Low_t);
    return PyArray_Return(atr);
}




PyMethodDef methods[] = {
  {"ATR",        (PyCFunction)ATR,          METH_VARARGS, "Computes the Average True Range."},
  {"ATRwma",     (PyCFunction)ATRwma,       METH_VARARGS, "Computes the Average True Range with Weighted Moving Average."},
  {"ATRema",     (PyCFunction)ATRema,       METH_VARARGS, "Computes the Average True Range with Exponential Moving Average."},
  {NULL, NULL, 0, NULL}
};

PyModuleDef ATRs = {
  PyModuleDef_HEAD_INIT,
  "ATRs",
  "This is the Average True Ranges module.",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_ATRs() {
  import_array(); // init numpy
  PyObject *module = PyModule_Create(&ATRs);
  printf("Imported ATRs module\n");
  return module;
}
