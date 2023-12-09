#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>


PyObject *SMA(PyObject *self, PyObject *args) {
    PyObject *input_array;
    npy_intp window_t;

    if (!PyArg_ParseTuple(args, "Oi", &input_array, &window_t) || PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Invalid argument. Expected a numpy array and an int.");
        return NULL;
    }
    if (window_t <= 0) {
        PyErr_SetString(PyExc_ValueError, "Window size must be a positive integer.");
        return NULL;
    }
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(input_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr == NULL) {
        return NULL;
    }

    double *data = (double *)PyArray_DATA(arr);
    npy_intp size = PyArray_SIZE(arr);

    PyObject *result = PyArray_NewLikeArray(arr, NPY_CORDER, NULL, 0);
    if (result == NULL) {
        Py_XDECREF(arr);
        return NULL;
    }
    double *result_data = (double *)PyArray_DATA((PyArrayObject *)result);

    // Modify the input array directly
    if (size > 1000) {
        #pragma omp parallel for reduction(+:data[i])
        for (int i = window_t - 1; i < size; i++) {
            double sum = 0;
            int valid_count = 0;
            for (int j = i - window_t + 1; j <= i; j++) {
                if (!isnan(data[j])) {
                    sum += data[j];
                    valid_count++;
                }
            }
            result_data[i] = valid_count > 0 ? sum / valid_count : NAN;
        }
    } else {
        for (int i = window_t - 1; i < size; i++) {
            double sum = 0;
            int valid_count = 0;
            for (int j = i - window_t + 1; j <= i; j++) {
                if (!isnan(data[j])) {
                    sum += data[j];
                    valid_count++;
                }
            }
            result_data[i] = valid_count > 0 ? sum / valid_count : NAN;
        }
    }

    if (PyErr_Occurred()) {
        Py_XDECREF(arr);
        Py_XDECREF(result);
        return NULL;
    }
    Py_XDECREF(arr);

    return result;
}


PyObject *EMA(PyObject *self, PyObject *args) {
    PyObject *input_array;
    PyObject *SMA_t;
    npy_int32 window_t;
    npy_float64 smooth;

    if (!PyArg_ParseTuple(args, "OOid", &input_array,  &SMA_t, &window_t, &smooth) || PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Invalid argument. Expected a numpy array and an int.");
        return NULL;
    }

    if (window_t < 1) {
        PyErr_SetString(PyExc_ValueError, "Window size must be a positive integer.");
        return NULL;
    }

    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(input_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_sma = (PyArrayObject *)PyArray_FROM_OTF(SMA_t, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (arr == NULL || arr_sma == NULL) {
        Py_XDECREF(arr);
        Py_XDECREF(arr_sma);
        return NULL;
    }

    npy_intp size = PyArray_SIZE(arr);

    if (size < window_t) {
        PyErr_SetString(PyExc_ValueError, "Array size is smaller than the specified window size.");
        Py_DECREF(arr);
        Py_DECREF(arr_sma);
        return NULL;
    }

    PyObject *result = PyArray_NewLikeArray(arr, NPY_CORDER, NULL, 0);

    if (result == NULL) {
        Py_DECREF(arr);
        Py_DECREF(arr_sma);
        return NULL;
    }

    double *data = (double *)PyArray_DATA(arr);
    double *data_sma = (double *)PyArray_DATA(arr_sma);
    double *result_data = (double *)PyArray_DATA((PyArrayObject *)result);

    double alpha = smooth / (window_t + 1);
    double ema_prev = data_sma[window_t - 1];  // Initialize with SMA[window_t - 1] for the first EMA value
    double ema;

    if (size > 1000) {
        #pragma omp parallel for private(ema, ema_prev) 
        for (npy_intp i = window_t - 1; i < size; i++) {
            ema = alpha * data[i] + (1 - alpha) * ema_prev;
            ema_prev = ema;
            result_data[i] = ema;
        }
    } else {
        for (npy_intp i = window_t - 1; i < size; i++) {
            ema = alpha * data[i] + (1 - alpha) * ema_prev;
            ema_prev = ema;
            result_data[i] = ema;
        }
    }

    if (PyErr_Occurred()) {
        Py_DECREF(arr);
        Py_DECREF(arr_sma);
        Py_DECREF(result);
        return NULL;
    }

    Py_DECREF(arr);
    Py_DECREF(arr_sma);

    return result;
}


PyObject *WMA(PyObject *self, PyObject *args) {
    PyObject *input_array;
    npy_int32 window_t;

    if (!PyArg_ParseTuple(args, "Oi", &input_array, &window_t) || PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Invalid argument. Expected a numpy array and an int.");
        return NULL;
    }
    if (window_t <= 0) {
        PyErr_SetString(PyExc_ValueError, "Window size must be a positive integer.");
        return NULL;
    }
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(input_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr == NULL) {
        return NULL;
    }

    double *data = (double *)PyArray_DATA(arr);
    npy_intp size = PyArray_SIZE(arr);

    PyObject *result = PyArray_NewLikeArray(arr, NPY_CORDER, NULL, 0);
    if (result == NULL) {
        Py_XDECREF(arr);
        return NULL;
    }
    double *result_data = (double *)PyArray_DATA((PyArrayObject *)result);

    double sum = 0;
    double counter = 0;

    for (npy_intp i = window_t - 1; i < size; i++) {
        sum = 0;
        counter = 1;
        for (npy_intp j = i - window_t + 1; j <= i; j++) {
            sum += data[j] * (counter / (window_t * (window_t + 1) / 2));
            counter += 1;
        }
        result_data[i] = sum;
    }

    if (PyErr_Occurred()) {
        Py_DECREF(arr);
        Py_DECREF(result);
        return NULL;
    }
    Py_DECREF(arr);

    return result;
}


PyMethodDef methods[] = {
  {"SMA",        (PyCFunction)SMA,          METH_VARARGS, "Computes the Simple Moving Average."},
  {"EMA",        (PyCFunction)EMA,          METH_VARARGS, "Computes the Exponential Moving Average."},
  {"WMA",        (PyCFunction)WMA,          METH_VARARGS, "Computes the Weighted Moving Average."},
  {NULL, NULL, 0, NULL}
};

PyModuleDef MAs = {
  PyModuleDef_HEAD_INIT,
  "MAs",
  "This is the Moving Averages module.",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_MAs() {
  import_array(); // init numpy
  PyObject *module = PyModule_Create(&MAs);
  printf("Imported MAs module\n");
  return module;
}
