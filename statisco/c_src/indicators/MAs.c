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

void calculate_ema(double *data, double *ema, int length, int window) {
    double alpha = 2.0 / (window + 1);
    ema[0] = data[0];
    for (int i = 1; i < length; i++) {
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1];
    }
}

PyObject *MACD(PyObject *self, PyObject *args) {
    PyObject *Close_t;
    npy_int32 short_window_t;
    npy_int32 long_window_t;
    npy_int32 signal_window_t;

    if (!PyArg_ParseTuple(args, "Oiii", &Close_t, &short_window_t, &long_window_t, &signal_window_t) || PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected Numpy array and three integers");
        return NULL;
    }
    if (short_window_t <= 0 || long_window_t <= 0 || signal_window_t <= 0) {
        PyErr_SetString(PyExc_ValueError, "Window lengths must be positive integers");
        return NULL;
    }

    PyArrayObject *close_arr = (PyArrayObject *)PyArray_FROM_OTF(Close_t, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (close_arr == NULL) {
        return NULL;
    }

    npy_intp length = PyArray_DIM(close_arr, 0);
    double *close_data = (double *)PyArray_DATA(close_arr);

    double *ema_short = (double *)malloc(length * sizeof(double));
    double *ema_long = (double *)malloc(length * sizeof(double));
    double *macd = (double *)malloc(length * sizeof(double));
    double *signal_line = (double *)malloc(length * sizeof(double));
    double *histogram = (double *)malloc(length * sizeof(double));

    if (!ema_short || !ema_long || !macd || !signal_line || !histogram) {
        free(ema_short);
        free(ema_long);
        free(macd);
        free(signal_line);
        free(histogram);
        Py_DECREF(close_arr);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }

    calculate_ema(close_data, ema_short, length, short_window_t);
    calculate_ema(close_data, ema_long, length, long_window_t);

    for (int i = 0; i < length; i++) {
        macd[i] = ema_short[i] - ema_long[i];
    }

    calculate_ema(macd, signal_line, length, signal_window_t);

    for (int i = 0; i < length; i++) {
        histogram[i] = macd[i] - signal_line[i];
    }

    npy_intp dims[1] = { length };
    PyObject *macd_arr = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, macd);
    PyObject *signal_arr = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, signal_line);
    PyObject *hist_arr = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, histogram);

    Py_DECREF(close_arr);
    free(ema_short);
    free(ema_long);
    
    PyObject *result = PyTuple_Pack(3, macd_arr, signal_arr, hist_arr);
    Py_DECREF(macd_arr);
    Py_DECREF(signal_arr);
    Py_DECREF(hist_arr);

    return result;
}

PyMethodDef methods[] = {
  {"SMA",        (PyCFunction)SMA,          METH_VARARGS, "Computes the Simple Moving Average."},
  {"EMA",        (PyCFunction)EMA,          METH_VARARGS, "Computes the Exponential Moving Average."},
  {"WMA",        (PyCFunction)WMA,          METH_VARARGS, "Computes the Weighted Moving Average."},
  {"MACD",       (PyCFunction)MACD,         METH_VARARGS, "Computes the MACD, Signal Line and Histogram."},
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
