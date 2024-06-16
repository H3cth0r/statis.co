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

PyObject *MACD(PyObject *self, PyObject * args){
    PyObject *Close_t;
    npy_int32 short_window_t;
    npy_int32 long_window_t;
    npy_int32 signal_window_t;

    if (!PyArg_ParseTuple(args, "Oiii", &Close_t, &short_window_t, &long_window_t, &signal_window_t) || PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected Numpy array and three integers");
        return NULL;
    }
    if (short_window_t <= 0 || long_window_t <= 0 || signal_window_t <= 0) {
        PyErr_SetString(PyExc_ValueError, "Window sizes must be positive integers.");
        return NULL;
    }

    PyArrayObject *close_arr = (PyArrayObject *)PyArray_FROM_OTF(Close_t, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (close_arr == NULL) {
        return NULL;
    }

    // Calculate SMA for the long window
    PyObject *SMA_long = SMA(self, Py_BuildValue("Oi", close_arr, long_window_t));
    if (SMA_long == NULL) {
        Py_XDECREF(close_arr);
        return NULL;
    }

    // Calculate EMA for the short and long windows
    PyObject *EMA_short = EMA(self, Py_BuildValue("OOid", close_arr, SMA_long, short_window_t, 2.0));
    PyObject *EMA_long = EMA(self, Py_BuildValue("OOid", close_arr, SMA_long, long_window_t, 2.0));
    Py_XDECREF(SMA_long);
    if (EMA_short == NULL || EMA_long == NULL) {
        Py_XDECREF(close_arr);
        Py_XDECREF(EMA_short);
        Py_XDECREF(EMA_long);
        return NULL;
    }

    npy_intp size = PyArray_SIZE(close_arr);
    PyObject *result = PyTuple_New(3);
    if (result == NULL) {
        Py_XDECREF(close_arr);
        Py_XDECREF(EMA_short);
        Py_XDECREF(EMA_long);
        return NULL;
    }

    // Create arrays for MACD, Signal Line, and MACD Histogram
    PyObject *MACD_line = PyArray_NewLikeArray((PyArrayObject *)EMA_short, NPY_CORDER, NULL, 0);
    PyObject *Signal_line = PyArray_NewLikeArray((PyArrayObject *)EMA_short, NPY_CORDER, NULL, 0);
    PyObject *MACD_Histogram = PyArray_NewLikeArray((PyArrayObject *)EMA_short, NPY_CORDER, NULL, 0);

    if (MACD_line == NULL || Signal_line == NULL || MACD_Histogram == NULL) {
        Py_XDECREF(close_arr);
        Py_XDECREF(EMA_short);
        Py_XDECREF(EMA_long);
        Py_XDECREF(MACD_line);
        Py_XDECREF(Signal_line);
        Py_XDECREF(MACD_Histogram);
        return NULL;
    }

    double *data_short = (double *)PyArray_DATA((PyArrayObject *)EMA_short);
    double *data_long = (double *)PyArray_DATA((PyArrayObject *)EMA_long);
    double *macd_data = (double *)PyArray_DATA((PyArrayObject *)MACD_line);
    double *signal_data = (double *)PyArray_DATA((PyArrayObject *)Signal_line);
    double *hist_data = (double *)PyArray_DATA((PyArrayObject *)MACD_Histogram);

    // Calculate MACD line
    for (npy_intp i = 0; i < size; i++) {
        macd_data[i] = data_short[i] - data_long[i];
    }

    // Calculate Signal line (EMA of MACD line)
    PyObject *SMA_macd = SMA(self, Py_BuildValue("Oi", MACD_line, signal_window_t));
    if (SMA_macd == NULL) {
        Py_XDECREF(close_arr);
        Py_XDECREF(EMA_short);
        Py_XDECREF(EMA_long);
        Py_XDECREF(MACD_line);
        Py_XDECREF(Signal_line);
        Py_XDECREF(MACD_Histogram);
        return NULL;
    }
    PyObject *EMA_signal = EMA(self, Py_BuildValue("OOid", MACD_line, SMA_macd, signal_window_t, 2.0));
    Py_XDECREF(SMA_macd);
    if (EMA_signal == NULL) {
        Py_XDECREF(close_arr);
        Py_XDECREF(EMA_short);
        Py_XDECREF(EMA_long);
        Py_XDECREF(MACD_line);
        Py_XDECREF(Signal_line);
        Py_XDECREF(MACD_Histogram);
        return NULL;
    }

    double *signal_line_data = (double *)PyArray_DATA((PyArrayObject *)EMA_signal);
    for (npy_intp i = 0; i < size; i++) {
        signal_data[i] = signal_line_data[i];
        hist_data[i] = macd_data[i] - signal_data[i];
    }

    PyTuple_SET_ITEM(result, 0, MACD_line);
    PyTuple_SET_ITEM(result, 1, Signal_line);
    PyTuple_SET_ITEM(result, 2, MACD_Histogram);

    Py_XDECREF(close_arr);
    Py_XDECREF(EMA_short);
    Py_XDECREF(EMA_long);
    Py_XDECREF(EMA_signal);

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
