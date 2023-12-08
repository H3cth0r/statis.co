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



PyMethodDef methods[] = {
  {"SMA",        (PyCFunction)SMA,          METH_VARARGS, "Computes the Simple Moving Average."},
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
