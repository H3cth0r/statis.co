#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>

PyObject *compoundInterest(PyObject *self, PyObject *args){
    npy_float64 P_t, r_t, t_t;

    if (!PyArg_ParseTuple(args, "ddd", &P_t, &r_t, &t_t) || PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected float values.");
        return NULL;
    }

    if (t_t < 0) {
        PyErr_SetString(PyExc_ValueError, "Time cannot be negative.");
        return NULL;
    }

    return Py_BuildValue("d", P_t * pow((1 + r_t), t_t));
}
PyObject *moneyMadeInAYear(PyObject *self, PyObject *args) {
    npy_float64 P_t, r_t, t_t;

    if (!PyArg_ParseTuple(args, "ddd", &P_t, &r_t, &t_t) || PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected float values.");
        return NULL;
    }

    if (t_t < 0) {
        PyErr_SetString(PyExc_ValueError, "Time cannot be negative.");
        return NULL;
    }
    npy_float64 result = r_t * (P_t * pow((1 + r_t), t_t));

    return Py_BuildValue("d", result);
}

PyObject *compoundInterestTime(PyObject *self, PyObject *args) {
    npy_float64 r_t;

    if (!PyArg_ParseTuple(args, "d", &r_t)) {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected float values.");
        return NULL;
    }

    if (r_t <= 0 || r_t >= 1) {
        PyErr_SetString(PyExc_ValueError, "Interest rate must be between 0 and 1 (exclusive).");
        return NULL;
    }

    double compound_time = -log(r_t) / log(1 + r_t);
    PyObject *result = Py_BuildValue("d", compound_time);
    return result;
}

PyObject *expectedValue(PyObject *self, PyObject *args) {
    npy_float64 avgLoss_t;
    npy_float64 avgLP_t;
    npy_float64 avGain_t;
    npy_float64 avgGP_t;

    if (!PyArg_ParseTuple(args, "dddd", &avgLoss_t, &avgLP_t, &avGain_t, &avgGP_t)) {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected float values.");
        return NULL;
    }

    npy_float64 result = (avgLoss_t * avgLP_t) + (avGain_t * avgGP_t);

    return Py_BuildValue("d", result);
}

PyMethodDef methods[] = {
  {"compoundInterest",        (PyCFunction)compoundInterest,          METH_VARARGS, "Computes the compound interest."},
  {"moneyMadeInAYear",        (PyCFunction)moneyMadeInAYear,          METH_VARARGS, "Computes the money made in a year."},
  {"compoundInterestTime",    (PyCFunction)compoundInterestTime,      METH_VARARGS, "Computes the compound interest through time."},
  {"expectedValue",           (PyCFunction)expectedValue,             METH_VARARGS, "Computes the expected value."},
  {NULL, NULL, 0, NULL}
};

PyModuleDef finance = {
  PyModuleDef_HEAD_INIT,
  "finance",
  "This is the finance calculations module.",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_finance() {
  import_array(); // init numpy
  PyObject *module = PyModule_Create(&finance);
  printf("Imported finance module\n");
  return module;
}
