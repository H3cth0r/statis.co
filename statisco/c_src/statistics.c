#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>

PyObject *closingReturns(PyObject *self, PyObject *args){
  PyObject *input_array;
  if(!PyArg_ParseTuple(args, "O", &input_array) || PyErr_Occurred()){
    return NULL;
  }
  PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(input_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(arr == NULL){
    return NULL;
  }

  double *data = (double *)PyArray_DATA(arr);
  npy_intp size = PyArray_SIZE(arr);

  // npy_intp dims[] = {size};
  PyObject *result = PyArray_NewLikeArray(arr, NPY_CORDER, NULL, 0);
  if (result == NULL) {
      Py_XDECREF(arr);
      return NULL;
  }
  double *result_data = (double *)PyArray_DATA((PyArrayObject *)result);
  if (size > 1000) {
      #pragma omp parallel for
      for (npy_intp i = 0; i < size - 1; i++) {
          result_data[i] = data[i] / data[i + 1] - 1;
      }
  } else {
      for (npy_intp i = 0; i < size - 1; i++) {
            result_data[i] = data[i] / data[i + 1] - 1;
      }
  }
  
  result_data[size - 1] = 0;

  if (PyErr_Occurred()) {
      Py_XDECREF(arr);
      Py_XDECREF(result_data);
      return NULL;
  }

  Py_XDECREF(arr);
  return result;
}


PyMethodDef methods[] = {
  {"closingReturns",        (PyCFunction)closingReturns,          METH_VARARGS, "Computes the return column from dataframe."},
  {NULL, NULL, 0, NULL}
};

PyModuleDef statistics = {
  PyModuleDef_HEAD_INIT,
  "statistics",
  "This is the statistics module.",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_statistics() {
  import_array(); // init numpy
  PyObject *module = PyModule_Create(&statistics);
  printf("Imported statistics module\n");
  return module;
}
