#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <stdio.h>

int extractDoubleArray(PyObject *args, double **data, int64_t *size){
  PyArrayObject *arrData;
  if(!PyArg_ParseTuple(args, "O", &arrData)){
    return 0;
  }
  if(!PyArray_Check(arrData) || PyArray_TYPE(arrData) != NPY_DOUBLE || !PyArray_ISCARRAY(arrData)) {
    PyErr_SetString(PyExc_TypeError, "Argument must be a numpy C-contiguous array of type double.");
    return 0;
  }

  *size = PyArray_SIZE(arrData);
  *data = (double *)PyArray_DATA(arrData);

  return 1;
}

PyObject *closingReturns(PyObject *self, PyObject *args) {
  double *data;
  int64_t size;

  if(!extractDoubleArray(args, &data, &size) || PyErr_Occurred()){
    return NULL;
  }

  npy_intp dims[] = {[0] = size};
  PyObject *result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  double *result_data = PyArray_DATA((PyArrayObject *)result);

  if(size > 1000){
    #pragma omp parallel for
    for(npy_intp i = 0; i < size - 1; i++){
      result_data[i] = data[i] / data[i + 1] - 1;
    }
  }else {
    for(npy_intp i = 0; i < size - 1; i++){
      result_data[i] = data[i] / data[i + 1] - 1;
    }
  }

  result_data[size - 1] = 0;

  return result;
}

PyObject *averageReturns(PyObject *self, PyObject *args){
  int64_t size;
  double *data;
  if(!extractDoubleArray(args, &data, &size) || PyErr_Occurred()){
    return NULL;
  }
  
  double addition = 0;
  if(size > 1000) {
    #pragma omp parallel for reduction(+:addition)
    for(npy_intp i = 0; i < size; i++) {
      addition += data[i];
    }
  } else {
    for(npy_intp i = 0; i < size; i++) {
      addition += data[i];
    }
  }

  return Py_BuildValue("d", addition / size);
}

PyObject *varianceReturns (PyObject *self, PyObject *args) {
  PyArrayObject *returns_t;
  npy_float64 averageReturns_t;

  if(!PyArg_ParseTuple(args, "O!d", &PyArray_Type, &returns_t, &averageReturns_t) || PyErr_Occurred()){
    PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected a Numpy arr and a double.");
    return NULL;
  }
  printf("Parsed arguments: returns_t=%p, averageReturns_t=%lf\n", (void*)returns_t, averageReturns_t);

  double *returns         = PyArray_DATA(returns_t);
  npy_intp size           = PyArray_SIZE(returns_t);
  
  double averageDiffSqd = 0.0;
  double diff;
  if(size > 1000){
    #pragma omp parallel for private(diff) reduction(+:averageDiffSqd) 
    for(npy_intp i = 0; i < size; i++) {
      if (i >= PyArray_SIZE(returns_t)) {
          PyErr_SetString(PyExc_IndexError, "Index out of bounds");
          return NULL;
      }
      diff = returns[i] - averageReturns_t;
      averageDiffSqd += diff * diff;
    }
  }else {
    for(npy_intp i = 0; i < size; i++) {
      if (i >= PyArray_SIZE(returns_t)) {
          PyErr_SetString(PyExc_IndexError, "Index out of bounds");
          return NULL;
      }
      diff = returns[i] - averageReturns_t;
      averageDiffSqd += diff * diff;
    }
  }
  averageDiffSqd /= size;
  return Py_BuildValue("d", averageDiffSqd);
}


PyMethodDef methods[] = {
  {"closingReturns", (PyCFunction)closingReturns, METH_VARARGS, "Computes the return column from dataframe."},
  {"averageReturns", (PyCFunction)averageReturns, METH_VARARGS, "Computes the average of returns col."},
  {"varianceReturns", (PyCFunction)varianceReturns, METH_VARARGS, "Computes the varianceReturns of returns col and average returns."},
  {NULL, NULL, 0, NULL}
};

PyModuleDef processingFunctions = {
  PyModuleDef_HEAD_INIT,
  "processingFunctions",
  "This is the stocks operations module",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_processingFunctions() {
  import_array(); // init numpy
  PyObject *module = PyModule_Create(&processingFunctions);
  printf("Imported Stocks Operatios module\n");
  return module;
}
