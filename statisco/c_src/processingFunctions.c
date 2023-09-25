#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <stdio.h>


PyObject *add(PyObject *self, PyObject *args){
  double x;
  double y;
  PyArg_ParseTuple(args, "dd", &x, &y);
  return PyFloat_FromDouble(x + y);
}
static PyObject *sum(PyObject *self, PyObject *args){
  PyArrayObject *arr;
  PyArg_ParseTuple(args, "O", &arr);
  if(PyErr_Occurred()){
    return NULL;
  }
  // if(!PyArray_Check(arr) || PyArray_TYPE(arr) != NPY_DOUBLE || !PyArray_IS_C_CONTIGUOUS(arr)){
  //   PyErr_SetString(PyExc_TypeError, "Argument must be a numpy C-contiguous array of type double");
  //   return NULL;
  // }
  if(!PyArray_Check(arr)){
    PyErr_SetString(PyExc_TypeError, "Argument must be a numpy C-contiguous array of type double");
    return NULL;
  }
  
  // double *data = PyArray_DATA(arr);
  // int64_t size = PyArray_SIZE(arr);
  int64_t size = PyArray_SIZE(arr);
  double *data;
  npy_intp dims[] = {[0] = size};
  PyArray_AsCArray((PyObject **)&arr, &data, dims, 1, PyArray_DescrFromType(NPY_DOUBLE));
  if(PyErr_Occurred()){
    return NULL;
  }

  double total = 0;
  for(int i = 0; i < size; ++i){
    total += data[i];
  }
  return PyFloat_FromDouble(total);
}

// ==========================================================
// ==========================================================
// ==========================================================
static PyObject *double_array(PyObject *self, PyObject *args){
  PyArrayObject *arr;
  PyArg_ParseTuple(args, "O", &arr);
  if(PyErr_Occurred()){
    return NULL;
  }

  if(!PyArray_Check(arr)){
    PyErr_SetString(PyExc_TypeError, "Argument must be a numpy C-contiguous array of type double");
    return NULL;
  }

  int64_t size = PyArray_SIZE(arr);
  double *data;
  npy_intp dims[] = {[0] = size};
  PyArray_AsCArray((PyObject **)&arr, &data, dims, 1, PyArray_DescrFromType(NPY_DOUBLE));
  if(PyErr_Occurred()){
    return NULL;
  }

  PyObject *result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  double *result_data = PyArray_DATA((PyArrayObject *)result);
  for(int i = 0; i < size; ++i){
    result_data[i] = 2 * data[i];
  }
  return result;
}
// ==========================================================
static PyObject *closingReturns(PyObject *self, PyObject *args) {
  PyArrayObject *adjCloseData;
  PyArg_ParseTuple(args, "O", &adjCloseData);
  if(PyErr_Occurred()){
    return NULL;
  }

  if(!PyArray_Check(adjCloseData)){
    PyErr_SetString(PyExc_TypeError, "Argument must be a numpy C-contiguous array of type double");
    return NULL;
  }

  int64_t size = PyArray_SIZE(adjCloseData);
  double *data;
  npy_intp dims[] = {[0] = size};
  PyArray_AsCArray((PyObject **)&adjCloseData, &data, dims, 1, PyArray_DescrFromType(NPY_DOUBLE));
  if(PyErr_Occurred()){
    return NULL;
  }

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

static PyMethodDef methods[] = {
  {"add", add, METH_VARARGS, "Adds to numbers together"},
  {"sum", sum, METH_VARARGS, "Calculate sum of numpy array"},
  {"double_array", double_array, METH_VARARGS, "Double elements in numpy array"},
  {"closingReturns", closingReturns, METH_VARARGS, "Computes the return column from dataframe."},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef processingFunctions = {
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
