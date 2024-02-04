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
          result_data[i ] = data[i] / data[i - 1] - 1;
      }
  } else {
      for (npy_intp i = 0; i < size - 1; i++) {
            result_data[i] = data[i] / data[i - 1] - 1;
      }
  }
  
  result_data[size - 1] = 0;
  result_data[0] = 0;

  if (PyErr_Occurred()) {
      Py_XDECREF(arr);
      Py_XDECREF(result);
      return NULL;
  }

  Py_XDECREF(arr);
  return result;
}

PyObject *mean(PyObject *self, PyObject *args){
  PyObject *input_array;
  if(!PyArg_ParseTuple(args, "O", &input_array) || PyErr_Occurred()){
    return NULL;
  }

  PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(input_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if(arr == NULL){
    return NULL;
  }
  double *data = (double*)PyArray_DATA(arr);
  npy_intp size = PyArray_SIZE(arr);

  if (size == 0) {
      PyErr_SetString(PyExc_ValueError, "Input array is empty.");
      Py_XDECREF(arr);
      return NULL;
  }
  
  double sum = 0.0;
  #pragma omp parallel for reduction(+:sum)
  for(npy_intp i = 0; i  < size; i++) {
    sum += data[i];
  }
  Py_DECREF(arr);
  if (PyErr_Occurred()) {
      return NULL;
  }
  return Py_BuildValue("d", sum / size);
}


PyObject *variance(PyObject *self, PyObject *args){
    PyObject *input_array;
    npy_float64 average_t;

    if(!PyArg_ParseTuple(args, "Od", &input_array, &average_t) || PyErr_Occurred()){
        return NULL;
    }

    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(input_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(arr == NULL){
        return NULL;
    }

    double *data = (double*)PyArray_DATA(arr);
    npy_intp size = PyArray_SIZE(arr);

    if (size == 0) {
        PyErr_SetString(PyExc_ValueError, "Input array is empty.");
        Py_XDECREF(arr);
        return NULL;
    }

    double averageDiffSqd = 0.0;
    double diff;

    if(size > 1000){
        #pragma omp parallel for private(diff) reduction(+:averageDiffSqd) 
        for(npy_intp i = 0; i < size; i++) {
            if (i >= size) {
                PyErr_SetString(PyExc_IndexError, "Index out of bounds");
                Py_DECREF(arr);
                return NULL;
            }
            diff = data[i] - average_t;
            averageDiffSqd += diff * diff;
        }
    } else {
        for(npy_intp i = 0; i < size; i++) {
            if (i >= size) {
                PyErr_SetString(PyExc_IndexError, "Index out of bounds");
                Py_DECREF(arr);
                return NULL;
            }
            diff = data[i] - average_t;
            averageDiffSqd += diff * diff;
        }
    }

    Py_DECREF(arr);
    return Py_BuildValue("d", averageDiffSqd/size);
}

PyObject *stdDev(PyObject *self, PyObject *args){
  PyObject *input_array;
  npy_float64 average_t;
  if(!PyArg_ParseTuple(args, "Od", &input_array, &average_t) || PyErr_Occurred()){
    return NULL;
  }

  PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(input_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if(arr == NULL){
    return NULL;
  }

  double *data = (double *)PyArray_DATA(arr);
  npy_intp size = PyArray_SIZE(arr);

  if (size == 0) {
    PyErr_SetString(PyExc_ValueError, "Input array is empty.");
    Py_XDECREF(arr);
    return NULL;
  }

  double diff;
  double sum_squared_diff = 0.0;

  // Use a single loop body
  #pragma omp parallel for private(diff) reduction(+:sum_squared_diff)
  for (npy_intp i = 0; i < size; i++) {
    diff = data[i] - average_t;
    sum_squared_diff += diff * diff;
  }

  // Release the reference to the PyArrayObject
  Py_XDECREF(arr);

  return Py_BuildValue("d", sqrt(sum_squared_diff / size));
}

PyObject *covariance(PyObject *self, PyObject *args) {
    PyObject *input_array_one;
    npy_float64 mean_array_one;
    PyObject *input_array_two;
    npy_float64 mean_array_two;
    if (!PyArg_ParseTuple(args, "OdOd", &input_array_one, &mean_array_one, &input_array_two, &mean_array_two) || PyErr_Occurred()) {
        return NULL;
    }

    PyArrayObject *arr_one = (PyArrayObject *)PyArray_FROM_OTF(input_array_one, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_two = (PyArrayObject *)PyArray_FROM_OTF(input_array_two, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr_one == NULL || arr_two == NULL) {
        Py_XDECREF(arr_one);
        Py_XDECREF(arr_two);
        return NULL;
    }

    double *data_one = (double *)PyArray_DATA(arr_one);
    double *data_two = (double *)PyArray_DATA(arr_two);
    npy_intp size_one = PyArray_SIZE(arr_one);
    npy_intp size_two = PyArray_SIZE(arr_two);

    if (size_one == 0 || size_two == 0 || size_one != size_two) {
        PyErr_SetString(PyExc_ValueError, "Input arrays are empty or not the same size.");
        Py_XDECREF(arr_one);
        Py_XDECREF(arr_two);
        return NULL;
    }

    double covariance = 0.0;
    double diff_one;
    double diff_two;
    if (size_one > 1000) {
        #pragma omp parallel for private(diff_one, diff_two) reduction(+:covariance)
        for (npy_intp i = 0; i < size_one; i++) {
            diff_one = data_one[i] - mean_array_one;
            diff_two = data_two[i] - mean_array_two;
            covariance += diff_one * diff_two;
        }
    } else {
        for (npy_intp i = 0; i < size_one; i++) {
            diff_one = data_one[i] - mean_array_one;
            diff_two = data_two[i] - mean_array_two;
            covariance += diff_one * diff_two;
        }
    }

    Py_XDECREF(arr_one);
    Py_XDECREF(arr_two);

    return Py_BuildValue("d", covariance / (size_one - 1));
}

PyObject *correlation(PyObject *self, PyObject *args) {
    npy_float64 xyCovar_t, xVar_t, yVar_t;
    
    if (!PyArg_ParseTuple(args, "ddd", &xyCovar_t, &xVar_t, &yVar_t)) {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected float values.");
        return NULL;
    }

    return Py_BuildValue("d", xyCovar_t / (sqrt(xVar_t) * sqrt(yVar_t)));
}

PyObject *correlation_matrix(PyObject *self, PyObject *args)
{
    PyObject *input_obj;
    PyArrayObject *input_array;
    int n, m;

    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "O", &input_obj)) {
        return NULL;
    }

    // Convert the input object to a NumPy array
    input_array = (PyArrayObject*)PyArray_FROM_OTF(input_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (input_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Input must be a NumPy array of type float64.");
        return NULL;
    }

    // Check if the input array is a 2D array
    if (PyArray_NDIM(input_array) != 2) {
        Py_DECREF(input_array);
        PyErr_SetString(PyExc_ValueError, "Input array must be 2-dimensional");
        return NULL;
    }

    // Get the dimensions of the input array
    n = (int)PyArray_DIM(input_array, 0);
    m = (int)PyArray_DIM(input_array, 1);

    // Allocate memory for the correlation matrix
    npy_intp dims[2] = {m, m};
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (result_array == NULL) {
        Py_DECREF(input_array);
        return NULL;
    }

    // Get pointers to the data of input and result arrays
    double* input_data = (double*)PyArray_DATA(input_array);
    double* result_data = (double*)PyArray_DATA(result_array);


  // Calculate the correlation matrix
  for (int i = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j) {
          double sum_xy = 0.0, sum_x = 0.0, sum_y = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;
          for (int k = 0; k < n; ++k) {
              double x = input_data[k * m + i];
              double y = input_data[k * m + j];
              sum_xy += x * y;
              sum_x += x;
              sum_y += y;
              sum_x2 += x * x;
              sum_y2 += y * y;
          }
          double corr = (n * sum_xy - sum_x * sum_y) / sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
          result_data[i * m + j] = corr;
      }
  }

  Py_DECREF(input_array);
  return PyArray_Return(result_array);


}

PyMethodDef methods[] = {
  {"closingReturns",        (PyCFunction)closingReturns,          METH_VARARGS, "Computes the return column from dataframe."},
  {"mean",                  (PyCFunction)mean,                    METH_VARARGS, "Computes the mean/average."},
  {"variance",              (PyCFunction)variance,                METH_VARARGS, "Computes the variance based on the average returns."},
  {"stdDev",                (PyCFunction)stdDev,                  METH_VARARGS, "Computes the standard deviation based on the average returns."},
  {"covariance",            (PyCFunction)covariance,              METH_VARARGS, "Computes the covariance."},
  {"correlation",           (PyCFunction)correlation,             METH_VARARGS, "Computes the correlation."},
  {"correlation_matrix",    (PyCFunction)correlation_matrix,      METH_VARARGS, "Computes the correlation matrix."},
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
  return module;
}
