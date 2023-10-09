#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>


double calculateMean(double* data, int size){
  double sum = 0.0; 
  #pragma omp parallel for reduction(+:sum)
  for(npy_intp i = 0; i  < size; i++) {
    sum += data[i];
  }
  return sum / size;
}

/*
 * @brief   extracts the the array from the args
 *
 * @param   args    The arguments from python.
 * @param   data    double pointer.
 * @param   size    pointer to the size of the array.
 * */
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

/*
 * @brief     calculates the 'closing returns' column.
 *
 * @params    gets the 'Adj column' as numpy array input.
 * */
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

/*
 * @brief   method for calculating average of the returns columns.
 *
 * @params  gets numpy array of 'returns' columns.
 *
 * */
PyObject *averageReturns(PyObject *self, PyObject *args){
  int64_t size;
  double *data;
  if(!extractDoubleArray(args, &data, &size) || PyErr_Occurred()){
    return NULL;
  }
  
  // double addition = 0;
  // if(size > 1000) {
  //   #pragma omp parallel for reduction(+:addition)
  //   for(npy_intp i = 0; i < size; i++) {
  //     addition += data[i];
  //   }
  // } else {
  //   for(npy_intp i = 0; i < size; i++) {
  //     addition += data[i];
  //   }
  // }

  // return Py_BuildValue("d", addition / size);
  
  return Py_BuildValue("d", calculateMean(data, size));
}


/*
 * @brief   calculates the variance returns of the returns columns.
 *
 * @param   returns_t         The returns column.
 * @param   averageReturns_t  The averare returns.
 * */
PyObject *varianceReturns (PyObject *self, PyObject *args) {
  PyArrayObject *returns_t;
  npy_float64 averageReturns_t;

  if(!PyArg_ParseTuple(args, "O!d", &PyArray_Type, &returns_t, &averageReturns_t) || PyErr_Occurred()){
    PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected a Numpy arr and a double.");
    return NULL;
  }

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

/*
 * @brief     Calculates the standard deviation of a numerical column.
 * */
PyObject *stdDeviation (PyObject *self, PyObject *args) {
  PyArrayObject *returns_t;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &returns_t) || PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected a Numpy arr."); 
    return NULL;
  }
  
  double *returns   = PyArray_DATA(returns_t);
  npy_intp size     = PyArray_SIZE(returns_t);

  double mean = calculateMean(returns, size);
  double diff;
  double sum_squared_diff = 0.0;
  if(size > 1000) {
    #pragma omp parallel for private(diff) reduction(+:sum_squared_diff)
    for (npy_intp i = 0; i < size; i++) {
      diff = returns[i] - mean;
      sum_squared_diff += diff * diff;
    }
    
  } else {
    for (npy_intp i = 0; i < size; i++) {
      diff = returns[i] - mean;
      sum_squared_diff += diff * diff;
    }
  }

  return Py_BuildValue("d", sqrt(sum_squared_diff / size));
}

/*
 *
 * */
PyObject *covarianceReturns(PyObject *self, PyObject *args) {
  PyArrayObject *returns_one_t;
  PyArrayObject *returns_two_t;
  if(!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &returns_one_t, &PyArray_Type, &returns_two_t) || PyErr_Occurred()){
    PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected two numpy arrays.");
    return NULL;
  }

  double *returns_one   = PyArray_DATA(returns_one_t);
  double *returns_two   = PyArray_DATA(returns_two_t);
  npy_intp size_one     = PyArray_SIZE(returns_one_t);
  npy_intp size_two     = PyArray_SIZE(returns_two_t);

  if(size_one != size_two) {
    PyErr_SetString(PyExc_ValueError, "The two numpy arrays must have the same size.");
  }
  double mean_returns_one = calculateMean(returns_one, size_one);
  double mean_returns_two = calculateMean(returns_two, size_two);

  double covariance = 0.0;
  double diff_one;
  double diff_two;
  if(size_one > 1000){
    #pragma omp parallel for private(diff_one, diff_two) reduction(+:covariance)
    for (npy_intp i = 0; i < size_one; i++) {
      diff_one = returns_one[i]  - mean_returns_one;
      diff_two = returns_two[i]  - mean_returns_two;
      covariance += diff_one * diff_two;
    }
  } else {
    for (npy_intp i = 0; i < size_one; i++) {
      diff_one = returns_one[i]  - mean_returns_one;
      diff_two = returns_two[i]  - mean_returns_two;
      covariance += diff_one * diff_two;
    }
  }
  
  covariance /= (size_one - 1);
  return Py_BuildValue("d", covariance);
}

PyObject *correlationReturns(PyObject *self, PyObject *args) {
  npy_float64 xyCovar_t;
  npy_float64 xVar_t;
  npy_float64 yVar_t;
  if(!PyArg_ParseTuple(args, "ddd", &xyCovar_t, &xVar_t, &yVar_t) || PyErr_Occurred()){
    PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected float values.");
    return NULL;
  }

  double result = xyCovar_t / (sqrt(xVar_t) * sqrt(yVar_t));
  return Py_BuildValue("d", result);
}

PyObject *compoundInterest(PyObject *self, PyObject *args) {
  npy_float64 P_t;
  npy_float64 r_t;
  npy_float64 t_t;
  if(!PyArg_ParseTuple(args, "ddd", &P_t, &r_t, &t_t) || PyErr_Occurred()){
    PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected float values.");
    return NULL;
  }
  return Py_BuildValue("d", P_t * pow((1 + r_t), t_t));
}

PyObject *moneyMadeInAYear(PyObject *self, PyObject *args){
  npy_float64 P_t;
  npy_float64 r_t;
  npy_float64 t_t;
  if(!PyArg_ParseTuple(args, "ddd", &P_t, &r_t, &t_t) || PyErr_Occurred()){
    PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected float values.");
    return NULL;
  }
  return Py_BuildValue("d", r_t * (P_t * pow((1 + r_t), t_t)));
}

PyObject *compoundInterestTime(PyObject *self, PyObject *args){
  npy_float64 r_t;
  if(!PyArg_ParseTuple(args, "d", &r_t) || PyErr_Occurred()){
    PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected float values.");
    return NULL;
  }
  return Py_BuildValue("d", -log(r_t) / log(1 + r_t));
}
PyObject *expectedValue(PyObject *self, PyObject *args) {
  npy_float64 avgLoss_t;
  npy_float64 avgLP_t;
  npy_float64 avGain_t;
  npy_float64 avgGP_t;
  if(!PyArg_ParseTuple(args, "dddd", &avgLoss_t, &avgLP_t, &avGain_t, &avgGP_t) || PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected float values.");
    return NULL;
  }
  return Py_BuildValue("d", (avgLoss_t*avgLP_t) + (avGain_t*avgGP_t));
}

PyObject *calculateSMA(PyObject *self, PyObject *args) {
  PyArrayObject *returns_t;
  int window_t;

  if(!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &returns_t, &window_t) || PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "Invalid argument. Expected a numpy array and an int.");
    return NULL;
  }

  double *returns         = PyArray_DATA(returns_t);
  npy_intp size           = PyArray_SIZE(returns_t);

  npy_intp size_array[1] = {size};
  // PyObject *result = PyArray_Zeros(1, size_array, NPY_DOUBLE, 0);
  PyObject *result = PyArray_Zeros(1, size_array, PyArray_DescrFromType(NPY_DOUBLE), 0);
  double sum;
  if(size > 1000){
    #pragma omp parallel for reduction(+:sum) 
    for(int i = window_t-1 ; i < size; i++) {
      sum = 0;
      for(int j = i-window_t+1; j <= i; j++) {
        sum += returns[j];
      }
      PyArray_SETITEM((PyArrayObject *)result, PyArray_GETPTR1((PyArrayObject *)result, i), PyFloat_FromDouble(sum / window_t));
    } 
  }else {
    for(int i = window_t-1 ; i < size; i++) {
      sum = 0;
      for(int j = i-window_t+1; j <= i; j++) {
        sum += returns[j];
      }
      PyArray_SETITEM((PyArrayObject *)result, PyArray_GETPTR1((PyArrayObject *)result, i), PyFloat_FromDouble(sum / window_t));
    } 
  }

  return result;
}

// PyObject *calculateEMA(PyObject *self, PyObject *args){
//   PyArrayObject *returns_t;
//   PyArrayObject *SMA_t;
//   int window_t;
//   if(!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &returns_t, &PyArray_Type, &SMA_t, &window_t) || PyErr_Occurred()){
//     PyErr_SetString(PyExc_TypeError, "Invalid Argument. Expected a numpy array and a int.");
//     return NULL;
//   }
//   double *returns   = PyArray_DATA(returns_t);
//   npy_intp size     = PyArray_SIZE(returns_t);
//   double *SMA       = PyArray_DATA(SMA_t);
//
//   npy_intp size_array[1]  = {size};
//   PyObject *result  = PyArray_Zeros(1, size_array, PyArray_DescrFromType(NPY_DOUBLE), 0);
//   double multipler  = 2/(window_t+1);
//   double res;
//   npy_intp prevIndex;
//   bool first = true;
//   if(size > 1000) {
//     #pragma omp parallel for private(prevIndex, first) reduction(+:res)
//     for(npy_intp i = window_t-1; i < size; i++) {
//       if(first){
//         res   = returns[i] * multipler + SMA[i] * (1-multipler);
//         first = false;
//       }else{
//         prevIndex = i - 1;
//         // res   = returns[i] * multipler + *((double *)PyArray_GetPtr((PyArrayObject*)result, &prevIndex)) * (1-multipler);
//         res   = returns[i] * multipler + res  * (1-multipler);
//       }
//        *((double *)PyArray_GetPtr((PyArrayObject *)result, &i)) = res;
//     }
//   }else{
//     for(npy_intp i = window_t-1; i < size; i++) {
//       if(first){
//         res   = returns[i] * multipler + SMA[i] * (1-multipler);
//         first = false;
//       }else{
//         prevIndex = i - 1;
//         // res   = returns[i] * multipler + *((double *)PyArray_GetPtr((PyArrayObject*)result, &prevIndex)) * (1-multipler);
//         res   = returns[i] * multipler + res  * (1-multipler);
//       }
//        *((double *)PyArray_GetPtr((PyArrayObject *)result, &i)) = res;
//     }
//   }
//   return result;
// }
PyObject *calculateEMA(PyObject *self, PyObject *args){
    PyArrayObject *returns_t;
    PyArrayObject *SMA_t;
    int window_t;

    if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &returns_t, &PyArray_Type, &SMA_t, &window_t) || PyErr_Occurred()){
        PyErr_SetString(PyExc_TypeError, "Invalid Argument. Expected a numpy array and an int.");
        return NULL;
    }

    double *returns = PyArray_DATA(returns_t);
    npy_intp size = PyArray_SIZE(returns_t);
    double *SMA = PyArray_DATA(SMA_t);

    npy_intp size_array[1] = {size};
    PyObject *result = PyArray_Zeros(1, size_array, PyArray_DescrFromType(NPY_DOUBLE), 0);
    double multiplier = 2.0 / (window_t + 1);
    double ema_prev = SMA[window_t - 1];  // Initialize with SMA[window_t - 1] for the first EMA value
    double ema;

    if(size > 1000) {
      #pragma omp parallel for private(ema, ema_prev) 
      for (npy_intp i = window_t-1; i < size; i++) {
          ema = returns[i] * multiplier + ema_prev * (1 - multiplier);
          ema_prev = ema; 
          ((double *)PyArray_DATA((PyArrayObject*)result))[i] = ema;
      }
    }else{
      for (npy_intp i = window_t-1; i < size; i++) {
          ema = returns[i] * multiplier + ema_prev * (1 - multiplier);
          ema_prev = ema;
          ((double *)PyArray_DATA((PyArrayObject*)result))[i] = ema;
      }
    }
    return result;
}



PyMethodDef methods[] = {
  {"closingReturns",        (PyCFunction)closingReturns,          METH_VARARGS, "Computes the return column from dataframe."},
  {"averageReturns",        (PyCFunction)averageReturns,          METH_VARARGS, "Computes the average of returns col."},
  {"varianceReturns",       (PyCFunction)varianceReturns,         METH_VARARGS, "Computes the varianceReturns of returns col and average returns."},
  {"stdDeviation",          (PyCFunction)stdDeviation,            METH_VARARGS, "Computes the standard deviation of the returns column."},
  {"covarianceReturns",     (PyCFunction)covarianceReturns,       METH_VARARGS, "Computes the covariance returns."},
  {"correlationReturns",    (PyCFunction)correlationReturns,      METH_VARARGS, "Computes the correlation between stocks."},
  {"compoundInterest",      (PyCFunction)compoundInterest,        METH_VARARGS, "Computes the compound interest."},
  {"moneyMadeInAYear",      (PyCFunction)moneyMadeInAYear,        METH_VARARGS, "Computes the money made in a year."},
  {"compoundInterestTime",  (PyCFunction)compoundInterestTime,    METH_VARARGS, "Computes the compoung interest per year."},
  {"expectedValue",         (PyCFunction)expectedValue,           METH_VARARGS, "Computes the expected value given averages."},
  {"calculateSMA",          (PyCFunction)calculateSMA,            METH_VARARGS, "Computes the simple moving average of a column."},
  {"calculateEMA",          (PyCFunction)calculateEMA,            METH_VARARGS, "Computes the exponential moving average."},
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
