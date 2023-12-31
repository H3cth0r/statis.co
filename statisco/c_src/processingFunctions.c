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

PyObject *calculateWMA(PyObject *self, PyObject *args) {
  PyArrayObject *returns_t;
  int window_t;
  if(!PyArg_ParseTuple(args, "0!i", &PyArray_Type, &returns_t, &window_t) || PyErr_Occurred()){
    PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected a numpy array.");
    return NULL;
  }

  npy_intp size     = PyArray_SIZE(returns_t);
  npy_intp size_array[1] = {size};
  PyObject *result  = PyArray_Zeros(1, size_array, PyArray_DescrFromType(NPY_DOUBLE), 0);


  double sum      = 0;
  double counter  = 0;
  for(npy_intp i = window_t-1; i < size; i++){
    sum = 0;
    counter = 1;
    for(npy_intp j = i-window_t; j < i; j++){
        sum += *(double*)PyArray_GetPtr(returns_t, &j) * (counter/window_t);
        counter += 1;
    }
    ((double *)PyArray_DATA((PyArrayObject*)result))[i] = sum;
  }

  return result;
}

PyObject *calculateATR(PyObject *self, PyObject *args) {
    PyObject *Close_t_obj, *High_t_obj, *Low_t_obj;
    int window_t;

    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "OOOi", &Close_t_obj, &High_t_obj, &Low_t_obj, &window_t)) {
        PyErr_SetString(PyExc_TypeError, "Invalid input arguments");
        return NULL;
    }

    // Convert Python objects to NumPy arrays
    PyArrayObject *Close_t = (PyArrayObject *)PyArray_FROM_OTF(Close_t_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *High_t = (PyArrayObject *)PyArray_FROM_OTF(High_t_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *Low_t = (PyArrayObject *)PyArray_FROM_OTF(Low_t_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (Close_t == NULL || High_t == NULL || Low_t == NULL) {
        PyErr_SetString(PyExc_TypeError, "Invalid input arrays");
        Py_XDECREF(Close_t);
        Py_XDECREF(High_t);
        Py_XDECREF(Low_t);
        return NULL;
    }

    int len = PyArray_SIZE(Close_t);

    if (len != PyArray_SIZE(High_t) || len != PyArray_SIZE(Low_t)) {
        PyErr_SetString(PyExc_ValueError, "Close_t, High_t, and Low_t should have the same length");
        Py_XDECREF(Close_t);
        Py_XDECREF(High_t);
        Py_XDECREF(Low_t);
        return NULL;
    }

    double *Close_t_data = (double *)PyArray_DATA(Close_t);
    double *High_t_data = (double *)PyArray_DATA(High_t);
    double *Low_t_data = (double *)PyArray_DATA(Low_t);

    PyArrayObject *atr = (PyArrayObject *)PyArray_SimpleNew(1, PyArray_DIMS(Close_t), NPY_DOUBLE);
    double *atr_data = (double *)PyArray_DATA(atr);

    for (int i = 0; i < window_t - 1; i++) {
        atr_data[i] = 0.0;
    }

    for (int i = window_t - 1; i < len; i++) {
        double high_low = High_t_data[i] - Low_t_data[i];
        double high_close = fabs(High_t_data[i] - Close_t_data[i]);
        double low_close = fabs(Low_t_data[i] - Close_t_data[i]);
        double true_range = fmax(high_low, fmax(high_close, low_close));
        
        atr_data[i] = ((window_t - 1) * atr_data[i-1] + true_range) / window_t;
    }

    Py_XDECREF(Close_t);
    Py_XDECREF(High_t);
    Py_XDECREF(Low_t);
    return PyArray_Return(atr);
}

PyObject *calculateATRwma(PyObject *self, PyObject *args){
    PyObject *Close_t_obj, *High_t_obj, *Low_t_obj;
    int window_t;

    if (!PyArg_ParseTuple(args, "OOOi", &Close_t_obj, &High_t_obj, &Low_t_obj, &window_t)) {
        PyErr_SetString(PyExc_TypeError, "Invalid input arguments");
        return NULL;
    }

    PyArrayObject *Close_t = (PyArrayObject *)PyArray_FROM_OTF(Close_t_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *High_t = (PyArrayObject *)PyArray_FROM_OTF(High_t_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *Low_t = (PyArrayObject *)PyArray_FROM_OTF(Low_t_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (Close_t == NULL || High_t == NULL || Low_t == NULL) {
        PyErr_SetString(PyExc_TypeError, "Invalid input arrays");
        Py_XDECREF(Close_t);
        Py_XDECREF(High_t);
        Py_XDECREF(Low_t);
        return NULL;
    }

    int len = PyArray_SIZE(Close_t);

    if (len != PyArray_SIZE(High_t) || len != PyArray_SIZE(Low_t)) {
        PyErr_SetString(PyExc_ValueError, "Close_t, High_t, and Low_t should have the same length");
        Py_XDECREF(Close_t);
        Py_XDECREF(High_t);
        Py_XDECREF(Low_t);
        return NULL;
    }

    double *Close_t_data = (double *)PyArray_DATA(Close_t);
    double *High_t_data = (double *)PyArray_DATA(High_t);
    double *Low_t_data = (double *)PyArray_DATA(Low_t);

    PyArrayObject *atr = (PyArrayObject *)PyArray_SimpleNew(1, PyArray_DIMS(Close_t), NPY_DOUBLE);
    double *atr_data = (double *)PyArray_DATA(atr);

    double denominator = window_t * (window_t + 1) / 2.0;  

    double sum = 0.0;
    for (int i = 0; i < window_t; i++) {
        double high_low = High_t_data[i] - Low_t_data[i];
        double high_close = fabs(High_t_data[i] - Close_t_data[i]);
        double low_close = fabs(Low_t_data[i] - Close_t_data[i]);
        double true_range = fmax(high_low, fmax(high_close, low_close));
        sum += true_range;
    }
    atr_data[window_t-1] = sum / window_t;

    for (int i = window_t; i < len; i++) {
        double weighted_sum = 0.0;
        for (int j = 1; j <= window_t; j++) {
            int idx = i - window_t + j;
            double high_low = High_t_data[idx] - Low_t_data[idx];
            double high_close = fabs(High_t_data[idx] - Close_t_data[idx]);
            double low_close = fabs(Low_t_data[idx] - Close_t_data[idx]);
            double true_range = fmax(high_low, fmax(high_close, low_close));
            weighted_sum += j * true_range;
        }
        atr_data[i] = weighted_sum / denominator;
    }

    Py_XDECREF(Close_t);
    Py_XDECREF(High_t);
    Py_XDECREF(Low_t);
    return PyArray_Return(atr);
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
  {"calculateWMA",          (PyCFunction)calculateWMA,            METH_VARARGS, "Computes the exponential moving average."},
  {"calculateATR",          (PyCFunction)calculateATR,            METH_VARARGS, "Computes the averege true rate."},
  {"calculateATRwma",       (PyCFunction)calculateATRwma,         METH_VARARGS, "Computes the weighter moving average true rate."},
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
