#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

typedef struct {
  double min_val;
  double max_val;
} MinMaxScalerState;

typedef struct {
  PyObject_HEAD MinMaxScalerState* state;
} MinMaxScalerObject;

static int MinMaxScaler_init(MinMaxScalerObject* self, PyObject* args, PyObject* kwds){
  static char* keywords[] = {NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kwds, "", keywords)){
    return -1;
  }
  self->state = NULL;
  return 0;
}

static void MinMaxScaler_dealloc(MinMaxScalerObject* self){
  if(self->state != NULL){
    PyMem_Free(self->state);
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* MinMaxScaler_fit(MinMaxScalerObject* self, PyObject* args){
  PyArrayObject *input_array;

  if(!PyArg_ParseTuple(args, "O", &input_array)){
    return NULL;
  }

  // check input is 1d
  if(!PyArray_ISCONTIGUOUS(input_array) || PyArray_TYPE(input_array) != NPY_DOUBLE || PyArray_NDIM(input_array) != 1){
      PyErr_SetString(PyExc_TypeError, "Input must be a 1D contiguous array of doubles.");
      return NULL;
  }

  // Allocate memory for the state 
  if(self->state != NULL){
    PyMem_Free(self->state);
  }
  self->state = (MinMaxScalerState*)PyMem_Malloc(sizeof(MinMaxScalerState));
  if(state == NULL){
      PyErr_SetString(PyExc_TypeError, "Failed to allocate memory for the scaler state.");
      return NULL;
  }

  // Get dimensions and data pointer of the input array
  npy_intp size = PyArray_SIZE(input_array);
  double *data = (double*)PyArray_DATA(input_array);

  // Find Min and max values
  self->min_val = self->max_val = data[0];
  for(npy_intp i = 1; i < size; ++i){
    if(data[i] < self->min_val){
      self->min_val = data[i];
    }else if(data[i]>self->max_val) {
      self->max_val = data[i];
    }
  }

  // Return the scaler state
  Py_RETURN_NONE;
}

static PyObject* MinMaxScaler_transform(MinMaxScalerObject *self, PyObject* args){
  PyArrayObject* input_array;

  // Parse the input arguments 
  if(!PyArg_ParseTuple(args, "O", &input_array)){
    return NULL;
  }

  // Check if the input is a 1D contiguous array of doubles.
  if (!PyArray_ISCONTIGUOUS(input_array) || PyArray_TYPE(input_array) != NPY_DOUBLE || PyArray_NDIM(input_array) != 1) {
    PyErr_SetString(PyExc_TypeError, "Input must be a 1D contiguous array of doubles.");
    return NULL;
  }

  if(self->state == NULL){
    PyErr_SetString(PyExc_TypeError, "Scaler must be fitted before transforming.");
    return NULL;
  }

  // Get dimensions and data pointer of the input array
  npy_intp size = PyArray_SIZE(input_array);
  double* data = (double*)PyArray_DATA(input_array);

  // Transform the data using min max scaling
  for(npy_intp i = 0; i < size; ++i){
    data[i] = (data[i] - self->min_val) / (self->max_val - self->min_val);
  }

  Py_RETURN_NONE;
}

PyMethodDef MinMaxScaler_methods[] = {
  {"fit",       (PyCFunction)MinMaxScaler_fit,        METH_VARARGS, "Fit the Min-Max scaler"},
  {"transform", (PyCFunction)MinMaxScaler_transform,  METH_VARARGS, "Transform data using the Min-Max scaler"},
  {NULL, NULL, 0, NULL}  /* Sentinel */
};


static PyTypeObject MinMaxScalerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "MinMaxScaler",
    .tp_basicsize = sizeof(MinMaxScalerObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)MinMaxScaler_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "MinMaxScaler object",
    .tp_methods = MinMaxScaler_methods,
    .tp_init = (initproc)MinMaxScaler_init,
    .tp_new = MinMaxScaler_new,
};

static PyModuleDef MinMaxScaler_module = {
    PyModuleDef_HEAD_INIT,
    "MinMaxScaler",
    NULL,
    -1,
    NULL
};

PyMODINIT_FUNC PyInit_MinMaxScaler(void) {
    PyObject* m;

    if (PyType_Ready(&MinMaxScalerType) < 0) {
        return NULL;
    }

    m = PyModule_Create(&MinMaxScaler_module);
    if (m == NULL) {
        return NULL;
    }

    Py_INCREF(&MinMaxScalerType);
    PyModule_AddObject(m, "MinMaxScaler", (PyObject*)&MinMaxScalerType);

    import_array();

    return m;
}
