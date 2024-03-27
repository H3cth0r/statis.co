#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

typedef struct {
    PyObject_HEAD
    PyObject *min_vals;
    PyObject *max_vals;
} MinMaxScalerObject;

static int MinMaxScaler_init(MinMaxScalerObject* self, PyObject* args, PyObject* kwds) {
    static char* keywords[] = {NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "", keywords)) {
        return -1;
    }
    self->min_vals   = PyList_New(0);
    self->max_vals   = PyList_New(0);

    return 0;
}

static PyObject* MinMaxScaler_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    MinMaxScalerObject *self;

    self = (MinMaxScalerObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->min_vals = PyList_New(0);
        self->max_vals = PyList_New(0);
    }

    return (PyObject *)self;
}

static void MinMaxScaler_dealloc(MinMaxScalerObject* self) {
  Py_XDECREF(self->min_vals);
  Py_XDECREF(self->max_vals);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* MinMaxScaler_get_min_vals(MinMaxScalerObject* self, void* closure) {
    Py_INCREF(self->min_vals);
    return (PyObject*)self->min_vals;
}

static PyObject* MinMaxScaler_get_max_vals(MinMaxScalerObject* self, void* closure) {
    Py_INCREF(self->max_vals);
    return (PyObject*)self->max_vals;
}

PyObject* MinMaxScaler_fit(MinMaxScalerObject* self, PyObject* args) {
    PyArrayObject* input_array;

    if (!PyArg_ParseTuple(args, "O", &input_array)) {
        return NULL;
    }

    // Check input is 1d
    // if (!PyArray_ISCONTIGUOUS(input_array) || PyArray_TYPE(input_array) != NPY_DOUBLE || PyArray_NDIM(input_array) != 2) {
    if ( PyArray_TYPE(input_array) != NPY_DOUBLE || PyArray_NDIM(input_array) != 2) {
        char error_message[1000];
        snprintf(error_message, sizeof(error_message), "Input must be a 2D contiguous array of doubles. Type: %d, Dimensions: %d, Contiguous: %d", PyArray_TYPE(input_array), PyArray_NDIM(input_array), PyArray_ISCONTIGUOUS(input_array));
        PyErr_SetString(PyExc_TypeError, error_message);
        // PyErr_SetString(PyExc_TypeError, "Input must be a 2D contiguous array of doubles.");
        return NULL;
    }

    npy_intp rows = PyArray_DIM(input_array, 0);
    npy_intp cols = PyArray_DIM(input_array, 1);
    double* data = (double*)PyArray_DATA(input_array);

    for(npy_intp i = 0; i < cols; i++)
    {
      double min_val = NPY_INFINITY;
      double max_val = -NPY_INFINITY;
      for(npy_intp j = 0; j < rows; j++)
      {
        double val = data[j + i * rows];
        if(min_val > val) min_val = val;
        if(max_val < val)max_val = val;
      }
        PyList_Append(self->min_vals, PyFloat_FromDouble(min_val));
        PyList_Append(self->max_vals, PyFloat_FromDouble(max_val));
    }
    Py_RETURN_NONE;
}

PyObject* MinMaxScaler_transform(MinMaxScalerObject* self, PyObject* args) {
    PyArrayObject* input_array;

    if (!PyArg_ParseTuple(args, "O", &input_array)) {
        return NULL;
    }

    // Check input is 2d
    if ( PyArray_TYPE(input_array) != NPY_DOUBLE || PyArray_NDIM(input_array) != 2) {
        PyErr_SetString(PyExc_TypeError, "Input must be a 2D contiguous array of doubles.");
        return NULL;
    }
    // Get dimensions and data pointer of the input array
    npy_intp rows = PyArray_DIM(input_array, 0);
    npy_intp cols = PyArray_DIM(input_array, 1);
    double* data = (double*)PyArray_DATA(input_array);

    PyArrayObject* output_array = (PyArrayObject*)PyArray_SimpleNew(2, PyArray_DIMS(input_array), NPY_DOUBLE);
    double* output_array_data = (double*)PyArray_DATA(output_array);

    for(npy_intp i = 0; i < cols; i++)
    {
      double min_val = PyFloat_AsDouble(PyList_GetItem(self->min_vals, i));
      double max_val = PyFloat_AsDouble(PyList_GetItem(self->max_vals, i));
      for(npy_intp j = 0; j < rows; j++)
      {
          double val     = data[j + i * rows];
          output_array_data[j * cols + i] = ((val - min_val) / (max_val - min_val));
      }
    }

    return PyArray_Return(output_array);
}


static PyMethodDef MinMaxScaler_methods[] = {
    {"fit",         (PyCFunction)MinMaxScaler_fit,        METH_VARARGS, "Fit the Min-Max scaler"},
    {"transform",   (PyCFunction)MinMaxScaler_transform,  METH_VARARGS, "Transform data using the Min-Max scaler"},
    {NULL,          NULL,                                 0,            NULL} 
};

static PyGetSetDef MinMaxScaler_getset[] = {
    {"min_vals", (getter)MinMaxScaler_get_min_vals, NULL, "Min values array", NULL},
    {"max_vals", (getter)MinMaxScaler_get_max_vals, NULL, "Max values array", NULL},
    {NULL}
};

static PyTypeObject MinMaxScalerType = {
    .ob_base       = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "normalization.MinMaxScaler",
    .tp_basicsize = sizeof(MinMaxScalerObject),
    .tp_itemsize  = 0,
    .tp_dealloc   = (destructor)MinMaxScaler_dealloc,
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc       = PyDoc_STR("MinMaxScaler object"),
    .tp_methods   = MinMaxScaler_methods,
    .tp_getset    = MinMaxScaler_getset,
    .tp_init      = (initproc)MinMaxScaler_init,
    .tp_new       = MinMaxScaler_new,  // Add the new method here
};
