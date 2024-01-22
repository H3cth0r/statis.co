#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

typedef struct {
    PyObject_HEAD
    double min_val;
    double max_val;
} MinMaxScalerObject;

static int MinMaxScaler_init(MinMaxScalerObject* self, PyObject* args, PyObject* kwds) {
    static char* keywords[] = {NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "", keywords)) {
        return -1;
    }
    self->min_val   = 0;
    self->max_val   = 0;

    return 0;
}

static PyObject* MinMaxScaler_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    MinMaxScalerObject *self;

    self = (MinMaxScalerObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->min_val = 0;
        self->max_val = 0;
    }

    return (PyObject *)self;
}

static void MinMaxScaler_dealloc(MinMaxScalerObject* self) {
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* MinMaxScaler_fit(MinMaxScalerObject* self, PyObject* args) {
    PyArrayObject* input_array;

    if (!PyArg_ParseTuple(args, "O", &input_array)) {
        return NULL;
    }

    // Check input is 1d
    if (!PyArray_ISCONTIGUOUS(input_array) || PyArray_TYPE(input_array) != NPY_DOUBLE || PyArray_NDIM(input_array) != 1) {
        PyErr_SetString(PyExc_TypeError, "Input must be a 1D contiguous array of doubles.");
        return NULL;
    }

    // Get dimensions and data pointer of the input array
    npy_intp size = PyArray_SIZE(input_array);
    double* data = (double*)PyArray_DATA(input_array);

    // Find Min and max values
    self->min_val = self->max_val = data[0];
    for (npy_intp i = 1; i < size; ++i) {
        if (data[i] < self->min_val) {
            self->min_val = data[i];
        } else if (data[i] > self->max_val) {
            self->max_val = data[i];
        }
    }

    Py_RETURN_NONE;
}

static PyObject* MinMaxScaler_transform(MinMaxScalerObject* self, PyObject* args) {
    PyArrayObject* input_array;

    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "O", &input_array)) {
        return NULL;
    }

    // Check if the input is a 1D contiguous array of doubles.
    if (!PyArray_ISCONTIGUOUS(input_array) || PyArray_TYPE(input_array) != NPY_DOUBLE || PyArray_NDIM(input_array) != 1) {
        PyErr_SetString(PyExc_TypeError, "Input must be a 1D contiguous array of doubles.");
        return NULL;
    }

    if (self->min_val == 0 && self->max_val == 0) {
        PyErr_SetString(PyExc_TypeError, "Scaler must be fitted before transforming.");
        return NULL;
    }

    // Get dimensions and data pointer of the input array
    npy_intp size = PyArray_SIZE(input_array);
    double* data = (double*)PyArray_DATA(input_array);

    // Transform the data using min max scaling
    for (npy_intp i = 0; i < size; ++i) {
        data[i] = (data[i] - self->min_val) / (self->max_val - self->min_val);
    }

    Py_RETURN_NONE;
}

static PyMethodDef MinMaxScaler_methods[] = {
    {"fit",         (PyCFunction)MinMaxScaler_fit,        METH_VARARGS, "Fit the Min-Max scaler"},
    {"transform",   (PyCFunction)MinMaxScaler_transform,  METH_VARARGS, "Transform data using the Min-Max scaler"},
    {NULL,          NULL,                                 0,            NULL} 
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
    .tp_init      = (initproc)MinMaxScaler_init,
    .tp_new       = MinMaxScaler_new,  // Add the new method here
};

static PyModuleDef Normalization_module = {
    PyModuleDef_HEAD_INIT,
    .m_name   = "normalization",
    .m_doc    = "Normalization tools module.",
    .m_size   = -1,
};

PyMODINIT_FUNC 
PyInit_normalization(void) {
    import_array();

    if (PyType_Ready(&MinMaxScalerType) < 0) {
        return NULL;
    }

    // Create a new module object
    PyObject* module = PyModule_Create(&Normalization_module);
    if (module == NULL) {
        return NULL;
    }

    // Add the class to the module
    Py_INCREF(&MinMaxScalerType);
    PyModule_AddObject(module, "MinMaxScaler", (PyObject*)&MinMaxScalerType);

    // Return the module
    return module;
}

