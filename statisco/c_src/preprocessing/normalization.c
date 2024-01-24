#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "MinMaxScaler.c"


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

