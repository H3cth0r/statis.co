#include <Python.h>
#include "utils/TrainTestSplit.c"

static PyMethodDef methods[] = {
  {"TrainTestSplit", (PyCFunction)TrainTestSplit, METH_VARARGS | METH_KEYWORDS, "Custom train-test split function"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef utils = {
  PyModuleDef_HEAD_INIT,
  "utils",
  "This is the utils module",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_utils() {
  import_array();
  PyObject *module = PyModule_Create(&utils);
  return module;
};
