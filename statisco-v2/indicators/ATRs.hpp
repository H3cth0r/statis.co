#pragma once


inline double* createResultArray(const NumpyArrayRef& input, NumpyArrayRef& output) {
    const npy_intp size = input.size();
    npy_intp dims[] = {size};
    PyObject* result_obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!result_obj) {
        throw std::runtime_error("Failed to create output array");
    }
    PyArrayObject* result_arr = reinterpret_cast<PyArrayObject*>(result_obj);
    output = NumpyArrayRef(result_arr);
    return static_cast<double*>(PyArray_DATA(result_arr));
}

NumpyArrayRef ATR(NumpyArrayRef& Close_t_obj, NumpyArrayRef& High_t_obj, NumpyArrayRef& Low_t_obj) {
    NumpyArrayRef atr;
    double* result_data = createResultArray(Close_t_obj, atr);

    return atr;
}
