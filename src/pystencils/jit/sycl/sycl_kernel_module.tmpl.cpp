#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <sycl/sycl.hpp>
#include <Python.h>
#include "dpctl_capi.h"
#include "dpctl_sycl_interface.h"

#include <numpy/ndarrayobject.h>

#include <array>
#include <vector>
#include <string>
#include <sstream>
#include <utility>
#include <initializer_list>
#include <cstdint>
#include <type_traits>

${includes}

#define RESTRICT ${restrict_qualifier}

namespace internal {

${kernel_definition}

}

struct PythonError {};

struct KernelModuleError {
    PyObject * exception;
    std::string msg;
};


/**
 * Proxy with an Pointer to USM array object.
 */
struct ArrayProxy {
private:
    PyUSMArrayObject *arr_ { nullptr };
    npy_intp itemsize_;
    std::string name_;

    ArrayProxy(PyUSMArrayObject * array, npy_intp itemsize, const std::string& name) : arr_{array}, itemsize_{itemsize}, name_{name} {}

public:
    static ArrayProxy fromPyObject(const std::string& name, PyObject * obj, int ndim, int typeno, npy_intp itemsize){
        if(!PyObject_TypeCheck(obj, &PyUSMArrayType)){
            throw KernelModuleError { PyExc_TypeError, "Invalid array argument" };
        }
        auto array_object = reinterpret_cast< PyUSMArrayObject * >(obj);

        ArrayProxy proxy { array_object, itemsize, name };

        if( UsmNDArray_GetTypenum(proxy.arr_) != typeno){
            std::stringstream err;
            err << "Invalid element type of array argument " << name;
            throw KernelModuleError { PyExc_TypeError, err.str() };
        }

        return proxy;
    }

    const std::string& name() const {
        return name_;
    }

    template< typename T >
    T * data() {
        T * ptr = (T*) UsmNDArray_GetData(arr_);
        return ptr;
    }

    int ndim() const {
        return UsmNDArray_GetNDim(arr_);
    }

    template< typename index_type = ssize_t >
    index_type shape(int c) const {
        return static_cast< index_type >(UsmNDArray_GetShape(arr_)[c]);
    }
    
    bool is_c_contiguous() const {
        int flags = UsmNDArray_GetFlags(arr_);
        return static_cast<bool>(flags & USM_ARRAY_C_CONTIGUOUS);
    }

    template< typename index_type = ssize_t >
    index_type stride(int c) const {
        Py_ssize_t *strides = UsmNDArray_GetStrides(arr_);
        // returns null if c-or f-contiguous
        if (strides != NULL){
            return static_cast< index_type >(strides[c]/itemsize_);
        }
        index_type ret {1};
        if (is_c_contiguous()) {
            for (int d = c + 1; d < ndim(); d++){
                ret *= shape<index_type>(d);
            }
        } else {
            for (int d = 0; d < c; d++){
                ret *= shape<index_type>(d);
            }
        }
        return ret;
    }

    sycl::queue *get_queue() const {
        DPCTLSyclQueueRef q_ptr =  UsmNDArray_GetQueueRef(arr_);
        return reinterpret_cast<sycl::queue *>(q_ptr);
    }
};


template< typename T, int TYPENO >
T scalarFromPyObject(PyObject * obj, std::string name = ""){
    // obj must be a NumPy or Python scalar
    if(!PyArray_IsAnyScalar(obj)){
        std::stringstream err;
        err << "Invalid type of scalar kernel argument " << name;
        throw KernelModuleError { PyExc_TypeError, err.str() };
    }

    //  Convert the given object to the desired NumPy array scalar type
    PyArray_Descr * dtype = PyArray_DescrFromType(TYPENO);
    PyObject * arrayScalar = PyObject_CallOneArg((PyObject *) dtype->typeobj, obj);
    Py_DECREF(dtype);

    //  Check if cast was successful
    if( arrayScalar == NULL ){
        throw PythonError{};
    }

    //  Extract as C type
    T val;
    PyArray_ScalarAsCtype(arrayScalar, &val);
    Py_DECREF(arrayScalar);
    return val;
}

void checkFieldShape(const std::string& expected, const ArrayProxy& arr, int coord, ssize_t desired) {
    if(arr.ndim() <= coord || arr.shape(coord) != desired){
        std::stringstream err;
        err << "Invalid shape of array argument '" << arr.name()
            << "': expected " << expected << ".";
        throw KernelModuleError{ PyExc_ValueError, err.str() };
    }
}

void checkFieldStride(const std::string& expected, const ArrayProxy& arr, int coord, ssize_t desired) {
    if(arr.ndim() <= coord || arr.stride(coord) != desired){
        std::stringstream err;
        err << "Invalid stride of array argument '" << arr.name()
            << "': expected " << expected << ".";
        throw KernelModuleError{ PyExc_ValueError, err.str() };
    }
}

void checkTrivialIndexShape(const std::string& expected, const ArrayProxy& arr, int spatial_rank) {
    const int ndim = arr.ndim();
    if(ndim > spatial_rank){
        for(int c = spatial_rank; c < ndim; ++c){
            if(arr.shape(c) != 1) {
                std::stringstream err;
                err << "Invalid shape of array argument '" << arr.name()
                    << "': expected " << expected << ".";
                throw KernelModuleError{ PyExc_ValueError, err.str() };
            }
        }
    }
}


void checkSameShape(std::initializer_list< const ArrayProxy * > arrays, int spatial_dims) {
    const ArrayProxy * fst { nullptr };
    for(const auto arr : arrays) {
        if(arr->ndim() < spatial_dims) {
            std::stringstream err;
            err << "Invalid dimensionality of array argument '" << arr->name()
                << "'. Expected " << spatial_dims;
            throw KernelModuleError{ PyExc_ValueError, err.str() };
        }

        if(fst == nullptr) {
            fst = arr;
        } else {
            for(int c = 0; c < spatial_dims; ++c){
                if(fst->shape(c) != arr->shape(c)){
                    throw KernelModuleError{
                        PyExc_ValueError,
                        "Incompatible shapes of array arguments: "
                        "All domain field arrays must have the same spatial shape."
                    };
                }
            }
        }
    }
}

void checkQueue(sycl::queue *q, sycl::queue *array_q){
    if (*q != *array_q)
    {
        throw KernelModuleError{ PyExc_ValueError, "Invalid Queue"};
    }

}


PyObject * getKwarg(PyObject * kwargs, const std::string& key) {
    PyObject * keyUnicode = PyUnicode_FromString(key.c_str());
    PyObject * obj = PyDict_GetItemWithError(kwargs, keyUnicode);
    Py_DECREF(keyUnicode);

    if(obj == NULL) {
        if( PyErr_Occurred() ){
            throw PythonError {};
        } else {
            std::stringstream err;
            err << "Missing kernel argument: " << key;
            throw KernelModuleError{ PyExc_KeyError, err.str() };
        }
    }

    return obj;
}


PyObject * getQueueFromKwargs(PyObject * kwargs) {
    const std::string queue_key {"queue"};
    try {
        return getKwarg(kwargs, queue_key);
    } catch (const KernelModuleError & e) {
        return nullptr;
    }
}


template<ssize_t dim, typename = std::enable_if_t<(dim >= 1 && dim <= 3)>>
sycl::range<dim> getRangefromPyObject(PyObject *kwargs, const std::string& name){
    PyObject *obj = getKwarg(kwargs, name);
    if (!PyTuple_Check(obj)) {
        throw KernelModuleError { PyExc_TypeError, "Invalid tuple argument" };
    }

    std::array<size_t, dim> dims;
    for (ssize_t d = 0; d < dim; d++){
        PyObject* item = PyTuple_GetItem(obj, d);
        if (!PyLong_Check(item)){
            throw KernelModuleError { PyExc_TypeError, "Invalid type of range entry" };
        }
        auto entry = PyLong_AsLong(item);
        if (entry < 0){
            throw KernelModuleError { PyExc_ValueError, "Invalid range entry: must be positive" };
        }
        dims[static_cast<size_t>(d)] = static_cast<size_t>(entry);
    }

    if constexpr (dim == 1){
        return sycl::range<dim>(dims[0]);
    }
    else if constexpr (dim == 2){
        return sycl::range<dim>(dims[0], dims[1]);
    }
    else if constexpr (dim == 3){
        return sycl::range<dim>(dims[0], dims[1], dims[2]);
    }

}


struct KernelArgs_${kernel_name} {
    sycl::queue *q{nullptr};
${argstruct_members}

    KernelArgs_${kernel_name} (PyObject * posargs, PyObject * kwargs){

        //  Extract borrowed references to required kwargs
${kernel_kwarg_refs}

        //  Convert arrays to ArrayProxy
${array_proxy_defs}

        //  Extract scalar kernel arguments
${extract_kernel_args}

        //  Check preconditions
${precondition_checks}

        PyObject *q_ptr = getQueueFromKwargs(kwargs);

        if (q_ptr != nullptr) {
            if(!PyObject_TypeCheck(q_ptr, &PySyclQueueType)) {
                throw KernelModuleError { PyExc_TypeError, "Invalid queue argument" };
            }
            PySyclQueueObject *q_obj = reinterpret_cast<PySyclQueueObject *>(q_ptr);

            q = reinterpret_cast<sycl::queue *>(SyclQueue_GetQueueRef(q_obj));
        } else {
            q = ${array_queue};
        }


        // Check if queue is the right one
${queue_checks}
    }
};

extern "C"
{
    static PyObject *
    invoke(PyObject *module, PyObject *posargs, PyObject * kwargs)
    {
        try {
            KernelArgs_${kernel_name} kernel_args { posargs, kwargs };

${launch_config_args}

            kernel_args.q->wait();
            kernel_args.q->submit([${kernel_lambda_caption}](sycl::handler &cgh) {
                cgh.parallel_for<class ${kernel_class_name}>( ${sycl_range} (${sycl_range_arg}), [=](${sycl_item}) {
                    internal::${kernel_name} ( ${kernel_invocation_args} );
                });
            });
            kernel_args.q->wait();
        } catch (const KernelModuleError & err) {
            PyErr_SetString(err.exception, err.msg.c_str());
            return NULL;
        } catch (const PythonError& err) {
            //   Error condition was set by Python API - nothing to do
            return NULL;
        } catch (sycl::exception const& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            return NULL;
        }

        Py_RETURN_NONE;
    }

    static PyMethodDef ModuleMethods[] = {
        {"invoke", (PyCFunction)(void(*)(void)) invoke, METH_VARARGS | METH_KEYWORDS, "Invoke the kernel"},
        {NULL, NULL, 0, NULL}};

    static PyModuleDef Module = {
        PyModuleDef_HEAD_INIT,
        "${module_name}",
        NULL,
        -1,
        ModuleMethods
    };

    PyMODINIT_FUNC
    PyInit_${module_name} (void){
        import_array();
        import_dpctl();

        return PyModule_Create(&Module);
    }
}
