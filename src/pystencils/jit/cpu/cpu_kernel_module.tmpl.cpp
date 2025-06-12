#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/ndarrayobject.h>

#include <array>
#include <vector>
#include <string>
#include <sstream>
#include <utility>
#include <initializer_list>

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
 * RAII proxy for a NumPy array object.
 * Supports move, but not copy construction.
 */
struct ArrayProxy {
private:
    //  Don't forget to adapt move constructor / assignment!
    PyArrayObject * arr_ { nullptr }; // owned by this instance -> decref in destructor
    size_t itemsize_;
    std::string name_;

    ArrayProxy(PyArrayObject * array, size_t itemsize, const std::string& name) : arr_{array}, itemsize_{itemsize}, name_{name} {}

public:
    static ArrayProxy fromPyObject(const std::string& name, PyObject * obj, int ndim, int typeno, size_t itemsize){
        if(!PyArray_Check(obj)){
            throw KernelModuleError { PyExc_TypeError, "Invalid array argument" };
        }
        auto array_object = reinterpret_cast< PyArrayObject * >(PyArray_FromArray(reinterpret_cast< PyArrayObject * >(obj), NULL, 0));

        ArrayProxy proxy { array_object, itemsize, name };

        if( PyArray_TYPE(proxy.arr_) != typeno){
            std::stringstream err;
            err << "Invalid element type of array argument " << name;
            throw KernelModuleError { PyExc_TypeError, err.str() };
        }

        return std::move(proxy);
    }

    ArrayProxy(const ArrayProxy &) = delete;
    ArrayProxy(ArrayProxy && other) : arr_( std::exchange(other.arr_, nullptr) ), itemsize_{ other.itemsize_ } {}

    ArrayProxy& operator=(const ArrayProxy &) = delete;
    ArrayProxy& operator=(ArrayProxy && other) {
        std::swap(arr_, other.arr_);
        this->itemsize_ = other.itemsize_;
        return *this;
    }

    ~ArrayProxy() {
        Py_XDECREF(arr_);
    }

    const std::string& name() const {
        return name_;
    }

    template< typename T >
    T * data() {
        T * ptr = (T*) PyArray_DATA(arr_);
        return ptr;
    }

    size_t ndim() const {
        return static_cast< size_t >(PyArray_NDIM(arr_));
    } 

    template< typename index_type = ssize_t >
    index_type shape(size_t c) const {
        return static_cast< index_type >(PyArray_DIM(arr_, c));
    }

    template< typename index_type = ssize_t >
    index_type stride(size_t c) const {
        return static_cast< index_type >(PyArray_STRIDE(arr_, c) / itemsize_);
    }
};


template< typename T, int TYPENO >
T scalarFromPyObject(PyObject * obj, std::string name = ""){
    //  obj must be a NumPy or Python scalar
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

void checkFieldShape(const std::string& expected, const ArrayProxy& arr, size_t coord, ssize_t desired) {
    if(arr.ndim() <= coord || arr.shape(coord) != desired){
        std::stringstream err;
        err << "Invalid shape of array argument '" << arr.name()
            << "': expected " << expected << ".";
        throw KernelModuleError{ PyExc_ValueError, err.str() };
    }
}

void checkFieldStride(const std::string& expected, const ArrayProxy& arr, size_t coord, ssize_t desired) {
    if(arr.ndim() <= coord || arr.stride(coord) != desired){
        std::stringstream err;
        err << "Invalid stride of array argument '" << arr.name()
            << "': expected " << expected << ".";
        throw KernelModuleError{ PyExc_ValueError, err.str() };
    }
}

void checkTrivialIndexShape(const std::string& expected, const ArrayProxy& arr, size_t spatial_rank) {
    const size_t ndim = arr.ndim();
    if(ndim > spatial_rank){
        for(size_t c = spatial_rank; c < ndim; ++c){
            if(arr.shape(c) != 1) {
                std::stringstream err;
                err << "Invalid shape of array argument '" << arr.name()
                    << "': expected " << expected << ".";
                throw KernelModuleError{ PyExc_ValueError, err.str() };
            }
        }
    }
}


void checkSameShape(std::initializer_list< const ArrayProxy * > arrays, size_t spatial_dims) {
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
            for(size_t c = 0; c < spatial_dims; ++c){
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


struct KernelArgs_${kernel_name} {
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
    }
};

extern "C"
{
    static PyObject *
    invoke(PyObject *module, PyObject *posargs, PyObject * kwargs)
    {
        try {
            KernelArgs_${kernel_name} kernel_args { posargs, kwargs };
            internal::${kernel_name} ( ${kernel_invocation_args} );
        } catch (const KernelModuleError & err) {
            PyErr_SetString(err.exception, err.msg.c_str());
            return NULL;
        } catch (const PythonError& err) {
            //   Error condition was set by Python API - nothing to do
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

        return PyModule_Create(&Module);
    }
}