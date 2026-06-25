import dpctl
import dpctl.tensor as dpt
import numpy as np
from dpctl.tensor import *

__all__ = dpt.__all__  # Makes wildcard imports work as expected

# When an Nvidia / AMD GPU is used via the codeplay plugin, the standard functions like sin, cos, etc
# are not necessarily implemented, further some of these generic tests use fp64 which also not avaialbe on all GPUs
# so this enforces to used the cpu for all tests that use this compability layer
_cpu_queue = dpctl.SyclQueue("cpu")

def set_cpu_queue(func):
    def wrapper(*args, **kwargs):
        kwargs_new = kwargs.copy()
        kwargs_new["sycl_queue"] = _cpu_queue
        return func(*args, **kwargs_new)
    return wrapper


class TestingNamespace:
    @staticmethod
    def assert_array_equal(x, y, *args, **kwargs):
        x_np = dpt.asnumpy(x) if isinstance(x, usm_ndarray) else x
        y_np = dpt.asnumpy(y) if isinstance(y, usm_ndarray) else y
        np.testing.assert_array_equal(x_np, y_np, *args, **kwargs)

    @staticmethod
    def assert_allclose(x, y, *args, **kwargs):
        x_np = dpt.asnumpy(x) if isinstance(x, usm_ndarray) else x
        y_np = dpt.asnumpy(y) if isinstance(y, usm_ndarray) else y
        np.testing.assert_allclose(x_np, y_np, *args, **kwargs)


class NdarrayNamespace:
    @staticmethod
    def astype(*args, **kwargs):
        return dpt.astype(*args, **kwargs)


def asarray(array: dpt.usm_ndarray) -> np.ndarray:
    """ Converts dpt usm_array to an np.array.
        This is need form Field.create_from_numpy_array,
        as the strides attribute is not in dpt.usm_array not in byte but in sizeof(dtype)
    """
    return dpt.asnumpy(array)


def array(*args, **kwargs):
    np_array = np.array(*args, **kwargs)
    return dpt.from_numpy(np_array, sycl_queue=_cpu_queue)


def arcsin(*args, **kwargs):
    return dpt.asin(*args, **kwargs)


def arccos(*args, **kwargs):
    return dpt.acos(*args, **kwargs)


def arctan(*args, **kwargs):
    return dpt.atan(*args, **kwargs)

def arctan2(*args, **kwargs):
    return dpt.atan2(*args, **kwargs)


def fmin(*args, **kwargs):
    return dpt.minimum(*args, **kwargs)


def fmax(*args, **kwargs):
    return dpt.maximum(*args, **kwargs)


def power(*args, **kwargs):
    return dpt.pow(*args, **kwargs)

# override array creation functions
arange = set_cpu_queue(dpt.arange)
copy = set_cpu_queue(dpt.copy)
empty = set_cpu_queue(dpt.empty)
empty_like = set_cpu_queue(dpt.empty_like)
eye = set_cpu_queue(dpt.eye)
full = set_cpu_queue(dpt.full)
full_like = set_cpu_queue(dpt.full_like)
ones = set_cpu_queue(dpt.ones)
ones_like = set_cpu_queue(dpt.ones_like)
zeros = set_cpu_queue(dpt.zeros)
zeros_like = set_cpu_queue(dpt.zeros_like)

testing = TestingNamespace()
ndarray = NdarrayNamespace()
