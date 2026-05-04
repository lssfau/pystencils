import numpy as np
import pytest

from pystencils import (
    Assignment, AssignmentCollection, CreateKernelConfig, Field, FieldType, Target, create_kernel)


def test_indexed_kernel(target, xp):
    if target == Target.SYCL:
        pytest.skip("Complex dtypes with different types are not supported by dpctl")
    arr = xp.zeros((3, 4))
    dtype = np.dtype([('x', int), ('y', int), ('value', arr.dtype)], align=True)
    
    cpu_index_arr = np.zeros((3,), dtype=dtype)
    cpu_index_arr[0] = (0, 2, 3.0)
    cpu_index_arr[1] = (1, 3, 42.0)
    cpu_index_arr[2] = (2, 1, 5.0)

    if target.is_gpu() or target == Target.SYCL:
        gpu_index_arr = xp.empty(cpu_index_arr.shape, dtype=cpu_index_arr.dtype)
        gpu_index_arr.set(cpu_index_arr)
        index_arr = gpu_index_arr
    else:
        index_arr = cpu_index_arr

    index_field = Field.create_from_numpy_array('index', index_arr, field_type=FieldType.INDEXED)
    normal_field = Field.create_from_numpy_array('f', arr)
    update_rule = AssignmentCollection([
        Assignment(normal_field[0, 0], index_field('value'))
    ])

    options = CreateKernelConfig(index_field=index_field, target=target)
    ast = create_kernel(update_rule, options)
    kernel = ast.compile()

    kernel(f=arr, index=index_arr)

    if target.is_gpu() or target == Target.SYCL:
        arr = xp.asnumpy(arr)

    for i in range(cpu_index_arr.shape[0]):
        np.testing.assert_allclose(arr[cpu_index_arr[i]['x'], cpu_index_arr[i]['y']], cpu_index_arr[i]['value'], atol=1e-13)
