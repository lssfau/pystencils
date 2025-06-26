import pystencils as ps
import numpy as np
import pytest

try:
    import randomgen
except ImportError:
    pytest.skip("randomgen not available", allow_module_level=True)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "target", [t for t in ps.Target.available_targets() if not t.is_vector_cpu()]
)
@pytest.mark.parametrize(
    "offsets",
    [
        (0, 0),
        (12, 41),
        (ps.TypedSymbol("o0", "int64"), ps.TypedSymbol("o1", "int64")),
    ],
)
@pytest.mark.parametrize("c_value", [0, 41])
def test_philox(dtype, gen_config, xp, offsets, c_value):
    seed = 0xFEEDBEAF
    c = ps.TypedSymbol("c", "uint32")

    rng = ps.random.Philox("philox", dtype, seed=seed, offsets=offsets)
    q = rng.vector_size

    f, g = ps.fields(f"f({q}), g({q}): {dtype}[2D]")

    #   First invocation, write to f
    rx1, rasm1 = rng.get_random_vector(c)
    asms = [rasm1] + [ps.Assignment(f(i), rx1[i]) for i in range(q)]

    #   Second invocation, write to g
    rx2, rasm2 = rng.get_random_vector(c)
    asms += [rasm2] + [ps.Assignment(g(i), rx2[i]) for i in range(q)]

    ker = ps.create_kernel(asms, gen_config).compile()

    shape = (12, 12)
    f_arr = xp.zeros(shape + (q,), dtype=dtype)
    g_arr = xp.zeros_like(f_arr)

    if isinstance(offsets[0], ps.TypedSymbol):
        offset_args = {"o0": 15, "o1": -3}
        actual_offsets = (offset_args["o0"], offset_args["o1"])
    else:
        offset_args = dict()
        actual_offsets = offsets

    ker(f=f_arr, g=g_arr, c=c_value, **offset_args)

    def get_reference(invocation_key):
        int_reference = np.empty(shape + (4,), dtype=int)

        for x in range(shape[0]):
            for y in range(shape[1]):
                cx = x + actual_offsets[0]
                cy = y + actual_offsets[1]

                def cast(v):
                    return int(np.uint32(np.int32(v)))

                #   Reflect counter construction of implementation exactly
                counter = cast(c_value) + cast(cx) * 2**32 + cast(cy) * 2**64 - 1
                #   128-bit wrap-around if negative
                counter = (2**128 + counter) & (2**128 - 1)
                
                keys = (seed, invocation_key)
                philox = randomgen.Philox(
                    counter=counter,
                    key=keys[0] + keys[1] * 2**32,
                    number=4,
                    width=32,
                )
                int_reference[x, y, :] = philox.random_raw(size=4)

        reference = np.empty(shape + (q,), dtype=dtype)

        match dtype:
            case "float32":
                reference[:] = int_reference * 2.0**-32 + 2.0**-33
            case "float64":
                x = int_reference[:, :, 0::2]
                y = int_reference[:, :, 1::2]
                z = x ^ y << (53 - 32)
                reference[:] = z * 2.0**-53 + 2.0**-54

        return reference

    eps = np.finfo(np.dtype(dtype)).eps

    if gen_config.get_target().is_gpu():
        f_arr = f_arr.get()
        g_arr = g_arr.get()

    f_reference = get_reference(0)
    g_reference = get_reference(1)

    np.testing.assert_allclose(f_arr, f_reference, rtol=0, atol=eps)
    np.testing.assert_allclose(g_arr, g_reference, rtol=0, atol=eps)
