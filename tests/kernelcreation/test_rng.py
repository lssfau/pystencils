import pystencils as ps
import numpy as np
import sympy as sp
import pytest
import pickle

try:
    import randomgen
except ImportError:
    pytest.skip("randomgen not available", allow_module_level=True)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("target", ps.Target.available_targets())
@pytest.mark.parametrize(
    "offsets",
    [
        (0, 0),
        (12, 41),
        (
            ps.TypedSymbol("o0", ps.DynamicType.INDEX_TYPE),
            ps.TypedSymbol("o1", ps.DynamicType.INDEX_TYPE),
        ),
    ],
)
@pytest.mark.parametrize("c_value", [0, 41])
def test_philox(dtype, gen_config, xp, offsets, c_value):
    seed = 0x7EADBEEF
    c = ps.TypedSymbol("c", ps.DynamicType.INDEX_TYPE)

    rng = ps.random.Philox("philox", dtype, seed=seed, offsets=offsets)
    q = rng.vector_size

    target = gen_config.get_target()

    layout = "numpy" if target == ps.Target.SYCL else "fzyx"
    f, g = ps.fields(f"f({q}), g({q}): {dtype}[2D]", layout=layout)

    #   First invocation, write to f
    rx1, rasm1 = rng.get_random_vector(c)
    asms = [rasm1] + [ps.Assignment(f(i), rx1[i]) for i in range(q)]

    #   Second invocation, write to g
    rx2, rasm2 = rng.get_random_vector(c)
    asms += [rasm2] + [ps.Assignment(g(i), rx2[i]) for i in range(q)]

    cfg = gen_config.copy()
    cfg.default_dtype = dtype
    cfg.index_dtype = f"int{cfg.default_dtype.width}"

    ker = ps.create_kernel(asms, cfg).compile()

    shape = (12, 12)
    if target == ps.Target.SYCL:
        f_arr = xp.zeros(shape + (q,), dtype=dtype)
    else:
        layout_tuple = ps.field.layout_string_to_tuple(layout, 3)
        f_arr = ps.field.create_numpy_array_with_layout(
            shape + (q,), layout_tuple, dtype=dtype, xp=xp
        )

    f_arr[:] = np.dtype(dtype).type(0)
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

    f_reference = xp.array(get_reference(0))
    g_reference = xp.array(get_reference(1))

    xp.testing.assert_allclose(f_arr, f_reference, rtol=0, atol=eps)
    xp.testing.assert_allclose(g_arr, g_reference, rtol=0, atol=eps)


def test_compare_and_pickle():
    rng = ps.random.Philox(
        "phil", "float32", ps.TypedSymbol("seed", "uint32"), (13, 41, 2)
    )
    t = ps.TypedSymbol("t", "uint32")
    rx, rasm = rng.get_random_vector(t)

    assert isinstance(rasm.rhs, sp.Function)

    rasm_dump = pickle.dumps(rasm)

    rasm_reconstructed = pickle.loads(rasm_dump)
    assert rasm_reconstructed == rasm
    assert isinstance(rasm_reconstructed.rhs.state, ps.random.Philox.PhiloxState)
    assert isinstance(rasm_reconstructed.rhs, sp.Function)
    assert rasm_reconstructed.rhs.state == rasm.rhs.state

    rng2 = ps.random.Philox(
        "phil", "float32", ps.TypedSymbol("seed", "uint32"), (13, 41, 2)
    )
    rx2, rasm2 = rng2.get_random_vector(t)

    assert rx2 == rx
    assert rasm2 == rasm
    assert rasm2 == rasm_reconstructed

    rx3, rasm3 = rng2.get_random_vector(t)
    assert rx3 != rx2
    assert rasm3 != rasm2


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "target", [t for t in ps.Target.available_targets() if not t.is_vector_cpu()]
)
@pytest.mark.parametrize(
    "periodicity",
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_philox_periodicity(dtype, gen_config, xp, periodicity):
    """Test that periodic dimensions produce matching counters at wrapped positions.

    We create a kernel with ghost_layers=1.  The iteration counters run from 1
    to shape-1 (i.e. the inner domain).  With periodicity enabled for a
    dimension of inner size N, the counter fed into Philox is
    ``int_rem(ctr + offset, N)`` instead of ``ctr + offset``.

    To verify this we compare the compiled kernel output against a reference
    computed with ``randomgen.Philox`` using the same wrapped counters.
    """

    seed = 0xBEEFCAFE
    c = ps.TypedSymbol("c", ps.DynamicType.INDEX_TYPE)
    c_value = 7

    inner_shape = (10, 8)
    gl = 1
    # The field (including ghost layers) has this total shape
    total_shape = tuple(s + 2 * gl for s in inner_shape)

    # num_cells = inner domain size per dimension; only needed for periodic dims
    num_cells = tuple(
        s if p else None for s, p in zip(inner_shape, periodicity)
    )

    rng = ps.random.Philox(
        "philox", dtype, seed=seed,
        periodicity=periodicity,
        num_cells=num_cells,
    )
    q = rng.vector_size

    f = ps.fields(f"f({q}): {dtype}[2D]")

    rx, rasm = rng.get_random_vector(c)
    asms = [rasm] + [ps.Assignment(f(i), rx[i]) for i in range(q)]

    gen_config.ghost_layers = gl
    ker = ps.create_kernel(asms, gen_config).compile()

    f_arr = xp.zeros(total_shape + (q,), dtype=dtype)
    ker(f=f_arr, c=c_value)

    if gen_config.get_target().is_gpu():
        f_arr = f_arr.get()
    elif gen_config.get_target() == ps.Target.SYCL:
        import dpctl.tensor as dpt
        f_arr = dpt.asnumpy(f_arr)

    # Build reference: iterate over the inner domain (counters gl..total-gl)
    inner = f_arr[gl:-gl, gl:-gl, :]
    ref = np.empty(inner_shape + (4,), dtype=int)

    for x in range(inner_shape[0]):
        for y in range(inner_shape[1]):
            # The counter in the kernel is ctr_d which equals gl + x/y
            cx = gl + x
            cy = gl + y

            # Apply the same periodicity wrapping as get_counters
            if periodicity[0]:
                cx = cx % inner_shape[0]
            if periodicity[1]:
                cy = cy % inner_shape[1]

            def cast(v):
                return int(np.uint32(np.int32(v)))

            counter = cast(c_value) + cast(cx) * 2**32 + cast(cy) * 2**64 - 1
            counter = (2**128 + counter) & (2**128 - 1)

            keys = (seed, 0)
            philox = randomgen.Philox(
                counter=counter,
                key=keys[0] + keys[1] * 2**32,
                number=4,
                width=32,
            )
            ref[x, y, :] = philox.random_raw(size=4)

    reference = np.empty(inner_shape + (q,), dtype=dtype)
    match dtype:
        case "float32":
            reference[:] = ref * 2.0**-32 + 2.0**-33
        case "float64":
            a = ref[:, :, 0::2]
            b = ref[:, :, 1::2]
            reference[:] = (a ^ b << (53 - 32)) * 2.0**-53 + 2.0**-54

    eps = np.finfo(np.dtype(dtype)).eps
    np.testing.assert_allclose(inner, reference, rtol=0, atol=eps)

    # --- Verify ghost layers match their periodic inner counterparts ---
    # Run a second kernel with ghost_layers=0 so ALL cells (including ghost
    # layer positions) are written.  Then check that each ghost cell produced
    # the same random number as the inner cell it wraps to.
    gen_config_full = ps.CreateKernelConfig(target=gen_config.get_target())
    if gen_config.get_target().is_cpu():
        gen_config_full.jit = gen_config.jit
    gen_config_full.ghost_layers = 0

    rng_full = ps.random.Philox(
        "philox", dtype, seed=seed,
        periodicity=periodicity,
        num_cells=num_cells,
    )
    q_full = rng_full.vector_size
    f_full = ps.fields(f"f_full({q_full}): {dtype}[2D]")

    rx_f, rasm_f = rng_full.get_random_vector(c)
    asms_full = [rasm_f] + [ps.Assignment(f_full(i), rx_f[i]) for i in range(q_full)]

    ker_full = ps.create_kernel(asms_full, gen_config_full).compile()
    f_full_arr = xp.zeros(total_shape + (q_full,), dtype=dtype)
    ker_full(f_full=f_full_arr, c=c_value)

    if gen_config.get_target().is_gpu():
        f_full_arr = f_full_arr.get()
    elif gen_config.get_target() == ps.Target.SYCL:
        import dpctl.tensor as dpt
        f_full_arr = dpt.asnumpy(f_full_arr)

    if periodicity[0]:
        N0 = inner_shape[0]
        # Left ghost (pos 0) wraps to counter 0 % N0 = 0,
        # same as last inner cell (pos N0) whose counter is N0 % N0 = 0
        np.testing.assert_allclose(
            f_full_arr[0, gl:-gl, :], f_full_arr[N0, gl:-gl, :],
            rtol=0, atol=eps,
            err_msg="Periodic dim 0: left ghost != last inner cell",
        )
        # Right ghost (pos N0+1) wraps to counter (N0+1) % N0 = 1,
        # same as first inner cell (pos 1) whose counter is 1 % N0 = 1
        np.testing.assert_allclose(
            f_full_arr[N0 + 1, gl:-gl, :], f_full_arr[1, gl:-gl, :],
            rtol=0, atol=eps,
            err_msg="Periodic dim 0: right ghost != first inner cell",
        )

    if periodicity[1]:
        N1 = inner_shape[1]
        # Left ghost (pos 0) wraps to counter 0 % N1 = 0,
        # same as last inner cell (pos N1) whose counter is N1 % N1 = 0
        np.testing.assert_allclose(
            f_full_arr[gl:-gl, 0, :], f_full_arr[gl:-gl, N1, :],
            rtol=0, atol=eps,
            err_msg="Periodic dim 1: left ghost != last inner cell",
        )
        # Right ghost (pos N1+1) wraps to counter (N1+1) % N1 = 1,
        # same as first inner cell (pos 1) whose counter is 1 % N1 = 1
        np.testing.assert_allclose(
            f_full_arr[gl:-gl, N1 + 1, :], f_full_arr[gl:-gl, 1, :],
            rtol=0, atol=eps,
            err_msg="Periodic dim 1: right ghost != first inner cell",
        )


def test_compare_and_pickle_periodic():
    N = sp.Symbol("N", integer=True, positive=True)
    rng = ps.random.Philox(
        "phil", "float32", ps.TypedSymbol("seed", "uint32"),
        offsets=(13, 41, 2),
        periodicity=(True, False, True),
        num_cells=(N, None, 100),
    )
    t = ps.TypedSymbol("t", "uint32")
    rx, rasm = rng.get_random_vector(t)

    assert isinstance(rasm.rhs, sp.Function)

    rasm_dump = pickle.dumps(rasm)
    rasm_reconstructed = pickle.loads(rasm_dump)
    assert rasm_reconstructed == rasm
    assert isinstance(rasm_reconstructed.rhs.state, ps.random.Philox.PhiloxState)
    assert rasm_reconstructed.rhs.state == rasm.rhs.state
    assert rasm_reconstructed.rhs.state.periodicity == (True, False, True)
    assert rasm_reconstructed.rhs.state.num_cells == (N, None, 100)

    # Identical construction must yield equal states
    rng2 = ps.random.Philox(
        "phil", "float32", ps.TypedSymbol("seed", "uint32"),
        offsets=(13, 41, 2),
        periodicity=(True, False, True),
        num_cells=(N, None, 100),
    )
    rx2, rasm2 = rng2.get_random_vector(t)
    assert rasm2 == rasm

    # Different periodicity must yield different states
    rng3 = ps.random.Philox(
        "phil", "float32", ps.TypedSymbol("seed", "uint32"),
        offsets=(13, 41, 2),
        periodicity=(False, False, False),
    )
    rx3, rasm3 = rng3.get_random_vector(t)
    assert rasm3 != rasm


def test_philox_periodicity_validation():
    """Specifying periodicity=True without num_cells must raise."""
    with pytest.raises(ValueError, match="num_cells must be specified"):
        ps.random.Philox("phil", "float32", 42, periodicity=(True,))

    with pytest.raises(ValueError, match="num_cells must be specified"):
        ps.random.Philox(
            "phil", "float32", 42,
            periodicity=(False, True),
            num_cells=(10, None),
        )
