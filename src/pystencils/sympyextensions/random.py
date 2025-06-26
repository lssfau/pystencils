from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Sequence

import sympy as sp

from ..assignment import Assignment
from .typed_sympy import TypedSymbol, tcast
from ..types import UserTypeSpec, create_type, PsIeeeFloatType, PsNamedArrayType


class RngBase(ABC):
    """Abstract base class for counter-based random number generators.

    Args:
        name: Name of the RNG instance
        dtype: Data type of the generated random numbers. Must be a floating-point type.
    """

    class RngFunc(sp.core.function.AppliedUndef):
        rng: RngBase
        """RNG instance this function belongs to"""

        invocation_key: int
        """Key identifying this invocation of the RNG"""

    @classmethod
    @abstractmethod
    def _get_vector_type(cls, dtype: PsIeeeFloatType) -> PsNamedArrayType: ...

    def __init__(
        self,
        name: str,
        dtype: UserTypeSpec,
    ):
        dtype = create_type(dtype)

        if not isinstance(dtype, PsIeeeFloatType):
            raise ValueError("dtype must be a floating-point type")

        self._name = name
        self._dtype = dtype

        self._vector_type = self._get_vector_type(self._dtype)

    @property
    def dtype(self) -> PsIeeeFloatType:
        """Data type of the random numbers"""
        return self._dtype

    @property
    def vector_size(self) -> int:
        """Number of random numbers returned by a single invocation"""
        return self._vector_type.shape[0]

    @abstractmethod
    def get_keys(self, invocation: sp.Expr) -> tuple[sp.Expr | int, ...]:
        """Return the key tuple identifying the given RNG invocation's number stream.

        Args:
            invocation: Expression that is a call to this RNG's generator function

        .. note::
            Must match the ``key`` arguments of the underlying RNG engine;
            see `pystencils.backend.functions.PsRngEngineFunction`.
        """

    @abstractmethod
    def get_counters(self, invocation: sp.Expr, rank: int) -> tuple[sp.Expr | int, ...]:
        """Return the symbolic counter tuple identifying the position in the RNG's number stream
        of the given invocation.

        Args:
            invocation: Expression that is a call to this RNG's generator function
            rank: Rank of the kernel's iteration space

        .. note::
            Must match the ``counter`` arguments of the underlying RNG engine;
            see `pystencils.backend.functions.PsRngEngineFunction`.
        """

    @abstractmethod
    def _get_rng_func(self) -> type[RngFunc]: ...

    def get_random_vector(
        self, counter: sp.Expr | int
    ) -> tuple[sp.IndexedBase, Assignment]:
        """Get a symbolic invocation of the RNG and a symbol representing its result.

        Args:
            counter: An expression specifying the external counter part of the RNG state, optional
        """

        rng_func = self._get_rng_func()
        symb = TypedSymbol(f"{self._name}_{rng_func.invocation_key}", self._vector_type)
        asm = Assignment(symb, rng_func(counter))
        return sp.IndexedBase(symb, self._vector_type.shape), asm  # type: ignore


class Philox(RngBase):
    """Implementation of the Philox counter-based RNG with W = 32 and N = 4.

    See https://doi.org/10.1145/2063384.2063405.

    This Philox RNG computes 128 random bits using a bijection ``b_k``, as ``b_k (n)``.
    Here, ``k`` is a subsequence key, and ``n`` is a 128-bit unsigned integer counter.
    The subsequence key of this implementation is given by the ``seed`` argument
    and an internal invocation key that is incremented  by one
    at each call to ``get_random_vector``.

    The counter ``n`` is constructed by bitwise concatenation of a user-provided
    32-bit external counter (passed to ``get_random_vector``)
    and the up to three 32-bit spatial iteration indices of the kernel.
    The spatial indices may be shifted by user-defined offsets,
    set in the ``offsets`` parameter.

    Args:
        name: Name of this RNG
        dtype: Data type of the random numbers; must be either ``float32`` or ``float64``
        seed: Expression or integer; the 32-bit unsigned integer seed for this RNG
        offsets: Optionally, offsets that are added to the iteration space counters when
            computing the random values
    """

    def __init__(
        self,
        name: str,
        dtype: UserTypeSpec,
        seed: sp.Expr | int,
        offsets: Sequence[sp.Expr | int] = (),
    ):
        super().__init__(name, dtype)

        self._seed = seed
        self._offsets = tuple(offsets)
        self._next_key = 0

    def get_keys(self, invocation) -> tuple[sp.Expr | int, ...]:
        if not isinstance(invocation, RngBase.RngFunc) or invocation.rng != self:
            raise ValueError("Given invocation does not belong to this RNG.")

        return (tcast(self._seed, "uint32"), invocation.invocation_key)

    def get_counters(self, invocation, rank) -> tuple[sp.Expr | int, ...]:
        if not isinstance(invocation, RngBase.RngFunc) or invocation.rng != self:
            raise ValueError("Given invocation does not belong to this RNG.")

        counters = [tcast(invocation.args[0], "uint32")]

        offsets = self._offsets + (0,) * (rank - len(self._offsets))

        from ..defaults import DEFAULTS

        for spatial_ctr, offset in zip(DEFAULTS.spatial_counters[:rank], offsets):
            counters.append(tcast(spatial_ctr + offset, "uint32"))

        counters += [0 for _ in range(3 - rank)]

        return tuple(counters)

    def _get_rng_func(self) -> type[RngBase.RngFunc]:
        key = self._next_key
        self._next_key += 1

        return sp.Function(
            f"{self._name}_{key}",
            nargs=1,
            bases=(RngBase.RngFunc,),
            rng=self,
            invocation_key=key,
        )  # type: ignore

    @classmethod
    def _get_vector_type(cls, dtype: PsIeeeFloatType):
        name_pattern = "pystencils::runtime::Fp{}x{}"

        match dtype.width:
            case 32:
                return PsNamedArrayType(
                    name_pattern.format(32, 4), PsIeeeFloatType(32), (4,)
                )
            case 64:
                return PsNamedArrayType(
                    name_pattern.format(64, 2), PsIeeeFloatType(64), (2,)
                )

        raise ValueError(f"Philox RNG not available for type {dtype}")
