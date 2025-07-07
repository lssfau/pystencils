from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Sequence
from dataclasses import dataclass

import sympy as sp

from ..assignment import Assignment
from .typed_sympy import TypedSymbol, tcast
from ..types import UserTypeSpec, create_type, PsIeeeFloatType, PsNamedArrayType


@dataclass(frozen=True)
class RngState(ABC):
    """State of an RNG object.

    Instances of `RngState` must be immutable, hashable, and comparable
    to facilitate comparison, hashing, and pickle serialization of RNG invocation expressions.
    """

    name: str
    dtype: PsIeeeFloatType
    invocation_key: int

    @abstractmethod
    def next(self) -> RngState:
        """Return the RNG state for the next invocation"""

    @abstractmethod
    def get_keys(self) -> tuple[sp.Expr | int, ...]:
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


class RngBase(ABC):
    """Abstract base class for counter-based random number generators."""

    @classmethod
    def is_invocation(cls, expr: sp.Expr) -> bool:
        """Determine if the given expression is an RNG invocation"""
        return isinstance(expr, sp.Function) and isinstance(
            getattr(expr, "state", None), RngState
        )

    @classmethod
    def get_invocation_state(cls, expr: sp.Expr) -> RngState | None:
        """If ``expr`` is an RNG invocation, return its state; return `None` otherwise."""
        if isinstance(expr, sp.Function):
            state = getattr(expr, "state", None)
            if isinstance(state, RngState):
                return state

        return None

    @classmethod
    @abstractmethod
    def _get_vector_type(cls, dtype: PsIeeeFloatType) -> PsNamedArrayType: ...

    def __init__(self, state: RngState):
        self._state = state
        self._vector_type = self._get_vector_type(self._state.dtype)

    @property
    def dtype(self) -> PsIeeeFloatType:
        """Data type of the random numbers"""
        return self._state.dtype

    @property
    def vector_size(self) -> int:
        """Number of random numbers returned by a single invocation"""
        return self._vector_type.shape[0]

    def _get_rng_func(self) -> tuple[type[sp.Function], RngState]:
        state = self._state
        self._state = state.next()

        return sp.Function(
            f"{self._state.name}_{state.invocation_key}",
            nargs=1,
            state=state,
        ), state  # type: ignore

    def get_random_vector(
        self, counter: sp.Expr | int
    ) -> tuple[sp.IndexedBase, Assignment]:
        """Get a symbolic invocation of the RNG and a symbol representing its result.

        Args:
            counter: An expression specifying the external counter part of the RNG state,
                e.g. the time step counter
        """

        rng_func, state = self._get_rng_func()
        symb = TypedSymbol(
            f"{state.name}_{state.invocation_key}", self._vector_type
        )
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

    @dataclass(frozen=True)
    class PhiloxState(RngState):
        seed: sp.Expr | int
        offsets: tuple[sp.Expr | int, ...]

        def next(self) -> Philox.PhiloxState:
            return Philox.PhiloxState(
                self.name, self.dtype, self.invocation_key + 1, self.seed, self.offsets
            )

        def get_keys(self) -> tuple[sp.Expr | int, ...]:
            return (tcast(self.seed, "uint32"), self.invocation_key)

        def get_counters(self, invocation, rank) -> tuple[sp.Expr | int, ...]:
            if Philox.get_invocation_state(invocation) != self:
                raise ValueError(
                    "This RNG state does not belong to the given invocation."
                )

            counters = [tcast(invocation.args[0], "uint32")]

            offsets = self.offsets + (0,) * (rank - len(self.offsets))

            from ..defaults import DEFAULTS

            for spatial_ctr, offset in zip(DEFAULTS.spatial_counters[:rank], offsets):
                counters.append(tcast(spatial_ctr + offset, "uint32"))

            counters += [0 for _ in range(3 - rank)]

            return tuple(counters)

    def __init__(
        self,
        name: str,
        dtype: UserTypeSpec,
        seed: sp.Expr | int,
        offsets: Sequence[sp.Expr | int] = (),
    ):
        dtype = create_type(dtype)
        if not isinstance(dtype, PsIeeeFloatType):
            raise ValueError("Data type must be a floating-point type")

        state = Philox.PhiloxState(name, dtype, 0, seed, tuple(offsets))
        super().__init__(state)

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
