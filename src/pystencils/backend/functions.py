from __future__ import annotations
from typing import Any, Sequence, TYPE_CHECKING
from abc import ABC
from enum import Enum

from ..types import PsType, PsNumericType
from .exceptions import PsInternalCompilerError
from ..types import PsTypeError

if TYPE_CHECKING:
    from .ast.expressions import PsExpression


class PsFunction(ABC):
    """Base class for functions occuring in the IR"""

    __match_args__ = ("name", "arg_count")

    def __init__(self, name: str, num_args: int):
        self._name = name
        self._num_args = num_args

    @property
    def name(self) -> str:
        """Name of this function."""
        return self._name

    @property
    def arg_count(self) -> int:
        "Number of arguments this function takes"
        return self._num_args

    def __call__(self, *args: PsExpression) -> Any:
        from .ast.expressions import PsCall

        return PsCall(self, args)


class MathFunctions(Enum):
    """Mathematical functions supported by the backend.

    Each platform has to materialize these functions to a concrete implementation.
    """

    Exp = ("exp", 1)
    Log = ("log", 1)
    Sin = ("sin", 1)
    Cos = ("cos", 1)
    Tan = ("tan", 1)
    Sinh = ("sinh", 1)
    Cosh = ("cosh", 1)
    ASin = ("asin", 1)
    ACos = ("acos", 1)
    ATan = ("atan", 1)
    Sqrt = ("sqrt", 1)

    Abs = ("abs", 1)
    Floor = ("floor", 1)
    Ceil = ("ceil", 1)

    Min = ("min", 2)
    Max = ("max", 2)

    Pow = ("pow", 2)
    ATan2 = ("atan2", 2)

    def __init__(self, func_name, num_args):
        self.function_name = func_name
        self.num_args = num_args


class PsMathFunction(PsFunction):
    """Homogenously typed mathematical functions."""

    __match_args__ = ("func",)

    def __init__(self, func: MathFunctions) -> None:
        super().__init__(func.function_name, func.num_args)
        self._func = func

    @property
    def func(self) -> MathFunctions:
        return self._func

    def __str__(self) -> str:
        return f"{self._func.function_name}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PsMathFunction):
            return False

        return self._func == other._func

    def __hash__(self) -> int:
        return hash(self._func)


class ConstantFunctions(Enum):
    """Numerical constant functions.

    Each platform has to materialize these functions to a concrete implementation.
    """

    Pi = "pi"
    E = "e"
    PosInfinity = "pos_infinity"
    NegInfinity = "neg_infinity"

    def __init__(self, func_name):
        self.function_name = func_name


class PsConstantFunction(PsFunction):
    """Data-type-specific numerical constants.

    Represents numerical constants which need to be exactly represented,
    e.g. transcendental numbers and non-finite constants.

    Functions of this class are treated the same as `PsConstant` instances
    by most transforms.
    In particular, they are subject to the same contextual typing rules,
    and will be broadcast by the vectorizer.
    """

    __match_args__ = ("func,")

    def __init__(
        self, func: ConstantFunctions, dtype: PsNumericType | None = None
    ) -> None:
        super().__init__(func.function_name, 0)
        self._func = func
        self._set_dtype(dtype)

    @property
    def func(self) -> ConstantFunctions:
        return self._func

    @property
    def dtype(self) -> PsNumericType | None:
        """This constant function's data type, or ``None`` if it is untyped."""
        return self._dtype

    @dtype.setter
    def dtype(self, t: PsNumericType):
        self._set_dtype(t)

    def get_dtype(self) -> PsNumericType:
        """Retrieve this constant function's data type, throwing an exception if it is untyped."""
        if self._dtype is None:
            raise PsInternalCompilerError(
                "Data type of constant  function was not set."
            )
        return self._dtype

    def __str__(self) -> str:
        return f"{self._func.function_name}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PsConstantFunction):
            return False

        return (self._func, self._dtype) == (other._func, other._dtype)

    def __hash__(self) -> int:
        return hash((self._func, self._dtype))

    def _set_dtype(self, dtype: PsNumericType | None):
        if dtype is not None:
            match self._func:
                case (
                    ConstantFunctions.Pi
                    | ConstantFunctions.E
                    | ConstantFunctions.PosInfinity
                    | ConstantFunctions.NegInfinity
                ) if not dtype.is_float():
                    raise PsTypeError(
                        f"Invalid type for {self.func.function_name}: {dtype}"
                    )

        self._dtype = dtype


class CFunction(PsFunction):
    """A concrete C function.

    Instances of this class represent a C function by its name, parameter types, and return type.

    Args:
        name: Function name
        param_types: Types of the function parameters
        return_type: The function's return type
    """

    __match_args__ = ("name", "parameter_types", "return_type")

    @staticmethod
    def parse(obj) -> CFunction:
        """Parse the signature of a Python callable object to obtain a CFunction object.

        The callable must be fully annotated with type-like objects convertible by `create_type`.
        """
        import inspect
        from pystencils.types import create_type

        if not inspect.isfunction(obj):
            raise PsInternalCompilerError(f"Cannot parse object {obj} as a function")

        func_sig = inspect.signature(obj)
        func_name = obj.__name__
        arg_types = [
            create_type(param.annotation) for param in func_sig.parameters.values()
        ]
        ret_type = create_type(func_sig.return_annotation)

        return CFunction(func_name, arg_types, ret_type)

    def __init__(self, name: str, param_types: Sequence[PsType], return_type: PsType):
        super().__init__(name, len(param_types))

        self._param_types = tuple(param_types)
        self._return_type = return_type

    @property
    def parameter_types(self) -> tuple[PsType, ...]:
        return self._param_types

    @property
    def return_type(self) -> PsType:
        return self._return_type

    def __str__(self) -> str:
        param_types = ", ".join(str(t) for t in self._param_types)
        return f"{self._return_type} {self._name}({param_types})"

    def __repr__(self) -> str:
        return f"CFunction({self._name}, {self._param_types}, {self._return_type})"
