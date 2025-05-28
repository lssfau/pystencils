from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Any
from itertools import chain, count
from collections import namedtuple, defaultdict
import re

from ..ast.expressions import PsExpression, PsConstantExpr, PsCall
from ..functions import PsConstantFunction, ConstantFunctions
from ...defaults import DEFAULTS
from ...field import Field, FieldType
from ...sympyextensions import ReductionOp
from ...sympyextensions.typed_sympy import TypedSymbol, DynamicType

from ..memory import PsSymbol, PsBuffer
from ..constants import PsConstant
from ...types import (
    PsType,
    PsIntegerType,
    PsNumericType,
    PsPointerType,
    deconstify,
    constify,
)
from ..exceptions import PsInternalCompilerError, KernelConstraintsError

from .iteration_space import IterationSpace, FullIterationSpace, SparseIterationSpace


class FieldsInKernel:
    def __init__(self) -> None:
        self.domain_fields: set[Field] = set()
        self.index_fields: set[Field] = set()
        self.custom_fields: set[Field] = set()
        self.buffer_fields: set[Field] = set()

        self.archetype_field: Field | None = None

    def __iter__(self) -> Iterator[Field]:
        return chain(
            self.domain_fields,
            self.index_fields,
            self.custom_fields,
            self.buffer_fields,
        )


FieldBufferPair = namedtuple("FieldBufferPair", ("field", "buffer"))


@dataclass(frozen=True)
class ReductionInfo:
    """Information about a reduction operation, its neutral element in form of an initial value
    and the pointer used by the kernel as write-back argument.

    Attributes:
    ===========

    reduction_op : Reduction operation being performed
    init_val : Initial value used to initialize local symbol
    local_symbol : Kernel-local symbol used to accumulate intermediate reduction result
    writeback_ptr_symbol : Symbol that is used to export the final reduction result
    """

    op: ReductionOp
    init_val: PsExpression
    local_symbol: PsSymbol
    writeback_ptr_symbol: PsSymbol


class KernelCreationContext:
    """Manages the translation process from the SymPy frontend to the backend AST, and collects
    all necessary information for the translation:

    - *Data Types*: The kernel creation context manages the default data types for loop limits
      and counters, index calculations, and the typifier.
    - *Symbols*: The context maintains a symbol table, keeping track of all symbols encountered
      during kernel translation together with their types.
    - *Fields and Arrays*: The context collects all fields encountered during code generation,
      applies a few consistency checks to them, and manages their associated arrays.
    - *Iteration Space*: The context manages the iteration space of the kernel currently being
      translated.
    - *Constraints*: The context collects all kernel parameter constraints introduced during the
      translation process.
    - *Required Headers*: The context collects all header files required for the kernel to run.

    """

    def __init__(
        self,
        default_dtype: PsNumericType = DEFAULTS.numeric_dtype,
        index_dtype: PsIntegerType = DEFAULTS.index_dtype,
    ):
        self._default_dtype = deconstify(default_dtype)
        self._index_dtype = deconstify(index_dtype)

        self._symbols: dict[str, PsSymbol] = dict()

        self._symbol_ctr_pattern = re.compile(r"__[0-9]+$")
        self._symbol_dup_table: defaultdict[str, int] = defaultdict(lambda: 0)

        self._reduction_data: dict[str, ReductionInfo] = dict()

        self._fields_and_buffers: dict[str, FieldBufferPair] = dict()
        self._fields_collection = FieldsInKernel()

        self._ispace: IterationSpace | None = None

        self._req_headers: set[str] = set()

        self._metadata: dict[str, Any] = dict()

    @property
    def default_dtype(self) -> PsNumericType:
        """Data type used by default for numerical expressions"""
        return self._default_dtype

    @property
    def index_dtype(self) -> PsIntegerType:
        """Data type used by default for index expressions"""
        return self._index_dtype

    def resolve_dynamic_type(self, dtype: DynamicType | PsType) -> PsType:
        """Selects the appropriate data type for `DynamicType` instances, and returns all other types as they are."""
        match dtype:
            case DynamicType.NUMERIC_TYPE:
                return self._default_dtype
            case DynamicType.INDEX_TYPE:
                return self._index_dtype
            case _:
                return dtype

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    #   Symbols

    def get_symbol(self, name: str, dtype: PsType | None = None) -> PsSymbol:
        """Retrieve the symbol with the given name and data type from the symbol table.

        If no symbol named ``name`` exists, a new symbol with the given data type is created.

        If a symbol with the given ``name`` already exists and ``dtype`` is not `None`,
        the given data type will be applied to it, and it is returned.
        If the symbol already has a different data type, an error will be raised.

        If the symbol already exists and ``dtype`` is `None`, the existing symbol is returned
        without checking or altering its data type.

        Args:
            name: The symbol's name
            dtype: The symbol's data type, or `None`
        """
        if name not in self._symbols:
            symb = PsSymbol(name, None)
            self._symbols[name] = symb
        else:
            symb = self._symbols[name]

        if dtype is not None:
            symb.apply_dtype(dtype)

        return symb

    def get_new_symbol(self, name: str, dtype: PsType | None = None) -> PsSymbol:
        """Always create a new symbol, deduplicating its name if another symbol with the same name already exists."""

        if name in self._symbols:
            return self.duplicate_symbol(self._symbols[name], dtype)
        else:
            return self.get_symbol(name, dtype)

    def find_symbol(self, name: str) -> PsSymbol | None:
        """Find a symbol with the given name in the symbol table, if it exists.

        Returns:
            The symbol with the given name, or `None` if no such symbol exists.
        """
        return self._symbols.get(name, None)

    def add_symbol(self, symbol: PsSymbol):
        """Add an existing symbol to the symbol table.

        If a symbol with the same name already exists, an error will be raised.
        """
        if symbol.name in self._symbols:
            raise PsInternalCompilerError(f"Duplicate symbol: {symbol.name}")

        self._symbols[symbol.name] = symbol

    def replace_symbol(self, old: PsSymbol, new: PsSymbol):
        """Replace one symbol by another.

        The two symbols ``old`` and ``new`` must have the same name, but may have different data types.
        """
        if old.name != new.name:
            raise PsInternalCompilerError(
                "replace_symbol: Old and new symbol must have the same name"
            )

        if old.name not in self._symbols:
            raise PsInternalCompilerError("Trying to replace an unknown symbol")

        self._symbols[old.name] = new

    def add_reduction_info(
        self,
        lhs_name: str,
        lhs_dtype: PsNumericType,
        reduction_op: ReductionOp,
    ):
        """Create ReductionInfo instance and add to its corresponding lookup table for a given symbol name."""

        # make sure that lhs symbol never occurred before ReductionAssignment
        if self.find_symbol(lhs_name):
            raise KernelConstraintsError(
                f"Cannot create reduction with symbol {lhs_name}: "
                "Another symbol with the same name already exist."
            )

        # add symbol for lhs with pointer datatype for write-back mechanism
        pointer_symb = self.get_symbol(lhs_name, PsPointerType(lhs_dtype))

        # create kernel-local copy of lhs symbol
        local_symb = self.get_new_symbol(f"{lhs_name}_local", lhs_dtype)

        # match for reduction operation and set neutral init_val
        init_val: PsExpression
        match reduction_op:
            case ReductionOp.Add:
                init_val = PsConstantExpr(PsConstant(0))
            case ReductionOp.Sub:
                init_val = PsConstantExpr(PsConstant(0))
            case ReductionOp.Mul:
                init_val = PsConstantExpr(PsConstant(1))
            case ReductionOp.Min:
                init_val = PsCall(PsConstantFunction(ConstantFunctions.PosInfinity), [])
            case ReductionOp.Max:
                init_val = PsCall(PsConstantFunction(ConstantFunctions.NegInfinity), [])
            case _:
                raise PsInternalCompilerError(
                    f"Unsupported kind of reduction assignment: {reduction_op}."
                )

        # create reduction info and add to set
        reduction_info = ReductionInfo(reduction_op, init_val, local_symb, pointer_symb)
        self._reduction_data[lhs_name] = reduction_info

        return reduction_info

    def find_reduction_info(self, name: str) -> ReductionInfo | None:
        """Find a ReductionInfo with the given name in the lookup table, if it exists.

        Returns:
            The ReductionInfo with the given name, or `None` if it does not exist.
        """
        return self._reduction_data.get(name, None)

    def duplicate_symbol(
        self, symb: PsSymbol, new_dtype: PsType | None = None
    ) -> PsSymbol:
        """Canonically duplicates the given symbol.

        A new symbol with the new name ``symb.name + "__<counter>"`` and optionally a different data type
        is created, added to the symbol table, and returned.
        The ``counter`` reflects the number of previously created duplicates of this symbol.
        """
        basename = self.basename(symb)

        if new_dtype is None:
            new_dtype = symb.dtype

        initial_count = self._symbol_dup_table[basename]
        for i in count(initial_count):
            dup_name = f"{basename}__{i}"
            if self.find_symbol(dup_name) is None:
                self._symbol_dup_table[basename] = i + 1
                return self.get_symbol(dup_name, new_dtype)
        assert False, "unreachable code"

    def basename(self, symb: PsSymbol) -> str:
        """Returns the original name a symbol had before duplication."""
        if (result := self._symbol_ctr_pattern.search(symb.name)) is not None:
            span = result.span()
            return symb.name[: span[0]]
        else:
            return symb.name

    @property
    def symbols(self) -> Iterable[PsSymbol]:
        """Return an iterable of all symbols listed in the symbol table."""
        return self._symbols.values()

    @property
    def reduction_data(self) -> dict[str, ReductionInfo]:
        """Return a dictionary holding kernel-local reduction information for given symbol names."""
        return self._reduction_data

    #   Fields and Arrays

    @property
    def fields(self) -> FieldsInKernel:
        """Collection of fields that occured during the current kernel translation."""
        return self._fields_collection

    def add_field(self, field: Field, const: bool = False):
        """Add the given field to the context's fields collection.

        This method adds the passed ``field`` to the context's field collection, which is
        accesible through the `field <KernelCreationContext.fields>` member,
        and creates the underlying buffer for the field
        which is retrievable through `get_buffer`.
        Before adding the field to the collection, various sanity and constraint checks are applied.

        Args:
            field: The field to be registered
            const: If `True`, the field will be made read-only by setting its element type to const.
        """

        if field.name in self._fields_and_buffers:
            existing_pair: FieldBufferPair = self._fields_and_buffers[field.name]
            if existing_pair.field != field:
                raise KernelConstraintsError(
                    "Encountered two fields with the same name, but different properties: "
                    f"{field} and {existing_pair.field}"
                )
            elif existing_pair.buffer.element_type.const != const:
                raise KernelConstraintsError(
                    f"Field {field} was registered twice, once as writeable and once as read-only."
                )
            else:
                return

        #   Check field constraints, create buffer, and add them to the collection
        match field.field_type:
            case FieldType.GENERIC | FieldType.STAGGERED | FieldType.STAGGERED_FLUX:
                buf = self._create_regular_field_buffer(field, const)
                self._fields_collection.domain_fields.add(field)

            case FieldType.BUFFER:
                buf = self._create_buffer_field_buffer(field, const)
                self._fields_collection.buffer_fields.add(field)

            case FieldType.INDEXED:
                if field.spatial_dimensions != 1:
                    raise KernelConstraintsError(
                        f"Invalid spatial shape of index field {field.name}: {field.spatial_dimensions}. "
                        "Index fields must be one-dimensional."
                    )
                buf = self._create_regular_field_buffer(field, const)
                self._fields_collection.index_fields.add(field)

            case FieldType.CUSTOM:
                buf = self._create_regular_field_buffer(field, const)
                self._fields_collection.custom_fields.add(field)

            case _:
                assert False, "unreachable code"

        self._fields_and_buffers[field.name] = FieldBufferPair(field, buf)

    @property
    def buffers(self) -> Iterable[PsBuffer]:
        # return self._fields_and_arrays.values()
        yield from (item.buffer for item in self._fields_and_buffers.values())

    def get_buffer(self, field: Field) -> PsBuffer:
        """Retrieve the underlying array for a given field.

        If the given field was not previously registered using `add_field`,
        this method internally calls `add_field` to check the field for consistency.
        """
        if field.name in self._fields_and_buffers:
            if field != self._fields_and_buffers[field.name].field:
                raise KernelConstraintsError(
                    "Encountered two fields of the same name but with different properties."
                )
        else:
            self.add_field(field)
        return self._fields_and_buffers[field.name].buffer

    def find_field(self, name: str) -> Field:
        return self._fields_and_buffers[name].field

    #   Iteration Space

    def set_iteration_space(self, ispace: IterationSpace):
        """Set the iteration space used for the current kernel."""
        self._ispace = ispace

    def get_iteration_space(self) -> IterationSpace:
        if self._ispace is None:
            raise PsInternalCompilerError("No iteration space set in context.")
        return self._ispace

    def get_full_iteration_space(self) -> FullIterationSpace:
        if not isinstance(self._ispace, FullIterationSpace):
            raise PsInternalCompilerError("No full iteration space set in context.")
        return self._ispace

    def get_sparse_iteration_space(self) -> SparseIterationSpace:
        if not isinstance(self._ispace, SparseIterationSpace):
            raise PsInternalCompilerError("No sparse iteration space set in context.")
        return self._ispace

    #   Headers

    @property
    def required_headers(self) -> set[str]:
        return self._req_headers

    def require_header(self, header: str):
        self._req_headers.add(header)

    #   ----------- Internals ---------------------------------------------------------------------

    def _normalize_type(self, s: TypedSymbol) -> PsIntegerType:
        match s.dtype:
            case DynamicType.INDEX_TYPE:
                return self.index_dtype
            case DynamicType.NUMERIC_TYPE:
                if isinstance(self.default_dtype, PsIntegerType):
                    return self.default_dtype
                else:
                    raise KernelConstraintsError(
                        f"Cannot use non-integer default numeric type {self.default_dtype} "
                        f"in field indexing symbol {s}."
                    )
            case PsIntegerType():
                return deconstify(s.dtype)
            case _:
                raise KernelConstraintsError(
                    f"Invalid data type for field indexing symbol {s}: {s.dtype}"
                )

    def _create_regular_field_buffer(self, field: Field, const: bool) -> PsBuffer:
        idx_types = set(
            self._normalize_type(s)
            for s in chain(field.shape, field.strides)
            if isinstance(s, TypedSymbol)
        )

        entry_type = self.resolve_dynamic_type(field.dtype)
        if const:
            entry_type = constify(entry_type)

        if len(idx_types) > 1:
            raise KernelConstraintsError(
                f"Multiple incompatible types found in index symbols of field {field}: "
                f"{idx_types}"
            )

        idx_type = idx_types.pop() if len(idx_types) > 0 else self.index_dtype

        def convert_size(s: TypedSymbol | int) -> PsSymbol | PsConstant:
            if isinstance(s, TypedSymbol):
                return self.get_symbol(s.name, idx_type)
            else:
                return PsConstant(s, idx_type)

        buf_shape = [convert_size(s) for s in field.shape]
        buf_strides = [convert_size(s) for s in field.strides]

        # The frontend doesn't quite agree with itself on how to model
        # fields with trivial index dimensions. Sometimes the index_shape is empty,
        # sometimes its (1,). This is canonicalized here.
        if not field.index_shape:
            buf_shape += [convert_size(1)]
            buf_strides += [convert_size(1)]

        from ...codegen.properties import FieldShape, FieldStride

        for i, size in enumerate(buf_shape):
            if isinstance(size, PsSymbol):
                size.add_property(FieldShape(field, i))

        for i, stride in enumerate(buf_strides):
            if isinstance(stride, PsSymbol):
                stride.add_property(FieldStride(field, i))

        base_ptr = self.get_symbol(
            DEFAULTS.field_pointer_name(field.name),
            PsPointerType(entry_type, restrict=True),
        )

        return PsBuffer(field.name, entry_type, base_ptr, buf_shape, buf_strides)

    def _create_buffer_field_buffer(self, field: Field, const: bool) -> PsBuffer:
        if field.spatial_dimensions != 1:
            raise KernelConstraintsError(
                f"Invalid spatial shape of buffer field {field.name}: {field.spatial_dimensions}. "
                "Buffer fields must be one-dimensional."
            )

        if field.index_dimensions > 1:
            raise KernelConstraintsError(
                f"Invalid index shape of buffer field {field.name}: {field.spatial_dimensions}. "
                "Buffer fields can have at most one index dimension."
            )

        num_entries = field.index_shape[0] if field.index_shape else 1
        if not isinstance(num_entries, int):
            raise KernelConstraintsError(
                f"Invalid index shape of buffer field {field.name}: {num_entries}. "
                "Buffer fields cannot have variable index shape."
            )

        buffer_len = field.spatial_shape[0]
        buf_shape: list[PsSymbol | PsConstant]

        if isinstance(buffer_len, TypedSymbol):
            from ...codegen.properties import FieldShape

            idx_type = self._normalize_type(buffer_len)
            len_symb = self.get_symbol(buffer_len.name, idx_type)
            len_symb.add_property(FieldShape(field, 0))
            buf_shape = [len_symb, PsConstant(num_entries, idx_type)]
        else:
            idx_type = DEFAULTS.index_dtype
            buf_shape = [
                PsConstant(buffer_len, idx_type),
                PsConstant(num_entries, idx_type),
            ]

        buf_strides = [PsConstant(num_entries, idx_type), PsConstant(1, idx_type)]
        buf_dtype = self.resolve_dynamic_type(field.dtype)
        if const:
            buf_dtype = constify(buf_dtype)

        base_ptr = self.get_symbol(
            DEFAULTS.field_pointer_name(field.name),
            PsPointerType(buf_dtype, restrict=True),
        )

        return PsBuffer(field.name, buf_dtype, base_ptr, buf_shape, buf_strides)
