from __future__ import annotations
from dataclasses import dataclass

from ..field import Field
from ..grids.protocols import IField


@dataclass(frozen=True)
class PsSymbolProperty:
    """Base class for symbol properties, which can be used to add additional information to symbols"""


@dataclass(frozen=True)
class UniqueSymbolProperty(PsSymbolProperty):
    """Base class for unique properties, of which only one instance may be registered at a time."""


@dataclass(frozen=True)
class FieldShape(PsSymbolProperty):
    """Symbol acts as a shape parameter to a field."""

    field: Field | IField
    coordinate: int


@dataclass(frozen=True)
class FieldStride(PsSymbolProperty):
    """Symbol acts as a stride parameter to a field."""

    field: Field | IField
    coordinate: int


@dataclass(frozen=True)
class FieldBasePtr(UniqueSymbolProperty):
    """Symbol acts as a base pointer to a field."""

    field: Field | IField


@dataclass(frozen=True)
class SYCLItem(UniqueSymbolProperty):
    """Symbol acts as a sycl item."""
    rank: int


@dataclass(frozen=True)
class SYCLNDItem(UniqueSymbolProperty):
    """Symbol acts as a sycl nditem."""
    rank: int


@dataclass(frozen=True)
class SYCLId(UniqueSymbolProperty):
    """Symbol acts as a sycl id."""
    rank: int


FieldProperty = FieldShape | FieldStride | FieldBasePtr
_FieldProperty = (FieldShape, FieldStride, FieldBasePtr)
