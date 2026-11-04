# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


"""Arrow C data interface: structs, capsules, and ownership.

Everything here is ctypes on purpose: the compiled code must read a producer's
buffers without pyarrow building an object per call.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Hashable
from functools import cache
from typing import TypeVar

import pyarrow as pa

_StructT = TypeVar("_StructT", bound=ctypes.Structure)


class ArrowSchema(ctypes.Structure):
    """``ArrowSchema``: the C data interface's type descriptor."""

    _fields_ = [
        ("format", ctypes.c_char_p),
        ("name", ctypes.c_char_p),
        ("metadata", ctypes.c_char_p),
        ("flags", ctypes.c_int64),
        ("n_children", ctypes.c_int64),
        ("children", ctypes.c_void_p),
        ("dictionary", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    ]


class ArrowArray(ctypes.Structure):
    """``ArrowArray``: the C data interface's buffer descriptor.

    Field order and widths must match the Numba struct model in ``array.py``:
    ten word-sized fields, 80 bytes on a 64-bit platform.
    """

    _fields_ = [
        ("length", ctypes.c_int64),
        ("null_count", ctypes.c_int64),
        ("offset", ctypes.c_int64),
        ("n_buffers", ctypes.c_int64),
        ("n_children", ctypes.c_int64),
        ("buffers", ctypes.POINTER(ctypes.c_void_p)),
        ("children", ctypes.c_void_p),
        ("dictionary", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    ]


class ArrowArrayStream(ctypes.Structure):
    """``ArrowArrayStream``: the C stream interface's vtable."""

    _fields_ = [
        ("get_schema", ctypes.c_void_p),
        ("get_next", ctypes.c_void_p),
        ("get_last_error", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    ]


SchemaRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(ArrowSchema))
ArrayRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(ArrowArray))
StreamRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(ArrowArrayStream))
StreamGetSchema = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(ArrowArrayStream), ctypes.POINTER(ArrowSchema))
StreamGetNext = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(ArrowArrayStream), ctypes.POINTER(ArrowArray))
StreamGetError = ctypes.CFUNCTYPE(
    ctypes.c_char_p, ctypes.POINTER(ArrowArrayStream))

_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PyCapsule_GetPointer.restype = ctypes.c_void_p


def no_protocol(obj: object) -> TypeError:
    """``TypeError`` for an object that is not an Arrow producer."""
    return TypeError(f"{type(obj).__name__} implements neither "
                     "__arrow_c_array__ nor __arrow_c_stream__")


def capsule_pointer(capsule: object, name: bytes) -> int:
    """Address of the struct inside *capsule*, or ``TypeError`` if misnamed."""
    try:
        return _PyCapsule_GetPointer(capsule, name)
    except ValueError:
        raise TypeError(
            f"expected a PyCapsule named {name.decode()!r}, "
            f"got {type(capsule).__name__}") from None


def take(struct_type: type[_StructT], capsule: object,
         name: bytes) -> _StructT:
    """Copy the struct out of *capsule* and take over its release callback.

    The copy owns the buffers from here on: the producer's ``release`` is
    cleared so its capsule destructor cannot free them a second time.
    """
    addr = capsule_pointer(capsule, name)
    struct = struct_type.from_buffer_copy(
        ctypes.string_at(addr, ctypes.sizeof(struct_type)))
    struct_type.from_address(addr).release = None
    return struct


def release_struct(struct: ctypes.Structure,
                   release_type: Callable[..., None]) -> None:
    """Run the struct's release callback once, then clear it."""
    if struct.release:
        release_type(struct.release)(ctypes.byref(struct))
        struct.release = None


def store_array(array: ArrowArray, addr: int) -> None:
    """Copy *array* into the struct at *addr* (a Numba view's address)."""
    ctypes.memmove(addr, ctypes.addressof(array), ctypes.sizeof(ArrowArray))


def stream_error(stream: ArrowArrayStream) -> str:
    """The producer's message for a failed stream call."""
    if not stream.get_last_error:
        return "Arrow stream failed"
    message = StreamGetError(stream.get_last_error)(ctypes.byref(stream))
    return message.decode() if message else "Arrow stream failed"


def from_stream(capsule: object) -> tuple[ArrowSchema, ArrowArray | None]:
    """The single chunk a stream capsule yields, as ``(schema, array)``.

    Returns ``(schema, None)`` for a stream that yields no batch at all.
    ``TypeError`` for more than one chunk, ``ValueError`` when the stream
    reports an error.
    """
    stream = take(ArrowArrayStream, capsule, b"arrow_array_stream")
    get_schema = StreamGetSchema(stream.get_schema)
    get_next = StreamGetNext(stream.get_next)

    schema = ArrowSchema()
    array = ArrowArray()
    status = get_schema(ctypes.byref(stream), ctypes.byref(schema))
    if not status:
        status = get_next(ctypes.byref(stream), ctypes.byref(array))
    error = stream_error(stream) if status else None
    empty = error is None and not array.release

    has_more = False
    if error is None and not empty:
        extra = ArrowArray()
        probe = get_next(ctypes.byref(stream), ctypes.byref(extra))
        has_more = probe == 0 and bool(extra.release)
        if has_more:
            release_struct(extra, ArrayRelease)
        elif probe != 0:
            error = stream_error(stream)

    release_struct(stream, StreamRelease)

    if error is not None or has_more:
        release_struct(array, ArrayRelease)
        release_struct(schema, SchemaRelease)
        if has_more:
            raise TypeError(
                "Arrow stream yields more than one chunk; call from_any() "
                "on it to concatenate them (which copies)")
        raise ValueError(error)
    return schema, None if empty else array


@cache
def num_buffers(dtype: pa.DataType) -> int:
    """How many buffers a struct of *dtype* must carry: validity, offsets, data."""
    if pa.types.is_fixed_size_binary(dtype):
        return 2
    if (pa.types.is_string(dtype) or pa.types.is_large_string(dtype)
            or pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype)):
        return 3
    return 2


def _check_buffers(array: ArrowArray, dtype: pa.DataType | None) -> None:
    """Refuse a struct with fewer buffers than its type needs.

    The compiled code reads the validity, offsets and data pointers by index;
    a producer that exports too few would make it read past the end of the
    buffer array and then dereference whatever it found.
    """
    if dtype is None:
        return
    expected = num_buffers(dtype)
    if array.n_buffers < expected:
        raise ValueError(
            f"producer exported {array.n_buffers} buffers for {dtype}, "
            f"which needs {expected}")


def _column_of(obj):
    """Column 0 of a RecordBatch or Table, as a pa.Array."""
    column = obj.column(0)
    if isinstance(column, pa.ChunkedArray):
        column = (column.chunk(0) if column.num_chunks == 1
                  else column.combine_chunks())
    return column


def take_array_into(obj: object, addr: int,
                    dtype: pa.DataType | None = None) -> None:
    """Move *obj*'s ArrowArray into the Numba struct at *addr*.

    A ``pa.Array`` writes its struct straight out (0.2 us, no capsule
    allocation); everything else goes through ``__arrow_c_array__`` or a
    single-chunk ``__arrow_c_stream__``. Nothing is copied: the struct is
    moved, and the compiled code's cleanup calls its release callback.
    """
    if isinstance(obj, pa.Array):
        obj._export_to_c(addr)
        _check_buffers(ArrowArray.from_address(addr), dtype)
        return

    if isinstance(obj, (pa.RecordBatch, pa.Table)):
        _, column_capsule = _column_of(obj).__arrow_c_array__()
        column = take(ArrowArray, column_capsule, b"arrow_array")
        _check_buffers(column, dtype)
        store_array(column, addr)
        return

    protocol = _protocol_of(obj)
    if protocol == "__arrow_c_array__":
        schema_capsule, array_capsule = getattr(obj, protocol)()
        array = take(ArrowArray, array_capsule, b"arrow_array")
        del schema_capsule       # its destructor releases the schema
        _check_buffers(array, dtype)
    elif protocol is not None:
        schema, array = from_stream(getattr(obj, protocol)())
        if array is None:
            # an empty stream still has a type, known at compile time
            release_struct(schema, SchemaRelease)
            pa.array([], type=dtype)._export_to_c(addr)
            return
        if array.n_children:
            # a record-batch stream (duckdb): the batch is the struct array,
            # the column is what the compiled code reads
            batch_schema = pa.Schema._import_from_c(ctypes.addressof(schema))
            batch = pa.RecordBatch._import_from_c(
                ctypes.addressof(array), batch_schema)
            _, column_capsule = batch.column(0).__arrow_c_array__()
            column = take(ArrowArray, column_capsule, b"arrow_array")
            _check_buffers(column, dtype)
            store_array(column, addr)
            return
        release_struct(schema, SchemaRelease)   # type is known at compile time
    else:
        raise no_protocol(obj)
    _check_buffers(array, dtype)
    store_array(array, addr)


_DTYPE_CACHE: dict[tuple[type, Hashable], pa.DataType] = {}
_DTYPE_CACHE_LIMIT = 256
_PROTOCOLS: dict[type, str | None] = {}


def _protocol_of(obj: object) -> str | None:
    """The PyCapsule protocol *obj* exports, decided once per class.

    Producers such as Polars answer a missing attribute through their own
    ``__getattr__``, which costs microseconds per lookup; asking the class
    costs one dict hit after the first instance.
    """
    kind = type(obj)
    if kind not in _PROTOCOLS:
        name = None
        for candidate in ("__arrow_c_array__", "__arrow_c_stream__"):
            if hasattr(kind, candidate):
                name = candidate
                break
        else:
            # a class-level miss can still be a per-instance wrapper: ask once
            name = next(
                (candidate for candidate in ("__arrow_c_array__",
                                             "__arrow_c_stream__")
                 if getattr(obj, candidate, None) is not None), None)
        _PROTOCOLS[kind] = name
    return _PROTOCOLS[kind]


def _dtype_hint(obj: object) -> tuple[type, Hashable] | None:
    """A cheap, hashable stand-in for *obj*'s Arrow type, or ``None``.

    Numba re-types every argument on every call, so reading a schema per call
    would open a stream (and, for duckdb, run the query) twice per call.

    ``None`` means "cannot be cached": no ``dtype``/``types`` attribute, a
    numpy object column (which says nothing about the Arrow type: two object
    columns can be a string and a struct), or an unhashable hint.
    """
    kind = type(obj)
    if not (hasattr(kind, "dtype") or hasattr(kind, "types")):
        # a per-instance protocol lookup on some producers (polars) costs
        # microseconds; a class-level miss means there is nothing to find
        return None
    hint = getattr(obj, "dtype", None)
    if hint is None:
        types_ = getattr(obj, "types", None)
        if types_ is None:
            return None
        hint = tuple(types_)
    if getattr(hint, "kind", "") == "O":
        return None
    return (kind, hint)


def dtype_of(obj: object) -> pa.DataType:
    """*obj*'s pyarrow DataType, read from its schema and nothing else.

    Streaming producers hand out record batches, so a one-field struct schema
    means the compiled code will read that field.
    """
    hint = _dtype_hint(obj)
    if hint is not None:
        try:
            return _DTYPE_CACHE[hint]      # TypeError for an unhashable hint
        except KeyError:
            pass
        except TypeError:
            hint = None
    dtype = _dtype_of_uncached(obj)
    if hint is not None:
        if len(_DTYPE_CACHE) >= _DTYPE_CACHE_LIMIT:
            _DTYPE_CACHE.clear()
        _DTYPE_CACHE[hint] = dtype
    return dtype


def _dtype_of_uncached(obj: object) -> pa.DataType:
    if isinstance(obj, pa.Array):
        return obj.type          # free, and Numba re-types arguments per call
    if isinstance(obj, pa.ChunkedArray):
        return obj.type
    if isinstance(obj, (pa.RecordBatch, pa.Table)):
        return obj.schema.field(0).type

    export = getattr(obj, "__arrow_c_array__", None)
    if export is not None:
        schema_capsule, _ = export()      # the array capsule is dropped
        return pa.DataType._import_from_c_capsule(schema_capsule)

    stream = getattr(obj, "__arrow_c_stream__", None)
    if stream is None:
        raise no_protocol(obj)

    handle = take(ArrowArrayStream, stream(), b"arrow_array_stream")
    schema = ArrowSchema()
    try:
        status = StreamGetSchema(handle.get_schema)(
            ctypes.byref(handle), ctypes.byref(schema))
        if status:
            raise ValueError(stream_error(handle))
        dtype = pa.DataType._import_from_c(ctypes.addressof(schema))
    finally:
        release_struct(handle, StreamRelease)
    return dtype.field(0).type if pa.types.is_struct(dtype) else dtype
