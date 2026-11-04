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

from __future__ import annotations

import ctypes
import functools
import os

import llvmlite.binding
from llvmlite import ir
from numba import types as nbt
from numba.core import cgutils

import pyarrow as pa

from ._capsule import _protocol_of
from ._capsule import no_protocol as _no_protocol


def array_from(obj: object) -> pa.Array:
    """A ``pa.Array`` for *obj*, through the Arrow PyCapsule interface.

    Zero copy: the capsules hand over the producer's buffers. Producers that
    only implement ``__arrow_c_stream__`` (polars, duckdb) are read as a
    one-column table; a single-chunk column keeps the buffers.
    """
    if isinstance(obj, pa.Array):
        return obj
    protocol = _protocol_of(obj)
    if protocol == "__arrow_c_array__":
        return pa.Array._import_from_c_capsule(*getattr(obj, protocol)())
    if protocol is not None:
        # a stream of record batches (duckdb) or of arrays (polars); both
        # import zero copy, but combine_chunks() copies even for one chunk
        try:
            chunked = pa.chunked_array(obj)
        except pa.ArrowInvalid:
            chunked = pa.table(obj).column(0)
        if chunked.num_chunks == 0:
            return pa.array([], type=chunked.type)
        array = (chunked.chunk(0) if chunked.num_chunks == 1
                 else chunked.combine_chunks())
        if pa.types.is_struct(array.type):
            # a record-batch stream: one struct array, one column
            array = array.field(0)
        return array
    raise _no_protocol(obj)


def _is_var_width(ty: pa.DataType) -> bool:
    """True for string/binary types: elements are views, not scalars."""
    return (pa.types.is_string(ty) or pa.types.is_large_string(ty)
            or pa.types.is_binary(ty) or pa.types.is_large_binary(ty)
            or pa.types.is_fixed_size_binary(ty))


def _arrow_type_to_nb_type(ty: pa.DataType) -> nbt.Type:
    if pa.types.is_signed_integer(ty):
        return nbt.Integer(str(ty), bitwidth=ty.bit_width, signed=True)
    if pa.types.is_unsigned_integer(ty):
        return nbt.Integer(str(ty), bitwidth=ty.bit_width, signed=False)
    if pa.types.is_float32(ty):
        return nbt.float32
    if pa.types.is_float64(ty):
        return nbt.float64
    if _is_var_width(ty):
        # element reads go through StringViewType, there is no scalar type
        return nbt.uint8
    if pa.types.is_decimal(ty) and ty.bit_width == 128:
        # (high, low) of the 128-bit two's complement integer, exact;
        # use decimal_to_float() when a double is what you want
        return nbt.Tuple((nbt.int64, nbt.uint64))
    # temporal types are their storage integers: no unit conversion here
    if pa.types.is_date32(ty) or pa.types.is_time32(ty):
        return nbt.int32
    if (pa.types.is_date64(ty) or pa.types.is_time64(ty)
            or pa.types.is_timestamp(ty) or pa.types.is_duration(ty)):
        return nbt.int64
    if pa.types.is_boolean(ty):
        # bit-packed values: readable as validity, not as elements
        return nbt.boolean
    raise TypeError(f'Unsupported PyArrow type {ty}')


int64_t = ir.IntType(64)

pyarrow_cpp_funcs = {
    'PyArrow_AllocatePoolBuffer': ir.FunctionType(
        cgutils.voidptr_t,
        [int64_t],
    ),
    'PyArrow_PoolBufferReserve': ir.FunctionType(
        cgutils.int32_t, [cgutils.voidptr_t, int64_t]),
    'PyArrow_PoolBufferResizeExactly': ir.FunctionType(
        cgutils.int32_t, [cgutils.voidptr_t, int64_t]),
    'PyArrow_PoolBufferDestroy': ir.FunctionType(
        ir.VoidType(), [cgutils.voidptr_t]),
    'PyArrow_MakeAndExportArray': ir.FunctionType(
        cgutils.int32_t, [int64_t, int64_t, cgutils.int32_t,
                          cgutils.voidptr_t.as_pointer(), cgutils.voidptr_t]),
    'PyArrow_ReleaseArray': ir.FunctionType(ir.VoidType(), [cgutils.voidptr_t]),
}


@functools.cache
def get_pyarrow_dll() -> ctypes.CDLL:
    exc = None
    dll = None
    for path in pa.get_library_dirs():
        try:
            dll = ctypes.CDLL(os.path.join(path, 'libarrow_python.so'))
            break
        except OSError as e:
            exc = e
    if dll is None:
        if exc is not None:
            raise exc
        raise OSError('Could not locate libarrow_python.so')
    # symbols must be registered before the first compilation in this process
    for func_name in pyarrow_cpp_funcs:
        func = dll[func_name]
        func_addr = ctypes.cast(func, ctypes.c_void_p).value
        llvmlite.binding.add_symbol(func_name, func_addr)
    return dll


dll = get_pyarrow_dll()


def get_pyarrow_cpp_func(builder: ir.IRBuilder,
                         func_name: str) -> ir.Function:
    fnty = pyarrow_cpp_funcs[func_name]
    fn = cgutils.get_or_insert_function(builder.module, fnty,
                                        func_name)
    return fn
