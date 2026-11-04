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

"""Growable Arrow buffers backed by the C++ pool.

``TypedBufferBuilder`` appends values (or bits, for booleans) one at a time
into a ``pa.Buffer`` owned by the Arrow memory pool.  Growth doubles the
capacity, so appending n items costs one reallocation per doubling.

Only the pieces the array builders need are exposed: ``reserve``, ``append``,
``size`` and the code-generation helpers the builder lowering calls.
"""

from __future__ import annotations

import ctypes

from llvmlite import ir
from numba import extending as nbext
from numba import types as nbt
from numba.core import cgutils
from numba.core.base import BaseContext
from numba.core.typing.templates import (
    AttributeTemplate,
    bound_function,
    infer_getattr,
    signature,
)

import pyarrow as pa

from ._utils import (
    _arrow_type_to_nb_type,
    get_pyarrow_cpp_func,
    get_pyarrow_dll,
    int64_t,
)

#
# 1. Typing
#


class PoolBufferType(nbt.Type):

    def __init__(self):
        super().__init__(name='PoolBuffer')


class TypedBufferBuilderType(nbt.Type):

    def __init__(self, value_type):
        self._value_type = value_type
        self._nb_type = _arrow_type_to_nb_type(self._value_type)
        super().__init__(name=f'TypedBufferBuilder({self._value_type})')

    @property
    def arrow_type(self):
        return self._value_type

    @property
    def numba_type(self):
        return self._nb_type


@infer_getattr
class TypedBufferBuilderAttribute(AttributeTemplate):
    key = TypedBufferBuilderType

    @bound_function("TypedBufferBuilder.reserve")
    def resolve_reserve(self, buffer_builder, args, kws):
        if kws or len(args) != 1:
            return
        return signature(nbt.none, nbt.int64)

    @bound_function("TypedBufferBuilder.append")
    def resolve_append(self, buffer_builder, args, kws):
        if kws or len(args) != 1:
            return
        return signature(nbt.none, buffer_builder.numba_type)

    @bound_function("TypedBufferBuilder.append_bytes")
    def resolve_append_bytes(self, buffer_builder, args, kws):
        if kws or len(args) != 2:
            return
        return signature(nbt.none, nbt.voidptr, nbt.int64)


#
# 2. Data model
#


@nbext.register_model(PoolBufferType)
class PoolBufferModel(nbext.models.StructModel):
    cpp_ptr_type = nbt.RawPointer('PyArrow_PoolBufferPtr')

    def __init__(self, dmm, fe_type):
        members = [
            ('cpp_ptr', self.cpp_ptr_type),
        ]
        super().__init__(dmm, fe_type, members)


@nbext.register_model(TypedBufferBuilderType)
class TypedBufferBuilderModel(nbext.models.StructModel):
    data_ptr_type = nbt.RawPointer('uint8_t*')

    def __init__(self, dmm, fe_type):
        members = [
            ('pool_buffer', PoolBufferType()),
            ('index_ptr', nbt.EphemeralPointer(nbt.int64)),
            ('capacity_ptr', nbt.EphemeralPointer(nbt.int64)),
            ('data_ptr_ptr', nbt.EphemeralPointer(self.data_ptr_type)),
        ]
        super().__init__(dmm, fe_type, members)


#
# 3. Code generation
#

class _PoolBufferMeta(ctypes.Structure):
    """The C++ pool buffer's field offsets, in the order numba.cc fills them."""

    _fields_ = [
        ("offset_of_data", ctypes.c_int32),
        ("offset_of_size", ctypes.c_int32),
        ("offset_of_capacity", ctypes.c_int32),
    ]


def _pool_buffer_offsets() -> tuple[int, int, int]:
    """``(data, size, capacity)`` byte offsets, read from the C++ layout."""
    meta = _PoolBufferMeta()
    get_pyarrow_dll().PyArrow_GetPoolBufferMeta(ctypes.byref(meta))
    return (meta.offset_of_data, meta.offset_of_size, meta.offset_of_capacity)


(_pool_buffer_offsetof_data, _pool_buffer_offsetof_size,
 _pool_buffer_offsetof_capacity) = _pool_buffer_offsets()

if not (len({_pool_buffer_offsetof_data, _pool_buffer_offsetof_size,
             _pool_buffer_offsetof_capacity}) == 3
        and 0 < _pool_buffer_offsetof_data < 64
        and 0 < _pool_buffer_offsetof_size < 64
        and 0 < _pool_buffer_offsetof_capacity < 64):
    raise RuntimeError(
        "unexpected PoolBuffer layout: "
        f"data={_pool_buffer_offsetof_data}, size={_pool_buffer_offsetof_size}, "
        f"capacity={_pool_buffer_offsetof_capacity}")


def pool_buffer_get_properties(
        context: BaseContext, builder: ir.IRBuilder,
        pool_buffer_ty: nbt.Type,
        pool_buffer: ir.Value) -> tuple[ir.Value, ir.Value, ir.Value]:
    """Load ``(size, capacity, data)`` out of the C++ pool buffer."""
    pool_buffer = context.make_helper(builder, pool_buffer_ty, value=pool_buffer)

    def field_ptr(offset, ty):
        return builder.inttoptr(
            builder.add(builder.ptrtoint(pool_buffer.cpp_ptr, int64_t),
                        int64_t(offset)), ty.as_pointer())

    size_ptr = field_ptr(_pool_buffer_offsetof_size, int64_t)
    capa_ptr = field_ptr(_pool_buffer_offsetof_capacity, int64_t)
    data_ptr = field_ptr(_pool_buffer_offsetof_data, cgutils.voidptr_t)

    return builder.load(size_ptr), builder.load(capa_ptr), builder.load(data_ptr)


def pool_buffer_realloc_ll(context, builder, func_name, pool_buffer_ty,
                           pool_buffer, byte_size, check_errors=False):
    pool_buffer = context.make_helper(builder, pool_buffer_ty, value=pool_buffer)

    fn = get_pyarrow_cpp_func(builder, func_name)
    status = builder.call(fn, [pool_buffer.cpp_ptr, byte_size])
    if not check_errors:
        return
    # the C++ returns a status and leaves a Python exception behind on failure
    with builder.if_then(builder.icmp_signed('<', status, status.type(0))):
        context.call_conv.return_exc(builder)


def get_item_capacity(builder: ir.IRBuilder, item_type: pa.DataType,
                      byte_capacity: ir.Value) -> ir.Value:
    if pa.types.is_boolean(item_type):
        return builder.mul(byte_capacity, byte_capacity.type(8))
    else:
        return builder.udiv(byte_capacity, byte_capacity.type(item_type.byte_width))


def get_byte_capacity(builder: ir.IRBuilder, item_type: pa.DataType,
                      item_capacity: ir.Value) -> ir.Value:
    if pa.types.is_boolean(item_type):
        return builder.udiv(builder.add(item_capacity, item_capacity.type(7)),
                            item_capacity.type(8))
    else:
        return builder.mul(item_capacity, item_capacity.type(item_type.byte_width))


@nbext.overload_attribute(TypedBufferBuilderType, 'size')
def typed_buffer_builder_size(buffer_builder):
    def get(buffer_builder):
        return typed_buffer_builder_size_ll(buffer_builder)

    return get


@nbext.intrinsic
def typed_buffer_builder_size_ll(typingctx, buffer_builder_type):
    assert isinstance(buffer_builder_type, TypedBufferBuilderType)
    sig = nbt.int64(buffer_builder_type)

    def codegen(context, builder, signature, args):
        buffer_builder = context.make_helper(
            builder, signature.args[0], value=args[0])
        return builder.load(buffer_builder.index_ptr)

    return sig, codegen


def typed_buffer_builder_ctor_ll(context, builder, buffer_builder_type):
    value_type = buffer_builder_type.arrow_type

    pool_buffer = context.make_helper(builder, PoolBufferType())
    fn = get_pyarrow_cpp_func(builder, 'PyArrow_AllocatePoolBuffer')
    # start empty and let the first append double the capacity
    pool_buffer.cpp_ptr = builder.call(fn, [int64_t(0)])
    with builder.if_then(cgutils.is_null(builder, pool_buffer.cpp_ptr)):
        context.call_conv.return_exc(builder)

    buffer_builder = context.make_helper(builder, buffer_builder_type)
    buffer_builder.pool_buffer = pool_buffer._getvalue()
    zero = context.get_constant(nbt.int64, 0)
    buffer_builder.index_ptr = cgutils.alloca_once_value(builder, zero)
    _, capa, data = pool_buffer_get_properties(
        context, builder, PoolBufferType(), buffer_builder.pool_buffer)

    buffer_builder.capacity_ptr = cgutils.alloca_once_value(
        builder, get_item_capacity(builder, value_type, capa))
    buffer_builder.data_ptr_ptr = cgutils.alloca_once_value(builder, data)
    return buffer_builder._getvalue()


def typed_buffer_builder_finish_ll(context, builder, buffer_builder_type,
                                   buffer_builder):
    value_type = buffer_builder_type.arrow_type
    buffer_builder = context.make_helper(
        builder, buffer_builder_type, value=buffer_builder)
    final_size = builder.load(buffer_builder.index_ptr)
    final_byte_size = get_byte_capacity(builder, value_type, final_size)
    pool_buffer_realloc_ll(
        context, builder, 'PyArrow_PoolBufferResizeExactly', PoolBufferType(),
        buffer_builder.pool_buffer, final_byte_size, check_errors=True)
    return buffer_builder.pool_buffer


def grow_ll(context: BaseContext, builder: ir.IRBuilder,
            value_type: pa.DataType, buffer_builder: ir.Value,
            new_capa: ir.Value, check_errors: bool = False) -> None:
    """Reserve *new_capa* items and publish the pool buffer's new state."""
    new_byte_capacity = get_byte_capacity(builder, value_type, new_capa)
    pool_buffer_realloc_ll(context, builder, 'PyArrow_PoolBufferReserve',
                           PoolBufferType(), buffer_builder.pool_buffer,
                           new_byte_capacity, check_errors)
    _, new_capa, new_data = pool_buffer_get_properties(
        context, builder, PoolBufferType(), buffer_builder.pool_buffer)
    builder.store(get_item_capacity(builder, value_type, new_capa),
                  buffer_builder.capacity_ptr)
    builder.store(new_data, buffer_builder.data_ptr_ptr)


@nbext.lower_builtin('TypedBufferBuilder.reserve', TypedBufferBuilderType, nbt.int64)
def typed_buffer_builder_reserve(context, builder, sig, args):
    buffer_builder_type = sig.args[0]
    value_type = buffer_builder_type.arrow_type

    buffer_builder = context.make_helper(builder, sig.args[0], value=args[0])
    current_size = builder.load(buffer_builder.index_ptr)
    additional_capa = args[1]
    new_capa = builder.add(current_size, additional_capa)

    grow_ll(context, builder, value_type, buffer_builder, new_capa,
            check_errors=True)


def typed_buffer_builder_append_ll(context, builder, buffer_builder_type,
                                   buffer_builder, data_value):
    """Append *data_value* to a pool-backed buffer; returns the new value.

    *buffer_builder* is a buffer-builder struct value: the result must be
    assigned back where it came from, since a reallocation may have replaced
    the pool buffer.
    """
    value_type = buffer_builder_type.arrow_type
    buffer_builder = context.make_helper(
        builder, buffer_builder_type, value=buffer_builder)

    index = builder.load(buffer_builder.index_ptr)
    index_plus_one = builder.add(index, index.type(1))
    capa = builder.load(buffer_builder.capacity_ptr)

    needed_capa = index_plus_one
    not_enough_capa = builder.icmp_signed('<', capa, needed_capa)
    with builder.if_then(not_enough_capa, likely=False):
        grow_ll(context, builder, value_type, buffer_builder,
                builder.mul(needed_capa, capa.type(2)))

    data = builder.load(buffer_builder.data_ptr_ptr)
    if pa.types.is_boolean(value_type):
        byte_index = builder.udiv(index, index.type(8))
        bit_index = builder.urem(index, index.type(8))
        mask = builder.shl(cgutils.int8_t(1),
                           builder.trunc(bit_index, cgutils.int8_t))
        flipped_mask = builder.not_(mask)
        byte_ptr = builder.gep(data, [byte_index])
        byte = builder.load(byte_ptr)
        byte = builder.select(data_value, builder.or_(byte, mask),
                              builder.and_(byte, flipped_mask))
        builder.store(byte, byte_ptr)
    else:
        data = builder.bitcast(data, data_value.type.as_pointer())
        builder.store(data_value, builder.gep(data, [index]))
    builder.store(index_plus_one, buffer_builder.index_ptr)
    return buffer_builder._getvalue()


@nbext.lower_builtin('TypedBufferBuilder.append', TypedBufferBuilderType, nbt.Any)
def typed_buffer_builder_append(context, builder, sig, args):
    return typed_buffer_builder_append_ll(
        context, builder, sig.args[0], args[0], args[1])


def typed_buffer_builder_append_bytes_ll(
        context: BaseContext, builder: ir.IRBuilder,
        buffer_builder_type: TypedBufferBuilderType,
        buffer_builder: ir.Value, source: ir.Value,
        nbytes: ir.Value) -> ir.Value:
    """Copy *nbytes* from *source* in one capacity check and one memcpy.

    Appending one byte at a time costs a load/compare/store/index update per
    byte (measured: 56 instructions per 8-byte string); this is the bulk form
    the string and binary builders use. Like ``append``, the returned builder
    must be assigned back: growing may replace the pool buffer.
    """
    value_type = buffer_builder_type.arrow_type
    buffer_builder = context.make_helper(
        builder, buffer_builder_type, value=buffer_builder)

    index = builder.load(buffer_builder.index_ptr)
    capa = builder.load(buffer_builder.capacity_ptr)
    needed_capa = builder.add(index, nbytes)
    not_enough_capa = builder.icmp_signed('<', capa, needed_capa)
    with builder.if_then(not_enough_capa, likely=False):
        grow_ll(context, builder, value_type, buffer_builder,
                builder.mul(needed_capa, capa.type(2)))

    data = builder.load(buffer_builder.data_ptr_ptr)
    destination = builder.gep(builder.bitcast(data, ir.IntType(8).as_pointer()),
                              [index])
    cgutils.raw_memcpy(builder, destination, source, nbytes, 1)
    builder.store(needed_capa, buffer_builder.index_ptr)
    return buffer_builder._getvalue()


@nbext.lower_builtin('TypedBufferBuilder.append_bytes',
                     TypedBufferBuilderType, nbt.voidptr, nbt.int64)
def typed_buffer_builder_append_bytes(context, builder, sig, args):
    return typed_buffer_builder_append_bytes_ll(
        context, builder, sig.args[0], args[0], args[1], args[2])
