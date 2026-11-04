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

from collections.abc import Callable

from llvmlite import ir
from numba import extending as nbext
from numba import types as nbt
from numba.core import cgutils
from numba.core.base import BaseContext
from numba.core.typing.context import BaseContext as TypingContext
from numba.core.typing.templates import (
    AttributeTemplate,
    Signature,
    bound_function,
    infer_getattr,
    signature,
)

import pyarrow as pa

from ._utils import _arrow_type_to_nb_type, _is_var_width, get_pyarrow_cpp_func
from .array import PyArrowArrayType, StringViewType, string_view_data
from .poolbuffer import (
    PoolBufferType,
    TypedBufferBuilderType,
    typed_buffer_builder_append_ll,
    typed_buffer_builder_ctor_ll,
    typed_buffer_builder_finish_ll,
)

#
# 0. Pure Python
#


def _data_buffer_type(arrow_type):
    """What a builder appends into: raw bytes for var-width types.

    ``pa.string().byte_width`` raises ``ValueError``, so the payload buffer of
    a var-width builder is typed ``uint8`` and driven one byte at a time.
    """
    return pa.uint8() if _is_var_width(arrow_type) else arrow_type


def _offsets_type(arrow_type):
    """``int64`` offsets for the ``large_*`` types, ``int32`` otherwise."""
    return (pa.int64()
            if arrow_type in (pa.large_string(), pa.large_binary())
            else pa.int32())


class ArrayBuilder:
    def __init__(self, value_type: pa.DataType) -> None:
        supported = (pa.types.is_integer(value_type)
                     or pa.types.is_floating(value_type)
                     or pa.types.is_boolean(value_type)
                     or pa.types.is_string(value_type)
                     or pa.types.is_large_string(value_type)
                     or pa.types.is_binary(value_type)
                     or pa.types.is_large_binary(value_type))
        if not supported:
            # fixed-size binary lands here too: no offsets buffer, so the
            # var-width layout would be wrong
            raise TypeError(f'Cannot create ArrayBuilder for {value_type}')
        self.value_type = value_type


class Int8Builder(ArrayBuilder):
    def __init__(self):
        super().__init__(pa.int8())


class Int16Builder(ArrayBuilder):
    def __init__(self):
        super().__init__(pa.int16())


class Int32Builder(ArrayBuilder):
    def __init__(self):
        super().__init__(pa.int32())


class Int64Builder(ArrayBuilder):
    def __init__(self):
        super().__init__(pa.int64())


class UInt8Builder(ArrayBuilder):
    def __init__(self):
        super().__init__(pa.uint8())


class UInt16Builder(ArrayBuilder):
    def __init__(self):
        super().__init__(pa.uint16())


class UInt32Builder(ArrayBuilder):
    def __init__(self):
        super().__init__(pa.uint32())


class UInt64Builder(ArrayBuilder):
    def __init__(self):
        super().__init__(pa.uint64())


class Float32Builder(ArrayBuilder):
    def __init__(self):
        super().__init__(pa.float32())


class Float64Builder(ArrayBuilder):
    def __init__(self):
        super().__init__(pa.float64())


class BoolBuilder(ArrayBuilder):
    def __init__(self):
        super().__init__(pa.bool_())


class StringBuilder(ArrayBuilder):
    def __init__(self):
        super().__init__(pa.string())


class LargeStringBuilder(ArrayBuilder):
    def __init__(self):
        super().__init__(pa.large_string())


class BinaryBuilder(ArrayBuilder):
    def __init__(self):
        super().__init__(pa.binary())


class LargeBinaryBuilder(ArrayBuilder):
    def __init__(self):
        super().__init__(pa.large_binary())


#
# 1. Typing
#

class ArrayBuilderType(nbt.Type):

    def __init__(self, value_type):
        self._value_type = value_type
        self._nb_type = _arrow_type_to_nb_type(self._value_type)
        super().__init__(name=f'ArrayBuilderType({self._value_type})')

    @property
    def arrow_type(self):
        return self._value_type

    @property
    def numba_type(self):
        return self._nb_type


# lets a builder created in Python (outside @njit) be passed to one
@nbext.typeof_impl.register(ArrayBuilder)
def typeof_builder(val: object, c: TypingContext) -> ArrayBuilderType:
    return ArrayBuilderType(val.value_type)


# one typer per class: the Arrow type is baked in, since PyArrow's own types
# are not Numba types
def _builder_typer(arrow_type: pa.DataType
                   ) -> Callable[[TypingContext], Callable[[], nbt.Type]]:
    def type_callable(context):
        def typer():
            return ArrayBuilderType(arrow_type)
        return typer
    return type_callable


_BUILDERS = (
    (Int8Builder, pa.int8()), (Int16Builder, pa.int16()),
    (Int32Builder, pa.int32()), (Int64Builder, pa.int64()),
    (UInt8Builder, pa.uint8()), (UInt16Builder, pa.uint16()),
    (UInt32Builder, pa.uint32()), (UInt64Builder, pa.uint64()),
    (Float32Builder, pa.float32()), (Float64Builder, pa.float64()),
    (BoolBuilder, pa.bool_()),
    (StringBuilder, pa.string()), (LargeStringBuilder, pa.large_string()),
    (BinaryBuilder, pa.binary()), (LargeBinaryBuilder, pa.large_binary()),
)

for _builder_cls, _arrow_type in _BUILDERS:
    nbext.type_callable(_builder_cls)(_builder_typer(_arrow_type))


@infer_getattr
class ArrayBuilderAttribute(AttributeTemplate):
    key = ArrayBuilderType

    @bound_function("ArrayBuilder.finish")
    def resolve_finish(self, buffer_builder, args, kws):
        if args or kws:
            return
        return signature(PyArrowArrayType(buffer_builder.arrow_type))


#
# 2. Data model
#

@nbext.register_model(ArrayBuilderType)
class ArrayBuilderModel(nbext.models.StructModel):

    def __init__(self, dmm, fe_type):
        arrow_type = fe_type.arrow_type
        members = [
            ('null_bitmap_builder', TypedBufferBuilderType(pa.bool_())),
            ('data_builder',
             TypedBufferBuilderType(_data_buffer_type(arrow_type))),
            ('null_count_ptr', nbt.EphemeralPointer(nbt.int64)),
        ]
        if _is_var_width(arrow_type):
            members.append(('offsets_builder',
                            TypedBufferBuilderType(_offsets_type(arrow_type))))
        super().__init__(dmm, fe_type, members)


#
# 3. Code generation
#


nbext.make_attribute_wrapper(
    ArrayBuilderType, 'null_bitmap_builder', 'null_bitmap_builder')
nbext.make_attribute_wrapper(ArrayBuilderType, 'data_builder', 'data_builder')
nbext.make_attribute_wrapper(ArrayBuilderType, 'offsets_builder',
                             'offsets_builder')


# a plain lowering, not an @intrinsic: the model embeds EphemeralPointers,
# which the intrinsic path cannot handle
def builder_ctor(context: BaseContext, builder: ir.IRBuilder,
                 sig: Signature, args: tuple[ir.Value, ...]) -> ir.Value:
    value_type = sig.return_type.arrow_type
    array_builder = context.make_helper(builder, sig.return_type)
    array_builder.null_bitmap_builder = typed_buffer_builder_ctor_ll(
        context, builder, TypedBufferBuilderType(pa.bool_()))
    array_builder.data_builder = typed_buffer_builder_ctor_ll(
        context, builder, TypedBufferBuilderType(_data_buffer_type(value_type)))
    zero = context.get_constant(nbt.int64, 0)
    array_builder.null_count_ptr = cgutils.alloca_once_value(builder, zero)
    if _is_var_width(value_type):
        offsets_ty = TypedBufferBuilderType(_offsets_type(value_type))
        offsets_builder = typed_buffer_builder_ctor_ll(
            context, builder, offsets_ty)
        offset_zero = builder.trunc(
            zero, context.get_value_type(
                _arrow_type_to_nb_type(_offsets_type(value_type))))
        array_builder.offsets_builder = typed_buffer_builder_append_ll(
            context, builder, offsets_ty, offsets_builder, offset_zero)

    return array_builder._getvalue()


for _builder_cls, _ in _BUILDERS:
    nbext.lower_builtin(_builder_cls)(builder_ctor)


@nbext.intrinsic
def builder_increment_null_count(typingctx: TypingContext,
                                 array_builder: nbt.Type
                                 ) -> tuple[Signature, Callable[..., None]]:
    assert isinstance(array_builder, ArrayBuilderType)
    sig = nbt.none(array_builder)

    def codegen(context, builder, sig, args):
        array_builder = context.make_helper(builder, sig.args[0], value=args[0])
        null_count = builder.load(array_builder.null_count_ptr)
        builder.store(builder.add(null_count, null_count.type(1)),
                      array_builder.null_count_ptr)

    return sig, codegen


@nbext.intrinsic
def bytes_data(typingctx: TypingContext, value_type: nbt.Type) -> object:
    """The first byte of a ``bytes`` value, for the bulk builder copy."""

    def codegen(context, builder, signature, args):
        value = context.make_helper(builder, value_type, value=args[0])
        return builder.bitcast(value.data, ir.IntType(8).as_pointer())

    return signature(nbt.voidptr, value_type), codegen


@nbext.overload_method(ArrayBuilderType, 'append')
def builder_append(builder, value):
    if _is_var_width(builder.arrow_type):
        if value == nbt.none:
            def append_null(builder, value):
                builder.null_bitmap_builder.append(False)
                builder.offsets_builder.append(builder.data_builder.size)
                builder_increment_null_count(builder)
            return append_null

        if isinstance(value, nbt.Bytes):
            def append_bytes(builder, value):
                builder.data_builder.append_bytes(bytes_data(value), len(value))
                builder.null_bitmap_builder.append(True)
                builder.offsets_builder.append(builder.data_builder.size)
            return append_bytes

        if isinstance(value, StringViewType):
            def append_view(builder, value):
                if value.valid:
                    builder.data_builder.append_bytes(
                        string_view_data(value), value.end - value.start)
                else:
                    builder_increment_null_count(builder)
                builder.null_bitmap_builder.append(value.valid)
                builder.offsets_builder.append(builder.data_builder.size)
            return append_view

        return

    if value == nbt.none:
        # the data buffer still needs a placeholder of its own type
        element = builder.numba_type
        if isinstance(element, nbt.Float):
            zero = 0.0
        elif element == nbt.boolean:
            zero = False
        else:
            zero = 0

        def append_impl(builder, value):
            builder.null_bitmap_builder.append(False)
            builder.data_builder.append(zero)
            builder_increment_null_count(builder)
        return append_impl
    else:
        def append_impl(builder, value):
            builder.null_bitmap_builder.append(True)
            builder.data_builder.append(value)
        return append_impl


@nbext.overload_method(ArrayBuilderType, 'reserve')
def builder_reserve(builder, capacity):
    if _is_var_width(builder.arrow_type):
        def impl_var_width(builder, capacity):
            builder.null_bitmap_builder.reserve(capacity)
            builder.data_builder.reserve(capacity)
            builder.offsets_builder.reserve(capacity)
        return impl_var_width

    def impl(builder, capacity):
        builder.null_bitmap_builder.reserve(capacity)
        builder.data_builder.reserve(capacity)
    return impl


@nbext.lower_builtin('ArrayBuilder.finish', ArrayBuilderType)
def builder_finish(context: BaseContext, builder: ir.IRBuilder,
                   sig: Signature,
                   args: tuple[ir.Value, ...]) -> ir.Value:
    value_type = sig.args[0].arrow_type
    array_builder = context.make_helper(builder, sig.args[0], value=args[0])

    null_bitmap_type = TypedBufferBuilderType(pa.bool_())
    length = builder.load(context.make_helper(
        builder, null_bitmap_type,
        value=array_builder.null_bitmap_builder).index_ptr)
    null_count = builder.load(array_builder.null_count_ptr)

    null_bitmap_buffer = typed_buffer_builder_finish_ll(
        context, builder, null_bitmap_type, array_builder.null_bitmap_builder)
    data_buffer = typed_buffer_builder_finish_ll(
        context, builder, TypedBufferBuilderType(_data_buffer_type(value_type)),
        array_builder.data_builder)

    def get_buffer_ptr(pool_buffer):
        pool_buffer = context.make_helper(builder, PoolBufferType(), value=pool_buffer)
        return pool_buffer.cpp_ptr

    buffers = [get_buffer_ptr(null_bitmap_buffer)]
    if _is_var_width(value_type):
        offsets_ty = TypedBufferBuilderType(_offsets_type(value_type))
        offsets_builder = array_builder.offsets_builder
        buffers.append(get_buffer_ptr(
            typed_buffer_builder_finish_ll(context, builder, offsets_ty,
                                           offsets_builder)))
    buffers.append(get_buffer_ptr(data_buffer))

    buffer_ptrs = cgutils.pack_array(builder, buffers)
    # LLVM doesn't implicitly cast `[n x i8]*` to `i8**`
    buffer_ptrs = builder.bitcast(
        cgutils.alloca_once_value(builder, buffer_ptrs), cgutils.voidptr_t.as_pointer())
    num_buffers = cgutils.int32_t(len(buffers))

    out_array = cgutils.create_struct_proxy(
        PyArrowArrayType(value_type))(context, builder)
    out_array_ptr = builder.bitcast(out_array._getpointer(), cgutils.voidptr_t)

    fn = get_pyarrow_cpp_func(builder, 'PyArrow_MakeAndExportArray')
    status = builder.call(
        fn, [length, null_count, num_buffers, buffer_ptrs, out_array_ptr])
    # the helper returns the Arrow status as a signed int: negative is an error
    # and leaves a Python exception set
    with builder.if_then(builder.icmp_signed('<', status, status.type(0))):
        context.call_conv.return_user_exc(
            builder, RuntimeError,
            ("failed to export the Arrow array",))
    return out_array._getvalue()
