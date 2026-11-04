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

import functools
import operator
from collections.abc import Callable
from contextlib import ExitStack

import numpy as np
from llvmlite import ir
from numba import extending as nbext
from numba import types as nbt
from numba.core import cgutils
from numba.core.base import BaseContext
from numba.core.datamodel.manager import DataModelManager
from numba.core.pythonapi import PythonAPI
from numba.core.typing.context import BaseContext as TypingContext
from numba.core.typing.templates import Signature, signature

import pyarrow as pa

from ._capsule import dtype_of, num_buffers, take_array_into
from ._utils import _arrow_type_to_nb_type, _is_var_width, get_pyarrow_cpp_func, int64_t

#
# 1. Typing
#


class PyArrowArrayType(nbt.Type):
    def __init__(self, type: pa.DataType, nullable: bool = True) -> None:
        self._type = type
        self._nullable = nullable
        self._nb_type = _arrow_type_to_nb_type(type)
        super().__init__(name=f"PyArrowArray({type})")

    @property
    def key(self) -> tuple:
        # decides whether the null test is emitted at all, so it is part of the
        # type; producers that cannot say cheaply leave it True and pay the test
        return (self.name, self._nullable)

    @property
    def nullable(self) -> bool:
        return self._nullable

    @property
    def arrow_type(self):
        return self._type

    @property
    def numba_type(self):
        return self._nb_type


@functools.lru_cache(maxsize=None)
def _array_type(arrow_type: pa.DataType, nullable: bool = True) -> PyArrowArrayType:
    """One Numba type per (Arrow type, nullability): typeof() runs per call."""
    return PyArrowArrayType(arrow_type, nullable)


@nbext.typeof_impl.register(pa.Array)
def typeof_array(val: object, c: TypingContext) -> PyArrowArrayType:
    assert isinstance(val, pa.Array)
    return _array_type(val.type, val.null_count != 0)


def register_producer(cls: type) -> type:
    """Let ``@njit`` accept arrays of *cls* via the Arrow PyCapsule interface.

    Call once per producer class before compiling::

        register_producer(polars.Series)

    The type is derived from the producer's own dtype, so a producer of a
    supported Arrow type needs no further setup.
    """

    @nbext.typeof_impl.register(cls)
    def typeof_producer(val, c):
        return _array_type(dtype_of(val))

    return cls


for _field in ("length", "null_count", "offset"):
    nbext.make_attribute_wrapper(PyArrowArrayType, _field, _field)


class StringViewType(nbt.Type):
    """A non-owning view of one string or binary element."""

    def __init__(self) -> None:
        super().__init__(name="ArrowStringView")


#
# 2. Data model
#


@nbext.register_model(PyArrowArrayType)
class ArrayModel(nbext.models.StructModel):
    def __init__(self, dmm: DataModelManager, fe_type: nbt.Type) -> None:
        # Mirrors the C ABI; _capsule.ArrowArray is the authority. Everything
        # is word-sized, so a mismatch only shows up as wrong data, not a crash.
        # By value on purpose: unboxing writes the struct in place and its
        # cleanup releases the buffers, so there is nothing to heap-allocate.
        members = [
            ("length", nbt.int64),
            ("null_count", nbt.int64),
            ("offset", nbt.int64),
            ("n_buffers", nbt.int64),
            ("n_children", nbt.int64),
            ("buffers", nbt.CPointer(nbt.CPointer(nbt.uint8))),
            ("children", nbt.CPointer(nbt.CPointer(nbt.uint8))),
            ("dictionary", nbt.CPointer(nbt.uint8)),
            ("release_cb", nbt.CPointer(nbt.uint8)),
            ("private_data", nbt.CPointer(nbt.uint8)),
        ]
        super().__init__(dmm, fe_type, members)


@nbext.register_model(StringViewType)
class StringViewModel(nbext.models.StructModel):
    """chars/start/end: a byte range inside the array's data buffer."""

    def __init__(self, dmm: DataModelManager, fe_type: nbt.Type) -> None:
        super().__init__(
            dmm,
            fe_type,
            [
                ("chars", nbt.CPointer(nbt.uint8)),
                ("start", nbt.int64),
                ("end", nbt.int64),
                ("valid", nbt.boolean),
            ],
        )


for _field in ("chars", "start", "end", "valid"):
    nbext.make_attribute_wrapper(StringViewType, _field, _field)


#
# 3. Code generation
#


def _safe_buffers_ll(
    context: BaseContext, builder: ir.IRBuilder, array: type, n_buffers: int
) -> None:
    """Copy the buffer pointers, putting a zero word where Arrow has none.

    A dense array leaves ``buffers[0]`` null. Pointing our copy at zeroed
    memory makes the per-element bitmap load unconditional, which is what lets
    the null test vectorise; ``null_count`` then decides if the bit counts.
    """
    ptr_t = context.get_value_type(nbt.uint8).as_pointer()
    source = array.buffers
    copy = cgutils.alloca_once(builder, ptr_t, size=n_buffers)
    zero_word = cgutils.alloca_once(builder, ir.IntType(64), zfill=True)
    zero_ptr = builder.bitcast(zero_word, ptr_t)

    has_source = builder.icmp_unsigned(
        "!=", builder.ptrtoint(source, int64_t), int64_t(0)
    )
    with builder.if_then(has_source):
        for i in range(n_buffers):
            pointer = builder.load(cgutils.gep(builder, source, i))
            builder.store(
                builder.select(
                    builder.icmp_unsigned(
                        "!=", builder.ptrtoint(pointer, int64_t), int64_t(0)
                    ),
                    pointer,
                    zero_ptr,
                ),
                cgutils.gep(builder, copy, i),
            )

    # a failed export leaves no buffers: keep the null pointer, the caller is
    # raising anyway
    array.buffers = builder.select(has_source, copy, source)


@nbext.unbox(PyArrowArrayType)
def unbox_array(
    typ: PyArrowArrayType, obj: ir.Value, c: PythonAPI
) -> nbext.NativeValue:
    """
    Unbox a pa.Array object as a ArrowArray struct wrapper.
    """
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    c_array = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    with ExitStack() as stack:
        # one Python call per argument: _capsule.take_array_into moves the
        # producer's ArrowArray into this struct (0.2 us for a pa.Array)
        array_ptr_obj = c.pyapi.long_from_unsigned_int(
            c.builder.ptrtoint(c_array._getpointer(), cgutils.intp_t)
        )
        # the struct is moved out of the producer's capsule: no pa.Array here
        export_obj = c.pyapi.unserialize(c.pyapi.serialize_object(take_array_into))
        dtype_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.arrow_type))
        ret_obj = c.pyapi.call_function_objargs(
            export_obj, (obj, array_ptr_obj, dtype_obj)
        )
        c.pyapi.decref(export_obj)
        c.pyapi.decref(dtype_obj)
        c.pyapi.decref(array_ptr_obj)
        with cgutils.early_exit_if_null(c.builder, stack, ret_obj):
            c.builder.store(cgutils.true_bit, is_error_ptr)
        c.pyapi.decref(ret_obj)

    _safe_buffers_ll(c.context, c.builder, c_array, num_buffers(typ.arrow_type))

    def cleanup():
        c_array_ptr = c.builder.bitcast(c_array._getpointer(), cgutils.voidptr_t)
        fn = get_pyarrow_cpp_func(c.builder, "PyArrow_ReleaseArray")
        c.builder.call(fn, [c_array_ptr])

    return nbext.NativeValue(
        c_array._getvalue(), cleanup=cleanup, is_error=c.builder.load(is_error_ptr)
    )


@nbext.box(PyArrowArrayType)
def box_array(typ: PyArrowArrayType, val: ir.Value, c: PythonAPI) -> ir.Value:
    """
    Box a ArrowArray struct as a pa.Array object.

    Only structs created inside compiled code (a builder's ``finish()``) may be
    boxed. Boxing a struct that came from an *argument* re-uses that argument's
    ArrowArray: pyarrow's import keeps pointing at it, so unboxing's cleanup
    releases the buffers the returned array still reads. Clearing the source's
    release callback was tried and segfaults; returning an array argument stays
    unsupported until the struct itself is owned by the caller.
    """
    c_array = c.context.make_helper(c.builder, typ, value=val)

    with ExitStack():
        # a handful of Python calls, once per built array, not per element
        pa_name = c.context.insert_const_string(c.builder.module, "pyarrow")
        pa_mod = c.pyapi.import_module(pa_name)
        array_cls = c.pyapi.object_getattr_string(pa_mod, "Array")
        c.pyapi.decref(pa_mod)
        datatype_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.arrow_type))
        array_ptr_obj = c.pyapi.long_from_unsigned_int(
            c.builder.ptrtoint(c_array._getpointer(), cgutils.intp_t)
        )
        ret_obj = c.pyapi.call_method(
            array_cls, "_import_from_c", (array_ptr_obj, datatype_obj)
        )
        c.pyapi.decref(array_cls)
        c.pyapi.decref(array_ptr_obj)
        c.pyapi.decref(datatype_obj)
        return ret_obj


def make_array(typ: PyArrowArrayType) -> type:
    """
    Return the Structure representation of the given array type.
    (an instance of types.PyArrowArrayType).
    """
    # NOTE see numba/np/arrayobj.py to perhaps add annotations for speed
    assert isinstance(typ, PyArrowArrayType)
    return cgutils.create_struct_proxy(typ)


@nbext.intrinsic
def ll_array_is_null(
    typingctx: TypingContext, arr_type: nbt.Type, index_type: nbt.Type
) -> tuple[Signature, Callable[..., ir.Value]] | None:
    assert isinstance(arr_type, PyArrowArrayType)
    result_type = nbt.bool_
    # Force index to int64
    sig = result_type(arr_type, nbt.int64)

    def codegen(context, builder, signature, args):
        (arr_val, index) = args
        return _array_is_null_ll(context, builder, arr_type, arr_val, index)

    return sig, codegen


def _array_is_null_ll(context, builder, arr_type, arr_val, index):
    """True when element *index* is null; straight-line, no branch.

    The validity pointer is never null here (unbox substitutes a zero word for
    the slot a dense array leaves empty), so the bit load is unconditional and
    the ``null_count`` gate is a plain ``and``: a numeric loop stays
    vectorisable, and an array with no nulls pays only that.
    """
    if not arr_type.nullable:
        # no nulls: no load, no test, no branch
        return cgutils.false_bit

    array = make_array(arr_type)(context, builder, arr_val)

    has_nulls = builder.icmp_unsigned("!=", array.null_count, array.null_count.type(0))
    bitmap = builder.load(cgutils.gep(builder, array.buffers, 0, inbounds=True))

    numeric = isinstance(arr_type.numba_type, nbt.Number)
    word_t = ir.IntType(64) if numeric else cgutils.int8_t
    shift = word_t.width.bit_length() - 1  # 6 for u64, 3 for i8
    within = word_t.width - 1

    bit_index = builder.add(array.offset, index)
    word = builder.load(
        cgutils.gep(
            builder,
            builder.bitcast(bitmap, word_t.as_pointer()),
            builder.lshr(bit_index, bit_index.type(shift)),
        )
    )
    mask = builder.shl(
        ir.Constant(word_t, 1),
        builder.trunc(builder.and_(bit_index, bit_index.type(within)), word_t),
    )
    return builder.and_(
        has_nulls,
        builder.icmp_unsigned("==", builder.and_(word, mask), ir.Constant(word_t, 0)),
    )


def nb_type_to_ll_type(ty: nbt.Type) -> ir.Type:
    if ty in nbt.integer_domain:
        return ir.IntType(ty.bitwidth)
    if ty == nbt.float32:
        return ir.FloatType()
    if ty == nbt.float64:
        return ir.DoubleType()
    raise TypeError(f"Unsupported type {ty}")


@nbext.intrinsic
def ll_array_get_optional(
    typingctx: TypingContext, arr_type: nbt.Type, index_type: nbt.Type
) -> tuple[Signature, Callable[..., ir.Value]]:
    """Element *index* as an ``Optional``, value read unconditionally.

    A fixed-width data buffer has a slot per element whether or not that
    element is null, so the load can sit outside the null test. Keeping it
    straight-line is what lets LLVM turn a caller's ``if v is None`` into a
    select instead of a mispredicted branch per element.
    """
    assert isinstance(arr_type, PyArrowArrayType)
    result_type = nbt.Optional(arr_type.numba_type)
    sig = result_type(arr_type, nbt.int64)

    def codegen(context, builder, signature, args):
        (arr_val, index) = args
        array = make_array(arr_type)(context, builder, arr_val)
        data = builder.load(cgutils.gep(builder, array.buffers, 1, inbounds=True))
        data = builder.bitcast(
            data, nb_type_to_ll_type(arr_type.numba_type).as_pointer()
        )
        element = builder.add(array.offset, index)

        out = cgutils.create_struct_proxy(result_type)(context, builder)
        out.data = builder.load(cgutils.gep(builder, data, element))
        out.valid = builder.xor(
            _array_is_null_ll(context, builder, arr_type, arr_val, index),
            cgutils.true_bit,
        )
        return out._getvalue()

    return sig, codegen


@nbext.intrinsic
def ll_array_get_value(
    typingctx: TypingContext, arr_type: nbt.Type, index_type: nbt.Type
) -> tuple[Signature, Callable[..., ir.Value]] | None:
    assert isinstance(arr_type, PyArrowArrayType)
    decimal = pa.types.is_decimal(arr_type.arrow_type)
    packed_bool = arr_type.arrow_type == pa.bool_()
    result_type = arr_type.numba_type
    if packed_bool:
        sig = nbt.boolean(arr_type, nbt.int64)
    elif not decimal and not isinstance(result_type, nbt.Number):
        raise nbt.TypingError(
            f"{arr_type.arrow_type} values are bit-packed: there is no element "
            "accessor, but is_valid() and the raw buffers still work"
        )
    # Force index to int64
    if not packed_bool:
        sig = result_type(arr_type, nbt.int64)

    def codegen(context, builder, signature, args):
        (arr_val, index) = args
        arrayty = make_array(arr_type)
        array = arrayty(context, builder, arr_val)

        data_buffer = builder.load(
            cgutils.gep(builder, array.buffers, 1, inbounds=True)
        )
        index = builder.add(array.offset, index)

        if packed_bool:
            # bit-packed values: one bit per element, no byte per value
            byte = builder.load(
                cgutils.gep(builder, data_buffer, builder.lshr(index, index.type(3)))
            )
            mask = builder.shl(
                cgutils.int8_t(1),
                builder.trunc(builder.and_(index, index.type(7)), cgutils.int8_t),
            )
            return builder.icmp_unsigned("!=", builder.and_(byte, mask), byte.type(0))

        if decimal:
            # 128-bit little-endian two's complement, returned as (high, low):
            # exact, and tuple order compares like the value
            words = builder.bitcast(data_buffer, ir.IntType(64).as_pointer())
            pair = builder.mul(index, ir.Constant(index.type, 2))
            low = builder.load(cgutils.gep(builder, words, pair))
            high = builder.load(
                cgutils.gep(
                    builder, words, builder.add(pair, ir.Constant(index.type, 1))
                )
            )
            return context.make_tuple(builder, result_type, [high, low])

        data_buffer = builder.bitcast(
            data_buffer, nb_type_to_ll_type(result_type).as_pointer()
        )
        return builder.load(cgutils.gep(builder, data_buffer, index))

    return sig, codegen


@nbext.intrinsic
def ll_string_view(
    typingctx: TypingContext,
    arr_type: nbt.Type,
    index_type: nbt.Type,
    valid_type: nbt.Type,
) -> tuple[Signature, Callable[..., ir.Value]] | None:
    """The byte range of element *index*, offsets width taken from the type."""
    assert isinstance(arr_type, PyArrowArrayType)
    arrow_type = arr_type.arrow_type
    fixed_width = (
        arrow_type.byte_width if pa.types.is_fixed_size_binary(arrow_type) else 0
    )
    wide = pa.types.is_large_string(arrow_type) or pa.types.is_large_binary(arrow_type)
    off_t = ir.IntType(64 if wide else 32)
    sig = StringViewType()(arr_type, nbt.int64, nbt.boolean)

    def codegen(context, builder, signature, args):
        arr_val, index, valid = args
        array = context.make_helper(builder, signature.args[0], value=arr_val)
        i = builder.add(array.offset, index)
        if fixed_width:
            # fixed-size binary has no offsets buffer: [validity, data]
            chars = builder.load(cgutils.gep(builder, array.buffers, 1, inbounds=True))
            start = builder.mul(i, ir.Constant(i.type, fixed_width))
            stop = builder.add(start, ir.Constant(i.type, fixed_width))
        else:
            # string/binary: [validity, offsets, data]
            offsets = builder.load(
                cgutils.gep(builder, array.buffers, 1, inbounds=True)
            )
            chars = builder.load(cgutils.gep(builder, array.buffers, 2, inbounds=True))
            offsets = builder.bitcast(offsets, off_t.as_pointer())
            start = builder.load(cgutils.gep(builder, offsets, i))
            stop = builder.load(
                cgutils.gep(builder, offsets, builder.add(i, i.type(1)))
            )
        out = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        out.valid = valid
        out.chars = builder.bitcast(chars, ir.IntType(8).as_pointer())
        if fixed_width:
            out.start = start
            out.end = stop
        else:
            out.start = builder.sext(start, ir.IntType(64))
            out.end = builder.sext(stop, ir.IntType(64))
        return out._getvalue()

    return sig, codegen


@nbext.intrinsic
def string_view_data(typingctx: TypingContext, view_type: nbt.Type) -> object:
    """The first byte of a view, so a builder can copy the range in one call."""

    def codegen(context, builder, signature, args):
        view = context.make_helper(builder, view_type, value=args[0])
        return builder.gep(view.chars, [view.start])

    return signature(nbt.voidptr, view_type), codegen


@nbext.box(StringViewType)
def box_string_view(typ: StringViewType, val: ir.Value, c: PythonAPI) -> ir.Value:
    """Box a view as ``bytes``: one copy of the element, no decoding."""
    view = c.context.make_helper(c.builder, typ, value=val)
    ptr = c.builder.gep(view.chars, [view.start])
    length = c.builder.sub(view.end, view.start)
    fn = cgutils.get_or_insert_function(
        c.builder.module,
        ir.FunctionType(
            ir.IntType(8).as_pointer(), [cgutils.voidptr_t, cgutils.intp_t]
        ),
        "PyBytes_FromStringAndSize",
    )
    return c.builder.call(
        fn,
        [
            c.builder.bitcast(ptr, cgutils.voidptr_t),
            c.builder.trunc(length, cgutils.intp_t),
        ],
    )


def decimal_to_float(words: tuple[int, int], scale: int) -> float:
    """A double for a ``(high, low)`` decimal128 value.

    Exact only to 53 bits, which is all a double holds; ``words`` itself is
    the exact value: ``high * 2**64 + low``, unscaled by ``scale``.
    """
    raise NotImplementedError("only callable from nopython code")


@nbext.overload(decimal_to_float)
def decimal_to_float_impl(words, scale):
    if not isinstance(words, nbt.BaseTuple) or len(words) != 2:
        return
    if not isinstance(scale, nbt.Integer):
        return

    def impl(words, scale):
        high, low = words
        # the halves separately would cancel to 0 for a value like -1, so the
        # 64-bit cases are converted from the low word alone
        if high == 0:
            unscaled = np.float64(low)
        elif high == -1 and low >= 9223372036854775808:
            unscaled = np.float64(np.int64(low))
        else:
            unscaled = np.float64(high) * 2.0**64 + np.float64(low)
        return unscaled * 10.0 ** (-scale)

    return impl


def string_len(view: object) -> int:
    """Byte length of a string view. Implemented in compiled code."""
    raise NotImplementedError("only callable from nopython code")


@nbext.overload(string_len)
def string_len_impl(view):
    if not isinstance(view, StringViewType):
        return

    def impl(view):
        return view.end - view.start

    return impl


@nbext.overload(operator.getitem)
def array_getitem(arr, index):
    if not isinstance(arr, PyArrowArrayType):
        return
    if index not in nbt.integer_domain:
        raise nbt.TypingError("array indices must be integers")

    if _is_var_width(arr.arrow_type):

        def getitem_var(arr, index):
            # inlined on purpose: Numba never inlines a callee raising
            i = index if index >= 0 else index + arr.length
            if i < 0 or i >= arr.length:
                raise IndexError("Arrow array index out of range")
            return ll_string_view(arr, i, not ll_array_is_null(arr, i))

        return getitem_var

    if pa.types.is_decimal(arr.arrow_type) or arr.arrow_type == pa.bool_():

        def getitem_slow(arr, index):
            # a tuple or a bit-packed value has no slot to load speculatively
            i = index if index >= 0 else index + arr.length
            if i < 0 or i >= arr.length:
                raise IndexError("Arrow array index out of range")
            if ll_array_is_null(arr, i):
                return None
            return ll_array_get_value(arr, i)

        return getitem_slow

    def getitem_impl(arr, index):
        # inlined on purpose: Numba never inlines a callee containing ``raise``
        i = index if index >= 0 else index + arr.length
        if i < 0 or i >= arr.length:
            raise IndexError("Arrow array index out of range")
        return ll_array_get_optional(arr, i)

    return getitem_impl


def is_valid(arr: object, index: int) -> bool:
    """True when element *index* is not null. Implemented in compiled code."""
    raise NotImplementedError("only callable from nopython code")


@nbext.overload(is_valid)
def is_valid_impl(arr, index):
    if not isinstance(arr, PyArrowArrayType):
        return
    if index not in nbt.integer_domain:
        raise nbt.TypingError("array indices must be integers")

    def impl(arr, index):
        i = index if index >= 0 else index + arr.length
        if i < 0 or i >= arr.length:
            raise IndexError("Arrow array index out of range")
        return not ll_array_is_null(arr, i)

    return impl


@nbext.overload(len)
def array_len(arr):
    if not isinstance(arr, PyArrowArrayType):
        return

    def len_impl(arr):
        return arr.length

    return len_impl


def _export_into_ll(
    c: PythonAPI,
    obj: ir.Value,
    addr: ir.Value,
    dtype: pa.DataType,
    array_type: nbt.Type,
    is_error_ptr: ir.Value,
) -> None:
    """Emit ``take_array_into(obj, addr, dtype)``; flag a NULL result."""
    addr_obj = c.pyapi.long_from_unsigned_int(c.builder.ptrtoint(addr, cgutils.intp_t))
    fn_obj = c.pyapi.unserialize(c.pyapi.serialize_object(take_array_into))
    dtype_obj = c.pyapi.unserialize(c.pyapi.serialize_object(dtype))
    ret = c.pyapi.call_function_objargs(fn_obj, (obj, addr_obj, dtype_obj))
    for done in (fn_obj, dtype_obj, addr_obj):
        c.pyapi.decref(done)
    with c.builder.if_then(cgutils.is_null(c.builder, ret)):
        c.builder.store(cgutils.true_bit, is_error_ptr)
    with c.builder.if_then(cgutils.is_not_null(c.builder, ret)):
        c.pyapi.decref(ret)

    # the slot now holds the producer's struct: make its null buffer slots safe
    proxy = c.context.make_helper(c.builder, array_type, value=c.builder.load(addr))
    _safe_buffers_ll(c.context, c.builder, proxy, num_buffers(dtype))
    c.builder.store(proxy._getvalue(), addr)


def _child_obj(c: PythonAPI, obj: ir.Value, method: str, k: int) -> ir.Value:
    """``obj.<method>(k)``, the k-th column or chunk, as a Python object."""
    index = c.pyapi.long_from_unsigned_int(ir.Constant(cgutils.intp_t, k))
    child = c.pyapi.call_method(obj, method, (index,))
    c.pyapi.decref(index)
    return child
