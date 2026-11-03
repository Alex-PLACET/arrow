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

from contextlib import ExitStack
import functools
import operator

from numba import types as nbt
from numba import extending as nbext
from numba.core import cgutils

from llvmlite import ir

import pyarrow as pa
import numpy as np

from ._capsule import dtype_of, take_array_into
from ._utils import (_arrow_type_to_nb_type, _is_var_width,
                     get_pyarrow_cpp_func)

#
# 1. Typing
#


class PyArrowArrayType(nbt.Type):

    def __init__(self, type):
        self._type = type
        self._nb_type = _arrow_type_to_nb_type(type)
        super().__init__(name=f'PyArrowArray({type})')

    @property
    def arrow_type(self):
        return self._type

    @property
    def numba_type(self):
        return self._nb_type


@functools.lru_cache(maxsize=None)
def _array_type(arrow_type):
    """One Numba type per Arrow type: typeof() runs on every call."""
    return PyArrowArrayType(arrow_type)


@nbext.typeof_impl.register(pa.Array)
def typeof_array(val, c):
    assert isinstance(val, pa.Array)
    return _array_type(val.type)


def register_producer(cls):
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


for _field in ('length', 'null_count', 'offset'):
    nbext.make_attribute_wrapper(PyArrowArrayType, _field, _field)


class StringViewType(nbt.Type):
    """A non-owning view of one string or binary element."""

    def __init__(self):
        super().__init__(name="ArrowStringView")


#
# 2. Data model
#

@nbext.register_model(PyArrowArrayType)
class ArrayModel(nbext.models.StructModel):
    def __init__(self, dmm, fe_type):
        # Mirrors the C ABI; _capsule.ArrowArray is the authority. Everything
        # is word-sized, so a mismatch only shows up as wrong data, not a crash.
        # XXX Should this instead carry a pointer to a heap-allocated struct?
        members = [
            ('length', nbt.int64),
            ('null_count', nbt.int64),
            ('offset', nbt.int64),
            ('n_buffers', nbt.int64),
            ('n_children', nbt.int64),
            ('buffers', nbt.CPointer(nbt.CPointer(nbt.uint8))),
            ('children', nbt.CPointer(nbt.CPointer(nbt.uint8))),
            ('dictionary', nbt.CPointer(nbt.uint8)),
            ('release_cb', nbt.CPointer(nbt.uint8)),
            ('private_data', nbt.CPointer(nbt.uint8)),
        ]
        super().__init__(dmm, fe_type, members)


@nbext.register_model(StringViewType)
class StringViewModel(nbext.models.StructModel):
    """chars/start/end: a byte range inside the array's data buffer."""

    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [
            ("chars", nbt.CPointer(nbt.uint8)),
            ("start", nbt.int64),
            ("end", nbt.int64),
        ])


for _field in ("chars", "start", "end"):
    nbext.make_attribute_wrapper(StringViewType, _field, _field)


#
# 3. Code generation
#

@nbext.unbox(PyArrowArrayType)
def unbox_array(typ, obj, c):
    """
    Unbox a pa.Array object as a ArrowArray struct wrapper.
    """
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    c_array = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    with ExitStack() as stack:
        # one Python call per argument: _capsule.take_array_into moves the
        # producer's ArrowArray into this struct (0.2 us for a pa.Array)
        array_ptr_obj = c.pyapi.long_from_unsigned_int(
            c.builder.ptrtoint(c_array._getpointer(), cgutils.intp_t))
        # the struct is moved out of the producer's capsule: no pa.Array here
        export_obj = c.pyapi.unserialize(
            c.pyapi.serialize_object(take_array_into))
        dtype_obj = c.pyapi.unserialize(
            c.pyapi.serialize_object(typ.arrow_type))
        ret_obj = c.pyapi.call_function_objargs(
            export_obj, (obj, array_ptr_obj, dtype_obj))
        c.pyapi.decref(export_obj)
        c.pyapi.decref(dtype_obj)
        c.pyapi.decref(array_ptr_obj)
        with cgutils.early_exit_if_null(c.builder, stack, ret_obj):
            c.builder.store(cgutils.true_bit, is_error_ptr)
        c.pyapi.decref(ret_obj)

    def cleanup():
        c_array_ptr = c.builder.bitcast(c_array._getpointer(), cgutils.voidptr_t)
        fn = get_pyarrow_cpp_func(c.builder, 'PyArrow_ReleaseArray')
        c.builder.call(fn, [c_array_ptr])

    return nbext.NativeValue(c_array._getvalue(),
                             cleanup=cleanup,
                             is_error=c.builder.load(is_error_ptr))


@nbext.box(PyArrowArrayType)
def box_array(typ, val, c):
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
        # TODO this sequence of calls through Python is slow, should implement
        # it as a PyArrow native function
        pa_name = c.context.insert_const_string(c.builder.module, 'pyarrow')
        pa_mod = c.pyapi.import_module(pa_name)
        array_cls = c.pyapi.object_getattr_string(pa_mod, 'Array')
        c.pyapi.decref(pa_mod)
        datatype_obj = c.pyapi.unserialize(
            c.pyapi.serialize_object(typ.arrow_type))
        array_ptr_obj = c.pyapi.long_from_unsigned_int(
            c.builder.ptrtoint(c_array._getpointer(), cgutils.intp_t))
        ret_obj = c.pyapi.call_method(
            array_cls, "_import_from_c", (array_ptr_obj, datatype_obj))
        c.pyapi.decref(array_cls)
        c.pyapi.decref(array_ptr_obj)
        c.pyapi.decref(datatype_obj)
        return ret_obj


def make_array(typ):
    """
    Return the Structure representation of the given array type.
    (an instance of types.PyArrowArrayType).
    """
    # NOTE see numba/np/arrayobj.py to perhaps add annotations for speed
    assert isinstance(typ, PyArrowArrayType)
    return cgutils.create_struct_proxy(typ)


@nbext.intrinsic
def ll_array_is_null(typingctx, arr_type, index_type):
    assert isinstance(arr_type, PyArrowArrayType)
    result_type = nbt.bool_
    # Force index to int64
    sig = result_type(arr_type, nbt.int64)

    def codegen(context, builder, signature, args):
        (arr_val, index) = args
        arrayty = make_array(arr_type)
        array = arrayty(context, builder, arr_val)

        null_bitmap = builder.load(cgutils.gep(
            builder, array.buffers, 0, inbounds=True))
        has_null_bitmap = builder.icmp_unsigned(
            '!=', array.null_count, array.null_count.type(0))

        with builder.if_else(has_null_bitmap) as (then_block, else_block):
            with then_block:
                bit_index = builder.add(array.offset, index)
                byte_offset = builder.lshr(bit_index, bit_index.type(3))
                bit_offset = builder.and_(bit_index, bit_index.type(7))
                bitmap_byte = builder.load(cgutils.gep(
                    builder, null_bitmap, byte_offset))
                mask = builder.shl(cgutils.int8_t(
                    1), builder.trunc(bit_offset, cgutils.int8_t))
                masked_byte = builder.and_(bitmap_byte, mask)
                is_null_with_bitmap = builder.icmp_unsigned(
                    '==', masked_byte, masked_byte.type(0))
                bb_then = builder.basic_block
            with else_block:
                is_null_without_bitmap = cgutils.false_bit
                bb_else = builder.basic_block

        is_null = builder.phi(is_null_with_bitmap.type)
        is_null.add_incoming(is_null_with_bitmap, bb_then)
        is_null.add_incoming(is_null_without_bitmap, bb_else)
        return is_null

    return sig, codegen


def nb_type_to_ll_type(ty):
    if ty in nbt.integer_domain:
        return ir.IntType(ty.bitwidth)
    if ty == nbt.float32:
        return ir.FloatType()
    if ty == nbt.float64:
        return ir.DoubleType()
    raise TypeError(f'Unsupported type {ty}')


@nbext.intrinsic
def ll_array_get_value(typingctx, arr_type, index_type):
    assert isinstance(arr_type, PyArrowArrayType)
    decimal = pa.types.is_decimal(arr_type.arrow_type)
    result_type = arr_type.numba_type
    if not decimal and not isinstance(result_type, nbt.Number):
        raise nbt.TypingError(
            f"{arr_type.arrow_type} values are bit-packed: there is no element "
            "accessor, but is_valid() and the raw buffers still work")
    # Force index to int64
    sig = result_type(arr_type, nbt.int64)

    def codegen(context, builder, signature, args):
        (arr_val, index) = args
        arrayty = make_array(arr_type)
        array = arrayty(context, builder, arr_val)

        data_buffer = builder.load(cgutils.gep(
            builder, array.buffers, 1, inbounds=True))
        index = builder.add(array.offset, index)

        if decimal:
            # 128-bit little-endian two's complement, returned as (high, low):
            # exact, and tuple order compares like the value
            words = builder.bitcast(data_buffer, ir.IntType(64).as_pointer())
            pair = builder.mul(index, ir.Constant(index.type, 2))
            low = builder.load(cgutils.gep(builder, words, pair))
            high = builder.load(cgutils.gep(
                builder, words, builder.add(pair, ir.Constant(index.type, 1))))
            return context.make_tuple(builder, result_type, [high, low])

        data_buffer = builder.bitcast(
            data_buffer, nb_type_to_ll_type(result_type).as_pointer())
        return builder.load(cgutils.gep(builder, data_buffer, index))

    return sig, codegen


@nbext.intrinsic
def ll_string_view(typingctx, arr_type, index_type):
    """The byte range of element *index*, offsets width taken from the type."""
    assert isinstance(arr_type, PyArrowArrayType)
    arrow_type = arr_type.arrow_type
    fixed_width = (arrow_type.byte_width
                   if pa.types.is_fixed_size_binary(arrow_type) else 0)
    wide = (pa.types.is_large_string(arrow_type)
            or pa.types.is_large_binary(arrow_type))
    off_t = ir.IntType(64 if wide else 32)
    sig = StringViewType()(arr_type, nbt.int64)

    def codegen(context, builder, signature, args):
        arr_val, index = args
        array = context.make_helper(builder, signature.args[0], value=arr_val)
        i = builder.add(array.offset, index)
        if fixed_width:
            # fixed-size binary has no offsets buffer: [validity, data]
            chars = builder.load(cgutils.gep(builder, array.buffers, 1,
                                             inbounds=True))
            start = builder.mul(i, ir.Constant(i.type, fixed_width))
            stop = builder.add(start, ir.Constant(i.type, fixed_width))
        else:
            # string/binary: [validity, offsets, data]
            offsets = builder.load(cgutils.gep(builder, array.buffers, 1,
                                               inbounds=True))
            chars = builder.load(cgutils.gep(builder, array.buffers, 2,
                                             inbounds=True))
            offsets = builder.bitcast(offsets, off_t.as_pointer())
            start = builder.load(cgutils.gep(builder, offsets, i))
            stop = builder.load(cgutils.gep(builder, offsets,
                                            builder.add(i, i.type(1))))
        out = cgutils.create_struct_proxy(signature.return_type)(
            context, builder)
        out.chars = builder.bitcast(chars, ir.IntType(8).as_pointer())
        if fixed_width:
            out.start = start
            out.end = stop
        else:
            out.start = builder.sext(start, ir.IntType(64))
            out.end = builder.sext(stop, ir.IntType(64))
        return out._getvalue()

    return sig, codegen


@nbext.box(StringViewType)
def box_string_view(typ, val, c):
    """Box a view as ``bytes``: one copy of the element, no decoding."""
    view = c.context.make_helper(c.builder, typ, value=val)
    ptr = c.builder.gep(view.chars, [view.start])
    length = c.builder.sub(view.end, view.start)
    fn = cgutils.get_or_insert_function(
        c.builder.module,
        ir.FunctionType(ir.IntType(8).as_pointer(),
                        [cgutils.voidptr_t, cgutils.intp_t]),
        "PyBytes_FromStringAndSize")
    return c.builder.call(fn, [c.builder.bitcast(ptr, cgutils.voidptr_t),
                               c.builder.trunc(length, cgutils.intp_t)])


def decimal_to_float(words, scale):
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
            unscaled = np.float64(high) * 2.0 ** 64 + np.float64(low)
        return unscaled * 10.0 ** (-scale)

    return impl


def string_len(view):
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
            return ll_string_view(arr, i)

        return getitem_var

    def getitem_impl(arr, index):
        # inlined on purpose: Numba never inlines a callee containing ``raise``
        i = index if index >= 0 else index + arr.length
        if i < 0 or i >= arr.length:
            raise IndexError("Arrow array index out of range")
        if ll_array_is_null(arr, i):
            return None
        else:
            return ll_array_get_value(arr, i)

    return getitem_impl


def is_valid(arr, index):
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
