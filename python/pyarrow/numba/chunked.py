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

"""``ChunkedArray`` support: one column, several ArrowArrays, no copy.

A chunked column keeps every chunk's ``ArrowArray`` and walks the chunk starts
on each access.  ``Table`` is the same thing with the chunks one level deeper
(under ``column(0)``), so it shares this type with ``from_column=True``.
"""

from __future__ import annotations

import functools
import operator

from llvmlite import ir
from numba import extending as nbext
from numba import types as nbt
from numba.core import cgutils
from numba.core.datamodel.manager import DataModelManager
from numba.core.pythonapi import PythonAPI
from numba.core.typing.context import BaseContext as TypingContext

import pyarrow as pa

from ._utils import _is_var_width, get_pyarrow_cpp_func
from .array import (
    PyArrowArrayType,
    _array_type,
    _child_obj,
    _export_into_ll,
    is_valid,
    ll_array_get_optional,
    ll_array_get_value,
    ll_array_is_null,
    ll_string_view,
)


class PyArrowChunkedType(nbt.Type):
    """A ``ChunkedArray``: several ArrowArrays read as one column.

    Chunks are never concatenated: ``getitem`` walks the chunk starts, so a
    two-chunk column costs one extra compare per element, not a copy.
    """

    def __init__(self, arrow_type: pa.DataType, n_chunks: int,
                 nullables: tuple[bool, ...], from_column: bool = False) -> None:
        self._arrow_type = arrow_type
        self._n_chunks = n_chunks
        self._nullables = nullables
        self._from_column = from_column
        self._array_type = _array_type(arrow_type, any(nullables))
        super().__init__(name=f'PyArrowChunked({arrow_type}, {n_chunks})')

    @property
    def key(self) -> tuple:
        return (self.name, self._n_chunks, self._nullables, self._from_column)

    @property
    def from_column(self) -> bool:
        """True for a Table: the chunks live under ``column(0)``."""
        return self._from_column

    @property
    def arrow_type(self) -> pa.DataType:
        return self._arrow_type

    @property
    def array_type(self) -> PyArrowArrayType:
        return self._array_type

    @property
    def numba_type(self) -> nbt.Type:
        return self._array_type.numba_type

    @property
    def n_chunks(self) -> int:
        return self._n_chunks


@functools.cache
def _chunked_type(arrow_type: pa.DataType, n_chunks: int,
                  nullables: tuple[bool, ...],
                  from_column: bool = False) -> PyArrowChunkedType:
    return PyArrowChunkedType(arrow_type, n_chunks, nullables, from_column)


@nbext.typeof_impl.register(pa.ChunkedArray)
def typeof_chunked(val: pa.ChunkedArray,
                   c: TypingContext) -> nbt.Type:
    if val.num_chunks <= 1:
        # nothing to walk: the plain array path handles 0 and 1 chunks
        return _array_type(val.type, val.null_count != 0)
    return _chunked_type(val.type, val.num_chunks,
                         tuple(ch.null_count != 0 for ch in val.chunks))


@nbext.typeof_impl.register(pa.Table)
def typeof_table(val: pa.Table, c: TypingContext) -> nbt.Type:
    column = val.column(0)
    if column.num_chunks <= 1:
        return _array_type(val.schema.field(0).type, column.null_count != 0)
    return _chunked_type(val.schema.field(0).type, column.num_chunks,
                         tuple(ch.null_count != 0 for ch in column.chunks),
                         from_column=True)


@nbext.register_model(PyArrowChunkedType)
class ChunkedModel(nbext.models.StructModel):
    def __init__(self, dmm: DataModelManager, fe_type: PyArrowChunkedType):
        n = fe_type.n_chunks
        super().__init__(dmm, fe_type, [
            ('chunks', nbt.UniTuple(fe_type.array_type, n)),
            ('starts', nbt.UniTuple(nbt.int64, n)),
            ('length', nbt.int64),
        ])


for _field in ('length', 'starts', 'chunks'):
    nbext.make_attribute_wrapper(PyArrowChunkedType, _field, _field)


@nbext.unbox(PyArrowChunkedType)
def unbox_chunked(typ: PyArrowChunkedType, obj: ir.Value,
                  c: PythonAPI) -> nbext.NativeValue:
    """Move every chunk's ArrowArray into the struct; no concatenation."""
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    model = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    base = model._getpointer()

    # a Table keeps its chunks under column(0); a ChunkedArray has them directly
    source = obj
    if typ.from_column:
        zero = c.pyapi.long_from_unsigned_int(ir.Constant(cgutils.intp_t, 0))
        source = c.pyapi.call_method(obj, "column", (zero,))
        c.pyapi.decref(zero)

    for k in range(typ.n_chunks):
        child = _child_obj(c, source, 'chunk', k)
        _export_into_ll(c, child,
                        cgutils.gep(c.builder, base, 0, 0, k),
                        typ.arrow_type, typ.array_type, is_error_ptr)
        c.pyapi.decref(child)

        if k == 0:
            start = ir.Constant(ir.IntType(64), 0)
        else:
            # ArrayModel field 0 is the length: a chunk starts where the
            # previous one ended
            prev_length = cgutils.gep(
                c.builder, cgutils.gep(c.builder, base, 0, 0, k - 1), 0, 0)
            start = c.builder.add(start, c.builder.load(prev_length))
        c.builder.store(start, cgutils.gep(c.builder, base, 0, 1, k))

    last_length = cgutils.gep(
        c.builder, cgutils.gep(c.builder, base, 0, 0, typ.n_chunks - 1), 0, 0)
    c.builder.store(c.builder.add(start, c.builder.load(last_length)),
                    cgutils.gep(c.builder, base, 0, 2))

    if typ.from_column:
        c.pyapi.decref(source)

    def cleanup():
        fn = get_pyarrow_cpp_func(c.builder, 'PyArrow_ReleaseArray')
        for k in range(typ.n_chunks):
            slot = cgutils.gep(c.builder, base, 0, 0, k)
            c.builder.call(fn, [c.builder.bitcast(slot, cgutils.voidptr_t)])

    return nbext.NativeValue(model._getvalue(), cleanup=cleanup,
                             is_error=c.builder.load(is_error_ptr))


@nbext.overload(len)
def chunked_len(arr):
    if not isinstance(arr, PyArrowChunkedType):
        return

    def impl(arr):
        return arr.length

    return impl


@nbext.overload(operator.getitem)
def chunked_getitem(arr, index):
    if not isinstance(arr, PyArrowChunkedType):
        return
    if index not in nbt.integer_domain:
        raise nbt.TypingError("array indices must be integers")

    n_chunks = arr.n_chunks
    var_width = _is_var_width(arr.arrow_type)
    slow = (pa.types.is_decimal(arr.arrow_type)
            or arr.arrow_type == pa.bool_())

    def impl(arr, index):
        # inlined on purpose: Numba never inlines a callee containing ``raise``
        i = index if index >= 0 else index + arr.length
        if i < 0 or i >= arr.length:
            raise IndexError("Arrow array index out of range")
        k = 0
        while k < n_chunks - 1 and i >= arr.starts[k + 1]:
            k += 1
        j = i - arr.starts[k]
        chunk = arr.chunks[k]
        if var_width:
            return ll_string_view(chunk, j,
                                  not ll_array_is_null(chunk, j))
        if slow:
            if ll_array_is_null(chunk, j):
                return None
            return ll_array_get_value(chunk, j)
        return ll_array_get_optional(chunk, j)

    return impl


@nbext.overload(is_valid)
def chunked_is_valid(arr, index):
    if not isinstance(arr, PyArrowChunkedType):
        return
    if index not in nbt.integer_domain:
        raise nbt.TypingError("array indices must be integers")

    n_chunks = arr.n_chunks

    def impl(arr, index):
        i = index if index >= 0 else index + arr.length
        if i < 0 or i >= arr.length:
            raise IndexError("Arrow array index out of range")
        k = 0
        while k < n_chunks - 1 and i >= arr.starts[k + 1]:
            k += 1
        return not ll_array_is_null(arr.chunks[k], i - arr.starts[k])

    return impl
