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

"""``RecordBatch`` support: every column as an array, picked by a literal index."""

from __future__ import annotations

import functools

from llvmlite import ir
from numba import extending as nbext
from numba import types as nbt
from numba.core import cgutils
from numba.core.datamodel.manager import DataModelManager
from numba.core.pythonapi import PythonAPI
from numba.core.typing.context import BaseContext as TypingContext

import pyarrow as pa

from ._utils import get_pyarrow_cpp_func
from .array import PyArrowArrayType, _array_type, _child_obj, _export_into_ll


class PyArrowBatchType(nbt.Type):
    """A ``RecordBatch``: its columns, each one an ArrowArray.

    ``batch.column(k)`` needs a literal *k*, because the columns need not
    share a type and Numba resolves the call at compile time.
    """

    def __init__(self, field_types: tuple[pa.DataType, ...],
                 nullables: tuple[bool, ...]) -> None:
        self._field_types = field_types
        self._nullables = nullables
        self._column_types = tuple(_array_type(t, nullable)
                                   for t, nullable in zip(field_types,
                                                          nullables))
        super().__init__(name=f'PyArrowBatch({len(field_types)})')

    @property
    def key(self) -> tuple:
        return (self.name, self._field_types, self._nullables)

    @property
    def field_types(self) -> tuple[pa.DataType, ...]:
        return self._field_types

    @property
    def column_types(self) -> tuple[PyArrowArrayType, ...]:
        return self._column_types

    @property
    def n_columns(self) -> int:
        return len(self._field_types)


@functools.cache
def _batch_type(field_types: tuple[pa.DataType, ...],
                nullables: tuple[bool, ...]) -> PyArrowBatchType:
    return PyArrowBatchType(field_types, nullables)


@nbext.typeof_impl.register(pa.RecordBatch)
def typeof_batch(val: pa.RecordBatch, c: TypingContext) -> PyArrowBatchType:
    if not val.num_columns:
        raise nbt.TypingError("a RecordBatch needs at least one column")
    return _batch_type(tuple(val.schema.types),
                       tuple(column.null_count != 0
                             for column in val.columns))


@nbext.register_model(PyArrowBatchType)
class BatchModel(nbext.models.StructModel):
    def __init__(self, dmm: DataModelManager, fe_type: PyArrowBatchType):
        columns = nbt.Tuple(fe_type.column_types)
        super().__init__(dmm, fe_type, [
            ('columns', columns),
            ('num_columns', nbt.int64),
        ])


nbext.make_attribute_wrapper(PyArrowBatchType, 'num_columns', 'num_columns')
nbext.make_attribute_wrapper(PyArrowBatchType, 'columns', 'columns')


@nbext.unbox(PyArrowBatchType)
def unbox_batch(typ: PyArrowBatchType, obj: ir.Value,
                c: PythonAPI) -> nbext.NativeValue:
    """Move every column's ArrowArray into the struct."""
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    model = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    base = model._getpointer()

    for k in range(typ.n_columns):
        child = _child_obj(c, obj, 'column', k)
        _export_into_ll(c, child,
                        cgutils.gep(c.builder, base, 0, 0, k),
                        typ.field_types[k], typ.column_types[k], is_error_ptr)
        c.pyapi.decref(child)

    c.builder.store(ir.Constant(ir.IntType(64), typ.n_columns),
                    cgutils.gep(c.builder, base, 0, 1))

    def cleanup():
        for k in range(typ.n_columns):
            slot = cgutils.gep(c.builder, base, 0, 0, k)
            fn = get_pyarrow_cpp_func(c.builder, 'PyArrow_ReleaseArray')
            c.builder.call(fn, [c.builder.bitcast(slot, cgutils.voidptr_t)])

    return nbext.NativeValue(model._getvalue(), cleanup=cleanup,
                             is_error=c.builder.load(is_error_ptr))


@nbext.overload(len)
def batch_len(batch):
    if not isinstance(batch, PyArrowBatchType):
        return

    def impl(batch):
        return batch.columns[0].length

    return impl


@nbext.overload_method(PyArrowBatchType, 'column')
def batch_column(batch, index):
    if not isinstance(index, nbt.IntegerLiteral):
        raise nbt.TypingError(
            "RecordBatch.column() needs a literal index, e.g. batch.column(0)")
    k = index.literal_value
    if not 0 <= k < batch.n_columns:
        raise nbt.TypingError(
            f"column {k} is out of range for a {batch.n_columns}-column batch")

    def impl(batch, index):
        return batch.columns[k]

    return impl
