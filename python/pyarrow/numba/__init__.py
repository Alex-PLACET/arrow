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

"""Run Numba compiled code on Arrow arrays, and build Arrow arrays in it.

Reading is zero-copy: the producer's buffers are passed to the compiled code,
never copied. Building appends into pool-backed buffers and hands them to
Arrow on ``finish()``. The null test is emitted only for arrays that have
nulls, and it is branchless, so a loop over a nullable array vectorises.

>>> import numba as nb
>>> import pyarrow as pa
>>> @nb.njit
... def sum_and_nulls(a):
...     total = 0
...     nulls = 0
...     for i in range(len(a)):
...         if a[i] is None:
...             nulls += 1
...         else:
...             total += a[i]
...     return total, nulls
>>> sum_and_nulls(pa.array([2, 3, None] * 50))
(250, 50)

Element types
-------------
=====================  =================================================
Arrow type             ``a[i]`` in ``@njit``
=====================  =================================================
int8 .. int64          ``Optional[int]``
uint8 .. uint64        ``Optional[int]``
float32, float64       ``Optional[float]``
date32/64, time32/64   storage integer (days, ms, s, us)
timestamp, duration    storage integer
string, binary,        ``StringView``: ``string_len(v)``, ``v.valid``,
fixed_size_binary      boxes to bytes
decimal128             ``(high, low)``, exact; see ``decimal_to_float``
bool                   ``Optional[bool]``, read from the bit-packed buffer
=====================  =================================================

Any other type (nested, dictionary, decimal256) raises ``TypeError`` when it
is first seen, rather than reading the wrong buffer.

Producers
---------
pyarrow arrays are accepted as they are. A ``ChunkedArray`` keeps its chunks:
``len`` and ``a[i]`` walk the chunk starts, so nothing is concatenated (the
walk is scalar, so a chunked read is a few times slower than one array). A
``RecordBatch`` keeps every column: ``batch.column(k)`` with a literal *k*
returns column *k* as a regular array, and ``len(batch)`` is the row count::

    >>> @nb.njit
    ... def total(batch):
    ...     col = batch.column(1)
    ...     acc = 0
    ...     for i in range(len(batch)):
    ...         acc += col[i]
    ...     return acc

A ``Table`` is read as its first column, which keeps its chunks too. Anything
else implementing ``__arrow_c_array__`` or ``__arrow_c_stream__`` is accepted
once its class is registered, or per call through ``from_any()`` which needs
no registration::

    >>> from pyarrow.numba import register_producer
    >>> register_producer(polars.Series)      # doctest: +SKIP

Performance
-----------
Test ``a[i] is None`` and read ``a[i]`` again inside the branch, rather than
binding the element to a local: the bound form materialises an ``Optional``
pair for every element and costs about twice as much (1 M int64: 0.67 ms
against 1.35 ms), because the value and its validity are both live before the
branch is taken.

Limits
------
* A ``StringView`` carries its own validity as ``v.valid``: a null element
  still yields a view, so nothing silently reads a null's offsets.
* A stream yielding more than one chunk raises; ``from_any()`` concatenates,
  which copies.
* ``decimal128`` has no 128-bit arithmetic in ``@njit``: the two words are
  exact, ``decimal_to_float()`` is a double.
* Returning an array *argument* from compiled code is unsupported: its
  buffers are released when the call returns.

Building
--------
One builder per supported type: the eleven scalar builders plus
``StringBuilder``/``LargeStringBuilder``/``BinaryBuilder``/``LargeBinaryBuilder``.
A var-width builder keeps ``[validity, offsets, data]`` and copies bytes one
at a time, so it is slower than the scalar builders by design.

>>> from pyarrow.numba import Float64Builder
>>> @nb.njit
... def build():
...     b = Float64Builder()
...     b.reserve(3)
...     b.append(1.5)
...     b.append(None)
...     return b.finish()
>>> build().to_pylist()
[1.5, None]
"""

from __future__ import annotations

import pyarrow as pa  # noqa: F401  (used by the doctest above)

from . import batch, chunked  # noqa: F401  (registers their types)
from ._utils import array_from as from_any
from .array import decimal_to_float, is_valid, register_producer, string_len
from .builder import (
    BinaryBuilder,
    BoolBuilder,
    Float32Builder,
    Float64Builder,
    Int8Builder,
    Int16Builder,
    Int32Builder,
    Int64Builder,
    LargeBinaryBuilder,
    LargeStringBuilder,
    StringBuilder,
    UInt8Builder,
    UInt16Builder,
    UInt32Builder,
    UInt64Builder,
)

__all__ = [
    "BinaryBuilder",
    "BoolBuilder",
    "Float32Builder",
    "Float64Builder",
    "Int8Builder",
    "Int16Builder",
    "Int32Builder",
    "Int64Builder",
    "LargeBinaryBuilder",
    "LargeStringBuilder",
    "StringBuilder",
    "UInt8Builder",
    "UInt16Builder",
    "UInt32Builder",
    "UInt64Builder",
    "decimal_to_float",
    "from_any",
    "is_valid",
    "register_producer",
    "string_len",
]


def _register() -> bool:
    """Entry point: importing this module is what registers the extension."""
    return True
