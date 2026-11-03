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

Reading is zero-copy: an array's buffers are passed to the compiled code, not
copied. Building appends into pool-backed buffers and hands them to Arrow on
``finish()``.

>>> import numba as nb
>>> import pyarrow as pa
>>> from pyarrow.numba import Int64Builder
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

Arrays from other libraries are read as-is once their class is registered:
the Arrow PyCapsule interface hands over the buffers, so nothing is copied.

>>> from pyarrow.numba import register_producer
>>> register_producer(polars.Series)      # doctest: +SKIP
>>> sum_and_nulls(polars.Series([2, 3, None] * 50))   # doctest: +SKIP
(250, 50)
"""

import pyarrow as pa  # noqa: F401  (used by the doctest below)
from ._utils import array_from as from_any  # noqa: F401
from .array import (decimal_to_float, is_valid,  # noqa: F401
                    register_producer, string_len)
from .builder import (  # noqa: F401
    BoolBuilder, Float32Builder, Float64Builder, Int8Builder, Int16Builder,
    Int32Builder, Int64Builder, UInt8Builder, UInt16Builder, UInt32Builder,
    UInt64Builder)
from .poolbuffer import (PoolBuffer, TypedBufferBuilder,  # noqa: F401
                         Int64BufferBuilder, BooleanBufferBuilder)


def _register():
    """Entry point: importing this module is what registers the extension."""
    return True
