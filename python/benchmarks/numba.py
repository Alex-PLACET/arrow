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

"""Compiled (Numba) code reading and building Arrow arrays, per type."""

from decimal import Decimal

import pyarrow as pa

N = 10 ** 6

READ_TYPES = {
    "int8": pa.int8(), "int16": pa.int16(),
    "int32": pa.int32(), "int64": pa.int64(),
    "uint8": pa.uint8(), "uint16": pa.uint16(),
    "uint32": pa.uint32(), "uint64": pa.uint64(),
    "float32": pa.float32(), "float64": pa.float64(),
    "date32": pa.date32(), "date64": pa.date64(),
    "time32_s": pa.time32("s"), "time64_us": pa.time64("us"),
    "timestamp_us": pa.timestamp("us"), "duration_s": pa.duration("s"),
    "bool": pa.bool_(),
    "decimal128": pa.decimal128(20, 2),
    "string": pa.string(), "large_string": pa.large_string(),
    "binary": pa.binary(), "large_binary": pa.large_binary(),
    "fixed_size_binary": pa.binary(8),
}

BUILD_TYPES = {
    "int8": "Int8Builder", "int16": "Int16Builder",
    "int32": "Int32Builder", "int64": "Int64Builder",
    "uint8": "UInt8Builder", "uint16": "UInt16Builder",
    "uint32": "UInt32Builder", "uint64": "UInt64Builder",
    "float32": "Float32Builder", "float64": "Float64Builder",
    "bool": "BoolBuilder",
    "string": "StringBuilder", "binary": "BinaryBuilder",
}


def _numba():
    try:
        import numba
    except ImportError:
        raise NotImplementedError("numba is not installed")
    return numba


def _values(arrow_type):
    """One million values of *arrow_type*, null every third."""
    if arrow_type == "bool":
        return [(i % 2 == 0) if i % 3 else None for i in range(N)]
    if arrow_type == "decimal128":
        return [Decimal(i % 10000) / 100 if i % 3 else None for i in range(N)]
    if arrow_type == "fixed_size_binary":
        return [b"abcdefgh" if i % 3 else None for i in range(N)]
    if arrow_type in ("string", "large_string", "binary", "large_binary"):
        return [b"abcdefgh"[: (i % 8) + 1] if i % 3 else None
                for i in range(N)]
    return [i % 100 if i % 3 else None for i in range(N)]


class NumbaRead:
    """Read one array of every supported type, element by element."""

    param_names = ["arrow_type"]
    params = [sorted(READ_TYPES)]

    def setup(self, arrow_type):
        nb = _numba()
        import pyarrow.numba  # noqa: F401  (registers the array type)
        from pyarrow.numba import decimal_to_float, string_len
        from pyarrow.numba._utils import _is_var_width

        self.array = pa.array(_values(arrow_type), type=READ_TYPES[arrow_type])

        if arrow_type == "bool":
            # element read twice on purpose: binding it first materialises an
            # Optional pair and costs about 2x (see the module docstring)
            @nb.njit
            def scan(a):
                n = 0
                for i in range(len(a)):
                    if a[i] is not None and a[i]:
                        n += 1
                return n
        elif pa.types.is_decimal(READ_TYPES[arrow_type]):
            @nb.njit
            def scan(a):
                total = 0.0
                for i in range(len(a)):
                    if a[i] is not None:
                        total += decimal_to_float(a[i], 2)
                return total
        elif _is_var_width(READ_TYPES[arrow_type]):
            @nb.njit
            def scan(a):
                n = 0
                for i in range(len(a)):
                    v = a[i]
                    if v.valid:
                        n += string_len(v)
                return n
        else:
            @nb.njit
            def scan(a):
                total = 0
                nulls = 0
                for i in range(len(a)):
                    if a[i] is None:
                        nulls += 1
                    else:
                        total += a[i]
                return total, nulls

        self.func = scan
        self.func(self.array)          # compile outside the measured region

    def time_read(self, arrow_type):
        self.func(self.array)


class NumbaBuild:
    """Append one million values of every builder type, then finish."""

    param_names = ["arrow_type"]
    params = [sorted(BUILD_TYPES)]

    def setup(self, arrow_type):
        nb = _numba()
        import pyarrow.numba as apn

        builder_cls = getattr(apn, BUILD_TYPES[arrow_type])

        if arrow_type in ("string", "binary"):
            @nb.njit
            def build(n):
                b = builder_cls()
                b.reserve(n)
                for i in range(n):
                    if i % 3:
                        b.append(b"abcdefgh")
                    else:
                        b.append(None)
                return b.finish()
        elif arrow_type == "bool":
            @nb.njit
            def build(n):
                b = builder_cls()
                b.reserve(n)
                for i in range(n):
                    if i % 3:
                        b.append(i % 2 == 0)
                    else:
                        b.append(None)
                return b.finish()
        else:
            # one append per element, no arithmetic in the loop: the append
            # (reserve check, cast, store) is what is being measured
            @nb.njit
            def build(n):
                b = builder_cls()
                b.reserve(n)
                for i in range(n):
                    if i % 3:
                        b.append(i)
                    else:
                        b.append(None)
                return b.finish()

        self.func = build
        self.func(16)                  # compile outside the measured region

    def time_build(self, arrow_type):
        self.func(N)


class NumbaReadSmall:
    """Fixed cost of passing one tiny array into compiled code."""

    def setup(self):
        nb = _numba()
        import pyarrow.numba  # noqa: F401

        @nb.njit
        def first(a):
            return a[0]

        self.array = pa.array([1, 2, 3])
        self.func = first
        self.func(self.array)          # compile outside the measured region

    def time_tiny_array(self):
        self.func(self.array)


class NumbaProducerCall:
    """Fixed cost of calling compiled code with a foreign producer."""

    def setup(self):
        nb = _numba()
        try:
            import polars
        except ImportError:
            raise NotImplementedError("polars is not installed")
        from pyarrow.numba import register_producer

        register_producer(polars.Series)

        @nb.njit
        def first(a):
            return a[0]

        self.series = polars.Series([1, 2, 3])
        self.func = first
        self.func(self.series)

    def time_polars_series(self):
        self.func(self.series)
