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

"""One loop per primitive: where the per-element time actually goes."""

import functools

import numpy as np
import pyarrow as pa

N = 1_000_000


def _numba():
    import numba
    import pyarrow.numba  # noqa: F401  (registers the types)
    return numba


def _nullable(n=N):
    data = np.arange(n, dtype=np.int64)
    return pa.array(np.where(np.arange(n) % 3, data, 0),
                    mask=np.arange(n) % 3 == 0)


def _dense(n=N):
    return pa.array(np.arange(n, dtype=np.int64))


def _strings(n=N):
    return pa.array([b"abcdefgh" if i % 3 else None for i in range(n)])


def _chunked(n=N):
    data = np.arange(n, dtype=np.int64)
    return pa.chunked_array([pa.array(data[: n // 2]), pa.array(data[n // 2:])],
                            type=pa.int64())


def _batch(n=N):
    return pa.record_batch({"x": pa.array(np.arange(n, dtype=np.int64))})


READ_OPS = ("getitem_optional", "getitem_dense", "is_valid",
            "string_len", "chunked", "batch_column")
BUILD_OPS = ("append_int", "append_null", "append_bool",
             "append_bytes", "append_view", "reserve", "append_no_reserve")
CALL_PRODUCERS = ("pyarrow_dense", "pyarrow_nullable", "polars_series",
                  "from_any")


@functools.cache
def _read_loops():
    """op -> (jitted function, argument factory)"""
    nb = _numba()
    from pyarrow.numba import is_valid, string_len

    @nb.njit
    def loop_getitem_optional(a):
        total = 0
        nulls = 0
        for i in range(len(a)):
            v = a[i]
            if v is None:
                nulls += 1
            else:
                total += v
        return total, nulls

    @nb.njit
    def loop_is_valid(a):
        total = 0
        for i in range(len(a)):
            if is_valid(a, i):
                total += a[i]
        return total

    @nb.njit
    def loop_string_len(a):
        total = 0
        for i in range(len(a)):
            v = a[i]
            if v.valid:
                total += string_len(v)
        return total

    @nb.njit
    def loop_chunked(a):
        total = 0
        for i in range(len(a)):
            total += a[i]
        return total

    @nb.njit
    def loop_batch_column(b):
        column = b.column(0)
        total = 0
        for i in range(len(b)):
            total += column[i]
        return total

    return {
        "getitem_optional": (loop_getitem_optional, _nullable),
        "getitem_dense": (loop_getitem_optional, _dense),
        "is_valid": (loop_is_valid, _nullable),
        "string_len": (loop_string_len, _strings),
        "chunked": (loop_chunked, _chunked),
        "batch_column": (loop_batch_column, _batch),
    }


class ReadPrimitives:
    """Per-element cost of each read accessor, ns per element."""

    param_names = ["op"]
    params = [list(READ_OPS)]

    def setup(self, op):
        self.func, factory = _read_loops()[op]
        self.arg = factory()
        self.func(self.arg)          # compile outside the timed region

    def time_read(self, op):
        self.func(self.arg)


@functools.cache
def _build_loops():
    nb = _numba()
    import pyarrow.numba as apn

    @nb.njit
    def append_int(n):
        b = apn.Int64Builder()
        b.reserve(n)
        for i in range(n):
            b.append(i % 100)
        return b.finish()

    @nb.njit
    def append_null(n):
        b = apn.Int64Builder()
        b.reserve(n)
        for i in range(n):
            b.append(None)
        return b.finish()

    @nb.njit
    def append_bool(n):
        b = apn.BoolBuilder()
        b.reserve(n)
        for i in range(n):
            b.append(i % 3 == 0)
        return b.finish()

    @nb.njit
    def append_bytes(n):
        b = apn.StringBuilder()
        b.reserve(n)
        for _ in range(n):
            b.append(b"abcdefgh")
        return b.finish()

    @nb.njit
    def append_view(source):
        b = apn.StringBuilder()
        b.reserve(len(source))
        for i in range(len(source)):
            b.append(source[i])
        return b.finish()

    @nb.njit
    def reserve(n):
        b = apn.Int64Builder()
        b.reserve(n)
        return b.finish()

    @nb.njit
    def no_reserve(n):
        b = apn.Int64Builder()
        for i in range(n):
            b.append(i % 100)
        return b.finish()

    return {
        "append_int": (append_int, lambda: N),
        "append_null": (append_null, lambda: N),
        "append_bool": (append_bool, lambda: N),
        "append_bytes": (append_bytes, lambda: N),
        "append_view": (append_view, _strings),
        "reserve": (reserve, lambda: N),
        "append_no_reserve": (no_reserve, lambda: N),
    }


class BuildPrimitives:
    """Per-element cost of each builder method, ns per element."""

    param_names = ["op"]
    params = [list(BUILD_OPS)]

    def setup(self, op):
        self.func, factory = _build_loops()[op]
        self.arg = factory()
        self.func(self.arg)          # compile, and time the growth path once

    def time_build(self, op):
        self.func(self.arg)


class CallOverhead:
    """Python glue per call: typeof + unbox + box, microseconds."""

    param_names = ["producer"]
    params = [list(CALL_PRODUCERS)]

    def setup(self, producer):
        nb = _numba()
        import polars as pl
        from pyarrow.numba import from_any, register_producer
        register_producer(pl.Series)

        @nb.njit
        def tiny(a):
            total = 0
            for i in range(len(a)):
                if a[i] is not None:
                    total += a[i]
            return total

        self.func = tiny
        if producer == "pyarrow_dense":
            self.arg = pa.array([1, 2, 3])
        elif producer == "pyarrow_nullable":
            self.arg = pa.array([1, None, 3], type=pa.int64())
        elif producer == "polars_series":
            self.arg = pl.Series("x", [1, 2, 3])
        else:
            # from_any() on a pa.Array is an isinstance check and nothing
            # else; a Polars Series is the real foreign-producer path.
            self.arg = pl.Series("x", [1, 2, 3])
            self.func = lambda arg: from_any(arg)
        self.func(self.arg)

    def time_call(self, producer):
        self.func(self.arg)
