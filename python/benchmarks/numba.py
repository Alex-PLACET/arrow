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

"""Compiled (Numba) code reading and building Arrow arrays."""

import pyarrow as pa

N = 10 ** 6


def _numba():
    try:
        import numba
    except ImportError:
        raise NotImplementedError("numba is not installed")
    return numba


class NumbaRead:
    """Sum a nullable int64 array and count its nulls."""

    def setup(self):
        nb = _numba()
        import pyarrow.numba  # noqa: F401  (registers the array type)

        @nb.njit
        def sum_and_nulls(a):
            total = 0
            nulls = 0
            for i in range(len(a)):
                if a[i] is None:
                    nulls += 1
                else:
                    total += a[i]
            return total, nulls

        values = [i if i % 3 else None for i in range(N)]
        self.array = pa.array(values, type=pa.int64())
        self.func = sum_and_nulls
        self.func(self.array)          # compile outside the measured region

    def time_sum_and_nulls(self):
        self.func(self.array)


class NumbaBuild:
    """Append N values into a pool-backed builder, then hand them to Arrow."""

    def setup(self):
        nb = _numba()
        from pyarrow.numba import Int64Builder

        @nb.njit
        def build(n):
            b = Int64Builder()
            b.reserve(n)
            for i in range(n):
                if i % 3:
                    b.append(i)
                else:
                    b.append(None)
            return b.finish()

        self.func = build
        self.func(16)                  # compile outside the measured region

    def time_build_int64(self):
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
