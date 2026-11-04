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


"""Reading Arrow arrays: element types, nulls, producers, layouts."""

import numpy as np
import pytest

import pyarrow as pa

nb = pytest.importorskip("numba")

from pyarrow.numba import _register  # noqa: E402  (registers the extension)
from pyarrow.numba import is_valid, string_len  # noqa: E402

_register()


def test_doc_example_reads_a_pyarrow_array():
    @nb.njit
    def sum_and_nulls(a):
        non_null_sum = 0
        null_count = 0
        for i in range(len(a)):
            if a[i] is None:
                null_count += 1
            else:
                non_null_sum += a[i]
        return non_null_sum, null_count

    arr = pa.array([2, 3, None] * 50)
    assert sum_and_nulls(arr) == (250, 50)


def test_getitem_on_a_slice_and_out_of_range():
    source = pa.array([10, None, 30, 40, 50])
    view = source.slice(2)

    @nb.njit
    def at(a, i):
        return a[i]

    assert at(view, 0) == 30
    assert at(view, 1) == 40
    with pytest.raises(IndexError):
        at(view, len(view))
    with pytest.raises(IndexError):
        at(view, -len(view) - 1)


def test_is_valid_matches_pyarrow():
    from pyarrow.numba import is_valid

    arr = pa.array([1, None, 3, None])

    @nb.njit
    def flags(a):
        out = np.zeros(len(a), np.bool_)
        for i in range(len(a)):
            out[i] = is_valid(a, i)
        return out

    assert flags(arr).tolist() == arr.is_valid().to_pylist()


def test_accepts_a_non_pyarrow_capsule_producer():
    from pyarrow.numba import from_any

    class Foreign:
        """Minimal producer: hands out the capsules of a real array."""

        def __init__(self, array):
            self._array = array

        def __arrow_c_array__(self, requested_schema=None):
            return self._array.__arrow_c_array__(requested_schema)

    @nb.njit
    def total(a):
        acc = 0
        for i in range(len(a)):
            if a[i] is not None:
                acc += a[i]
        return acc

    arr = pa.array([1, None, 3, 4])
    assert total(from_any(Foreign(arr))) == 8
    assert from_any(arr) is arr
    with pytest.raises(TypeError):
        from_any(object())


def _polars_producer():
    pl = pytest.importorskip("polars")
    return pl.Series("x", [2, 3, None, 5])


def _pandas_producer():
    pd = pytest.importorskip("pandas")
    return pd.Series([2, 3, None, 5])


def _ibis_producer():
    ibis = pytest.importorskip("ibis")
    return ibis.memtable({"x": [2, 3, None, 5]})


def _duckdb_producer():
    duckdb = pytest.importorskip("duckdb")
    return duckdb.sql("select * from (values (2), (3), (null), (5)) t(x)")


@pytest.mark.parametrize("make_producer", [
    _polars_producer, _pandas_producer, _ibis_producer, _duckdb_producer,
], ids=["polars", "pandas", "ibis", "duckdb"])
def test_reads_a_third_party_producer(make_producer):
    from pyarrow.numba import register_producer

    producer = make_producer()
    register_producer(type(producer))

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

    assert sum_and_nulls(producer) == (10, 1)


def test_polars_series_is_read_zero_copy():
    pl = pytest.importorskip("polars")
    from pyarrow.numba import from_any

    series = pl.Series([2, 3, None, 5])
    assert (from_any(series).buffers()[1].address
            == series.to_arrow().buffers()[1].address)


def test_decimal128_reads_exactly():
    from decimal import Decimal

    from pyarrow.numba import decimal_to_float

    values = [Decimal("12345.67"), None, Decimal("-0.01")]
    array = pa.array(values, type=pa.decimal128(20, 2))

    @nb.njit
    def words(a, i):
        return a[i]

    @nb.njit
    def as_float(a, i):
        return decimal_to_float(a[i], 2)

    @nb.njit
    def ordered(a, i, j):
        return a[i] < a[j]

    def unscaled(i):
        raw = words(array, i)
        return None if raw is None else Decimal(raw[0] * 2 ** 64 + raw[1])

    assert [unscaled(i) for i in range(3)] == [Decimal(1234567), None, Decimal(-1)]
    assert as_float(array, 0) == float(values[0])
    assert as_float(array, 2) == float(values[2])
    assert ordered(array, 2, 0)          # -0.01 < 12345.67

    big = Decimal("123456789012345678901234567890.12")
    wide = pa.array([big], type=pa.decimal128(38, 2))

    @nb.njit
    def wide_float(a):
        return decimal_to_float(a[0], 2)

    from decimal import localcontext

    raw = words(wide, 0)
    with localcontext() as ctx:
        ctx.prec = 60
        # exact past 53 bits: the two words hold every digit
        assert Decimal(raw[0] * 2 ** 64 + raw[1]).scaleb(-2) == big
    assert wide_float(wide) == float(big)                       # lossy by choice


SCALAR_TYPES = [
    pa.int8(), pa.int16(), pa.int32(), pa.int64(),
    pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64(),
    pa.float32(), pa.float64(),
    pa.date32(), pa.date64(), pa.time32("s"), pa.time64("us"),
    pa.timestamp("us"), pa.duration("s"),
]


@pytest.mark.parametrize("pa_type", SCALAR_TYPES, ids=str)
@pytest.mark.parametrize("offset", [0, 1])
def test_scalar_types_read_values_and_nulls(pa_type, offset):
    values = [1, None, 3, 4]
    view = pa.array(values, type=pa_type).slice(offset)
    expected = values[offset:]

    @nb.njit
    def read(a):
        out = np.zeros(len(a), np.float64)
        valid = np.zeros(len(a), np.bool_)
        for i in range(len(a)):
            valid[i] = is_valid(a, i)
            v = a[i]
            if v is not None:
                out[i] = v
        return out, valid

    out, valid = read(view)
    assert valid.tolist() == [v is not None for v in expected]
    assert out.tolist() == [0.0 if v is None else float(v) for v in expected]


@pytest.mark.parametrize("pa_type,payloads", [
    (pa.string(), [b"abc", None, b"", b"de"]),
    (pa.large_string(), [b"abc", None, b"", b"de"]),
    (pa.binary(), [b"abc", None, b"", b"de"]),
    (pa.large_binary(), [b"abc", None, b"", b"de"]),
    (pa.binary(3), [b"abc", None, b"xyz", b"def"]),
], ids=["string", "large_string", "binary", "large_binary",
        "fixed_size_binary"])
@pytest.mark.parametrize("offset", [0, 1])
def test_string_types_read_views(pa_type, payloads, offset):
    view = pa.array(payloads, type=pa_type).slice(offset)
    expected = payloads[offset:]

    @nb.njit
    def read(a):
        valid = np.zeros(len(a), np.bool_)
        lengths = np.zeros(len(a), np.int64)
        for i in range(len(a)):
            valid[i] = is_valid(a, i)
            if valid[i]:
                lengths[i] = string_len(a[i])
        return valid, lengths

    valid, lengths = read(view)
    assert valid.tolist() == [p is not None for p in expected]
    assert lengths.tolist() == [0 if p is None else len(p) for p in expected]

    @nb.njit
    def at(a, i):
        return a[i]

    for i, payload in enumerate(expected):
        if payload is not None:
            assert at(view, i) == payload


BOOLEANS = [True, False, True, True, False, False, True, False,
            None, True, False, None, True, True, False, True, False]


@pytest.mark.parametrize("offset", range(8))
def test_boolean_read_every_bit_offset(offset):
    """Every bit position inside a byte, across the array's bytes."""
    view = pa.array(BOOLEANS, type=pa.bool_()).slice(offset)
    expected = BOOLEANS[offset:]

    @nb.njit
    def read(a):
        values = np.zeros(len(a), np.bool_)
        valid = np.zeros(len(a), np.bool_)
        missing = np.zeros(len(a), np.bool_)
        for i in range(len(a)):
            valid[i] = is_valid(a, i)
            missing[i] = a[i] is None
            v = a[i]
            if v is not None:
                values[i] = v
        return values, valid, missing

    values, valid, missing = read(view)
    assert missing.tolist() == [v is None for v in expected]
    assert valid.tolist() == [v is not None for v in expected]
    assert values.tolist() == [bool(v) if v is not None else False
                               for v in expected]


def test_boolean_read_without_a_null_bitmap():
    view = pa.array([True, False, True], type=pa.bool_())
    assert view.null_count == 0

    @nb.njit
    def read(a):
        out = np.zeros(len(a), np.int64)
        for i in range(len(a)):
            valid = 1 if is_valid(a, i) else 0
            present = 2 if a[i] is not None else 0
            out[i] = valid + present
        return out

    assert read(view).tolist() == [3, 3, 3]


def test_boolean_read_all_null():
    view = pa.nulls(5, type=pa.bool_())

    @nb.njit
    def count_valid(a):
        n = 0
        for i in range(len(a)):
            if is_valid(a, i):
                n += 1
        return n

    assert count_valid(view) == 0


def test_boolean_index_forms():
    view = pa.array([True, False, True], type=pa.bool_())

    @nb.njit
    def at(a, i):
        return a[i]

    assert at(view, 0) is True
    assert at(view, 1) is False
    assert at(view, -1) is True
    with pytest.raises(IndexError):
        at(view, 3)


def test_boolean_read_empty():
    view = pa.array([], type=pa.bool_())
    assert len(view) == 0

    @nb.njit
    def count(a):
        n = 0
        for i in range(len(a)):
            n += 1
        return n

    assert count(view) == 0


@pytest.mark.parametrize("pa_type", [pa.string(), pa.binary(),
                                     pa.binary(3)],
                         ids=["string", "binary", "fixed_size_binary"])
def test_string_view_carries_validity(pa_type):
    payloads = ([b"abc", None, b"", b"de"] if not pa.types.is_fixed_size_binary(
        pa_type) else [b"abc", None, b"xyz", b"def"])
    view = pa.array(payloads, type=pa_type).slice(1)

    @nb.njit
    def flags(a):
        out = np.zeros(len(a), np.bool_)
        for i in range(len(a)):
            out[i] = a[i].valid
        return out

    assert flags(view).tolist() == [p is not None for p in payloads[1:]]


def test_chunked_array_with_one_chunk():
    chunked = pa.chunked_array([[1, 2, 3]], type=pa.int64())

    @nb.njit
    def total(a):
        acc = 0
        for i in range(len(a)):
            acc += a[i]
        return acc

    assert total(chunked) == 6


@pytest.mark.parametrize("chunks", [
    [[1, 2, 3], [4, 5], [6]],
    [[1], [], [2, 3, 4, 5, 6]],
], ids=["three", "with_empty"])
def test_chunked_array_walks_every_chunk(chunks):
    chunked = pa.chunked_array(chunks, type=pa.int64())
    expected = [v for chunk in chunks for v in chunk]

    @nb.njit
    def total(a):
        acc = 0
        for i in range(len(a)):
            acc += a[i]
        return acc

    @nb.njit
    def read(a):
        return [a[i] for i in range(len(a))]

    assert len(chunked) == len(expected)
    assert total(chunked) == sum(expected)
    assert read(chunked) == expected


def test_chunked_array_with_nulls_and_negative_index():
    from pyarrow.numba import is_valid
    chunked = pa.chunked_array([[1, None], [None, 4]], type=pa.int64())

    @nb.njit
    def scan(a):
        total = 0
        nulls = 0
        for i in range(len(a)):
            v = a[i]
            if v is None:
                nulls += 1
            else:
                total += v
        return total, nulls, a[-1], is_valid(a, 1)

    assert scan(chunked) == (5, 2, 4, False)


def test_chunked_string_column():
    from pyarrow.numba import string_len
    chunked = pa.chunked_array([["ab", "cde"], ["f", None]],
                               type=pa.string())

    @nb.njit
    def sizes(a):
        out = []
        for i in range(len(a)):
            v = a[i]
            out.append(string_len(v) if v.valid else -1)
        return out

    assert sizes(chunked) == [2, 3, 1, -1]


def test_record_batch_columns_are_all_reachable():
    from pyarrow.numba import string_len
    batch = pa.record_batch({
        "x": pa.array([1, 2, 3]),
        "y": pa.array([10.5, 20.5, 30.5]),
        "z": pa.array(["a", "bb", "ccc"]),
    })

    @nb.njit
    def sums(batch):
        xs = 0
        ys = 0.0
        zs = 0
        x = batch.column(0)
        y = batch.column(1)
        z = batch.column(2)
        for i in range(len(batch)):
            xs += x[i]
            ys += y[i]
            zs += string_len(z[i])
        return xs, ys, zs, batch.num_columns

    assert sums(batch) == (6, 61.5, 6, 3)


def test_record_batch_needs_a_literal_column():
    batch = pa.record_batch({"x": pa.array([1, 2])})

    @nb.njit
    def nth(batch, k):
        return batch.column(k)[0]

    with pytest.raises(nb.core.errors.TypingError, match="literal index"):
        nth(batch, 0)


@pytest.mark.parametrize("chunks", [1, 2], ids=["one_chunk", "two_chunks"])
def test_table_reads_column_zero(chunks):
    if chunks == 1:
        column = pa.chunked_array([[1, None, 3]], type=pa.int64())
    else:
        column = pa.chunked_array([[1], [None, 3]], type=pa.int64())
    table = pa.table({"x": column, "y": pa.array(["a", "b", "c"])})

    @nb.njit
    def total(a):
        acc = 0
        for i in range(len(a)):
            if a[i] is not None:
                acc += a[i]
        return acc

    assert total(table) == 4

    if chunks > 1:
        # the chunks are kept, not concatenated into a temp array
        assert "PyArrowChunked" in str(total.signatures[0])
    else:
        assert "PyArrowChunked" not in str(total.signatures[0])


@pytest.mark.parametrize("null_count", [0, -1], ids=["known_dense", "unknown"])
def test_validity_bitmap_with_no_nulls(null_count):
    """A bitmap that marks nothing null must not change the values."""
    import pyarrow.numba  # noqa: F401
    data = pa.py_buffer(np.array([10, 20, 30], dtype=np.int64))
    bitmap = pa.py_buffer(np.array([0b111], dtype=np.uint8))   # all valid
    array = pa.Array.from_buffers(pa.int64(), 3, [bitmap, data],
                                  null_count=null_count)

    @nb.njit
    def total(a):
        acc = 0
        for i in range(len(a)):
            acc += a[i]
        return acc

    assert total(array) == 60


@pytest.mark.parametrize("builder_name,n", [("StringBuilder", 4),
                                            ("LargeStringBuilder", 4),
                                            ("BinaryBuilder", 0)])
def test_var_width_offsets_have_one_entry_per_value_plus_one(builder_name, n):
    """The leading offset is seeded once; a second seed would shift them all."""
    import pyarrow.numba as apn

    builder_cls = getattr(apn, builder_name)

    @nb.njit
    def build(count):
        b = builder_cls()
        for _ in range(count):
            b.append(b"xy")
        return b.finish()

    out = build(n)
    offsets = out.buffers()[1]
    width = 8 if "Large" in builder_name else 4
    assert len(offsets) == width * (n + 1)
    assert offsets.to_pybytes()[:width] == b"\x00" * width
    expected = [b"xy"] * n if builder_name.endswith("BinaryBuilder") else ["xy"] * n
    assert out.to_pylist() == expected
    out.validate(full=True)


def test_pool_buffer_offsets_come_from_the_cpp_layout():
    """The offsets are read from numba.cc, not hardcoded: three sane fields."""
    from pyarrow.numba import poolbuffer

    offsets = poolbuffer._pool_buffer_offsets()
    assert len(set(offsets)) == 3, offsets
    assert all(0 < offset < 64 for offset in offsets), offsets
    assert offsets == (poolbuffer._pool_buffer_offsetof_data,
                       poolbuffer._pool_buffer_offsetof_size,
                       poolbuffer._pool_buffer_offsetof_capacity)
