import gc

import numpy as np
import pytest

import pyarrow as pa

nb = pytest.importorskip("numba")

from pyarrow.numba import _register  # noqa: E402  (imports the extension)
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


def test_int_builder_round_trip():
    from pyarrow.numba import Int64Builder

    @nb.njit
    def build():
        b = Int64Builder()
        b.append(1)
        b.append(None)
        b.append(3)
        return b.finish()

    out = build()
    assert isinstance(out, pa.Array)
    assert out.type == pa.int64()
    assert out.to_pylist() == [1, None, 3]


def test_float_builder_round_trip():
    from pyarrow.numba import Float64Builder

    @nb.njit
    def build():
        b = Float64Builder()
        b.append(1.5)
        b.append(None)
        b.append(3.5)
        return b.finish()

    out = build()
    assert isinstance(out, pa.Array)
    assert out.type == pa.float64()
    assert out.to_pylist() == [1.5, None, 3.5]


def test_many_builds_do_not_leak_pool_memory():
    from pyarrow.numba import Float64Builder

    @nb.njit
    def build(n):
        b = Float64Builder()
        b.reserve(n)
        for i in range(n):
            b.append(i * 0.5)
        return b.finish()

    assert build(4).to_pylist() == [0.0, 0.5, 1.0, 1.5]   # warm up

    pool = pa.default_memory_pool()
    before = pool.bytes_allocated()
    for _ in range(2000):
        build(16)                       # each result is dropped immediately
    gc.collect()
    after = pool.bytes_allocated()

    assert after - before < 1 << 16, (before, after)


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


def test_capsule_producer_is_read_without_materialising_it():
    """Unboxing fills the view from the capsule: no pa.Array per call."""
    import ctypes

    from pyarrow.numba._capsule import ArrowArray, take_array_into

    class Producer:
        """A producer that is not a pa.Array."""

        def __init__(self, array):
            self._array = array

        def __arrow_c_array__(self, requested_schema=None):
            return self._array.__arrow_c_array__(requested_schema)

    array = pa.array([2, 3, None, 5])
    view = ArrowArray()
    take_array_into(Producer(array), ctypes.addressof(view))

    assert view.length == 4
    assert view.n_buffers == 2
    assert view.buffers[1] == array.buffers()[1].address    # zero copy
    assert view.release


def test_take_moves_the_release_callback():
    from pyarrow.numba._capsule import ArrowArray, take

    source = pa.array([1, 2, 3])
    _, array_capsule = source.__arrow_c_array__()
    moved = take(ArrowArray, array_capsule, b"arrow_array")

    assert moved.length == 3
    assert moved.offset == 0
    assert moved.buffers[1]                  # data buffer, non-null
    assert moved.release                     # we own it now

    # the capsule's own struct was neutralised: dropping it must not free
    del array_capsule, source
    gc.collect()
    assert moved.length == 3                 # still readable, no crash


def test_multi_chunk_stream_raises_and_from_any_copies():
    table = pa.table({"x": pa.chunked_array([[1, 2], [3, 4]],
                                            type=pa.int64())})

    class StreamOnly:
        """Polars/duckdb shape: a stream, no array capsule."""

        def __init__(self, table):
            self._table = table

        def __arrow_c_stream__(self, requested_schema=None):
            return self._table.__arrow_c_stream__(requested_schema)

    stream = StreamOnly(table)
    from pyarrow.numba import from_any, register_producer
    register_producer(StreamOnly)

    @nb.njit
    def total(a):
        acc = 0
        for i in range(len(a)):
            if a[i] is not None:
                acc += a[i]
        return acc

    with pytest.raises(TypeError, match="more than one chunk"):
        total(stream)

    assert total(from_any(stream)) == 10      # the documented copying route


def test_empty_stream_reads_as_an_empty_array():
    empty = pa.table({"x": pa.chunked_array([], type=pa.int64())})

    class StreamOnly:
        def __arrow_c_stream__(self, requested_schema=None):
            return empty.__arrow_c_stream__(requested_schema)

    from pyarrow.numba import register_producer
    register_producer(StreamOnly)

    @nb.njit
    def total(a):
        acc = 0
        for i in range(len(a)):
            acc += a[i]
        return acc

    assert total(StreamOnly()) == 0


def test_producer_buffers_are_not_kept_alive_by_the_call():
    import weakref

    seen = []

    class Producer:
        def __init__(self, array):
            self._array = array

        def __arrow_c_array__(self, requested_schema=None):
            seen.append(weakref.ref(self._array))
            return self._array.__arrow_c_array__(requested_schema)

    from pyarrow.numba import register_producer
    register_producer(Producer)

    @nb.njit
    def total(a):
        acc = 0
        for i in range(len(a)):
            acc += a[i]
        return acc

    array = pa.array([1, 2, 3])
    assert total(Producer(array)) == 6

    del array
    gc.collect()
    assert seen[0]() is None        # nothing from the call is kept alive


def test_capsule_struct_matches_the_c_layout():
    import ctypes

    from pyarrow.numba._capsule import (ArrowArray, ArrowArrayStream,
                                        ArrowSchema)

    assert ctypes.sizeof(ArrowArray) == 80
    assert [name for name, _ in ArrowArray._fields_] == [
        "length", "null_count", "offset", "n_buffers", "n_children",
        "buffers", "children", "dictionary", "release", "private_data",
    ]
    assert ctypes.sizeof(ArrowArrayStream) == 40
    assert [name for name, _ in ArrowArrayStream._fields_] == [
        "get_schema", "get_next", "get_last_error", "release", "private_data",
    ]
    assert [name for name, _ in ArrowSchema._fields_] == [
        "format", "name", "metadata", "flags", "n_children", "children",
        "dictionary", "release", "private_data",
    ]


def test_boolean_arrays_have_no_element_access():
    from numba.core.errors import TypingError

    array = pa.array([True, False, None])

    @nb.njit
    def at(a, i):
        return a[i]

    with pytest.raises(TypingError, match="bit-packed"):
        at(array, 0)


@pytest.mark.parametrize("builder_name,pa_type,values", [
    ("Int8Builder", pa.int8(), [1, None, 3]),
    ("Int16Builder", pa.int16(), [1, None, 3]),
    ("Int32Builder", pa.int32(), [1, None, 3]),
    ("Int64Builder", pa.int64(), [1, None, 3]),
    ("UInt8Builder", pa.uint8(), [1, None, 3]),
    ("UInt16Builder", pa.uint16(), [1, None, 3]),
    ("UInt32Builder", pa.uint32(), [1, None, 3]),
    ("UInt64Builder", pa.uint64(), [1, None, 3]),
    ("Float32Builder", pa.float32(), [0.5, None, 1.5]),
    ("Float64Builder", pa.float64(), [0.5, None, 1.5]),
    ("BoolBuilder", pa.bool_(), [True, None, False]),
])
def test_every_builder_round_trips(builder_name, pa_type, values):
    import pyarrow.numba as apn

    builder_cls = getattr(apn, builder_name)

    # literals inside the compiled function: freevar lists do not lower
    if pa.types.is_boolean(pa_type):
        @nb.njit
        def build():
            b = builder_cls()
            b.reserve(3)
            b.append(True)
            b.append(None)
            b.append(False)
            return b.finish()
    elif pa.types.is_floating(pa_type):
        @nb.njit
        def build():
            b = builder_cls()
            b.reserve(3)
            b.append(0.5)
            b.append(None)
            b.append(1.5)
            return b.finish()
    else:
        @nb.njit
        def build():
            b = builder_cls()
            b.reserve(3)
            b.append(1)
            b.append(None)
            b.append(3)
            return b.finish()

    out = build()
    assert out.type == pa_type
    assert out.to_pylist() == values


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


@pytest.mark.parametrize("offset", [0, 1])
def test_boolean_values_are_reachable_through_the_bitmap(offset):
    array = pa.array([True, None, False, True])
    view = array.slice(offset)

    @nb.njit
    def flags(a):
        out = np.zeros(len(a), np.bool_)
        for i in range(len(a)):
            out[i] = is_valid(a, i)
        return out

    assert flags(view).tolist() == array.is_valid().to_pylist()[offset:]


def test_typing_a_stream_producer_is_cached():
    """Numba re-types per call: the schema must not be read every time."""
    opened = []

    class Producer:
        dtype = "int64"          # cheap, hashable, stands in for the schema

        def __init__(self, array):
            self._array = array

        def __arrow_c_stream__(self, requested_schema=None):
            opened.append(1)
            return self._array.__arrow_c_stream__(requested_schema)

    from pyarrow.numba import register_producer
    register_producer(Producer)

    @nb.njit
    def total(a):
        acc = 0
        for i in range(len(a)):
            acc += a[i]
        return acc

    producer = Producer(pa.chunked_array([[1, 2, 3]]))
    assert total(producer) == 6          # one stream for typing, one for data

    before = len(opened)
    assert total(producer) == 6
    assert len(opened) - before == 1     # typing came from the cache
