"""Building Arrow arrays: the builders, their buffers and their limits."""

import gc

import pytest

import pyarrow as pa

nb = pytest.importorskip("numba")

from pyarrow.numba import _register  # noqa: E402  (registers the extension)

_register()


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


def test_bool_builder_crosses_a_byte():
    from pyarrow.numba import BoolBuilder

    @nb.njit
    def build():
        b = BoolBuilder()
        b.reserve(10)
        for i in range(10):
            if i % 3 == 1:
                b.append(None)
            else:
                b.append(i % 2 == 0)
        return b.finish()

    out = build()
    assert out.type == pa.bool_()
    # nulls every third (i % 3 == 1), value is i % 2 == 0
    assert out.to_pylist() == [True, None, True, False, None, False, True,
                               None, True, False]


@pytest.mark.parametrize("builder_name,pa_type,expected", [
    ("StringBuilder", pa.string(), ["ab", None, "", "cde"]),
    ("LargeStringBuilder", pa.large_string(), ["ab", None, "", "cde"]),
    ("BinaryBuilder", pa.binary(), [b"ab", None, b"", b"cde"]),
    ("LargeBinaryBuilder", pa.large_binary(), [b"ab", None, b"", b"cde"]),
])
def test_var_width_builders(builder_name, pa_type, expected):
    import pyarrow.numba as apn

    builder_cls = getattr(apn, builder_name)

    @nb.njit
    def build():
        b = builder_cls()
        b.append(b"ab")
        b.append(None)
        b.append(b"")
        b.append(b"cde")
        return b.finish()

    out = build()
    assert out.type == pa_type
    assert out.to_pylist() == expected
    assert out.null_count == 1
    out.validate(full=True)


def test_string_builder_appends_from_a_view():
    from pyarrow.numba import StringBuilder

    source = pa.array([b"abc", None, b"de"], type=pa.string())

    @nb.njit
    def copy_valid(a):
        b = StringBuilder()
        for i in range(len(a)):
            v = a[i]
            if v.valid:
                b.append(v)
            else:
                b.append(None)
        return b.finish()

    out = copy_valid(source)
    assert out.to_pylist() == ["abc", None, "de"]
    out.validate(full=True)


def test_string_builder_copies_long_and_sliced_views():
    """One append can outgrow the buffer, and a slice starts mid-buffer."""
    from pyarrow.numba import StringBuilder

    source = pa.array([b"x" * 100, b"y", b"z" * 300], type=pa.string())[1:]

    @nb.njit
    def copy(a):
        b = StringBuilder()
        for i in range(len(a)):
            b.append(a[i])
        return b.finish()

    out = copy(source)
    assert out.to_pylist() == ["y", "z" * 300]
    out.validate(full=True)


def test_string_builder_can_finish_empty():
    from pyarrow.numba import StringBuilder

    @nb.njit
    def build():
        b = StringBuilder()
        return b.finish()

    out = build()
    assert out.to_pylist() == []
    out.validate(full=True)


def test_string_builder_keeps_a_null_view():
    """Appending a view without checking .valid must keep the null."""
    from pyarrow.numba import StringBuilder

    source = pa.array(["ab", None, "c"], type=pa.string())

    @nb.njit
    def copy(a):
        b = StringBuilder()
        for i in range(len(a)):
            b.append(a[i])
        return b.finish()

    out = copy(source)
    assert out.to_pylist() == ["ab", None, "c"]
    assert out.null_count == 1


def test_string_builder_reserve_then_append():
    from pyarrow.numba import StringBuilder

    @nb.njit
    def build(n):
        b = StringBuilder()
        b.reserve(n)
        for _ in range(n):
            b.append(b"xy")
        return b.finish()

    assert build(5).to_pylist() == ["xy"] * 5


@pytest.mark.parametrize("arrow_type", [pa.binary(4), pa.date32()])
def test_types_without_a_builder(arrow_type):
    from pyarrow.numba.builder import ArrayBuilder

    with pytest.raises(TypeError, match="Cannot create ArrayBuilder"):
        ArrayBuilder(arrow_type)


@pytest.mark.parametrize("builder_name,pa_type,expected", [
    ("StringBuilder", pa.string(), ["ab", None, "c"]),
    ("BinaryBuilder", pa.binary(), [b"ab", None, b"c"]),
])
def test_var_width_builder_appends_null_views_as_nulls(builder_name, pa_type,
                                                       expected):
    """A null view is a null, not an empty value (it used to be ``""``)."""
    import pyarrow.numba as apn

    builder_cls = getattr(apn, builder_name)
    source = pa.array([b"ab", None, b"c"], type=pa_type)

    @nb.njit
    def copy(a):
        b = builder_cls()
        for i in range(len(a)):
            b.append(a[i])          # no .valid check on purpose
        return b.finish()

    out = copy(source)
    assert out.null_count == 1
    assert out.to_pylist() == expected
    out.validate(full=True)


def test_empty_var_width_builder_still_has_a_zero_offset():
    import pyarrow.numba as apn

    @nb.njit
    def build():
        b = apn.StringBuilder()
        return b.finish()

    out = build()
    assert out.to_pylist() == []
    assert len(out.buffers()[1]) == 4          # one int32 offset, zero
    out.validate(full=True)
