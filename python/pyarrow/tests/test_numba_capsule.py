"""The Arrow C data interface: structs, capsules and buffer ownership."""

import gc

import pytest

import pyarrow as pa

nb = pytest.importorskip("numba")

from pyarrow.numba import _register  # noqa: E402  (registers the extension)
_register()


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


def test_producer_with_too_few_buffers_is_refused():
    """A struct with fewer buffers than its type needs must raise, not read
    past the end of the buffer array."""
    import ctypes

    from pyarrow.numba._capsule import ArrowArray, take_array_into

    class Liar:
        """Says "string", exports an int64 struct (two buffers, not three)."""

        def __arrow_c_array__(self):
            schema, _ = pa.array(["a"]).__arrow_c_array__()
            _, array = pa.array([1, 2]).__arrow_c_array__()
            return schema, array

    slot = ArrowArray()
    with pytest.raises(ValueError, match="needs 3"):
        take_array_into(Liar(), ctypes.addressof(slot), pa.string())


def test_object_dtype_columns_do_not_share_a_cached_type():
    """Two pandas object columns can hold a string and a struct: the dtype
    cache must not answer the second with the first one's type."""
    pandas = pytest.importorskip("pandas")

    from pyarrow.numba._capsule import _DTYPE_CACHE, dtype_of

    _DTYPE_CACHE.clear()
    words = pandas.Series([{"x": "a"}, {"x": "b"}], dtype=object)
    numbers = pandas.Series([{"x": 1}, {"x": 2}], dtype=object)

    assert dtype_of(words) == pa.string()
    assert dtype_of(numbers) == pa.int64()
