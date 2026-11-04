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

"""Direct tests for the pure helpers behind the extension."""

import pytest

import pyarrow as pa
from pyarrow.numba._utils import _arrow_type_to_nb_type, _is_var_width


@pytest.mark.parametrize("arrow_type,expected", [
    (pa.string(), True),
    (pa.large_string(), True),
    (pa.binary(), True),
    (pa.large_binary(), True),
    (pa.binary(8), True),
    (pa.int64(), False),
    (pa.date32(), False),
    (pa.bool_(), False),
])
def test_is_var_width(arrow_type, expected):
    assert _is_var_width(arrow_type) is expected


@pytest.mark.parametrize("arrow_type,bitwidth,signed", [
    (pa.int8(), 8, True),
    (pa.int64(), 64, True),
    (pa.uint16(), 16, False),
    (pa.uint64(), 64, False),
])
def test_integer_type_mapping(arrow_type, bitwidth, signed):
    mapped = _arrow_type_to_nb_type(arrow_type)
    assert mapped.bitwidth == bitwidth
    assert mapped.signed is signed


@pytest.mark.parametrize("arrow_type,expected", [
    (pa.float32(), "float32"),
    (pa.float64(), "float64"),
    (pa.date32(), "int32"),
    (pa.time32("s"), "int32"),
    (pa.date64(), "int64"),
    (pa.time64("us"), "int64"),
    (pa.timestamp("us"), "int64"),
    (pa.duration("s"), "int64"),
    (pa.bool_(), "bool"),
])
def test_scalar_type_mapping(arrow_type, expected):
    assert str(_arrow_type_to_nb_type(arrow_type)) == expected


def test_var_width_and_decimal_mapping():
    import numba.core.types as nbt

    assert _arrow_type_to_nb_type(pa.string()) == nbt.uint8
    assert _arrow_type_to_nb_type(pa.decimal128(20, 2)) == nbt.Tuple(
        (nbt.int64, nbt.uint64))
    with pytest.raises(TypeError, match="Unsupported PyArrow type"):
        _arrow_type_to_nb_type(pa.decimal256(40, 2))
    with pytest.raises(TypeError, match="Unsupported PyArrow type"):
        _arrow_type_to_nb_type(pa.list_(pa.int64()))


def test_builder_buffer_types():
    from pyarrow.numba.builder import _data_buffer_type, _offsets_type

    assert _data_buffer_type(pa.string()) == pa.uint8()
    assert _data_buffer_type(pa.int64()) == pa.int64()
    assert _offsets_type(pa.string()) == pa.int32()
    assert _offsets_type(pa.large_string()) == pa.int64()
    assert _offsets_type(pa.large_binary()) == pa.int64()


def test_num_buffers_per_type():
    from pyarrow.numba._capsule import num_buffers

    assert num_buffers(pa.int64()) == 2
    assert num_buffers(pa.bool_()) == 2
    assert num_buffers(pa.binary(4)) == 2
    assert num_buffers(pa.string()) == 3
    assert num_buffers(pa.large_binary()) == 3


def test_dtype_hint_rejects_what_it_cannot_represent():
    from pyarrow.numba._capsule import _dtype_hint

    class Opaque:
        pass

    assert _dtype_hint(Opaque()) is None          # no dtype, no types
    assert _dtype_hint(pa.array([1, 2])) is None  # pyarrow array: no hint either
