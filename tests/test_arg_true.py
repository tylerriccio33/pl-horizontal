import polars as pl
import pytest
from pl_horizontal import arg_true_horizontal


def test_basic_case():
    df = pl.DataFrame(
        {
            "a": [True, False, True],
            "b": [False, True, True],
            "c": [False, False, True],
        }
    )
    result = df.select(arg_true_horizontal(pl.all()))
    assert result.to_series().to_list() == [
        [0],  # row0: only 'a' is True
        [1],  # row1: only 'b' is True
        [0, 1, 2],  # row2: all True
    ]


def test_multiple_true_per_row():
    df = pl.DataFrame(
        {
            "x": [True, True],
            "y": [True, False],
            "z": [False, True],
        }
    )
    result = df.select(arg_true_horizontal(pl.all()))
    assert result.to_series().to_list() == [
        [0, 1],  # row0: x,y
        [0, 2],  # row1: x,z
    ]


def test_all_false():
    df = pl.DataFrame(
        {
            "a": [False, False],
            "b": [False, False],
        }
    )
    result = df.select(arg_true_horizontal(pl.all()))
    assert result.to_series().to_list() == [
        [],
        [],
    ]


def test_with_nulls():
    df = pl.DataFrame(
        {
            "a": [True, None],
            "b": [None, True],
        }
    )
    result = df.select(arg_true_horizontal(pl.all()))
    # Nulls are treated as False
    assert result.to_series().to_list() == [
        [0],  # row0: a=True, b=null
        [1],  # row1: a=null, b=True
    ]


def test_int_coercion():
    df = pl.DataFrame(
        {
            "a": [1, 0, 2],
            "b": [0, 1, 0],
        }
    )
    result = df.select(arg_true_horizontal(pl.all()))
    assert result.to_series().to_list() == [
        [0],  # 1 -> True
        [1],  # 1 -> True
        [0],  # 2 -> True
    ]


if __name__ == "__main__":
    pytest.main([__file__])
