import polars as pl
import pytest
from pl_horizontal import arg_true_horizontal
import numpy as np


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


## Benchmaks:
@pytest.fixture
def df():
    n_rows = 100_000
    n_cols = 20

    rng = np.random.default_rng(seed=42)  # reproducible

    data = {}
    for i in range(n_cols):
        # Generate random bools (True/False)
        bools = rng.choice([True, False], size=n_rows)

        # Randomly set ~10% to None
        mask = rng.random(n_rows) < 0.1
        col = np.where(mask, None, bools)

        data[f"col{i}"] = col.tolist()

    return pl.DataFrame(data)


def test_arg_true_bench(benchmark, df):
    benchmark.group = "arg_true"
    benchmark(lambda: df.select(arg_true_horizontal(pl.all())))


if __name__ == "__main__":
    pytest.main([__file__])
