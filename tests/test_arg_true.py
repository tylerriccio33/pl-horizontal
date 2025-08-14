import polars as pl
import pytest
from pl_horizontal import arg_true_horizontal, arg_first_true_horizontal
import numpy as np


@pytest.mark.parametrize("stop_on_first", [False, True])
def test_basic_case(stop_on_first):
    df = pl.DataFrame(
        {
            "a": [True, False, True],
            "b": [False, True, True],
            "c": [False, False, True],
        }
    )
    if stop_on_first:
        result = df.select(arg_first_true_horizontal(pl.all()))
        expected = [0, 1, 0]  # first True per row
    else:
        result = df.select(arg_true_horizontal(pl.all()))
        expected = [
            [0],  # row0
            [1],  # row1
            [0, 1, 2],  # row2
        ]
    assert result.to_series().to_list() == expected


@pytest.mark.parametrize("stop_on_first", [False, True])
def test_multiple_true_per_row(stop_on_first):
    df = pl.DataFrame(
        {
            "x": [True, True],
            "y": [True, False],
            "z": [False, True],
        }
    )
    if stop_on_first:
        result = df.select(arg_first_true_horizontal(pl.all()))
        expected = [0, 0]  # first True per row
    else:
        result = df.select(arg_true_horizontal(pl.all()))
        expected = [
            [0, 1],  # row0
            [0, 2],  # row1
        ]
    assert result.to_series().to_list() == expected


@pytest.mark.parametrize("stop_on_first", [False, True])
def test_all_false(stop_on_first):
    df = pl.DataFrame(
        {
            "a": [False, False],
            "b": [False, False],
        }
    )
    if stop_on_first:
        result = df.select(arg_first_true_horizontal(pl.all()))
        expected = [None, None]  # no True found
    else:
        result = df.select(arg_true_horizontal(pl.all()))
        expected = [
            [],
            [],
        ]
    assert result.to_series().to_list() == expected


@pytest.mark.parametrize("stop_on_first", [False, True])
def test_with_nulls(stop_on_first):
    df = pl.DataFrame(
        {
            "a": [True, None],
            "b": [None, True],
        }
    )
    if stop_on_first:
        result = df.select(arg_first_true_horizontal(pl.all()))
        expected = [0, 1]  # first True per row
    else:
        result = df.select(arg_true_horizontal(pl.all()))
        expected = [
            [0],  # row0
            [1],  # row1
        ]
    assert result.to_series().to_list() == expected


@pytest.mark.parametrize("stop_on_first", [False, True])
def test_int_coercion(stop_on_first):
    df = pl.DataFrame(
        {
            "a": [1, 0, 2],
            "b": [0, 1, 0],
        }
    )
    if stop_on_first:
        result = df.select(arg_first_true_horizontal(pl.all()))
        expected = [0, 1, 0]  # first True per row
    else:
        result = df.select(arg_true_horizontal(pl.all()))
        expected = [
            [0],  # row0
            [1],  # row1
            [0],  # row2
        ]
    assert result.to_series().to_list() == expected


## Benchmaks:
@pytest.fixture
def df():
    n_rows = 100_000
    n_cols = 20

    rng = np.random.default_rng(seed=42)

    data = {}
    for i in range(n_cols):
        bools = rng.choice([True, False], size=n_rows)

        # Randomly set ~10% to None
        mask = rng.random(n_rows) < 0.1
        col = np.where(mask, None, bools)

        data[f"col{i}"] = col.tolist()

    return pl.DataFrame(data)


def test_arg_true_bench(benchmark, df):
    benchmark.group = "arg_true"
    benchmark(lambda: df.select(arg_true_horizontal(pl.all())))


def test_arg_first_true(benchmark, df):
    benchmark.group = "arg_true"
    benchmark(lambda: df.select(arg_first_true_horizontal(pl.all())))


if __name__ == "__main__":
    pytest.main([__file__])
