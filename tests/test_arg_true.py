import polars as pl
import pytest
from pl_horizontal import arg_true_horizontal, arg_first_true_horizontal
import numpy as np


@pytest.mark.parametrize(
    ("stop_on_first", "selector"),
    [(True, pl.all()), (True, ["a", "b", "c"]), (False, pl.all())],
)
def test_basic_case(stop_on_first: bool, selector: pl.Expr):
    df = pl.DataFrame(
        {
            "a": [True, False, True],
            "b": [False, True, True],
            "c": [False, False, True],
        }
    )
    if stop_on_first:
        result = df.select(arg_first_true_horizontal(selector))
        expected = [0, 1, 0]  # first True per row
    else:
        result = df.select(arg_true_horizontal(selector))
        expected = [
            [0],  # row0
            [1],  # row1
            [0, 1, 2],  # row2
        ]
    assert result.to_series().to_list() == expected


@pytest.mark.parametrize(
    ("stop_on_first", "selector"),
    [(True, pl.all()), (True, ["x", "y", "z"]), (False, pl.all())],
)
def test_multiple_true_per_row(stop_on_first: bool, selector: pl.Expr):
    df = pl.DataFrame(
        {
            "x": [True, True],
            "y": [True, False],
            "z": [False, True],
        }
    )
    if stop_on_first:
        result = df.select(arg_first_true_horizontal(selector))
        expected = [0, 0]  # first True per row
    else:
        result = df.select(arg_true_horizontal(selector))
        expected = [
            [0, 1],  # row0
            [0, 2],  # row1
        ]
    assert result.to_series().to_list() == expected


@pytest.mark.parametrize(
    ("stop_on_first", "selector"),
    [(True, pl.all()), (True, ["a", "b"]), (False, pl.all())],
)
def test_all_false(stop_on_first: bool, selector: pl.Expr):
    df = pl.DataFrame(
        {
            "a": [False, False],
            "b": [False, False],
        }
    )
    if stop_on_first:
        result = df.select(arg_first_true_horizontal(selector))
        expected = [None, None]  # no True found
    else:
        result = df.select(arg_true_horizontal(selector))
        expected = [
            [],
            [],
        ]
    assert result.to_series().to_list() == expected


@pytest.mark.parametrize(
    ("stop_on_first", "selector"),
    [(True, pl.all()), (True, ["a", "b"]), (False, pl.all())],
)
def test_with_nulls(stop_on_first: bool, selector: pl.Expr):
    df = pl.DataFrame(
        {
            "a": [True, None],
            "b": [None, True],
        }
    )
    if stop_on_first:
        result = df.select(arg_first_true_horizontal(selector))
        expected = [0, 1]  # first True per row
    else:
        result = df.select(arg_true_horizontal(selector))
        expected = [
            [0],  # row0
            [1],  # row1
        ]
    assert result.to_series().to_list() == expected


## Benchmaks:
# TODO: These should all be conf fixtures
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


def test_arg_true_bench_old(benchmark, df):
    """Arg true horizontals if we know the columns."""
    benchmark.group = "arg_true"

    exprs: list[pl.Expr] = [
        pl.when(col).then(i).cast(int).alias(col) for i, col in enumerate(df.columns)
    ]

    res = df.select(pl.concat_list(exprs).list.drop_nulls())
    mine = df.select(arg_true_horizontal(pl.all()))

    assert res.equals(mine), "Invalid benchmark"

    benchmark(lambda: df.select(pl.concat_list(exprs).drop_nulls()))


def test_arg_first_true_bench_static(benchmark, df) -> None:
    """Benchmark for when columns are known at calltime; my function may fallback to this.."""
    benchmark.group = "arg_first_true"

    # TODO: Insert the builder function here
    col_iter = iter(df.columns)
    exprs = pl.when(next(col_iter)).then(0)
    for i, col in enumerate(col_iter, 1):
        exprs = exprs.when(col).then(i)

    res = df.select(exprs.alias("col0"))
    mine = df.select(arg_first_true_horizontal(pl.all()))

    assert res.equals(mine), "Invalid benchmark"

    benchmark(lambda: df.select(exprs))


def test_arg_first_true_bench_dynamic_old(benchmark, df) -> None:
    """Benchmark for when columns are not known at calltime."""
    benchmark.group = "arg_first_true"

    res = df.select(pl.concat_arr(pl.all()).arr.arg_max())
    mine = df.select(arg_first_true_horizontal(pl.all()))

    assert res.equals(mine), "Invalid benchmark"

    benchmark(lambda: df.select(pl.concat_arr(pl.all()).arr.arg_max()))


def test_arg_first_true_bench_dynamic(benchmark, df):
    benchmark.group = "arg_first_true"
    benchmark(lambda: df.select(arg_first_true_horizontal(pl.all())))


if __name__ == "__main__":
    pytest.main([__file__])
