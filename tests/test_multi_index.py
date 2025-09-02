import pytest
import polars as pl

from pl_horizontal import multi_index
import numpy as np
import string
import itertools
from collections.abc import Callable


## Setup Testing Parameters:
VALID_DTYPES = [pl.UInt32, pl.Int32, pl.Int64]
CONTEXTS = {
    "ldf_stream": lambda df, expr: df.lazy().select(expr).collect(engine="streaming"),
    "ldf_eager": lambda df, expr: df.lazy().select(expr).collect(engine="in-memory"),
    "eager": lambda df, expr: df.select(expr),
}
type Context = Callable[[pl.DataFrame, pl.Expr], pl.DataFrame]
args: list[tuple[pl.DataType, Context]] = list(
    itertools.product(VALID_DTYPES, CONTEXTS.values())
)


@pytest.mark.parametrize("_args", args)
def test_simple_lookup(_args: tuple[pl.DataType, Context]):
    df = pl.DataFrame(
        {
            "idx": [0, 2, 1],
            "lookup": ["a", "b", "c"],
        }
    )
    dtype, fn = _args
    expr = multi_index(pl.col("idx").cast(dtype), df["lookup"])
    out = fn(df, expr)
    assert out.to_series().to_list() == ["a", "c", "b"]


@pytest.mark.parametrize("_args", args)
def test_with_nulls(_args: tuple[pl.DataType, Context]):
    df = pl.DataFrame({"idx": [0, None, 1]})
    ser = pl.Series(["x", "y"])
    dtype, fn = _args
    expr = multi_index(pl.col("idx").cast(dtype), ser)
    out = fn(df, expr)
    assert out.to_series().to_list() == ["x", None, "y"]


@pytest.mark.parametrize("_args", args)
def test_out_of_bounds(_args: tuple[pl.DataType, Context]):
    df = pl.DataFrame({"idx": [0, 2, 1]})
    ser = pl.Series(["first", "second"])
    dtype, fn = _args
    err = "gather indices are out of bounds"
    expr = multi_index(pl.col("idx").cast(dtype), ser)
    with pytest.raises(pl.exceptions.ComputeError, match=err):
        fn(df, expr)


@pytest.mark.parametrize("_args", args)
def test_repeated_and_reverse(_args: tuple[pl.DataType, Context]):
    df = pl.DataFrame({"idx": [2, 0, 2, 1]})
    ser = pl.Series(["alpha", "beta", "gamma"])
    dtype, _ = _args
    out = df.select(multi_index(pl.col("idx").cast(dtype), ser))
    # idx 2 → "gamma", idx 0 → "alpha", idx 1 → "beta"
    assert out.to_series().to_list() == ["gamma", "alpha", "gamma", "beta"]


@pytest.mark.parametrize("_args", args)
def test_all_null_idx(_args: tuple[pl.DataType, Context]):
    df = pl.DataFrame({"idx": [None, None, None]})
    ser = pl.Series(["one", "two"])
    dtype, fn = _args
    expr = multi_index(pl.col("idx").cast(dtype), ser)
    out = fn(df, expr)
    assert out.to_series().to_list() == [None, None, None]


@pytest.mark.parametrize("_args", args)
def test_simple1(_args: tuple[pl.DataType, Context]) -> None:
    lookup = pl.Series(
        [
            "this",  # 0
            "is",  # 1
            "a",  # 2
            "sentence",  # 3
            "to",  # 4
            "test",  # 5
            "the",  # 6
            "lookups",  # 7
            "hi",  # 8
        ]
    )
    df = pl.DataFrame(
        {"amanda": [3, 4, 6, 8], "tyler": [0, 1, None, 3], "winnie": [-1, 0, None, 3]}
    )

    exp = pl.DataFrame(
        {
            "amanda": ["sentence", "to", "the", "hi"],
            "tyler": ["this", "is", None, "sentence"],
        }
    )

    dtype, fn = _args
    expr = multi_index(pl.all().exclude("winnie").cast(dtype), lookup)
    res = fn(df, expr)

    assert res.equals(exp)


@pytest.mark.parametrize("_args", args)
def test_multi_gather_simple(_args: tuple[pl.DataType, Context]) -> None:
    s = pl.Series(values=["hi", "how", "are", "you"])
    df = pl.DataFrame({"foo": [0, 1, 2], "duchess": [None, 3, 0]})

    expected = pl.DataFrame(
        {"foo": ["hi", "how", "are"], "duchess": [None, "you", "hi"]}
    )

    dtype, fn = _args
    expr = multi_index(pl.all().cast(dtype), lookup=s)
    res = fn(df, expr)

    assert expected.equals(res)


## Benchmarks:
@pytest.fixture
def df() -> tuple[pl.DataFrame, pl.Series]:
    n_rows = 1_000_000
    n_cols = 20

    rng = np.random.default_rng(seed=42)

    data = {}
    for i in range(n_cols):
        ints = rng.integers(low=0, high=1_000, size=n_rows)

        # Randomly set ~10% to None
        mask = rng.random(n_rows) < 0.1
        col = np.where(mask, None, ints)

        data[f"col{i}"] = col.tolist()

    rand_strings = [rng.choice(list(string.ascii_letters)) for _ in range(1_000)]

    return pl.DataFrame(data), pl.Series(values=rng.choice(rand_strings, n_rows))


def test_multi_index_bench(df, benchmark) -> None:
    benchmark.group = "multi_index"
    data, ser = df

    new = data.select(multi_index(pl.all(), ser))
    old = _old(data, ser)

    assert new.equals(old.select(data.columns)), "Invalid benchmark"

    data = data.select(pl.all().cast(pl.UInt32))  # try the fastest path

    benchmark(lambda: data.select(multi_index(pl.all(), ser)))


def test_multi_index_bench_old(df, benchmark) -> None:
    benchmark.group = "multi_index"
    data, ser = df

    old = _old(data, ser).select(data.columns)
    new = data.select(multi_index(pl.all(), ser))

    assert old.equals(new), "Invalid benchmark"

    benchmark(lambda: _old(data, ser))


def _old(data: pl.DataFrame, ser: pl.Series) -> pl.DataFrame:
    cols: list[str] = data.columns
    for col in cols:
        indices = data[col]
        new = ser[indices].rename(col)
        data = data.drop(col)
        data.insert_column(0, new)

    return data


if __name__ == "__main__":
    pytest.main([__file__])
