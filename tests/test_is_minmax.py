import pytest
import polars as pl
from pl_horizontal import is_max, is_min
import polars.testing.parametric as ptp
from hypothesis import given
import polars.selectors as cs


def test_simple_max() -> None:
    # https://stackoverflow.com/questions/70845035/is-there-any-simliar-function-of-idxmax-in-py-polars-in-groupby
    df = pl.DataFrame(
        {
            "cola": ["a", "a", "b", "b", "c", "c"],
            "colb": [1, 2, 3, 4, 99, 6],
        }
    )
    exp = pl.DataFrame({"top": [False, True, False, True, True, False]})

    res = df.select(top=is_max(pl.col("colb")).over("cola"))

    assert res.equals(exp)


def test_simple_min() -> None:
    df = pl.DataFrame(
        {
            "cola": ["a", "a", "b", "b", "c", "c"],
            "colb": [1, 2, 3, 4, 99, 6],
        }
    )
    exp = pl.DataFrame({"top": [True, False, True, False, False, True]})

    res = df.select(top=is_min(pl.col("colb")).over("cola"))

    assert res.equals(exp)


def test_with_nulls_max() -> None:
    df = pl.DataFrame(
        {
            "cola": ["a", "a", "b", "b", "c", "c"],
            "colb": [1, None, 3, 4, None, 6],
        }
    )
    exp = pl.DataFrame({"top": [True, False, False, True, False, True]})

    res = df.select(top=is_max(pl.col("colb")).over("cola"))

    assert res.equals(exp)


def test_with_nulls_min() -> None:
    df = pl.DataFrame(
        {
            "cola": ["a", "a", "b", "b", "c", "c"],
            "colb": [1, None, 3, 4, None, 6],
        }
    )
    exp = pl.DataFrame({"top": [True, False, True, False, False, True]})

    res = df.select(top=is_min(pl.col("colb")).over("cola"))

    assert res.equals(exp)


@given(df=ptp.dataframes(min_cols=1, allowed_dtypes={pl.Int64, pl.Float64, pl.UInt32}))
def test_correctness_hypothesis_max(df: pl.DataFrame) -> None:
    df = df.select(cs.float().fill_nan(None))

    expr = is_max(pl.all())
    res = df.select(expr)

    if df.height == 0:
        pytest.mark.skip("No rows to test")
        return

    for col in df.columns:
        ser: pl.Series = df[col]
        res_ser: pl.Series = res[col]
        exp_max = ser.max()
        real_max = df[col][res_ser.arg_max()]
        assert exp_max == real_max, f"Failed on column {col} with df:\n{df}"

    if df.width == 1:
        return


@given(df=ptp.dataframes(min_cols=1, allowed_dtypes={pl.Int64, pl.Float64, pl.UInt32}))
def test_correctness_hypothesis_min(df: pl.DataFrame) -> None:
    df = df.select(cs.float().fill_nan(None))

    expr = is_min(pl.all())
    res = df.select(expr)

    if df.height == 0:
        pytest.mark.skip("No rows to test")
        return

    for col in df.columns:
        ser: pl.Series = df[col]
        res_ser: pl.Series = res[col]
        exp_min = ser.min()
        real_min = df[col][res_ser.arg_max()]  # need arg max to get the True
        assert exp_min == real_min, f"Failed on column {col} with df:\n{df}"

    if df.width == 1:
        return


def test_no_over_max() -> None:
    df = pl.DataFrame(
        {
            "cola": ["a", "a", "b", "b", "c", "c"],
            "colb": [1, None, 3, 4, None, 6],
        }
    )
    exp = pl.DataFrame({"top": [False, False, False, False, False, True]})

    res = df.select(top=is_max(pl.col("colb")))

    assert res.equals(exp)


def test_no_over_min() -> None:
    df = pl.DataFrame(
        {
            "cola": ["a", "a", "b", "b", "c", "c"],
            "colb": [1, None, 3, 4, None, 6],
        }
    )
    exp = pl.DataFrame({"top": [True, False, False, False, False, False]})

    res = df.select(top=is_min(pl.col("colb")))

    assert res.equals(exp)


## -- Benchmarks
def test_bench_is_max(benchmark, df_ints) -> None:
    benchmark.group = "is_minmax"
    benchmark(lambda: df_ints.select(is_max(pl.all())))


def test_bench_is_max_over(benchmark, df_ints) -> None:
    benchmark.group = "is_minmax"
    benchmark(lambda: df_ints.select(is_max(pl.all()).over("col1")))


def test_bench_is_max_old(benchmark, df_ints) -> None:
    benchmark.group = "is_minmax"
    expr = pl.all().arg_max() == pl.arange(0, pl.len())
    assert df_ints.select(expr).equals(df_ints.select(is_max(pl.all())))
    benchmark(lambda: df_ints.select(expr))


def test_bench_is_max_over_old(benchmark, df_ints) -> None:
    benchmark.group = "is_minmax"
    exprs: tuple[pl.Expr, ...] = (
        pl.int_range(pl.len()).over("col1").alias("__idx"),
        pl.all()
        .exclude("__idx")
        .arg_max()
        .eq(pl.col("__idx"))
        .over("col1")
        .fill_null(False),
    )
    ref = df_ints.select(is_max(pl.all()).over("col1"))

    assert df_ints.with_columns(exprs[0]).select(exprs[1]).equals(ref)
    benchmark(lambda: df_ints.with_columns(exprs[0]).select(exprs[1]))


if __name__ == "__main__":
    pytest.main([__file__])
