import pytest
import polars as pl
from pl_horizontal import is_max


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


def test_with_nulls() -> None:
    df = pl.DataFrame(
        {
            "cola": ["a", "a", "b", "b", "c", "c"],
            "colb": [1, None, 3, 4, None, 6],
        }
    )
    exp = pl.DataFrame({"top": [True, False, False, True, False, True]})

    res = df.select(top=is_max(pl.col("colb")).over("cola"))

    assert res.equals(exp)

def test_no_over() -> None:
    df = pl.DataFrame(
        {
            "cola": ["a", "a", "b", "b", "c", "c"],
            "colb": [1, None, 3, 4, None, 6],
        }
    )
    exp = pl.DataFrame({"top": [False, False, False, False, False, True]})

    res = df.select(top=is_max(pl.col("colb")))

    assert res.equals(exp)


## -- Benchmarks
def test_bench_is_max(benchmark, df_ints) -> None:
    benchmark.group = "is_max"
    benchmark(lambda: df_ints.select(is_max(pl.all())))


def test_bench_is_max_over(benchmark, df_ints) -> None:
    benchmark.group = "is_max"
    benchmark(lambda: df_ints.select(is_max(pl.all()).over("col1")))


def test_bench_is_max_old(benchmark, df_ints) -> None:
    benchmark.group = "is_max"
    expr = pl.all().arg_max() == pl.arange(0, pl.len())
    assert df_ints.select(expr).equals(df_ints.select(is_max(pl.all())))
    benchmark(lambda: df_ints.select(expr))


def test_bench_is_max_over_old(benchmark, df_ints) -> None:
    benchmark.group = "is_max"
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
