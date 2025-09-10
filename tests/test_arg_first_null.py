import polars as pl
import pytest
from pl_horizontal import arg_first_null_horizontal


def test_basic_null_detection():
    """Test basic functionality with mixed null patterns"""
    df = pl.DataFrame({"a": [1, 2, None], "b": [None, 2, 3], "c": [1, None, 3]})
    result = df.select(arg_first_null_horizontal(pl.all())).to_series().to_list()
    expected = [1, 2, 0]  # First null at index 1, 2, 0 respectively
    assert result == expected


def test_no_nulls():
    """Test when no nulls are present in any row"""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    result = df.select(arg_first_null_horizontal(pl.all())).to_series().to_list()
    # Should return -1 or None when no nulls found (implementation dependent)
    # Assuming it returns None based on typical behavior
    expected = [None, None, None]
    assert result == expected


def test_all_nulls():
    """Test when entire rows are null"""
    df = pl.DataFrame(
        {"a": [None, None, 1], "b": [None, None, 2], "c": [None, None, 3]}
    )
    result = df.select(arg_first_null_horizontal(pl.all())).to_series().to_list()
    expected = [
        0,
        0,
        None,
    ]  # First null at index 0 for first two rows, no nulls in third
    assert result == expected


def test_single_column():
    """Test with single column DataFrame"""
    df = pl.DataFrame({"a": [1, None, 3, None]})
    result = df.select(arg_first_null_horizontal(pl.all())).to_series().to_list()
    expected = [None, 0, None, 0]  # Null at index 0 when present, None when not
    assert result == expected


def test_mixed_data_types():
    """Test with different data types including strings and numbers"""
    df = pl.DataFrame(
        {
            "numeric": [1, None, 3.5, 0],
            "string": ["hello", "world", None, None],
            "boolean": [True, False, True, None],
            "mixed": [None, 42, "test", None],
        },
        strict=False,
    )
    result = df.select(arg_first_null_horizontal(pl.all())).to_series().to_list()
    expected = [3, 0, 1, 1]  # First null positions: col 3, col 1, col 2, col 3
    assert result == expected


# Additional edge case test
def test_empty_dataframe():
    """Test with empty DataFrame"""
    df = pl.DataFrame({"a": [], "b": [], "c": []})
    result = df.select(arg_first_null_horizontal(pl.all())).to_series().to_list()
    expected = []  # Empty result for empty DataFrame
    assert result == expected


## -- Bench


def test_bench_arg_first_null(benchmark, df_ints):
    """Benchmark the arg_first_null_horizontal function"""
    benchmark.group = "arg_first_null"
    benchmark(lambda: df_ints.select(arg_first_null_horizontal(pl.all())))


def test_bench_arg_first_null_old(benchmark, df_ints):
    """Benchmark the arg_first_null_horizontal function"""
    benchmark.group = "arg_first_null"
    new = df_ints.select(arg_first_null_horizontal(pl.all()))

    # Compute the index of the first null in each row, or None if all are not null
    old = df_ints.select(
        pl.when(pl.concat_list(pl.all().is_null()).list.any())
        .then(pl.concat_list(pl.all().is_null()).list.arg_max())
        .otherwise(None)
    )

    assert new.equals(old)

    benchmark(
        lambda: df_ints.select(
            pl.when(pl.concat_list(pl.all().is_null()).list.any())
            .then(pl.concat_list(pl.all().is_null()).list.arg_max())
            .otherwise(None)
        )
    )


if __name__ == "__main__":
    pytest.main([__file__])
