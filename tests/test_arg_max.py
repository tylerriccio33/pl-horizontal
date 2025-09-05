import pytest
import polars as pl
from pl_horizontal import arg_max_horizontal


@pytest.mark.parametrize("colname", [True, False])
def test_arg_max_basic_streaming(colname: bool):
    """Test basic arg_max functionality with simple numeric data."""
    df = pl.DataFrame({"a": [1, 5, 3], "b": [4, 2, 6], "c": [2, 8, 1]})

    # Expected: max values are at indices [1, 2, 1] (0-based)
    # Row 0: [1, 4, 2] -> max is 4 at index 1
    # Row 1: [5, 2, 8] -> max is 8 at index 2
    # Row 2: [3, 6, 1] -> max is 6 at index 1
    result = (
        df.lazy()
        .select(
            arg_max_horizontal(pl.col("a", "b", "c"), return_colname=colname).alias(
                "arg_max"
            )
        )
        .collect(engine="streaming")
    )

    if colname:
        expected = ["b", "c", "b"]
    else:
        expected = [1, 2, 1]
    assert result["arg_max"].to_list() == expected


@pytest.mark.parametrize("colname", [True, False])
def test_arg_max_basic(colname: bool):
    """Test basic arg_max functionality with simple numeric data."""
    df = pl.DataFrame({"a": [1, 5, 3], "b": [4, 2, 6], "c": [2, 8, 1]})

    # Expected: max values are at indices [1, 2, 1] (0-based)
    # Row 0: [1, 4, 2] -> max is 4 at index 1
    # Row 1: [5, 2, 8] -> max is 8 at index 2
    # Row 2: [3, 6, 1] -> max is 6 at index 1
    result = df.select(
        arg_max_horizontal(pl.col("a", "b", "c"), return_colname=colname).alias(
            "arg_max"
        )
    )

    if colname:
        expected = ["b", "c", "b"]
    else:
        expected = [1, 2, 1]
    assert result["arg_max"].to_list() == expected


def test_arg_max_so() -> None:
    # https://stackoverflow.com/questions/77967334/getting-min-max-column-name-in-polars/
    df = pl.DataFrame(
        {
            "a": [1, 8, 3],
            "b": [4, 5, None],
        }
    )

    res = df.with_columns(max=arg_max_horizontal(pl.all(), return_colname=True))

    assert res["max"].to_list() == ["b", "a", "a"]


@pytest.mark.parametrize("colname", [True, False])
def test_arg_max_with_nulls(colname: bool):
    """Test arg_max with null values."""
    df = pl.DataFrame({"a": [1, None, 3], "b": [None, 2, None], "c": [2, 8, None]})

    # Row 0: [1, None, 2] -> max is 2 at index 2
    # Row 1: [None, 2, 8] -> max is 8 at index 2
    # Row 2: [3, None, None] -> max is 3 at index 0
    result = df.select(
        arg_max_horizontal(pl.col("a", "b", "c"), return_colname=colname).alias(
            "arg_max"
        )
    )

    if colname:
        expected = ["c", "c", "a"]
    else:
        expected = [2, 2, 0]
    assert result["arg_max"].to_list() == expected


@pytest.mark.parametrize("colname", [True, False])
def test_arg_max_all_nulls(colname: bool):
    """Test arg_max when all values in a row are null."""
    df = pl.DataFrame(
        {"a": [1, None, None], "b": [2, None, None], "c": [3, None, None]}
    )

    # Row 0: [1, 2, 3] -> max is 3 at index 2
    # Row 1: [None, None, None] -> all null, return None
    # Row 2: [None, None, None] -> all null, return None
    result = df.select(
        arg_max=arg_max_horizontal(pl.col("a", "b", "c"), return_colname=colname),
    )

    if colname:
        expected = ["c", None, None]
    else:
        expected = [2, None, None]
    assert result["arg_max"].to_list() == expected


@pytest.mark.parametrize("colname", [True, False])
def test_arg_max_ties(colname: bool):
    """Test arg_max with tied values (should return first occurrence)."""
    df = pl.DataFrame({"a": [5, 1, 7], "b": [5, 3, 7], "c": [3, 3, 2]})

    # Row 0: [5, 5, 3] -> max is 5, first at index 0
    # Row 1: [1, 3, 3] -> max is 3, first at index 1
    # Row 2: [7, 7, 2] -> max is 7, first at index 0
    result = df.select(
        arg_max_horizontal(pl.col("a", "b", "c"), return_colname=colname).alias(
            "arg_max"
        )
    )

    if colname:
        expected = ["a", "b", "a"]
    else:
        expected = [0, 1, 0]
    assert result["arg_max"].to_list() == expected


@pytest.mark.parametrize("colname", [True, False])
def test_arg_max_negative_numbers(colname: bool):
    """Test arg_max with negative numbers."""
    df = pl.DataFrame({"a": [-1, -5, -3], "b": [-4, -2, -6], "c": [-2, -8, -1]})

    # Row 0: [-1, -4, -2] -> max is -1 at index 0
    # Row 1: [-5, -2, -8] -> max is -2 at index 1
    # Row 2: [-3, -6, -1] -> max is -1 at index 2
    result = df.select(
        arg_max_horizontal(pl.col("a", "b", "c"), return_colname=colname).alias(
            "arg_max"
        )
    )

    if colname:
        expected = ["a", "b", "c"]
    else:
        expected = [0, 1, 2]
    assert result["arg_max"].to_list() == expected


@pytest.mark.parametrize("colname", [True, False])
def test_arg_max_float_values(colname: bool):
    """Test arg_max with float values."""
    df = pl.DataFrame(
        {"a": [1.5, 2.7, 3.1], "b": [2.1, 1.9, 2.8], "c": [1.8, 3.2, 2.5]}
    )

    # Row 0: [1.5, 2.1, 1.8] -> max is 2.1 at index 1
    # Row 1: [2.7, 1.9, 3.2] -> max is 3.2 at index 2
    # Row 2: [3.1, 2.8, 2.5] -> max is 3.1 at index 0
    result = df.select(
        arg_max_horizontal(pl.col("a", "b", "c"), return_colname=colname).alias(
            "arg_max"
        )
    )

    if colname:
        expected = ["b", "c", "a"]
    else:
        expected = [1, 2, 0]

    assert result["arg_max"].to_list() == expected


@pytest.mark.parametrize("colname", [True, False])
def test_arg_max_single_column(colname: bool):
    """Test arg_max with only one column."""
    df = pl.DataFrame({"a": [1, 2, 3]})

    result = df.select(
        arg_max_horizontal(pl.col("a"), return_colname=colname).alias("arg_max")
    )

    if colname:
        expected = ["a", "a", "a"]
    else:
        expected = [0, 0, 0]  # Always index 0 since there's only one column
    assert result["arg_max"].to_list() == expected


@pytest.mark.parametrize("colname", [True, False])
def test_arg_max_identical_values(colname: bool):
    """Test arg_max when all values in a row are identical."""
    df = pl.DataFrame({"a": [5, 5, 5], "b": [5, 5, 5], "c": [5, 5, 5]})

    result = df.select(
        arg_max_horizontal(pl.col("a", "b", "c"), return_colname=colname).alias(
            "arg_max"
        )
    )

    if colname:
        expected = ["a", "a", "a"]
    else:
        expected = [0, 0, 0]  # Should return first column index for ties
    assert result["arg_max"].to_list() == expected


@pytest.mark.parametrize("colname", [True, False])
def test_arg_max_mixed_types(colname: bool):
    """Test arg_max with different numeric types."""
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],  # i64
            "b": [1.5, 1.5, 2.5],  # f64
            "c": [2, 3, 1],  # i64
        }
    )

    # Row 0: [1, 1.5, 2] -> max is 2 at index 2
    # Row 1: [2, 1.5, 3] -> max is 3 at index 2
    # Row 2: [3, 2.5, 1] -> max is 3 at index 0
    with pytest.raises(
        pl.exceptions.ComputeError, match="All input Series must have the same"
    ):
        df.select(
            arg_max_horizontal(pl.col("a", "b", "c"), return_colname=colname).alias(
                "arg_max"
            )
        )

    result = df.select(
        arg_max_horizontal(
            pl.col("a", "b", "c").cast(float), return_colname=colname
        ).alias("arg_max")
    )

    if colname:
        expected = ["c", "c", "a"]
    else:
        expected = [2, 2, 0]
    assert result["arg_max"].to_list() == expected


def test_arg_max_bench(benchmark, df_ints):
    benchmark.group = "arg_star"
    benchmark(lambda: df_ints.select(arg_max_horizontal(pl.all())))


def test_arg_max_bench_colname(benchmark, df_ints):
    benchmark.group = "arg_star"
    benchmark(lambda: df_ints.select(arg_max_horizontal(pl.all(), return_colname=True)))


def test_arg_max_bench_old(benchmark, df_ints):
    benchmark.group = "arg_star"
    benchmark(lambda: df_ints.select(pl.concat_arr(pl.all()).arr.arg_max()))


if __name__ == "__main__":
    pytest.main([__file__])
