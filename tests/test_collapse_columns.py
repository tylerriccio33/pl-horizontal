import polars as pl
from pl_horizontal import collapse_columns
import pytest
import itertools
import polars.selectors as cs
from mimesis import Fieldset
from mimesis.keys import maybe


@pytest.mark.parametrize(
    ("is_null_sentinel", "selector"),
    itertools.product([True, False], [pl.all(), cs.string()]),
)
def test_simple1(is_null_sentinel: bool, selector: pl.Expr):
    """Test simple case that sould produce the same result regardless of is_null_sentinel."""
    df = pl.DataFrame(
        {
            "col1": ["this", "is", "not", "pig", "latin"],
            "col2": [None, "hi", None, "hello", "Amanda"],
        }
    )
    result = df.select(
        res=collapse_columns(pl.all(), is_null_sentinel=is_null_sentinel)
    ).select(pl.all().list.sort())

    expected_df = pl.DataFrame(
        {
            "res": [
                ["this"],
                ["is", "hi"],
                ["not"],
                ["pig", "hello"],
                ["latin", "Amanda"],
            ]
        }
    ).select(pl.all().list.sort())

    assert result.equals(expected_df)


@pytest.mark.parametrize("is_null_sentinel", [True, False])
def test_all_nulls(is_null_sentinel: bool):
    """All values are None — expect empty lists per row regardless of sentinel setting."""
    df = pl.DataFrame(
        {
            "a": [None, None],
            "b": [None, None],
        }
    )
    result = df.select(
        res=collapse_columns(pl.all().cast(str), is_null_sentinel=is_null_sentinel)
    )
    expected = pl.DataFrame({"res": [[] for _ in range(2)]})
    assert result.equals(expected)


@pytest.mark.parametrize("is_null_sentinel", [True, False])
def test_single_column(is_null_sentinel: bool):
    """Single column input — output should just be that column wrapped as list."""
    df = pl.DataFrame({"a": ["x", None, "y"]})
    result = df.select(
        res=collapse_columns(pl.all(), is_null_sentinel=is_null_sentinel)
    )
    expected = pl.DataFrame({"res": [["x"], [], ["y"]]})
    assert result.equals(expected)


@pytest.mark.parametrize("is_null_sentinel", [True, False])
def test_three_columns(is_null_sentinel: bool):
    """Multiple columns collapse in left-to-right order, dropping nulls unless sentinel changes behavior."""
    df = pl.DataFrame(
        {
            "c1": ["a", None, "c"],
            "c2": [None, "b", None],
            "c3": ["x", "y", None],
        }
    )
    result = df.select(
        res=collapse_columns(pl.all(), is_null_sentinel=is_null_sentinel)
    )
    if is_null_sentinel:
        # nulls are not preserved
        expected = pl.DataFrame({"res": [["a"], [], ["c"]]})
    else:
        expected = pl.DataFrame({"res": [["a", "x"], ["b", "y"], ["c"]]})
    assert result.equals(expected)


@pytest.mark.parametrize("is_null_sentinel", [True, False])
def test_empty_strings_and_nulls(is_null_sentinel: bool):
    """Empty strings should be preserved as values (not nulls)."""
    df = pl.DataFrame(
        {
            "c1": ["", None],
            "c2": [None, ""],
        }
    )
    result = df.select(
        res=collapse_columns(pl.all(), is_null_sentinel=is_null_sentinel)
    )
    if is_null_sentinel:
        expected = pl.DataFrame({"res": [[""], []]})
    else:
        expected = pl.DataFrame({"res": [[""], [""]]})
    assert result.equals(expected)


@pytest.mark.parametrize("is_null_sentinel", [True, False])
def test_mixed_types(is_null_sentinel: bool):
    """Mix strings, ints, and floats — should all be collapsed into the list."""
    df = pl.DataFrame(
        {
            "s": ["hi", None, "bye"],
            "i": [1, None, 3],
            "f": [1.5, 2.5, None],
        }
    )

    with pytest.raises(pl.exceptions.ComputeError, match="is not a string column"):
        df.select(res=collapse_columns(pl.all(), is_null_sentinel=is_null_sentinel))


## Benchmarks:
@pytest.fixture
def df() -> pl.DataFrame:
    width_scalar: int = 1_000_000  # x times more rows than columns
    length: int = 1_000_000
    width: int = int(length / width_scalar)

    # Create some random 5 character strings mixed in with None
    fs = Fieldset()
    df = (
        pl.DataFrame(
            {
                f"col{i}": fs(
                    "person.full_name", i=1_000, key=maybe(value=None, probability=0.02)
                )
                for i in range(width)
            }
        )
        # Concat all rows and put the nulls last
        .select(arr=pl.concat_arr(pl.all()).arr.sort(nulls_last=True))
        # Add a row index for the reshaping back to horizontal
        .with_row_index()
        .explode(pl.col("arr"))
        .with_columns(neighbor_n=pl.col("index").cum_count().over("index"))
        # Transpose it back to horizontal along the row index
        .pivot(on="neighbor_n", index="index", values="arr")
        .sample(n=length, with_replacement=True)
        .drop("index")
    )
    assert df.shape == (length, width)  # INTERNAL
    return df


def test_collapse_columns_bench(df, benchmark) -> None:
    benchmark.group = "collapse_columns"
    benchmark(lambda: df.select(collapse_columns(pl.all(), is_null_sentinel=True)))


def test_collapse_columns_old_bench(df, benchmark) -> None:
    benchmark.group = "collapse_columns"
    benchmark(lambda: df.select(pl.concat_list(pl.all()).list.drop_nulls()))


if __name__ == "__main__":
    pytest.main([__file__])
