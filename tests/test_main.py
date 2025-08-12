import polars as pl
from pl_col_collapse import collapse_columns
import pytest
import itertools
import polars.selectors as cs


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


if __name__ == "__main__":
    pytest.main([__file__])
