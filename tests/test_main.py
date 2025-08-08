import polars as pl
from pl_col_collapse import collapse_columns
import pytest


def test_simple():
    df = pl.DataFrame(
        {
            "col1": ["this", "is", "not", "pig", "latin"],
            "col2": [None, "hi", None, "hello", "Amanda"],
        }
    )
    result = df.select(res=collapse_columns(pl.all(), stop_on_first_null=True)).select(pl.all().list.sort())

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


if __name__ == "__main__":
    pytest.main([__file__])
