from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from pl_horizontal._internal import __version__ as __version__

if TYPE_CHECKING:
    from pl_horizontal.typing import IntoExprColumn

LIB = Path(__file__).parent


def collapse_columns(expr: IntoExprColumn, *, is_null_sentinel: bool) -> pl.Expr:
    """Collapse columns horizontally into a list column, while excluding Nulls.

    Args:
        expr (IntoExprColumn): Columns across the dataframe, evaluated in order.
        is_null_sentinel (bool): Whether the columns are arranged as a null-sentinel
            problem, where nulls are pushed to the back. This enables a fast path
            where the first null triggers an early stop.

    Returns:
        pl.Expr: Expression evaluating to a list column of collapsed values.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "a": ["x", None, "z"],
        ...     "b": ["y", "y2", None],
        ...     "c": [None, "c2", "c3"]
        ... })
        >>> res = df.select(f = collapse_columns(pl.col('a','b','c'), is_null_sentinel=False))
        >>> assert res["f"].to_list() == [['x','y'], ['y2','c2'], ['z','c3']]

        >>> df = pl.DataFrame({
        ...     "a": ["x", None, "z"],
        ...     "b": ["y", "y2", None],
        ...     "c": [None, "c2", "c3"]
        ... })
        >>> # Fast path: nulls are guaranteed to be after all non-nulls in each row
        >>> res = df.select(f = collapse_columns(pl.col("a", "b", "c"), is_null_sentinel=True))
        >>> assert res["f"].to_list() == [['x', 'y'], [], ['z']]

    """
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="collapse_columns",
        is_elementwise=True,
        input_wildcard_expansion=True,
        kwargs={"is_null_sentinel": is_null_sentinel},
    )


def arg_true_horizontal(expr: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="arg_true_horizontal",
        is_elementwise=True,
        input_wildcard_expansion=True,
    )


def arg_first_true_horizontal(expr: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="arg_first_true_horizontal",
        is_elementwise=True,
        input_wildcard_expansion=True,
    )
