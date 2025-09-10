from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from pl_horizontal._internal import __version__ as __version__
from pl_horizontal._expr_builders import build_arg_true_horizontal_first_known_col

if TYPE_CHECKING:
    from pl_horizontal.typing import IntoExprColumn
    from collections.abc import Iterable

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
    """
    Return a horizontal boolean expression that indicates whether each row has any True value.

    Args:
        expr: A Polars expression or column(s) to evaluate row-wise.

    Returns:
        pl.Expr: Expression returning True if any value in the row is True, False otherwise.

    Example:
        >>> df = pl.DataFrame({"a": [True, False], "b": [False, True]})
        >>> df.select(arg_true_horizontal(pl.all())).to_series().to_list()
        [[0], [1]]
    """
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="arg_true_horizontal",
        is_elementwise=True,
        input_wildcard_expansion=True,
    )


def arg_first_true_horizontal(expr: IntoExprColumn | Iterable[str]) -> pl.Expr:
    """
    Return the index of the first True value per row, or None if no True is found.

    Args:
        expr: A Polars expression or column(s) to evaluate row-wise.

    Returns:
        pl.Expr: Expression returning the index of the first True in each row.

    Example:
        >>> df = pl.DataFrame({"a": [False, False], "b": [True, False], "c": [False, True]})
        >>> df.select(arg_first_true_horizontal(pl.all())).to_series().to_list()
        [1, 2]
    """
    if not isinstance(expr, pl.Expr):
        # -> FP uses when-then (vectorized) which is almost always faster.
        return build_arg_true_horizontal_first_known_col(expr)

    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="arg_first_true_horizontal",
        is_elementwise=True,
        input_wildcard_expansion=True,
    )


def multi_index(expr: IntoExprColumn, lookup: pl.Series) -> pl.Expr:
    """Take integer cols and use them as an index against the lookup.

    This is functionally equivalent to using `gather` but against multiple columns
    in a dataframe context. Function is unsafe(!), and will panic on a bad index;
    done to increase performance during the indexing.

    Args:
        expr (IntoExprColumn): Integer column, ideally uint32.
        lookup (pl.Series): String series to index into.

    Returns:
        pl.Expr: Expression evaluating to the indexed string series.

    Example:
        >>> df = pl.DataFrame({"idx": [2, 0, 2, 1]})
        >>> ser = pl.Series(["alpha", "beta", "gamma"])
        >>> out = df.select(multi_index(pl.col("idx"), ser))
        >>> # idx 2 → "gamma", idx 0 → "alpha", idx 1 → "beta"
        >>> assert out.to_series().to_list() == ["gamma", "alpha", "gamma", "beta"]
    """
    if not lookup.dtype.is_(pl.String):
        raise TypeError(f"`lookup` must be a String series, not `{type(lookup)}`")

    # TODO: Let lookup be any old into expression too

    return register_plugin_function(
        args=[expr, lookup],
        plugin_path=LIB,
        function_name="multi_index",
        is_elementwise=True,  # Full `lookup` must be available for each row
    )
