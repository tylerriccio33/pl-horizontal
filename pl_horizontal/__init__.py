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

    return register_plugin_function(
        args=[expr, lookup],
        plugin_path=LIB,
        function_name="multi_index",
        is_elementwise=False,
        # ! BUG -> When `kwargs` is removed, there's a 7x performance penalty
        kwargs={"lookup": lookup},
    )


def arg_max_horizontal(
    expr: IntoExprColumn, *, return_colname: bool = False
) -> pl.Expr:
    """Return the index of the maximum value per row, or None if all values are null.

    Args:
        expr (IntoExprColumn): Columns across the dataframe, evaluated in order.
        return_colname (bool): Whether to return the column name instead of index.

    Returns:
        pl.Expr: Expression evaluating to the index or column name of the maximum value.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "a": [1, None, 3],
        ...     "b": [2, 2, None],
        ...     "c": [None, 3, 1]
        ... })
        >>> res = df.select(f = arg_max_horizontal(pl.col('a','b','c'), return_colname=False))
        >>> assert res["f"].to_list() == [1, 2, 0]  # indices of max values
        >>> res = df.select(f = arg_max_horizontal(pl.col('a','b','c'), return_colname=True))
        >>> assert res["f"].to_list() == ['b', 'c', 'a']  # names of max value columns
    """
    if return_colname:
        return register_plugin_function(
            args=[expr],
            plugin_path=LIB,
            function_name="arg_max_horizontal_colname",
            is_elementwise=True,
            input_wildcard_expansion=True,
        )
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="arg_max_horizontal",
        is_elementwise=True,
        input_wildcard_expansion=True,
    )


def arg_min_horizontal(
    expr: IntoExprColumn, *, return_colname: bool = False
) -> pl.Expr:
    """Return the index of the minimum value per row, or None if all values are null.

    Args:
        expr (IntoExprColumn): Columns across the dataframe, evaluated in order.
        return_colname (bool, optional): Return the column name instead of the index. Defaults to False.

    Returns:
        pl.Expr: Expression evaluating to the index or column name of the minimum value.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "a": [1, None, 3],
        ...     "b": [2, 2, None],
        ...     "c": [None, 3, 1]
        ... })
        >>> res = df.select(f = arg_min_horizontal(pl.col('a','b','c'), return_colname=False))
        >>> assert res["f"].to_list() == [0, 1, 2]  # indices of min values
        >>> res = df.select(f = arg_min_horizontal(pl.col('a','b','c'), return_colname=True))
        >>> assert res["f"].to_list() == ['a', 'b', 'c']  # names of min value columns
    """
    if return_colname:
        return register_plugin_function(
            args=[expr],
            plugin_path=LIB,
            function_name="arg_min_horizontal_colname",
            is_elementwise=True,
            input_wildcard_expansion=True,
        )
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="arg_min_horizontal",
        is_elementwise=True,
        input_wildcard_expansion=True,
    )


def is_max(expr: IntoExprColumn) -> pl.Expr:
    """Return a boolean mask indicating the maximum value(s) per row.

    Args:
        expr (IntoExprColumn): Columns across the dataframe, evaluated in order.

    Returns:
        pl.Expr: Expression evaluating to a boolean mask of maximum values.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "a": [1, None, 3],
        ...     "b": [2, 2, None],
        ...     "c": [None, 3, 1]
        ... })
        >>> res = df.select(is_max(pl.col('a','b','c')))
        >>> assert res.to_series().to_list() == [False, False, True]  # max values mask
    """
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="is_max",
        is_elementwise=False,  # ? must be `False` for `over` to work properly
    )


def is_min(expr: IntoExprColumn) -> pl.Expr:
    """Return a boolean mask indicating the minimum value(s) per row.

    Args:
        expr (IntoExprColumn): Columns across the dataframe, evaluated in order.

    Returns:
        pl.Expr: Expression evaluating to a boolean mask of minimum values.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "a": [1, None, 3],
        ...     "b": [2, 2, None],
        ...     "c": [None, 3, 1]
        ... })
        >>> res = df.select(is_min(pl.col('a','b','c')))
        >>> assert res.to_series().to_list() == [True, False, False]  # min values mask
    """
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="is_min",
        is_elementwise=False,  # ? must be `False` for `over` to work properly
    )


def arg_first_null_horizontal(expr: IntoExprColumn) -> pl.Expr:
    """
    Return the index of the first Null value per row, or None if no Null is found.

    Args:
        expr: A Polars expression or column(s) to evaluate row-wise.

    Returns:
        pl.Expr: Expression returning the index of the first Null in each row.

    Example:
        >>> df = pl.DataFrame({"a": [1, 2, None], "b": [None, 2, 3], "c": [1, None, 3]})
        >>> df.select(arg_first_null_horizontal(pl.all())).to_series().to_list()
        [1, 2, 0]
    """
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="arg_first_null_horizontal",
        is_elementwise=True,
        input_wildcard_expansion=True,
    )
