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
        # kwargs={"is_null_sentinel": is_null_sentinel},
    )
