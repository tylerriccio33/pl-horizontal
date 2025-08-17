from __future__ import annotations
import polars as pl

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def build_arg_true_horizontal_first_known_col(cols: Iterable[str]) -> pl.Expr:
    """Build when-then expression; returns uint32."""
    col_iter = iter(cols)
    exprs = pl.when(next(col_iter)).then(0)
    for i, col in enumerate(col_iter, 1):
        exprs = exprs.when(col).then(i)
    return exprs
