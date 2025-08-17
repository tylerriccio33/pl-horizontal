from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

    from polars.datatypes import DataType, DataTypeClass

    type IntoExprColumn = pl.Expr | str | pl.Series
    type PolarsDataType = DataType | DataTypeClass
