# pl-horizontal

A very small Polars plugin library providing horizontal operations on DataFrames. Library contains niche fast paths and shortcuts for more aggressive optimizations; useful when you know some information about your data that could be passed down.

---

### Motivation

Not all data is long, and not all data can be converted to long. Often times I deal with massively dimensional datasets that cannot be reasonably pivoted. This library is meant to deal with some of that.

I have only several functions that are hybrids of list/array operations with some little fast paths. This library is as much about performance in time as it is in memory.

## Installation

```bash
uv pip install pl-horizontal
```

## Features

- `collapse_columns`: Collapse multiple columns into a list column, optionally using a null-sentinel fast path.
- `arg_true_horizontal`: Check if any column in a row is True.
- `arg_first_true_horizontal`: Get the index of the first True value in a row.

## Usage

### Collapse Columns into List

```python
import polars as pl
from pl_horizontal import collapse_columns

df = pl.DataFrame({
    "a": ["x", None, "z"],
    "b": ["y", "y2", None],
    "c": [None, "c2", "c3"]
})

# Normal collapse
res = df.select(f=collapse_columns(pl.col("a", "b", "c"), is_null_sentinel=False))
print(res)
assert res.to_series().to_list() == [['x', 'y'], ['y2','c2'], ['z','c3']]

# Fast path: nulls are guaranteed to be at the end
res_fast = df.select(f=collapse_columns(pl.col("a", "b", "c"), is_null_sentinel=True))
print(res_fast)
assert res_fast.to_series().to_list() == [['x','y'], [], ['z']]
```

### Horizontal Boolean Checks
```python
from pl_horizontal import arg_true_horizontal, arg_first_true_horizontal

df = pl.DataFrame({
    "a": [True, False],
    "b": [False, True]
})

# Check if any value is True per row
res = df.select(arg_true_horizontal(pl.all()))
print(res)
assert res.to_series().to_list() == [[0], [1]]

# Index of first True value per row
df2 = pl.DataFrame({
    "a": [False, False],
    "b": [True, False],
    "c": [False, True]
})
res2 = df2.select(arg_first_true_horizontal(pl.all()))
print(res2)
assert res2.to_series().to_list() == [1, 2]
```

## Contributing

This is a simple project, would welcome any rust people, since I'm very new to it!

I use UV religiously, so it's commands (sync, add, uvx, ...) should handle everything you need.

Run the suite with `make test`, run the linter with `make lint`, along with some other targets.