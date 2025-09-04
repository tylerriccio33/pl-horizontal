import numpy as np
import polars as pl
import pytest


## -- Benchmarks
@pytest.fixture
def df_ints() -> pl.DataFrame:
    n_rows = 1_000_000
    n_cols = 20

    rng = np.random.default_rng(seed=42)

    data = {}
    for i in range(n_cols):
        ints = rng.integers(low=0, high=1_000, size=n_rows)

        # Randomly set ~10% to None
        mask = rng.random(n_rows) < 0.1
        col = np.where(mask, None, ints)

        data[f"col{i}"] = col.tolist()

    return pl.DataFrame(data)
