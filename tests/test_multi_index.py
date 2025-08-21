import pytest
import polars as pl

from pl_horizontal import multi_index
import numpy as np
import string


def test_multi_gather_simple() -> None:
    s = pl.Series(values=["hi", "how", "are", "you"])
    df = pl.DataFrame(
        {
            "foo": [0, 1, 2],
            "duchess": [None, 3, 0],
        }
    )

    expected = pl.DataFrame(
        {
            "foo": ["hi", "how", "are"],
            "duchess": [None, "you", "hi"],
        }
    )

    res = df.select(multi_index(pl.all(), lookup=s))

    assert expected.equals(res)


## Benchmaks:
@pytest.fixture
def df() -> tuple[pl.DataFrame, pl.Series]:
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

    rand_strings = [rng.choice(list(string.ascii_letters)) for _ in range(1_000)]

    return pl.DataFrame(data), pl.Series(values=rng.choice(rand_strings, n_rows))


def test_multi_index_bench(df, benchmark) -> None:
    benchmark.group = "multi_index"
    data, ser = df

    new = data.select(multi_index(pl.all(), ser))
    old = _old(data, ser)

    assert new.equals(old.select(data.columns)), "Invalid benchmark"

    data = data.select(pl.all().cast(pl.UInt32))  # try the fastest path

    benchmark(lambda: data.select(multi_index(pl.all(), ser)))


def test_multi_index_bench_old(df, benchmark) -> None:
    benchmark.group = "multi_index"
    data, ser = df

    old = _old(data, ser).select(data.columns)
    new = data.select(multi_index(pl.all(), ser))

    assert old.equals(new), "Invalid benchmark"

    benchmark(lambda: _old(data, ser))


def _old(data: pl.DataFrame, ser: pl.Series) -> pl.DataFrame:
    cols: list[str] = data.columns
    for col in cols:
        indices = data[col]
        new = ser[indices].rename(col)
        data = data.drop(col)
        data.insert_column(0, new)

    return data


if __name__ == "__main__":
    pytest.main([__file__])
