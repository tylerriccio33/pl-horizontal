import resource
import polars as pl
from pl_horizontal import collapse_columns
import timeit
from mimesis import Fieldset
from mimesis.keys import maybe
from collections.abc import Callable


def _new(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.select(collapse_columns(pl.all(), is_null_sentinel=True))


def _old(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.select(pl.concat_list(pl.all()).list.drop_nulls())


def _yield_df() -> pl.DataFrame:
    width_scalar: int = 1_000_000  # x times more rows than columns
    length: int = 10_000_000
    width: int = int(length / width_scalar)

    # Create some random 5 character strings mixed in with None
    fs = Fieldset()
    df = (
        pl.DataFrame(
            {
                f"col{i}": fs(
                    "person.full_name", i=1_000, key=maybe(value=None, probability=0.02)
                )
                for i in range(width)
            }
        )
        # Concat all rows and put the nulls last
        .select(arr=pl.concat_arr(pl.all()).arr.sort(nulls_last=True))
        # Add a row index for the reshaping back to horizontal
        .with_row_index()
        .explode(pl.col("arr"))
        .with_columns(neighbor_n=pl.col("index").cum_count().over("index"))
        # Transpose it back to horizontal along the row index
        .pivot(on="neighbor_n", index="index", values="arr")
        .sample(n=length, with_replacement=True)
        .drop("index")
    )
    assert df.shape == (length, width)  # INTERNAL
    return df


def bench_big_strings(df: pl.DataFrame, method=_new) -> tuple[float, float]:
    # Run the benchmark and record max memory during execution
    usage_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    start = timeit.default_timer()
    method(pl.LazyFrame(df)).collect(engine="streaming")
    end = timeit.default_timer()

    usage_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # On macOS, ru_maxrss is in bytes
    max_memory = max(usage_before, usage_after) / (1024**3)
    elapsed_time = end - start
    return elapsed_time, max_memory


if __name__ == "__main__":
    # Old Method (10m rows):
    # Time    -> 12.29s
    # Memory  -> 5.54 GB

    # Best Time (10m rows):
    # Time    -> .52s
    # Memory  -> 5.37 GB

    df = _yield_df()

    FN: Callable = _old
    # FN: Callable = _new

    print(f"Benchmarking {FN.__name__} @ {df.shape[0]:,} rows, {df.shape[1]} columns")

    times: list[float] = []
    mems: list[float] = []
    for _ in range(20):
        time, mem = bench_big_strings(df.sample(n=len(df), with_replacement=True), FN)
        times.append(time)
        mems.append(mem)
    print(f"Average time: {sum(times) / len(times):.2f} seconds")
    print(f"Average memory usage: {sum(mems) / len(mems):.2f} GB")
