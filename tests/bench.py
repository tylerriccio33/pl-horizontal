import resource
import polars as pl
from pl_col_collapse import collapse_columns
import timeit
from mimesis import Fieldset
from mimesis.keys import maybe


def _new(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(collapse_columns(pl.all(), stop_on_first_null=True))


def _old(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(pl.concat_list(pl.all()).list.drop_nulls())


def _yield_df() -> pl.DataFrame:
    width: int = 100
    length: int = 1_000_000

    # Create some random 5 character strings mixed in with None
    fs = Fieldset()
    df = pl.DataFrame(
        {
            # Gradually increase null probability from 0% to 20% across columns
            f"col{i}": fs(
                "person.full_name",
                i=1_000,
                key=maybe(
                    None, probability=(0.2 * i / (width - 1)) if width > 1 else 0.0
                ),
            )
            for i in range(width)
        }
    ).sample(n=length, with_replacement=True)
    assert df.shape == (length, width)  # INTERNAL
    return df


def bench_big_strings(df: pl.DataFrame, method=_new) -> tuple[float, float]:
    # Run the benchmark and record max memory during execution
    start = timeit.default_timer()
    usage_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    method(df)
    usage_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    end = timeit.default_timer()

    # On macOS, ru_maxrss is in bytes
    max_memory = max(usage_before, usage_after) / (1024**3)
    elapsed_time = end - start
    return elapsed_time, max_memory


if __name__ == "__main__":
    # Old Method (1m rows):
    # Time    -> 14.41s
    # Memory  -> 6.29 GB

    # Best Time (1m rows):
    # Time    -> 0.73s
    # Memory  -> 2.20 GB

    df = _yield_df()

    times = []
    mems = []
    for _ in range(10):
        time, mem = bench_big_strings(df.sample(n=len(df), with_replacement=True), _new)
        times.append(time)
        mems.append(mem)
    print(f"Average time: {sum(times) / len(times):.2f} seconds")
    print(f"Average memory usage: {sum(mems) / len(mems):.2f} GB")
