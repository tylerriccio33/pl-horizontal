import resource
import polars as pl
from pl_col_collapse import collapse_columns
import timeit
from mimesis import Fieldset
from mimesis.keys import maybe


def _method(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(collapse_columns(pl.all(), stop_on_first_null=True))


def _old(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(pl.concat_list(pl.all()).list.drop_nulls())


def bench_big_strings(method=_method) -> None:
    width: int = 100
    length: int = 2_000_000

    # Create some random 5 character strings mixed in with None
    print(f"Creating sample DataFrame with {width} columns and {length:,} rows...")
    fs = Fieldset()
    df = pl.DataFrame(
        {
            f"col{i}": fs("person.full_name", i=1_000, key=maybe(None, probability=0.6))
            for i in range(width)
        }
    ).sample(n=length, with_replacement=True)
    assert df.shape == (length, width)  # INTERNAL

    # Run the benchmark
    print("Running benchmark...")
    start = timeit.default_timer()
    method(df)
    end = timeit.default_timer()

    # Print stats
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(f"Time: {end - start:.2f} seconds")
    # On macOS, ru_maxrss is in bytes (not kilobytes as on Linux)
    print(f"Max Memory: {usage.ru_maxrss / (1024 ** 3):.2f} GB")


if __name__ == "__main__":
    bench_big_strings(_method)
    # bench_big_strings(_old)
