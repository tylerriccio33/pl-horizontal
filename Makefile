

build:
	uv run maturin develop --release

test:
	uv run pytest

bench:
	@uv run tests/bench.py