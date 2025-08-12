

build:
	@uv run maturin develop --release

test:
	@uv run pytest \
		--cov pl_col_collapse \
		--cov-report term-missing \
		--randomly-seed $(shell date +%s)

bench:
	@uv run tests/bench.py

lint:
	@uvx ruff format
	@uvx ruff check --fix
	@uvx ty check