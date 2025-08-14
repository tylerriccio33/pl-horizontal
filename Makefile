

build:
	@uv run maturin develop --release

test:
	@uv run pytest \
		--cov pl_horizontal \
		--cov-report term-missing \
		--randomly-seed $(shell date +%s) \
		--benchmark-columns=mean,rounds \
		--benchmark-sort=mean \
		--benchmark-group-by=group \
		--doctest-modules

bench:
	@uv run tests/bench.py

lint:
	@uvx ruff format
	@uvx ruff check --fix
	@uvx ty check

tox:
	@uvx tox -e py312,py313