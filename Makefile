

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
		--doctest-plus --doctest-glob '*.md'

bench:
	@uv run tests/bench.py

lint:
	@uvx ruff format
	@uvx ruff check --fix
	@uvx ty check

tox:
	@uvx tox -e py312,py313

gen-ci: ## Generate the CI File:
	@uvx maturin generate-ci github > .github/workflows/python-publish.yml