WORKFLOW_FILE := .github/workflows/python-publish.yml

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
		--doctest-plus \
		--codeblocks

bench:
	@uv run pytest \
		--benchmark-columns=mean,rounds \
		--benchmark-sort=mean \
		--benchmark-group-by=group \
		--benchmark-only
lint:
	@uvx ruff format
	@uvx ruff check --fix
	@uvx ty check

tox:
	@uvx tox -e py312,py313

gen-ci: ## Generate the CI File:
	@uvx maturin generate-ci github \
		-o .github/workflows/python-publish.yml
	@cp $(WORKFLOW_FILE) $(WORKFLOW_FILE).bak
	@sed -e '/- runner: ubuntu-22.04/{N;/\n[[:space:]]*target: aarch64/d;}' \
	    $(WORKFLOW_FILE).bak > $(WORKFLOW_FILE)
	@rm $(WORKFLOW_FILE).bak
	@cp $(WORKFLOW_FILE) $(WORKFLOW_FILE).bak
	@sed -e '/- runner: ubuntu-22.04/{N;/\n[[:space:]]*target: s390x/d;}' \
	    $(WORKFLOW_FILE).bak > $(WORKFLOW_FILE)
	@rm $(WORKFLOW_FILE).bak
	@cp $(WORKFLOW_FILE) $(WORKFLOW_FILE).bak
	@sed -e '/- runner: ubuntu-22.04/{N;/\n[[:space:]]*target: ppc64le/d;}' \
	    $(WORKFLOW_FILE).bak > $(WORKFLOW_FILE)
	@rm $(WORKFLOW_FILE).bak
	@cp $(WORKFLOW_FILE) $(WORKFLOW_FILE).bak
	@sed -e '/^[[:space:]]*branches:/,/^[[:space:]]*tags:/{/^[[:space:]]*branches:/d; /^[[:space:]]*- main/d; /^[[:space:]]*- master/d;}' \
	    $(WORKFLOW_FILE).bak > $(WORKFLOW_FILE)
	@rm $(WORKFLOW_FILE).bak