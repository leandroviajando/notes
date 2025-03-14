SHELL       := /usr/bin/env bash
MAKEFLAGS   += --silent

MD_TARGETS  := biology cognition computation knowledge language learning search uncertainty vision
TEX_TARGETS := algebra calculus complex diffeqs discrete linalg probability statistics

all: clean compile clean

.PHONY: help
help: ## Show the available commands
	@echo "Available commands:"
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install dependencies
	brew install --cask mactex-no-gui
	brew install pandoc pre-commit
	pre-commit install
	pre-commit autoupdate

.PHONY: compile
compile: ## Compile the documents
	for doc in $(TARGETS); do \
		pandoc -V geometry:margin=1in -V fontfamily=helvet --pdf-engine=xelatex -o ai/$$doc.pdf ai/$$doc.md; \
	done
	for doc in $(TARGETS); do \
		TEXINPUTS=$$doc:$$TEXINPUTS pdflatex -output-directory=$$doc $$doc/$$doc.tex; \
		biber $$doc/$$doc; \
		TEXINPUTS=$$doc:$$TEXINPUTS pdflatex -output-directory=$$doc $$doc/$$doc.tex; \
	done

.PHONY: clean
clean: ## Clean the repository
	find ./ai/ -type f \( -name '*.pdf' \) -delete
	find . -type f \( -name '*.aux' \
		-o -name '*.lof' \
		-o -name '*.log' \
		-o -name '*.lot' \
		-o -name '*.fls' \
		-o -name '*.out' \
		-o -name '*.toc' \
		-o -name '*.fmt' \
		-o -name '*.fot' \
		-o -name '*.cb' \
		-o -name '*.cb2' \
		-o -name '.*lb' \
		-o -name '*.run.xml' \
		-o -name '*.bcf' \
		-o -name '*.bbl' \
		-o -name '*.bbl-SAVE-ERROR' \
		-o -name '*.blg' \
		-o -name '*.fdb_latexmk' \
		-o -name '*.synctex.gz' \) -delete
