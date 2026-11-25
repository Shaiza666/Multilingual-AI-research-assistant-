PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
export PYTHONPATH := src:$(PYTHONPATH)
export TOKENIZERS_PARALLELISM := false

.PHONY: install download-models download-papers ingest evaluate test lint app sample hardware

install:
	python3 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

download-models:
	$(PYTHON) scripts/download_models.py

download-papers:
	$(PYTHON) scripts/download_papers.py

sample:
	$(PYTHON) scripts/build_sample_pdf.py

ingest:
	$(PYTHON) -m mira.cli ingest data/papers/pdf --index data/indexes/benchmark

evaluate:
	$(PYTHON) -m mira.cli evaluate --index data/indexes/benchmark --gold data/qa/gold.json --pilot data/qa/pilot.json --out results

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests

app:
	$(PYTHON) -m streamlit run src/mira/app.py --server.port 8741 --server.address 127.0.0.1

hardware:
	$(PYTHON) -m mira.cli hardware
