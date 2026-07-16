.PHONY: install demo app test lint check

install:
	python -m pip install -e ".[dev]"

demo:
	python -m signalgraph_aml.pipeline --demo

app:
	streamlit run app.py

test:
	pytest

lint:
	ruff check .

check: lint test
