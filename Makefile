.PHONY: install test run lint typecheck clean
install:
	python -m pip install -e ".[dev]"

test:
	pytest

run:
	sensor-fdir --config config/default.json --output artifacts/demo

lint:
	ruff check src tests

typecheck:
	mypy src/sensor_fdir

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
