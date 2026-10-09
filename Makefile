.PHONY: install test lint demo benchmark clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=rotorwatch --cov-report=term-missing

lint:
	ruff check rotorwatch tests

demo:
	python -m rotorwatch.cli demo

benchmark:
	python -m rotorwatch.real_benchmark

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
