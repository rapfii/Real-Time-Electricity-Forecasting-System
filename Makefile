.PHONY: train stream evaluate test lint clean

train:
	python -m scripts.train

stream:
	python -m scripts.stream

evaluate:
	python -m scripts.evaluate

test:
	python -m pytest tests/ -v

lint:
	python -m ruff check .
	python -m mypy config/ core/ models/ streaming/ api/

clean:
	rm -rf artifacts/*.lgb artifacts/*.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
