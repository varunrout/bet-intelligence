.PHONY: setup init-db ingest fetch-xg features test clean

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

init-db:
	python scripts/init_db.py

ingest:
	python scripts/run_pipeline.py

fetch-xg:
	python scripts/fetch_xg.py --all

features:
	python scripts/build_features.py --rebuild

test:
	pytest -q

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
