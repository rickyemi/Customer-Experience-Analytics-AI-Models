.PHONY: setup data ml analytics all test notebooks clean

setup:
	pip install -r requirements.txt

data:
	PYTHONPATH=src python -m cx_suite.data.generate_ml_data
	PYTHONPATH=src python -m cx_suite.data.generate_analytics_data

ml:
	python scripts/run_pipeline.py --only ml

analytics:
	python scripts/run_pipeline.py --only analytics

all:
	python scripts/run_pipeline.py

test:
	pytest -q

notebooks:
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

clean:
	rm -f models/*.joblib reports/metrics/*.json
