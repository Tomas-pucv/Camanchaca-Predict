.PHONY: setup verify test splits help

help:
	@echo "make setup   - instala dependencias (pip)"
	@echo "make verify  - verifica entorno (GPU y librerías)"
	@echo "make test    - ejecuta pruebas unitarias"
	@echo "make splits  - genera splits train/val/test desde labels.csv"

setup:
	python -m pip install -r requirements.txt

verify:
	python scripts/verify_env.py

test:
	PYTHONPATH=src python -m pytest tests -q

splits:
	PYTHONPATH=src python scripts/make_splits.py --labels data/processed/labels.csv --out data/splits
