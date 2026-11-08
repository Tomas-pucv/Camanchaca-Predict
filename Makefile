# Atajos para la ejecución local/CLI (en Colab usar los notebooks 00-06).
EXP ?= configs/experiment/main.yaml
PY  ?= python

.PHONY: help setup verify test test-fast smoke splits sweep train evaluate final interpret cars report demo

help:
	@echo "make setup      - instala el paquete en modo editable con extras (dev, demo, data)"
	@echo "make verify     - verifica el entorno (librerías y GPU)"
	@echo "make test       - todas las pruebas (incluye entrenamiento de humo en CPU)"
	@echo "make test-fast  - pruebas rápidas (sin las marcadas 'slow')"
	@echo "make smoke      - pipeline completo de punta a punta sobre datos sintéticos"
	@echo "make splits     - genera data/splits/v1 desde data/manifests/*.csv"
	@echo "make sweep      - barrido de LR (misma grilla para ambos modelos)"
	@echo "make train      - entrena ResNet-50 y DeiT-S x 3 semillas (idempotente)"
	@echo "make evaluate   - predicciones + baselines + métricas con IC en VALIDACIÓN"
	@echo "make final      - evaluación FINAL (test + externos), una sola vez"
	@echo "make interpret  - Grad-CAM / attention rollout + contrafactual"
	@echo "make cars       - análisis de autos: recall vs distancia por beta"
	@echo "make report     - reúne tablas en reports/RESULTADOS.md"
	@echo "make demo       - lanza la demo Gradio local"

setup:
	$(PY) -m pip install -e ".[dev,demo,data]"

verify:
	$(PY) scripts/verify_env.py

test:
	$(PY) -m pytest tests

test-fast:
	$(PY) -m pytest tests -m "not slow"

smoke:
	camanchaca smoke

splits:
	camanchaca split

sweep:
	camanchaca sweep --exp $(EXP)

train:
	camanchaca train-all --exp $(EXP) --lr auto

evaluate:
	camanchaca predict --exp $(EXP) --sets val
	camanchaca baselines --exp $(EXP) --sets val
	camanchaca evaluate --exp $(EXP) --sets val

final:
	camanchaca predict --exp $(EXP) --sets val test ext --final
	camanchaca baselines --exp $(EXP) --sets val test ext --final
	camanchaca evaluate --exp $(EXP) --sets val test ext --final
	camanchaca probes --exp $(EXP) --final
	camanchaca latency --exp $(EXP)

interpret:
	camanchaca interpret --exp $(EXP) --final
	camanchaca counterfactual --exp $(EXP) --final

cars:
	camanchaca cars --images data/manifests/cars_images.csv --gt data/manifests/cars_gt.csv

report:
	camanchaca report

demo:
	$(PY) app/app.py --exp main
