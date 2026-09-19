# Camanchaca-Predict — Visibilidad y seguridad vial bajo camanchaca (G5)

> Estimación de la \*\*visibilidad en metros\*\* en presencia de camanchaca para el
> \*\*Camino La Pólvora (Valparaíso)\*\* mediante visión por computador:
> \*\*ViT-B/16 (modelo protagonista)\*\* comparado contra \*\*ResNet-50 (baseline)\*\*.

**Curso:** Proyecto Aplicado 2026-2 (PUCV) · **Grupo:** 5


!\[Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python\&logoColor=white)
!\[PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch\&logoColor=white)
!\[Colab](https://img.shields.io/badge/Ejecutar%20en-Google%20Colab-F9AB00?logo=googlecolab\&logoColor=white)
!\[Licencia](https://img.shields.io/badge/Licencia-MIT-yellow)

\---

## 1\. Problema y motivación

La **camanchaca** es una niebla costera advectiva que afecta de forma recurrente a Valparaíso y su litoral, especialmente entre mayo y octubre. Se forma cuando el aire húmedo del océano se desplaza sobre aguas más frías, condensa en una capa de estratos bajos y penetra tierra adentro por quebradas y pasos de baja altura. A diferencia de una niebla uniforme, la camanchaca es **heterogénea y dinámica**: su densidad varía fuertemente en cientos de metros y en minutos, lo que la hace especialmente peligrosa para la conducción.

El **Camino La Pólvora** es la principal vía de carga del Gran Valparaíso: conecta el puerto con la Ruta 60-CH a través del túnel Las Chinchillas, con tramos de alta velocidad (80–110 km/h) y sectores expuestos directamente a la camanchaca. En condiciones de niebla, la **visibilidad disponible** puede caer por debajo de la **distancia de detención** del vehículo (≈ 168 m a 100 km/h en piso mojado), con el consiguiente riesgo de colisiones por alcance y choques múltiples. Hoy la decisión de restringir la circulación es manual y reactiva: no existe un sistema que **mida la visibilidad en metros** sobre la calzada en tiempo real.

Este proyecto construye el primer paso de ese "medidor de seguridad": un modelo de visión por computador que, a partir de una imagen de cámara routaria, estima la **visibilidad en metros** y la traduce a una **banda de seguridad vial** accionable (peligro / precaución / aceptable).

## 2\. Objetivo y formulación

**Objetivo general.** Entrenar y comparar dos arquitecturas de visión (ResNet-50 y ViT-B/16) para estimar la visibilidad en metros a partir de una sola imagen con niebla, evaluadas con métricas de regresión y de clasificación por bandas de seguridad.

**Formulación dual (multi-tarea):**

|Tarea|Salida|Métricas|
|-|-|-|
|Regresión|`log10(V)` → visibilidad estimada en metros|MAE (m), RMSE (m), MAPE, R²|
|Clasificación|Banda de seguridad (4 clases)|F1 macro, matriz de confusión, **tasa de falso-seguro**|

Bandas de seguridad (justificadas con la distancia de detención, ver [criterio de etiquetado](docs/labeling_criteria.md)):

|Banda|Rango|Interpretación operativa|
|-|-|-|
|`critico`|V < 50 m|Detención insegura incluso a baja velocidad|
|`alto\_riesgo`|50–100 m|Seguro solo a ≤ 60 km/h|
|`precaucion`|100–200 m|Precaución hasta \~100 km/h|
|`aceptable`|≥ 200 m|Margen para velocidad de diseño|

La **tasa de falso-seguro** (casos con V real < 100 m predichos como seguros) se monitorea por separado: es el error más peligroso del sistema (decir "seguro" cuando no lo es).

## 3\. Datos y etiquetado

|Dataset|Tipo|Uso en el proyecto|
|-|-|-|
|**RESIDE-OTS** (outdoor)|Sintético, β conocido por imagen|Train / val — fuente de las etiquetas en metros|
|**SOTS-OUT**|Sintético (test oficial RESIDE)|Test secundario (opcional)|
|**O-HAZE** (NTIRE 2018)|Niebla real, 45 escenas|Test externo cualitativo de generalización|
|Set cualitativo camanchaca|Fotos reales (Chile)|Solo inspección visual (roadmap)|

**Criterio de etiquetado (resumen):** ningún dataset de niebla entrega visibilidad en metros. RESIDE sintetiza niebla con el modelo atmosférico de dispersión `I = J·t + A(1−t)`, `t = exp(−β·d)`, y registra el **coeficiente de extinción β** en el nombre de archivo. Aplicando la **ley de Koschmieder** obtenemos la etiqueta física:

```
V = −ln(0.02) / β ≈ 3.912 / β      \[metros]
```

Detalles completos, validación y limitaciones en [docs/labeling\_criteria.md](docs/labeling_criteria.md).

**Split anti-fuga:** RESIDE genera varias versiones con niebla de cada escena limpia; el split 70/15/15 se hace **agrupado por escena** (una escena nunca puede aparecer en train y test), con tests automáticos que lo verifican.

## 4\. Modelos

||ResNet-50 (baseline)|ViT-B/16 (protagonista)|
|-|-|-|
|Rol|Baseline exigido por la rúbrica; rápido y robusto|Captura dependencias globales (niebla no homogénea) vía auto-atención de parches|
|Salida|Regresión `log10(V)`|**Dual:** regresión `log10(V)` + 4 bandas|
|Preentrenamiento|ImageNet-1k|ImageNet-21k → ImageNet-1k|
|Parámetros|≈ 25 M|≈ 87 M|
|Estrategia|Fine-tuning completo, LR 1e-4|Fase A: solo cabezas (tronco congelado) → Fase B: fine-tuning con LR diferenciales|

## 5\. Métricas de evaluación

* Regresión: **MAE (m)**, RMSE (m), MAPE (%), R².
* Clasificación: **F1 macro** sobre 4 bandas + matriz de confusión.
* Seguridad: **tasa de falso-seguro** (V real < 100 m predicha ≥ 100 m).
* Despliegue: latencia (ms/imagen en GPU T4 de Colab).

## 6\. Estructura del repositorio

```
camanchaca-predict/
├── notebooks/          # Colabs numerados, se ejecutan en orden 00 → 06
├── src/camanchaca/     # Paquete: labeling, data, models, training, evaluation
├── scripts/            # CLI: make\_splits, verify\_env, download\_ohaze
├── configs/            # Hiperparámetros por modelo (YAML)
├── data/               # raw (ignorado), processed (ignorado), splits (versionado)
├── models/             # checkpoints (ignorado)
├── reports/            # figures/ y tables/ generadas (versionadas)
├── docs/               # etiquetado, riesgos, data card, guion del avance
├── presentations/      # decks por fecha
├── tests/              # pytest: física del etiquetado + anti-fuga de splits
├── requirements.txt
├── Makefile
└── pyproject.toml
```

## 7\. Cómo ejecutar (Google Colab, GPU T4 gratis)

1. Abrir `notebooks/00\_setup\_colab.ipynb` en Colab (`Runtime → Change runtime type → T4 GPU`).
2. Tener un `kaggle.json` (kaggle.com → Account → Create New API Token) en la raíz de Google Drive.
3. Ejecutar los notebooks en orden:

|#|Notebook|Qué hace|Tiempo aprox.|
|-|-|-|-|
|00|setup\_colab|Monta Drive, instala deps, clona el repo|3 min|
|01|download\_reside|Descarga RESIDE-OTS (Kaggle) + manifiesto (o modo demo)|10–30 min|
|02|labeling\_koschmieder|Etiquetas V=3.912/β, bandas, validación de consistencia|10 min|
|03|eda\_splits|EDA + split 70/15/15 agrupado por escena|10 min|
|04|baseline\_resnet50|Entrena ResNet-50 + métricas + curvas|20–40 min|
|05|vit\_finetune|Entrena ViT-B/16 dual (2 fases) + métricas|40–90 min|
|06|eval\_compare|Comparación final, figuras y tablas para la presentación|10 min|

Los notebooks son autónomos (Colab-first); `src/` + `scripts/` permiten además ejecución local/CLI (`make verify`, `make splits`, `make test`).

## 8\. Resultados

*(se completa al terminar los entrenamientos — notebooks 04–06 escriben aquí `reports/tables/` y `reports/figures/`)*

|Modelo|MAE (m)|RMSE (m)|MAPE|R²|F1 macro|Falso-seguro|Latencia (ms)|
|-|-|-|-|-|-|-|-|
|ResNet-50|—|—|—|—|—|—|—|
|ViT-B/16|—|—|—|—|—|—|—|

## 9\. Riesgos y mitigación (resumen)

|ID|Riesgo|Mitigación|
|-|-|-|
|R1|Datos sintéticos ≠ camanchaca real|O-HAZE (real) como test externo + análisis de brecha + roadmap de datos chilenos|
|R2|β del filename sin documentación oficial de unidades|Validación física + consistencia con proxy de canal oscuro (Spearman)|
|R3|Desbalance de bandas (pocas imágenes < 50 m)|Class weights + re-síntesis `t^k` + reporte por banda|
|R4|Colab free: desconexiones / límite de GPU|Checkpoints por época en Drive + entrenamientos cortos|
|R5|OOM de ViT-B/16 en T4|AMP fp16 + batch/accum ajustables + congelado en fase A|

Tabla completa con probabilidad, impacto, señal de alarma y responsable en [docs/risks\_and\_mitigation.md](docs/risks_and_mitigation.md).

## 10\. Equipo y roles

|Integrante|Rol|Responsabilidad principal|
|-|-|-|
|*Tomás Moraga*|A — Datos|Notebooks 01–03, etiquetado, data card, Kaggle|
|*Nelson Fuentes*|B — Entrenamiento|Notebook 04, infraestructura Colab, checkpoints|
|*Fernando Guerrero*|C — ViT y evaluación|Notebooks 05–06, comparación, presentación|

Revisión cruzada obligatoria de PRs entre roles.



## 11\. Licencia

[MIT](LICENSE) — Grupo 5, Proyecto Aplicado 2026-2.

