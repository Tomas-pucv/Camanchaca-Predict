# Camanchaca-Predict: visibilidad bajo niebla con ResNet-50 vs DeiT-S (G5)

> Estimación de la **visibilidad meteorológica (MOR) en metros** a partir de
> una imagen con niebla, pensada para la camanchaca del **Camino La Pólvora
> (Valparaíso)**. Compara **ResNet-50** y **DeiT-S** bajo un **protocolo
> idéntico**, con incertidumbre, baselines, pruebas de atajo,
> interpretabilidad, un análisis de detectabilidad de autos y una demo.

**Curso:** Aprendizaje Automático II, Proyecto Aplicado 2026-2 (PUCV).
**Grupo:** 5.
**Equipo:** Tomás Ignacio Moraga y Nelson Mauricio Fuentes.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white)
![Colab](https://img.shields.io/badge/Ejecutar%20en-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![Licencia](https://img.shields.io/badge/Licencia-MIT-yellow)

---

## TL;DR

- **Tarea.** Imagen → V MOR en metros (`V = 2,996/β`, umbral OMM del 5 %) y
  una de 4 bandas de seguridad (< 50, 50–100, 100–200 y ≥ 200 m),
  justificadas con la distancia de detención.
- **Datos.**
  - RESIDE-OTS oficial: visibilidad *equivalente*.
  - Foggy Cityscapes y **CS-Fog**, nuestra niebla con distancia estéreo
    métrica: visibilidad *métrica*.
  - Externos: HazeRD (métrico) y O-HAZE (niebla real, sin etiqueta).
- **Modelos.** `resnet50.a1_in1k` (23,5 M) vs `deit_small_patch16_224.fb_in1k`
  (21,7 M), ambos ImageNet-1k, con la **misma cabeza** (regresión de log10 V
  + ordinal CORN) y el **mismo protocolo**. Barrido de LR igual para los dos,
  3 semillas y test bloqueado.
- **Evaluación.**
  - Métricas por fuente con IC por **bootstrap agrupado** y diferencia pareada
    ResNet − DeiT.
  - Baselines triviales y físico (sin deep learning).
  - Sondas de atajo; Grad-CAM y attention rollout sobre los mismos ejemplos.
  - Recall de autos vs distancia por β; latencia CPU/GPU; demo Gradio.
- **Estado (3-nov-2026).** **Código listo y probado** en CPU con datos
  sintéticos (`pytest`, `camanchaca smoke`). **Resultados pendientes de correr
  en Colab** con los datasets reales. Ninguna cifra de este README es un
  resultado todavía.

## Resultados

Las tablas y figuras las genera **solo el código**. `camanchaca report` las
reúne en **[`reports/RESULTADOS.md`](reports/RESULTADOS.md)**, que se crea
tras la evaluación final.

| Modelo | log10-MAE macro (test) | Falso-seguro | macro-F1 bandas | ms/img CPU |
|---|---|---|---|---|
| Baseline físico | [pendiente] | [pendiente] | [pendiente] | [pendiente] |
| ResNet-50 (3 semillas) | [pendiente] | [pendiente] | [pendiente] | [pendiente] |
| DeiT-S (3 semillas) | [pendiente] | [pendiente] | [pendiente] | [pendiente] |

Regla de decisión, fijada antes de ver resultados: ResNet-50 y DeiT-S solo
**difieren** si el IC 95 % pareado de la diferencia excluye 0
([informe §4.2](docs/informe/informe.md)).

Qué genera cada comando (todo en `reports/`, o en `$CAMANCHACA_REPORTS`):

| Archivo | Comando |
|---|---|
| `tables/data_counts.md` | `camanchaca split` (n por split × fuente × banda) |
| `tables/lr_sweep.md` | `camanchaca sweep` (barrido de LR de `main`) |
| `tables/metrics_<set>.md`, `tables/paired_<set>.md`, `tables/eps_sensitivity_<set>.md` (`<set>` = val, test, ext) | `camanchaca evaluate` (test y ext solo con `--final`) |
| `tables/ohaze_pairs.md` | `camanchaca evaluate --sets ext --final` |
| `tables/latency.md`, `figures/pareto_speed_accuracy.png` | `camanchaca latency` |
| `tables/probe_source.md` | `camanchaca probes` |
| `tables/attribution.md`, `figures/interpret_examples_{neg,pos}.png` | `camanchaca interpret` |
| `tables/counterfactual.md` | `camanchaca counterfactual` |
| `tables/cars_recall.md`, `tables/cars_vdet.md`, `figures/cars_recall_vs_distance.png` | `camanchaca cars` |
| `figures/{scatter,error_by_band,confusion,per_source}_<set>.png`, `figures/learning_curves.png` | `camanchaca evaluate` |
| `RESULTADOS.md` | `camanchaca report` (reúne todo lo anterior) |

Cada tabla `.md` tiene su `.csv` gemelo (salvo `eps_sensitivity_*`). La
prueba de humo escribe en una carpeta temporal, nunca en `reports/`, y sus
tablas llevan un **AVISO** explícito de datos sintéticos.

## 1. Problema y motivación

La **camanchaca** es una niebla costera advectiva: aire húmedo del Pacífico
que condensa sobre aguas frías y entra a la costa por quebradas y pasos bajos.
Es heterogénea y cambia en minutos. En el **Camino La Pólvora**, principal
acceso de los camiones al puerto de Valparaíso, con tramos de alta velocidad
expuestos a la niebla, el riesgo aparece cuando la visibilidad cae bajo la
**distancia de detención**:

```
SSD(v) = v·2,5 s + v² / (2·0,4·9,81)      (v en m/s; piso mojado)
```

Eso da ≈ 119 m a 80 km/h y ≈ 168 m a 100 km/h. Hoy no hay una medición
continua de la visibilidad en metros sobre la vía. Este proyecto evalúa si un
modelo de visión puede estimarla desde una cámara, y qué arquitectura (CNN o
ViT) lo hace mejor bajo una comparación justa.

## 2. Etiqueta: visibilidad MOR y bandas

Todas las fuentes se anclan en el **coeficiente de extinción β** [1/m] del
modelo atmosférico de dispersión, `I = J·t + A·(1 − t)` con `t = exp(−β·d)`.
La etiqueta es el **MOR** de la OMM (ley de Koschmieder con umbral de
contraste del 5 %):

```
V = −ln(0,05) / β ≈ 2,996 / β   [m]       (sensibilidad: ε = 0,02 → 3,912/β)
```

| Banda | Rango | Velocidad segura máx. (SSD ≤ V) |
|---|---|---|
| `critico` | V < 50 m | < 44 km/h |
| `alto_riesgo` | 50–100 m | 44–71 km/h |
| `precaucion` | 100–200 m | 71–111 km/h |
| `aceptable` | ≥ 200 m | ≥ 111 km/h |

Calidad de la etiqueta (`label_quality`):

- **métrica**: la distancia usada para sintetizar la niebla está en metros
  (Cityscapes, HazeRD);
- **equivalente**: la profundidad es estimada y relativa (RESIDE);
- **ninguna**: niebla real sin medición (O-HAZE, fotos propias).

La **tasa de falso-seguro** (V real < 100 m predicha ≥ 100 m) es la métrica
de seguridad. Detalle completo, tablas de conversión por dataset y etiquetas
en los bordes de banda en
**[`docs/labeling_criteria.md`](docs/labeling_criteria.md)**.

## 3. Datos

| Fuente | Tipo | Calidad de etiqueta | Uso | Licencia | Cómo obtener |
|---|---|---|---|---|---|
| **RESIDE-OTS** (Li et al., 2019) | ASM sobre fotos reales; profundidad estimada; 7 β | equivalente (15–75 m) | train/val/test por escena; 1 A por (escena, β), ≈ 14,4k | investigación / no comercial; la etiqueta de Kaggle no es autoritativa | Dropbox/Baidu oficiales o Kaggle `brunobelloni/outdoor-training-set-ots-reside`; verificar con `camanchaca prep reside --check-only` |
| **Foggy Cityscapes** (Sakaridis et al., 2018) | ASM con distancia estéreo; β = 0,005/0,01/0,02 | métrica (600/300/150 m) | train/val/test por ciudad | términos de Cityscapes: no comercial, **sin redistribución** | registro en cityscapes-dataset.com; `csDownload leftImg8bit_trainvaltest_foggy.zip` |
| **CS-Fog** (propia) | ASM sobre Cityscapes limpio con distancia estéreo; V log-uniforme 15–1000 m | métrica (4 bandas) | train/val/test por ciudad; rompe la confusión dataset ↔ banda | derivado de Cityscapes (no se redistribuye) | `camanchaca prep cs-fog` (requiere `leftImg8bit`, `disparity`, `camera`, `gtFine`) |
| **HazeRD** (Zhang et al., 2017) | ASM con profundidad métrica; 5 niveles nominales (50–1000 m) | métrica (umbral por confirmar; con ε_H = 0,02 → ≈ 38/77/153/383/766 m MOR) | **solo test externo** | IEEE DataPort open access (licencia exacta sin verificar) | IEEE DataPort (DOI 10.21227/h6q8-y165) o formulario del sitio |
| **O-HAZE** (Ancuti et al., 2018) | **niebla real** (máquina), 45 pares | ninguna | externo: cualitativo + V̂(hazy) < V̂(GT) | sin licencia explícita: investigación, sin redistribuir | sitio NTIRE 2018 (ETH) |
| **Fotos La Pólvora** (propias) | real | ninguna | externo cualitativo y demo | del equipo | `camanchaca prep folder --name polvora` |
| **Autos** (Cityscapes val) | limpio + Foggy (3 β) + CS-Fog (50 y 100 m) | distancia estéreo por auto | análisis complementario | términos de Cityscapes | `camanchaca prep cars` |

Particiones:

- **Cityscapes:** por **ciudad**. Test = frankfurt, lindau, munster (el val
  oficial); val = hanover, jena, tubingen; train = el resto.
- **RESIDE:** 70/15/15 **por escena**.
- **Externos:** HazeRD, O-HAZE y las fotos propias van en `ext` y nunca se
  usan para entrenar ni para elegir nada.
- **Congelamiento:** `camanchaca split` escribe en
  `data/splits/v1/SPLIT_INFO.json` los conteos y el sha256 de cada CSV.
  sha256 de `test.csv` congelado: `[COMPLETAR al congelar los datos]`.

AIR-VIEW (visibilidad medida) se consideró, pero casi todas sus etiquetas
superan 1,6 km y requiere suscripción. Queda como trabajo relacionado.

**Nunca se suben imágenes al repositorio**: solo CSV derivados (rutas
relativas y etiquetas). Data card completa: **[`docs/data_card.md`](docs/data_card.md)**.

## 4. Método: comparación justa

| | ResNet-50 | DeiT-S |
|---|---|---|
| Pesos (timm) | `resnet50.a1_in1k` | `deit_small_patch16_224.fb_in1k` |
| Preentrenamiento | ImageNet-1k | ImageNet-1k |
| Parámetros del backbone | 23,5 M (25,6 M con la capa de ImageNet) | 21,7 M (22,1 M con la capa de ImageNet) |
| GMACs (224 × 224, lote 1) | ≈ 4,1 | ≈ 4,6 |
| Cabeza | Dropout → regresión de log10 V estandarizado + 3 logits ordinales CORN | **idéntica** |
| Pérdida | SmoothL1 + 0,5·CORN | **idéntica** |
| Optimización | AdamW, wd 0,05 (sin decay en normas/sesgos), LR cabeza = 10 × backbone | **idéntica** |
| LR del backbone | barrido {3e-5, 1e-4, 3e-4}, 1.200 pasos, semilla 0, criterio en val | **mismo barrido** |
| Schedule | warmup 300 + coseno, **máximo 3.600 pasos** (el mismo presupuesto), evaluación cada 300 y early stopping tras 4 sin mejora; batch 64, AMP, clip 1,0 | **idéntico** |
| Entrada / aumentación | 224 px, media/desv. de ImageNet; crop + flip, **sin jitter de color** | **idéntica** |
| Selección | media de log10-MAE por fuente en val | **idéntica** |
| Semillas | 0, 1, 2 | 0, 1, 2 |

El protocolo vive en `configs/protocol/base.yaml`, y `assert_same_protocol`
verifica que solo cambie el backbone. DeiT-B (`configs/model/deit_b.yaml`) es
un *stretch* de 1 semilla, solo como punto de velocidad. Ficha del modelo:
**[`docs/model_card.md`](docs/model_card.md)**.

## 5. Métricas

- **Regresión (principal):** log10-MAE por fuente y su media, MdAPE, δ < 1,25
  y Spearman dentro de cada fuente; MAE en metros (global y por fuente) como
  métrica secundaria; error por banda real en `error_by_band_<set>.png`.
- **Bandas:** macro-F1 sobre una lista fija de bandas, F1 por clase, matriz
  de confusión y MAE ordinal, desde la regresión y desde CORN.
- **Seguridad:** falso-seguro (NaN y `n_peligrosos` cuando no está definido)
  y AUC-PR de V < 100 m.
- **Incertidumbre:** media ± d.e. sobre 3 semillas; IC 95 % por bootstrap
  agrupado por escena o secuencia; diferencia pareada ResNet − DeiT.
- **Atajos:**
  - baselines `mean_by_source` y físico;
  - sonda lineal de la fuente;
  - contrafactual de niebla uniforme vs dependiente de la distancia.
- **Interpretabilidad:** Grad-CAM con signo (ambos) y attention rollout
  (DeiT) sobre un conjunto **fijo** de ejemplos; atribución por tercil de
  distancia y clase semántica.
- **Autos:** recall del detector vs distancia métrica por β; $V_{det}$ frente
  a MOR y SSD.
- **Despliegue:** parámetros, GMACs y latencia en T4 y en CPU.

## 6. Reproducción

### 6.1 Colab (camino principal)

| # | Notebook | Entorno | Qué hace |
|---|---|---|---|
| 00 | [`00_setup_colab`](notebooks/00_setup_colab.ipynb) | GPU T4 | clona el repo en una referencia fija, instala el paquete, verifica el entorno y corre la prueba de humo opcional |
| 01 | [`01_prepare_data`](notebooks/01_prepare_data.ipynb) | **CPU** | descargas, `prep` de cada fuente, `split`, `pack` a tars en Drive |
| 02 | [`02_train`](notebooks/02_train.ipynb) | GPU | `sweep` + `train-all` (idempotente y reanudable) |
| 03 | [`03_evaluate`](notebooks/03_evaluate.ipynb) | GPU | `predict`, `baselines`, `evaluate` (val; test/ext con `--final`), `latency`, `probes`, `report` |
| 04 | [`04_interpretability`](notebooks/04_interpretability.ipynb) | GPU | `interpret` + `counterfactual` (hipótesis H1–H3) |
| 05 | [`05_cars_vs_fog`](notebooks/05_cars_vs_fog.ipynb) | GPU | `cars`: recall vs distancia y $V_{det}$ |
| 06 | [`06_demo`](notebooks/06_demo.ipynb) | CPU | demo Gradio + grabación del video de respaldo |

Los notebooks solo llaman a la CLI. La secuencia completa, en el orden en que
la ejecutan los notebooks, es la siguiente (rutas de ejemplo en `/content`;
cada comando tiene `--help`):

```bash
pip install -e ".[demo,data]"            # en Colab: NO reinstalar torch/torchvision
export CAMANCHACA_DATA=/content/data                         # imágenes preprocesadas (disco local)
export CAMANCHACA_RUNS=/content/drive/MyDrive/camanchaca/runs  # corridas en Drive (reanudables)
EXP=configs/experiment/main.yaml
TARS=/content/drive/MyDrive/camanchaca/data

# 1) Datos (runtime CPU; notebook 01)
camanchaca prep reside --raw /content/raw/ots.zip --check-only  # aborta si es un reempaque sin beta
camanchaca prep reside --raw /content/raw/ots.zip --one-a-per-beta --workers 2
camanchaca prep cs-fog --cs-root /content/cs --workers 2        # niebla propia, V MOR 15-1000 m
camanchaca prep foggy-cityscapes --zip /content/cs/leftImg8bit_trainvaltest_foggy.zip --workers 2
camanchaca prep cars --cs-root /content/cs --foggy-zip /content/cs/leftImg8bit_trainvaltest_foggy.zip --split val --workers 2
camanchaca prep hazerd --raw /content/raw/hazerd --eps 0.02     # eps A CONFIRMAR en demo_simu_haze.m
camanchaca prep ohaze --raw /content/raw/O-HAZE.zip
camanchaca prep folder --raw /content/raw/polvora --name polvora
camanchaca split                      # data/splits/v1/{train,val,test,ext}.csv + SPLIT_INFO.json + reports/tables/data_counts.md
for d in $CAMANCHACA_DATA/images/* $CAMANCHACA_DATA/aux; do
  camanchaca pack --src "$d" --tar "$TARS/$(basename "$d").tar"  # un tar por fuente (+ aux/)
done

# 2) Entrenamiento (GPU; notebook 02). En cada sesión nueva, primero:
for t in $TARS/*.tar; do camanchaca unpack --tar "$t"; done
camanchaca sweep --exp $EXP                   # misma grilla de LR y presupuesto para ambos -> selected_lr.json
camanchaca train-all --exp $EXP --lr auto     # 2 modelos x 3 semillas; idempotente y reanudable

# 3) Desarrollo: solo validación (notebook 03)
camanchaca predict   --exp $EXP --sets val
camanchaca baselines --exp $EXP --sets val
camanchaca evaluate  --exp $EXP --sets val

# 4) Evaluación FINAL (una sola vez, con todo congelado; notebooks 03-05)
camanchaca predict   --exp $EXP --sets val test ext --final
camanchaca baselines --exp $EXP --sets val test ext --final
camanchaca evaluate  --exp $EXP --sets val test ext --final --bootstrap 10000
camanchaca probes    --exp $EXP --final
camanchaca latency   --exp $EXP --device auto --batch-sizes 1 64   # CPU fp32 y, si hay GPU, T4 fp16 (una sola corrida)
camanchaca interpret --exp $EXP --seed 0 --n-per-cell 2 --final  # ejemplos fijos en data/splits/v1/interpret_examples.csv
camanchaca counterfactual --exp $EXP --n 200 --final
camanchaca cars --images data/manifests/cars_images.csv --gt data/manifests/cars_gt.csv --device cuda --score-thr 0.5 --iou 0.5
camanchaca report                             # -> reports/RESULTADOS.md
```

Notas:

- El test está **bloqueado**: sin `--final` solo se evalúa val. Sin
  `--final`, `interpret` y `counterfactual` usan val y guardan sus ejemplos en
  `interpret_examples_val.csv`; con `--final`, en
  `data/splits/v1/interpret_examples.csv` (se crea una vez y se reutiliza, así
  ambos modelos se explican sobre los mismos ejemplos).
- Las corridas viven en `runs/<exp>/<modelo>/seed<k>/` (o en
  `$CAMANCHACA_RUNS`). Un archivo `DONE` las hace idempotentes, y `last.pt`
  permite reanudar tras una desconexión. El barrido queda en
  `runs/<exp>/sweep/` con `selected_lr.json`.
- `latency` reescribe `latency.md` en cada llamada: por eso se mide CPU y GPU
  en una sola corrida con `--device auto`.
- **Coherencia del barrido.** `selected_lr.json` guarda, además del LR, el
  hash del protocolo con que se eligió (`sweep_protocol_hash`). Si se cambia
  `configs/protocol/base.yaml`, `train`/`train-all` se niegan a usar ese LR
  hasta repetir el barrido. Tampoco se permiten semillas del mismo modelo con
  LR distinto. Una corrida que diverge (pérdida no finita) se cierra con NaN y
  el barrido continúa con el siguiente LR.
- **GPU obligatoria.** `sweep`, `train` y `train-all` se niegan a entrenar sin
  CUDA, para evitar caer en CPU sin darse cuenta cuando Colab no asigna GPU.
  Para una prueba deliberada en CPU, pase `--device cpu`.
- **Espacio en Drive.** `last.pt` (~280 MB) se reescribe cada
  `schedule.save_last_every` evaluaciones (3). Drive envía cada versión
  reemplazada a la Papelera, que sigue ocupando cuota: vacíela entre sesiones.
- **Preprocesamiento reproducible.** Cada carpeta de imágenes guarda un
  `PARAMS.json` con los parámetros de render (semilla, rango de V, A, ε,
  tamaño, recorte). Si se re-ejecuta un `prep` con otros parámetros sobre
  imágenes ya existentes, el comando aborta para no mezclar píxeles viejos con
  etiquetas nuevas. Use `--overwrite` para re-renderizar, o borre la carpeta.
- **Autos.** El recall cuenta como acierto cualquier caja de vehículo COCO
  (`car`, `truck`, `bus`; `--classes car` sirve como sensibilidad). Los bins
  de distancia con menos de `--min-gt` autos GT (20 por defecto) no se usan
  para calcular $V_{det}$.
- Variables de entorno: `CAMANCHACA_DATA`, `CAMANCHACA_RUNS` y
  `CAMANCHACA_REPORTS`.
- Atajos: `make sweep`, `make train`, `make evaluate`, `make final`,
  `make interpret`, `make cars` y `make report`.

### 6.2 Local (verificación sin GPU ni datos)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,demo]"
camanchaca smoke              # pipeline completo sobre datos sintéticos (CPU, pesos aleatorios; ~15 min)
pytest -m "not slow"          # pruebas rápidas; `pytest` a secas incluye el entrenamiento de humo
```

Ni la prueba de humo ni las pruebas descargan pesos (`pretrained=False`) ni
escriben en `data/`, `runs/` o `reports/` del repositorio: usan carpetas
temporales (`camanchaca smoke --workdir DIR --keep` conserva la suya para
inspeccionarla). Equivalentes con `make`: `make setup`, `make smoke` y
`make test-fast`.

La prueba de humo recorre todo el pipeline con datos sintéticos: manifiesto →
splits → barrido → 2 pasos de entrenamiento de ResNet y DeiT → predict →
bootstrap → baselines → Grad-CAM y rollout → autos → reporte. La CI de GitHub
Actions corre `pytest` y `camanchaca smoke` en cada push.

## 7. Demo

```bash
pip install -e ".[demo]"
python scripts/fetch_weights.py --exp main     # pesos de app/weights.json, verificados con sha256
# alternativa: --drive-dir /content/drive/MyDrive/camanchaca/weights, o entrenar (camanchaca train-all)
python app/app.py --exp main                   # http://127.0.0.1:7860 (--ckpt RUTA para un best.pt concreto)
```

`app/weights.json` se publica después del entrenamiento final
(`fetch_weights.py --make-manifest`); hasta entonces la demo usa los `best.pt`
de `runs/main/<modelo>/seed<k>/`.

La demo corre en la CPU de un portátil. Entrega:

- V̂ MOR (ensamble de semillas) con su dispersión y la banda de seguridad;
- una velocidad segura **ilustrativa**;
- el mapa Grad-CAM o el attention rollout;
- avisos de fuera de distribución (noche, imagen diminuta, etc.).

Tiene además una pestaña para simular niebla sobre una foto despejada.
**No es apta para uso operativo.** Detalles en [`app/README.md`](app/README.md).
En la presentación se usa en el portátil, con un video de respaldo.

## 8. Estructura del repositorio

```
Camanchaca-Predict/
├── src/camanchaca/            # paquete único (fuente de verdad) + CLI `camanchaca`
│   ├── labeling/              # Koschmieder/MOR, SSD, síntesis de niebla (ASM), canal oscuro
│   ├── data/                  # esquema, manifiestos por fuente, prep, splits, dataset, transforms
│   ├── models/                # backbones timm + cabeza idéntica (regresión + CORN), Predictor
│   ├── training/              # pérdidas, optimizador, trainer por pasos, checkpoints, métricas
│   ├── evaluation/            # predict, bootstrap agrupado/pareado, tablas, figuras, latencia, reporte
│   ├── baselines/             # triviales, features físicas, Ridge/HGB, sonda de fuente
│   ├── interpret/             # Grad-CAM, attention rollout, atribución, contrafactual
│   ├── detection/             # detector de autos, emparejamiento, recall vs distancia
│   ├── demo/                  # lógica de la app Gradio
│   ├── testing/               # datos sintéticos para tests y prueba de humo
│   └── cli.py, config.py, paths.py, seed.py, runmeta.py, smoke.py
├── configs/
│   ├── protocol/              # base.yaml (protocolo compartido), lr_sweep.yaml
│   ├── model/                 # resnet50.yaml, deit_s.yaml, deit_b.yaml (stretch)
│   ├── data/                  # reside_ots, foggy_cityscapes, cs_fog, hazerd, ohaze
│   └── experiment/            # main.yaml, smoke.yaml
├── notebooks/                 # 00–06, delgados (solo llaman a la CLI)
├── app/                       # app.py (Gradio), examples/ (solo fotos propias o CC)
├── scripts/                   # fetch_weights.py, verify_env.py
├── tests/                     # pytest (CPU, pesos aleatorios, fixtures sintéticos)
├── docs/                      # etiquetado, data card, model card, riesgos, informe/, presentación, respuesta
├── reports/                   # tables/ y figures/ generadas por la CLI; RESULTADOS.md
├── data/
│   ├── manifests/             # CSV por fuente (versionados; sin imágenes)
│   └── splits/v1/             # train/val/test/ext.csv + SPLIT_INFO.json (versionados)
├── presentations/             # láminas por fecha
├── .github/workflows/ci.yml   # pytest + prueba de humo
├── pyproject.toml, requirements.txt, Makefile, .mailmap, LICENSE
```

## Equipo, roles y `.mailmap`

| Integrante | Responsabilidades | Partes de la presentación |
|---|---|---|
| **Tomás Ignacio Moraga** | datos y etiquetado, baselines, análisis de autos, demo, data card | problema, etiqueta, datos, autos, demo |
| **Nelson Mauricio Fuentes** | entrenamiento, evaluación e incertidumbre, interpretabilidad, latencia | cambios, método, resultados, interpretabilidad, velocidad vs precisión |

Los dos presentan unos 7 minutos cada uno y revisan el trabajo del otro.

**Identidades de git.** Los primeros commits del repositorio son de Tomás
(correo `tomas.moraga.g@mail.pucv.cl`) hechos desde equipos con distinto
`user.name` ("Paulita", "Tomás" y "Tomas Ignacio Moraga"). El archivo
[`.mailmap`](.mailmap) los unifica sin reescribir el historial
(`git shortlog -sne` muestra una sola persona).

## Uso de asistentes de IA

Parte del código, de las pruebas, de los notebooks y de la documentación se
escribió con apoyo de asistentes de IA, bajo instrucciones y revisión del
equipo y conforme a las normas del curso y de la PUCV. Las decisiones
experimentales son del equipo, los experimentos se ejecutan en Colab, y
**todas las cifras reportadas salen de las tablas generadas por el código**
(`reports/`). Ambos integrantes pueden explicar cada componente.

## Licencias

- **Código:** [MIT](LICENSE).
- **Datos:** no se redistribuyen. Cityscapes y Foggy Cityscapes tienen uso no
  comercial y prohíben redistribuir imágenes o versiones modificadas (incluida
  CS-Fog). RESIDE es para investigación y sus fotos tienen copyright de sus
  autores. HazeRD y O-HAZE se usan solo para investigación, citándolos. Solo
  se versionan CSV derivados, sin píxeles.
- **Modelos:** pesos de partida de timm (Apache-2.0 según sus metadatos) y
  detector de torchvision (BSD-3, pesos COCO). No se usa Ultralytics (AGPL).
  Los pesos entrenados derivan de datos no comerciales y se publican solo
  para uso académico.
- **Demo:** solo fotos propias o con licencia CC (`app/examples/README.md`).

## Citas

Si usas este repositorio, cita los datasets y métodos en los que se basa. El
BibTeX completo está en [`docs/informe/refs.bib`](docs/informe/refs.bib):

- RESIDE (Li et al., IEEE TIP 2019);
- Cityscapes (Cordts et al., CVPR 2016);
- Foggy Cityscapes (Sakaridis et al., IJCV 2018);
- HazeRD (Zhang et al., ICIP 2017);
- O-HAZE (Ancuti et al., CVPRW 2018);
- AIR-VIEW (Mourning et al., 2025);
- ResNet (He et al., CVPR 2016);
- ViT (Dosovitskiy et al., ICLR 2021);
- DeiT (Touvron et al., ICML 2021);
- ResNet strikes back (Wightman et al., 2021);
- timm (Wightman, 2019);
- CORN (Shi et al., 2023);
- Grad-CAM (Selvaraju et al., ICCV 2017);
- attention rollout (Abnar y Zuidema, ACL 2020);
- canal oscuro (He et al., TPAMI 2011);
- Koschmieder (1924) y OMM No. 8;
- Faster R-CNN (Ren et al., NeurIPS 2015) con la receta v2 de torchvision
  (Li et al., 2021);
- bootstrap (Efron y Tibshirani, 1993).

## Documentación

| Documento | Contenido |
|---|---|
| [`docs/labeling_criteria.md`](docs/labeling_criteria.md) | etiqueta MOR, β como ancla, equivalente vs métrica, conversión por dataset, bandas y SSD, CS-Fog |
| [`docs/data_card.md`](docs/data_card.md) | origen, licencia, conteos, splits y sesgos por dataset |
| [`docs/model_card.md`](docs/model_card.md) | modelos, uso previsto, métricas, advertencias |
| [`docs/risks_and_mitigation.md`](docs/risks_and_mitigation.md) | riesgos con responsables y cronograma con gates |
| [`docs/informe/`](docs/informe/) | informe técnico (pandoc) y bibliografía |
| [`docs/presentacion_final.md`](docs/presentacion_final.md) | guion de 15 min, checklist de láminas, FAQ |
