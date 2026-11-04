# Model card — Camanchaca-Predict (ResNet-50 y DeiT-S)

Ficha según *Model Cards for Model Reporting* (Mitchell et al., 2019).

> **Estado (3-nov-2026).** Arquitectura, protocolo y evaluación están
> implementados y probados en CPU con datos sintéticos y pesos aleatorios.
> **Todavía no hay métricas reales**: se generan en Colab
> (`camanchaca evaluate … --final`, luego `camanchaca report`) y se copian
> desde `reports/RESULTADOS.md`. Toda cifra marcada `[COMPLETAR]` está
> pendiente.

## 1. Detalles del modelo

| | ResNet-50 | DeiT-S |
|---|---|---|
| Backbone (timm) | `resnet50.a1_in1k` | `deit_small_patch16_224.fb_in1k` |
| Familia | CNN | Vision Transformer (parches 16×16) |
| Preentrenamiento | ImageNet-1k (receta A1 de "ResNet strikes back") | ImageNet-1k (DeiT, sin destilación) |
| Parámetros del backbone (sin la capa de ImageNet) | 23,5 M (25,6 M con ella) | 21,7 M (22,1 M con ella) |
| GMACs (224 × 224, lote 1) | ≈ 4,1 | ≈ 4,6 |
| Feature a la cabeza | GAP de 2048-d | token CLS de 384-d |
| Entrada | 224×224 RGB, media/desv. de ImageNet | igual |
| Licencia de los pesos de partida (metadatos de timm) | Apache-2.0 | Apache-2.0 |

**Cabeza idéntica** (`models/heads.py`): `Dropout(0,1)` seguido de

- `Linear → 1`: regresión de `log10(V MOR)` estandarizado;
- `Linear → 3`: logits ordinales **CORN** (Shi et al., 2023) sobre las 4
  bandas.

**Protocolo idéntico** (`configs/protocol/base.yaml`; `assert_same_protocol`):

- **Pérdida:** SmoothL1 (β = 1) sobre la regresión + 0,5 × CORN.
- **Optimizador:** AdamW con weight decay 0,05, sin decay en normas, sesgos ni
  tokens de posición. LR de la cabeza = 10 × LR del backbone. El LR del
  backbone se elige con un barrido {3e-5, 1e-4, 3e-4} de igual presupuesto
  (1.200 pasos, semilla 0, solo val).
- **Schedule:** warmup de 300 pasos + coseno con un presupuesto **máximo de
  3.600 pasos**, el mismo para ambos. Batch 64, AMP fp16 (solo CUDA), clip de
  gradiente 1,0.
- **Evaluación y selección:** se evalúa cada 300 pasos, con early stopping
  tras 4 evaluaciones sin mejora (por eso los pasos efectivos pueden diferir
  entre corridas; quedan en `history.csv` y en `config.yaml` de cada corrida).
  Selección del checkpoint por **media de log10-MAE por fuente en val**.
- **Aumentación:** Resize 256 → RandomCrop 224 + flip horizontal. **Sin**
  jitter fotométrico.
- **Semillas:** 0, 1 y 2.

**Salidas del `Predictor`**: `v_m` (V MOR en m), `log10_v`, `band_reg` (banda
digitalizada desde la regresión), `band_ord` y `ord_probs` (cabeza CORN). La
demo promedia las semillas con media geométrica y muestra su dispersión.

**Versión / pesos:** `[COMPLETAR: tag y sha256 de app/weights.json]`.
**Autores:** Tomás Ignacio Moraga y Nelson Mauricio Fuentes (G5, PUCV).
**Licencia del código:** MIT.

## 2. Uso previsto

- **Primario:** comparación académica de arquitecturas (CNN vs ViT) para
  estimar la visibilidad bajo niebla, y demostración en el curso.
- **Usuarios:** el equipo, el equipo docente y los compañeros del curso.
- **Fuera de alcance:** cualquier decisión operativa (cierre de vías, límites
  de velocidad, alertas a conductores). Tampoco sirve para imágenes nocturnas,
  lluvia, interiores ni cámaras con ángulos muy distintos a los de una cámara
  vial o fotográfica. La "velocidad segura" de la demo es **ilustrativa**.

## 3. Factores

Los resultados se desagregan por:

- **fuente** (`reside_ots`, `foggy_cs`, `cs_fog`, `hazerd`, `ohaze`);
- **`label_quality`** (`metrica` / `equivalente` / `ninguna`);
- **banda real** (figura `error_by_band_<set>.png` y matrices de confusión);
- **tercil de distancia** y **clase semántica** (solo en la atribución).

No hay factores demográficos. Factores ambientales que no están cubiertos por
los datos: noche, lluvia, niebla heterogénea, luz marina y cámaras chilenas.

## 4. Métricas

| Tipo | Métricas | Por qué |
|---|---|---|
| Regresión (principal) | log10-MAE (macro por fuente y por fuente), MdAPE, δ < 1,25, Spearman dentro de cada fuente, factor de error | error relativo y orden; robustas al rango 15–1000 m |
| Regresión (secundaria) | MAE en metros (global y por fuente) | interpretable, pero dominada por las V grandes |
| Bandas | macro-F1 sobre una lista **fija** de bandas presentes, F1 por clase, matriz de confusión, MAE ordinal; desde la regresión y desde CORN | la decisión operativa es por banda |
| Seguridad | tasa de falso-seguro (V < 100 m predicha ≥ 100 m; NaN si no hay casos peligrosos, junto a `n_peligrosos`), AUC-PR de "V < 100 m" | el error caro es decir "seguro" cuando no lo es |
| Incertidumbre | media ± d.e. entre 3 semillas; IC 95 % por **bootstrap agrupado** (escena / `ciudad_secuencia`), B = 2.000 (10.000 en la versión final); diferencia **pareada** ResNet − DeiT con la misma matriz de remuestreo | no afirmar diferencias sin evidencia |
| Despliegue | parámetros, GMACs, ms/imagen (batch 1) e imágenes/s (batch 64) en T4 (fp16) y CPU (fp32) | decisión velocidad vs precisión |

**Regla de decisión, fijada antes de ver resultados:** un modelo es "mejor"
solo si el IC 95 % pareado de la diferencia en log10-MAE macro por fuente en
**test** excluye 0.

## 5. Datos de evaluación

- **Test interno** (bloqueado con `--final`):
  - escenas de test de RESIDE-OTS;
  - ciudades frankfurt, lindau y munster (val oficial de Cityscapes) en
    Foggy Cityscapes y CS-Fog.
- **Externo** (`ext`): HazeRD (métrico), O-HAZE (real, sin etiqueta: orden por
  pares) y fotos de La Pólvora (cualitativo).

Detalle en [`docs/data_card.md`](data_card.md).

## 6. Datos de entrenamiento

Train de RESIDE-OTS (1 A por escena y β), Foggy Cityscapes y CS-Fog, con split
por escena o por ciudad. Los conteos están en
`reports/tables/data_counts.md`. La etiqueta es V MOR = 2,996/β
([`docs/labeling_criteria.md`](labeling_criteria.md)).

## 7. Análisis cuantitativo

| Tabla / figura | Archivo | Estado |
|---|---|---|
| Métricas por modelo y fuente (test) | `reports/tables/metrics_test.md` | `[COMPLETAR]` |
| Diferencia pareada ResNet − DeiT | `reports/tables/paired_test.md`, `paired_ext.md` | `[COMPLETAR]` |
| Externos (HazeRD; pares O-HAZE) | `reports/tables/metrics_ext.md`, `ohaze_pairs.md` | `[COMPLETAR]` |
| Sensibilidad a ε | `reports/tables/eps_sensitivity_test.md` | `[COMPLETAR]` |
| Sonda de fuente / contrafactual | `reports/tables/probe_source.md`, `counterfactual.md` | `[COMPLETAR]` |
| Atribución por distancia y clase (ejemplos fijos de `data/splits/v1/interpret_examples.csv`) | `reports/tables/attribution.md`, `reports/figures/interpret_examples_neg.png` | `[COMPLETAR]` |
| Barrido de LR | `reports/tables/lr_sweep.md` (`runs/main/sweep/selected_lr.json`) | `[COMPLETAR]` |
| Latencia | `reports/tables/latency.md`, `reports/figures/pareto_speed_accuracy.png` | `[COMPLETAR]` |

## 8. Consideraciones éticas

- **Riesgo de exceso de confianza.** Una estimación de "visibilidad aceptable"
  equivocada podría inducir a conducir más rápido. Por eso el modelo no es
  apto para uso operativo, la demo lo advierte y se reporta el falso-seguro.
- **Privacidad.** Cityscapes contiene personas y patentes. No se publican
  imágenes de los datasets. La demo usa fotos propias o con licencia CC, sin
  GPS.
- **Licencias.** Los pesos entrenados son derivados de datasets no
  comerciales (Cityscapes, RESIDE): se publican solo para uso académico.

## 9. Advertencias y recomendaciones

- La niebla de entrenamiento es sintética y homogénea. **Se espera una
  degradación en niebla real**, que este proyecto no puede cuantificar con
  visibilidad medida.
- RESIDE aporta "visibilidad equivalente", no metros.
- El modelo puede apoyarse en el contraste global o en la identidad del
  dataset (atajos). Antes de interpretar el desempeño, revisar
  `probe_source.md` y `counterfactual.md`.
- Para un uso real se necesitaría una cámara fija con referencias a distancia
  conocida o un visibilímetro para calibrar y validar en la ruta.
