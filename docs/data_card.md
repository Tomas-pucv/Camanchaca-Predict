# Data card — Camanchaca-Predict

Ficha de datos según *Datasheets for Datasets* (Gebru et al., 2021). Cubre
qué datos se usan, de dónde vienen, con qué licencia, cuántos son, cómo se
particionan, qué significa la etiqueta y qué sesgos tienen.

> **Estado (3-nov-2026).** El código de preparación está listo y probado con
> datos sintéticos. Las descargas reales se hacen en Colab, así que los
> conteos de abajo son **esperados** según la documentación de cada dataset.
> Los **conteos reales** los genera `camanchaca split` en
> [`reports/tables/data_counts.md`](../reports/tables/data_counts.md) y en
> [`data/splits/v1/SPLIT_INFO.json`](../data/splits/v1/), con el sha256 de
> cada CSV. Si los números difieren, mandan los generados.
> **Pendiente:** completar las fechas de acceso marcadas como `[COMPLETAR]`.

## 1. Motivación y uso previsto

- **Para qué.** Entrenar y comparar ResNet-50 y DeiT-S en la estimación de la
  visibilidad (MOR, metros) bajo niebla a partir de una sola imagen. Además,
  medir hasta qué distancia un detector sigue viendo autos con niebla.
- **Uso previsto.** Investigación académica y evaluación del curso
  Aprendizaje Automático II (PUCV, 2026-2).
- **Uso NO previsto.** Decisiones operativas de tránsito o seguridad vial. No
  hay validación con visibilidad medida en niebla real.

## 2. Resumen de fuentes

| Fuente (`source`) | Tipo de niebla | Etiqueta (`label_quality`) | Uso | Licencia (resumen) |
|---|---|---|---|---|
| RESIDE-OTS (`reside_ots`) | sintética (ASM) sobre fotos reales de internet; profundidad **estimada** | V MOR **equivalente** | train / val / test (por escena) | investigación, no comercial; fotos con copyright de sus autores |
| Foggy Cityscapes (`foggy_cs`) | sintética (ASM) sobre escenas urbanas reales; distancia **estéreo** | V MOR **métrica** (600/300/150 m) | train / val / test (por ciudad) | términos de Cityscapes: no comercial, **sin redistribución** |
| CS-Fog (`cs_fog`, propia) | sintética (ASM) generada por nosotros sobre Cityscapes limpio; distancia **estéreo** | V MOR **métrica** continua (15–1000 m) | train / val / test (por ciudad) | derivado de Cityscapes: mismos términos (no se redistribuyen las imágenes) |
| HazeRD (`hazerd`) | sintética sobre 14–15 escenas reales con profundidad **métrica** | V MOR **métrica** (convención por confirmar) | **solo test externo** (`ext`) | IEEE DataPort "open access" (licencia exacta no verificada) |
| O-HAZE (`ohaze`) | **real** (máquina de niebla), pares con/sin niebla | **ninguna** | **solo externo**: cualitativo + orden por pares | sin licencia explícita encontrada: uso de investigación, sin redistribuir |
| Fotos propias La Pólvora (`folder`) | **real** (camanchaca) o despejada | ninguna (opcional: V estimada con referencias) | **solo externo / demo** | del equipo (se pueden publicar si no muestran personas o patentes) |
| Autos (Cityscapes val) | limpio, Foggy Cityscapes (3 β) y CS-Fog (V = 50 y 100 m) | distancia por auto desde estéreo | análisis complementario (no entrena modelos) | términos de Cityscapes |
| AIR-VIEW | **real**, visibilidad medida por sensor | — | **no se usa** (ver §3.7) | suscripción IEEE DataPort |

Etiqueta común: `v_mor_m = 2,996/β` (MOR, ε = 5 %), con β como ancla. La
columna `v_eps002_m` (ε = 2 %) existe solo para el análisis de sensibilidad.
Detalle en [`docs/labeling_criteria.md`](labeling_criteria.md).

## 3. Fichas por dataset

### 3.1 RESIDE-OTS (RESIDE-β Outdoor Training Set)

- **Origen.** Li et al., *Benchmarking Single-Image Dehazing and Beyond*,
  IEEE TIP 28(1), 2019. Sitio: <https://sites.google.com/view/reside-dehaze-datasets>.
  Enlaces oficiales (Dropbox/Baidu) en el repositorio
  `Boyiliee/RESIDE-dataset-link`. Espejo práctico: Kaggle
  `brunobelloni/outdoor-training-set-ots-reside` (~12 GB, ~74,2k archivos).
- **Versión requerida.** Nombres oficiales `{id}_{A}_{beta}.jpg`, con 7 β
  {0,04 … 0,2} × 5 A {0,8 … 1,0} por escena. Los **reempaques** (RESIDE-6K,
  DehazeFormer, gUNet) renombran los archivos, pierden β y **no sirven**.
  `camanchaca prep reside --check-only` mide el % de nombres válidos y aborta
  bajo 90 %. Los nombres secuenciales de la primera versión del proyecto
  sugieren que se había usado un reempaque; esto se confirma al descargar de
  nuevo.
- **Fecha de acceso / espejo usado:** `[COMPLETAR]`.
- **Licencia.** El sitio de RESIDE indica uso de investigación / no comercial
  (no se pudo verificar desde el sandbox). Las fotos limpias vienen de
  internet y su copyright es de sus autores. La etiqueta "Apache-2.0" de Kaggle
  la pone quien sube el archivo y **no es autoritativa**.
- **Conteos esperados.** La versión RESIDE-β actual tiene 2.061 escenas y
  72.135 imágenes con niebla. Con **una A por (escena, β)**, elegida con
  semilla, quedan **14.427** imágenes. El split 70/15/15 por escena da
  ≈ 10,1k / 2,2k / 2,2k. Versiones antiguas tienen 8.970 u 8.477 escenas; el
  número de ids únicos identifica cuál se descargó.
- **Etiqueta.** `β` del nombre de archivo. V MOR de 15,0 a 74,9 m, en las
  bandas `critico` y `alto_riesgo`. `label_quality = equivalente`, porque la
  profundidad usada al sintetizar es monocular y relativa.
- **Split.** Agrupado por escena (`group` = id de escena). Las 7 variantes de
  β de una escena quedan siempre juntas.
- **Sesgos conocidos.** Fotos diurnas de internet con muchas escenas urbanas
  y de paisaje. Niebla homogénea con A ∈ [0,8; 1,0]. β, A y contenido de la
  escena están correlacionados. Solo cubre V < 75 m. La niebla de RESIDE se
  ha descrito como poco fiel a la profundidad real.

### 3.2 Foggy Cityscapes (oficial)

- **Origen.** Sakaridis, Dai y Van Gool, *Semantic Foggy Scene Understanding
  with Synthetic Data*, IJCV 126(9), 2018. Página:
  <https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/>. Imágenes **solo** en
  <https://www.cityscapes-dataset.com/downloads/>, con registro académico.
  Paquete `leftImg8bit_trainvaltest_foggy.zip` (~30 GB según la comunidad;
  verificar con `csDownload -l`).
- **Fecha de acceso:** `[COMPLETAR]`.
- **Licencia.** Términos de Cityscapes: uso no comercial, cita obligatoria,
  **prohibido redistribuir** las imágenes o versiones modificadas. Se permiten
  derivados abstractos (modelos, CSV de etiquetas). La lista "refined" (550
  frames) tiene una licencia que prohíbe redistribuirla: se descarga en tiempo
  de ejecución y no se versiona.
- **Conteos esperados.** 5.000 frames × 3 β = 15.000 imágenes (oficial: 2.975
  train / 500 val / 1.525 test frames). En nuestro split:
  - **test** = 500 frames del val oficial (frankfurt, lindau, munster) × 3 =
    1.500;
  - **val** = hanover, jena y tubingen. El conteo de ~459 frames es de memoria
    y debe verificarse.
  - **train** = el resto, que incluye las ciudades del test oficial (β
    conocido; nunca se mezclan con las ciudades de evaluación).
- **No usados.** Las ~20.000 imágenes extra (`trainextra`, solo β = 0,01)
  aportarían un único nivel (300 m) y cuestan disco. La variante DBF agrega
  otros ~20 GB sin nuevos niveles.
- **Etiqueta.** β ∈ {0,005; 0,01; 0,02} da V MOR = 600/300/150 m (la misma
  convención de 5 % que usan los autores). `label_quality = metrica`.
- **Split.** Por **ciudad**, para que los frames vecinos de una secuencia no
  crucen splits. `group` = `{ciudad}_{secuencia}` para el bootstrap.
- **Sesgos conocidos.** Ciudades alemanas y vecinas, de día y con buen
  tiempo. Niebla homogénea renderizada (con artefactos en el cielo y en zonas
  lejanas). Cámara montada en el auto (se recorta el 20 % inferior por el
  capó). Solo cubre V ≥ 150 m.

### 3.3 CS-Fog (niebla propia sobre Cityscapes)

- **Origen.** La generamos nosotros (`camanchaca prep cs-fog`) desde
  `leftImg8bit`, `disparity`, `camera` y `gtFine` de Cityscapes (Cordts et al.,
  CVPR 2016), generando niebla con β controlado sobre profundidad real.
- **Por qué existe.** Rompe la confusión dataset ↔ banda. Las cuatro bandas
  aparecen en el **mismo** dominio visual, en train, val y test.
- **Licencia.** Son imágenes modificadas de Cityscapes, con los mismos
  términos: **no se redistribuyen**. Solo se versiona el manifiesto CSV.
- **Conteos esperados.** Splits oficiales con gtFine (train + val = 3.475
  frames) × K = 2 visibilidades = 6.950 imágenes. Test = 500 × 2 = 1.000.
  Val ≈ 459 × 2. Train = el resto.
- **Etiqueta.** V MOR ~ log-uniforme en [15, 1000] m. La fracción esperada por
  banda es 28,7 / 16,5 / 16,5 / 38,3 %. A ~ U[0,75; 0,95].
  `label_quality = metrica`. El render se detalla en
  [`labeling_criteria.md` §8](labeling_criteria.md).
- **Split.** El mismo split por ciudad que Foggy Cityscapes. Todas las
  variantes de un frame quedan juntas.
- **Sesgos conocidos.** Hereda los de Cityscapes. Además: niebla homogénea con
  A gris, distancia estéreo ruidosa a > 100 m y huecos rellenados por vecino
  más cercano.

### 3.4 HazeRD

- **Origen.** Zhang, Ding y Sharma, *HazeRD: An outdoor scene dataset and
  benchmark for single image dehazing*, ICIP 2017. Página:
  <https://labsites.rochester.edu/gsharma/research/computer-vision/hazerd/>.
  IEEE DataPort, DOI 10.21227/h6q8-y165, que requiere cuenta gratuita. Existe
  también un registro antiguo de 10 escenas.
- **Fecha de acceso / versión:** `[COMPLETAR]`.
- **Licencia.** "Open access" en IEEE DataPort. La licencia exacta no se
  verificó (los registros abiertos de DataPort suelen ser CC BY 4.0).
- **Conteos esperados.** 14 escenas según la página y 15 según IEEE/DataPort,
  × 5 niveles = 70–75 imágenes. Todo es test.
- **Etiqueta.** Niveles nominales 50/100/200/500/1000 m sobre profundidad
  métrica. El paquete 2019 no trae las imágenes con niebla: `prep hazerd` las
  renderiza en Python con A = 0,76, composición en RGB lineal y profundidad
  inválida → 2 V, como el script original (los dos últimos detalles, por
  confirmar). La conversión `V_nom → β` usa el umbral de HazeRD (`--eps`, por
  defecto 0,02), que **está por confirmar** en `demo_simu_haze.m`. Con 0,02,
  la V MOR es 0,766 × V_nom: 50/100/200/500/1000 m nominales → ≈ 38/77/153/
  383/766 m MOR (bandas `critico`, `alto_riesgo`, `precaucion`, `aceptable`,
  `aceptable`).
  `label_quality = metrica`.
- **Split.** `ext` (solo evaluación final). El bootstrap agrupa por escena,
  así que los IC son anchos con 14–15 escenas.
- **Sesgos conocidos.** Escenas mayoritariamente de arquitectura, no de
  carreteras. Alta resolución. Luz atmosférica fija y render distinto al del
  entrenamiento, lo que introduce un cambio de dominio de render.

### 3.5 O-HAZE

- **Origen.** Ancuti et al., *O-HAZE: a dehazing benchmark with real hazy and
  haze-free outdoor images*, CVPRW (NTIRE) 2018. Página:
  <https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/>.
- **Fecha de acceso:** `[COMPLETAR]`.
- **Licencia.** No se encontró una licencia explícita. Se trata como uso de
  investigación y no se redistribuye.
- **Conteos esperados.** 45 pares (`NN_outdoor_hazy` / `NN_outdoor_GT`).
- **Etiqueta.** Ninguna (`label_quality = ninguna`). La GT lleva
  `is_clear = True`.
- **Uso.** Inspección cualitativa y chequeo de orden por pares
  `V̂(hazy) < V̂(GT)` (`reports/tables/ohaze_pairs.md`). La inversión física
  del modelo de dispersión (A, t, profundidad monocular) no se implementó: es
  trabajo futuro.
- **Sesgos conocidos.** Humo de máquina, no niebla marina. Niebla no
  homogénea en escenas de corto alcance (decenas de metros) que no son
  carreteras.

### 3.6 Fotos propias del Camino La Pólvora

- **Origen.** Fotos tomadas por el equipo con y sin camanchaca, con fecha y
  hora en el nombre. Se preparan con `camanchaca prep folder --raw <carpeta>
  --name polvora`.
- **Conteo:** `[COMPLETAR]` (objetivo: 20–50 fotos).
- **Etiqueta.** Ninguna, salvo un CSV opcional (`--labels`) con V estimada a
  partir de postes o señales a distancia conocida.
- **Uso.** Brecha de dominio cualitativa y ejemplos de la demo
  (`app/examples/`). Se quitan los metadatos GPS y no se incluyen personas ni
  patentes identificables.

### 3.7 AIR-VIEW (considerado, no usado)

Mourning, Wang y Murray (2025), arXiv:2506.20939: ~147k imágenes de 444
cámaras de la FAA, con visibilidad medida por sensores ASOS/AWOS. Es lo más
cercano a "niebla real con etiqueta real". No se usa por tres razones:

1. ~80 % de las etiquetas son de 9–10+ millas y, según sus autores, solo 28
   imágenes bajan de 1 milla (≈ 1,6 km), así que no cubre las bandas de
   0–200 m;
2. requiere suscripción a IEEE DataPort;
3. sus cámaras son meteorológicas y no viales.

Se cita como trabajo relacionado y como limitación: el proyecto **no** valida
con visibilidad medida.

## 4. Preprocesamiento común

- Las imágenes se reducen **una sola vez** a lado corto 288 px en JPEG y se
  empaquetan en tar (`camanchaca pack`) para copiarlas a `/content` al iniciar
  Colab. Nunca se entrena leyendo desde Drive.
- En Cityscapes se recorta el 20 % inferior (capó).
- Los dos modelos reciben la misma transformación: Resize 256×256 → RandomCrop
  224 + flip horizontal en train y CenterCrop 224 en evaluación. **No** hay
  jitter de color ni de contraste, porque el contraste es la señal.
- Los CSV usan rutas **relativas** a `$CAMANCHACA_DATA` y tienen el esquema
  canónico `relpath, source, group, frame, city, beta, A, v_mor_m, v_eps002_m,
  log10_v, band, band_idx, label_quality, is_clear, split`
  (`src/camanchaca/data/schema.py`).

## 5. Particiones (anti-fuga)

| Fuente | Regla | Unidad de agrupación (`group`) |
|---|---|---|
| RESIDE-OTS | 70/15/15 por escena, eligiendo entre permutaciones la de menor sesgo de bandas | escena |
| Foggy Cityscapes, CS-Fog | por ciudad: test = frankfurt, lindau, munster; val = hanover, jena, tubingen; train = resto | `{ciudad}_{secuencia}` |
| HazeRD, O-HAZE, fotos propias | `ext` (nunca se entrena ni se elige nada con ellas) | escena / par |

- `check_no_leakage` falla si un `group` o un `frame` aparece en más de un
  split (`tests/test_splits.py`).
- El **test está bloqueado**: los comandos que evalúan `test`/`ext` exigen
  `--final`. El LR, el checkpoint y cualquier umbral se eligen solo en val.
- `SPLIT_INFO.json` guarda parámetros, conteos y sha256 de cada CSV y de los
  manifiestos de entrada.

## 6. Distribución y mantenimiento

- **Se versiona:** `data/manifests/*.csv`, `data/splits/v1/*.csv`,
  `SPLIT_INFO.json` y los ejemplos fijos de interpretabilidad
  (`data/splits/v1/interpret_examples.csv`, que crea `camanchaca interpret
  --final`; sin `--final` se usa `interpret_examples_val.csv`). Son derivados abstractos (sin
  píxeles), compatibles con los términos de Cityscapes, RESIDE, HazeRD y
  O-HAZE.
- **No se versiona nunca:** imágenes, recortes, cachés, tars, mapas de
  distancia (`aux/`), la lista "refined" ni checkpoints. Los pesos se
  publican aparte con sha256 (`scripts/fetch_weights.py`, ver
  `app/README.md`).
- Si cambian los datos o los parámetros, se crea `data/splits/v2/` en vez de
  sobrescribir `v1/`.
- Responsable de los datos: Tomás Ignacio Moraga.

## 7. Sesgos y limitaciones transversales

1. **Toda la niebla con etiqueta es sintética y homogénea.** La camanchaca es
   advectiva, heterogénea y marina.
2. **No hay datos chilenos ni de carreteras costeras** con etiqueta.
3. Todo es **diurno**. La noche, la lluvia y el deslumbramiento quedan fuera
   de distribución (la demo advierte posible noche).
4. **Confusión dataset ↔ banda** en las fuentes oficiales. Se mitiga con
   CS-Fog, con el baseline por fuente y con sondas.
5. **Etiquetas cerca de los bordes de banda**: RESIDE β = 0,06 → 49,9 m;
   HazeRD 50/100/200 m si ε_H = 0,05.
6. **RESIDE es "visibilidad equivalente"**, no métrica.
7. Los datos de Cityscapes contienen personas y patentes. Las figuras del
   informe usan pocas imágenes, citadas, y nunca se publican en la demo.

## 8. Citas

RESIDE (Li et al., 2019), Cityscapes (Cordts et al., 2016), Foggy Cityscapes
(Sakaridis et al., 2018), HazeRD (Zhang et al., 2017), O-HAZE (Ancuti et al.,
2018), AIR-VIEW (Mourning et al., 2025), Datasheets for Datasets (Gebru et
al., 2021). BibTeX en [`docs/informe/refs.bib`](informe/refs.bib).
