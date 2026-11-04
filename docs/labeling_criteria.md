# Criterio de etiquetado: de β a visibilidad MOR en metros

> Versión final (octubre 2026). Reemplaza el criterio del avance (ε = 0,02), que
> se conserva solo como análisis de sensibilidad. Implementación:
> `src/camanchaca/labeling/koschmieder.py` (constantes y conversiones) y
> `src/camanchaca/labeling/fog_synthesis.py` (síntesis de niebla). Pruebas:
> `tests/test_koschmieder.py`, `tests/test_fog_synthesis.py`.

## 0. Resumen

| Decisión | Valor |
|---|---|
| Etiqueta principal | **V MOR** (Meteorological Optical Range, OMM No. 8): `V = −ln(0,05)/β = 2,996/β` [m] |
| Ancla común entre datasets | el coeficiente de extinción **β** [1/m]; cada fuente se convierte con la misma fórmula |
| Sensibilidad | `V(ε = 0,02) = 3,912/β` (columna `v_eps002_m`), solo para reportar cuánto cambian las conclusiones |
| Objetivo del modelo | `log10(V MOR)`, estandarizado con la media y desviación de train |
| Bandas de seguridad | 50 / 100 / 200 m, justificadas con la distancia de detención (SSD) |
| Calidad de la etiqueta | `label_quality` ∈ {`metrica`, `equivalente`, `ninguna`} por fuente |

## 1. Qué cambió respecto del avance

1. **El umbral pasa de ε = 0,02 a ε = 0,05 (MOR).** En el avance se dijo que
   ε = 0,02 era "conservador". Es al revés: `3,912/2,996 = 1,306`, o sea, con
   ε = 0,02 la visibilidad sale **~31 % mayor** para el mismo β. Sobreestimar
   la visibilidad es justamente el error peligroso en seguridad vial. El MOR
   es el estándar de la OMM (OMM, 2018) y la convención que usan los
   autores de Foggy Cityscapes (Sakaridis et al., 2018), así que es la opción
   coherente y la conservadora.
2. **Ya no es cierto que "ningún dataset trae metros".** Foggy Cityscapes
   publica β en 1/m renderizado con distancia estéreo métrica, HazeRD
   especifica la niebla directamente como visibilidad en metros sobre
   profundidad métrica (Zhang et al., 2017) y CS-Fog (sección 8) la generamos
   nosotros con distancia estéreo métrica. Solo RESIDE usa profundidad
   estimada, y por eso su etiqueta se declara **equivalente** (sección 4).
3. **La redacción de las bandas se corrigió.** El texto anterior contradecía
   su propia tabla de SSD. Por ejemplo, decía "≥ 200 m: margen incluso a
   120 km/h", pero SSD(120 km/h) ≈ 225 m. La sección 6 define la velocidad
   segura como la mayor velocidad cuya SSD cabe en V.

## 2. Marco físico

**Ley de Koschmieder** (Koschmieder, 1924; Middleton, 1952). Un
objeto negro visto contra el cielo del horizonte a distancia `d`, en una
atmósfera homogénea con coeficiente de extinción β [1/m], tiene contraste
aparente

```
C(d) = exp(−β·d)
```

La visibilidad es la distancia a la que ese contraste cae a un umbral ε:

```
V = −ln(ε) / β
```

**Definición MOR (OMM No. 8).** El *Meteorological Optical Range* es la
distancia que debe recorrer un haz colimado para que su flujo luminoso se
reduzca al 5 % del valor inicial. Esto equivale a ε = 0,05 en la expresión
anterior:

| Umbral | Fórmula | β = 0,02 1/m | Uso en el proyecto |
|---|---|---|---|
| ε = 0,05 (MOR, OMM) | `V = 2,996/β` | 149,8 m | **etiqueta principal** (`v_mor_m`) |
| ε = 0,02 (avance) | `V = 3,912/β` | 195,6 m | sensibilidad (`v_eps002_m`); +30,6 % |

**Modelo atmosférico de dispersión (ASM).** RESIDE, Foggy Cityscapes, HazeRD
y CS-Fog generan la niebla con

```
I(x) = J(x)·t(x) + A·(1 − t(x)),     t(x) = exp(−β·d(x))
```

donde `J` es la imagen limpia, `A` la luz atmosférica y `d(x)` la distancia
cámara–punto. Para un medio homogéneo, el β del ASM es el mismo β de
Koschmieder. Por eso un β conocido por imagen basta para fijar la
visibilidad, **siempre que `d` esté en metros**.

## 3. Construcción de la etiqueta (todas las fuentes)

Para cada imagen con niebla:

1. Se obtiene **β** de la fuente: nombre de archivo en RESIDE, nombre del
   paquete en Foggy Cityscapes, `β = −ln(ε_HazeRD)/V_nominal` en HazeRD y el β
   muestreado en CS-Fog.
2. `v_mor_m = 2,996/β` y `v_eps002_m = 3,912/β`
   (`koschmieder.visibility_from_beta`, `EPS_MOR`, `EPS_LEGACY`).
3. `log10_v = log10(v_mor_m)` es el objetivo de regresión. La escala
   logarítmica estabiliza el rango de ~15 a ~1000 m y hace que un error
   relativo pese lo mismo a 30 m que a 600 m.
4. `band`/`band_idx` sale de `np.digitize(V, [50, 100, 200])` (borde
   izquierdo cerrado: V = 100 m cae en `precaucion`).
5. `label_quality` según la fuente (sección 4).

El esquema completo de columnas está en `src/camanchaca/data/schema.py` y la
explicación por fuente en [`docs/data_card.md`](data_card.md).

## 4. Visibilidad "equivalente" vs "métrica"

| `label_quality` | Fuentes | Por qué |
|---|---|---|
| `metrica` | Foggy Cityscapes, CS-Fog, HazeRD | `d` en metros (estéreo calibrado en Cityscapes; mapas métricos en HazeRD), así que `V = 2,996/β` es una distancia física de la escena |
| `equivalente` | RESIDE-OTS | `d` la estimó una red monocular sobre fotos de internet: es **relativa** y sin escala métrica garantizada. El β del nombre es "1/m" solo nominalmente, y V es la visibilidad que tendría una escena cuya profundidad fuera la estimada |
| `ninguna` | O-HAZE, fotos propias (La Pólvora) | niebla real sin medición de visibilidad; uso cualitativo o de orden por pares |

Consecuencias:

- Las métricas se reportan **por fuente y por `label_quality`**. Un acierto
  en RESIDE no valida metros reales.
- La única validación métrica externa es HazeRD (sintético con profundidad
  métrica). **No hay niebla real con visibilidad medida** en el proyecto
  (sección 10).

## 5. Tabla de conversión por dataset

Bandas: `critico` < 50 m ≤ `alto_riesgo` < 100 m ≤ `precaucion` < 200 m ≤
`aceptable`. Los valores están calculados con `koschmieder.py`.

**RESIDE-OTS** (7 β; profundidad estimada → `equivalente`):

| β [1/m] | V MOR [m] | Banda (MOR) | V ε=0,02 [m] | Banda (ε=0,02) |
|---|---|---|---|---|
| 0,04 | 74,9 | alto_riesgo | 97,8 | alto_riesgo |
| 0,06 | **49,9** | critico (0,1 m bajo el borde) | 65,2 | alto_riesgo |
| 0,08 | 37,4 | critico | 48,9 | critico |
| 0,10 | 30,0 | critico | 39,1 | critico |
| 0,12 | 25,0 | critico | 32,6 | critico |
| 0,16 | 18,7 | critico | 24,5 | critico |
| 0,20 | 15,0 | critico | 19,6 | critico |

La lista de 7 β viene de la literatura y debe confirmarse con
`camanchaca prep reside --check-only`, que lista los β y A únicos.

**Foggy Cityscapes oficial** (3 β; distancia estéreo → `metrica`):

| β [1/m] | V MOR [m] | Banda (MOR) | V ε=0,02 [m] | Banda (ε=0,02) |
|---|---|---|---|---|
| 0,005 | 599,1 (≈ 600) | aceptable | 782,4 | aceptable |
| 0,01 | 299,6 (≈ 300) | aceptable | 391,2 | aceptable |
| 0,02 | 149,8 (≈ 150) | precaucion | 195,6 | precaucion (4,4 m bajo el borde) |

**HazeRD** (5 niveles nominales; profundidad métrica → `metrica`). HazeRD
especifica la niebla como visibilidad nominal `V_nom`. El β real depende del
umbral con el que su script la convierte (`β = −ln(ε_H)/V_nom`), y ese umbral
**está por confirmar** leyendo `demo_simu_haze.m`. Fuentes secundarias indican
ε_H = 0,02, que es el valor por defecto en `configs/data/hazerd.yaml`.

| V nominal [m] | Si ε_H = 0,02 (por defecto): β [1/m] → V MOR [m] (banda) | Si ε_H = 0,05: V MOR [m] (banda) |
|---|---|---|
| 50 | 0,0782 → 38,3 (critico) | 50 (alto_riesgo, justo en el borde) |
| 100 | 0,0391 → 76,6 (alto_riesgo) | 100 (precaucion, justo en el borde) |
| 200 | 0,0196 → 153,2 (precaucion) | 200 (aceptable, justo en el borde) |
| 500 | 0,0078 → 382,9 (aceptable) | 500 (aceptable) |
| 1000 | 0,0039 → 765,8 (aceptable) | 1000 (aceptable) |

**CS-Fog** (niebla propia; distancia estéreo → `metrica`): V MOR
log-uniforme en [15, 1000] m, es decir β ∈ [0,0030; 0,1997] 1/m. Por
construcción, la fracción esperada por banda es 28,7 % `critico`, 16,5 %
`alto_riesgo`, 16,5 % `precaucion` y 38,3 % `aceptable`. Los conteos reales
están en `reports/tables/data_counts.md`.

**O-HAZE y fotos propias**: sin β ni V (`ninguna`).

**Cobertura resultante.** Con solo los datasets oficiales, RESIDE cubre
15–75 m y Foggy Cityscapes 150–600 m, así que **la fuente determina la banda**
(sección 7.2). CS-Fog llena el hueco 75–150 m y cubre las cuatro bandas
dentro de un mismo dominio visual.

## 6. Bandas de seguridad y distancia de detención

Distancia de detención en piso mojado, con tiempo de percepción-reacción
t_r = 2,5 s (valor de diseño habitual (AASHTO, 2018)) y fricción
μ = 0,4:

```
SSD(v) = v·t_r + v² / (2·μ·g),     v en m/s, g = 9,81 m/s²
```

| Velocidad | 60 km/h | 80 km/h | 100 km/h | 110 km/h | 120 km/h |
|---|---|---|---|---|---|
| SSD | 77,1 m | 118,5 m | 167,8 m | 195,4 m | 224,9 m |

La **velocidad segura** para una visibilidad V es la mayor `v` con
`SSD(v) ≤ V` (`koschmieder.safe_speed_kmh`). Para los bordes de banda:

| V | 50 m | 100 m | 150 m | 200 m |
|---|---|---|---|---|
| Velocidad segura | 44,3 km/h | 71,5 km/h | 93,2 km/h | 111,6 km/h |

| Banda | Rango | Lectura operativa (corregida) |
|---|---|---|
| `critico` | V < 50 m | No es posible detenerse dentro de lo visible ni a ~44 km/h |
| `alto_riesgo` | 50–100 m | Velocidad segura máxima entre ~44 km/h (V = 50 m) y ~71 km/h (V = 100 m) |
| `precaucion` | 100–200 m | Velocidad segura máxima entre ~71 km/h (V = 100 m) y ~111 km/h (V = 200 m) |
| `aceptable` | ≥ 200 m | La SSD de ~111 km/h cabe en lo visible (cubre 80–110 km/h) |

`DANGER_CUT_M = 100 m` define el evento "peligroso" para la tasa de
falso-seguro: V real < 100 m predicha ≥ 100 m. Los bordes **no se ajustan a
los datos**. Vienen de la física del frenado y se mantienen aunque haya
etiquetas cerca de ellos (sección 7). La velocidad segura que muestra la demo
es **ilustrativa**: supone niebla homogénea, conductor atento y piso mojado
estándar.

## 7. Riesgos propios del etiquetado

### 7.1 Etiquetas en el borde de banda

- RESIDE β = 0,06 da **49,9 m**: `critico` por 0,1 m. Con ε = 0,02 sería
  `alto_riesgo`. Es el único nivel de entrenamiento que cambia de banda entre
  convenciones.
- Foggy Cityscapes β = 0,02 da 149,8 m, en el centro de `precaucion` con MOR.
  Con ε = 0,02 quedaba a 4,4 m del borde de 200 m.
- HazeRD: si ε_H = 0,05, tres de sus cinco niveles caen **exactamente** sobre
  los bordes (50/100/200 m), y la asignación depende del redondeo de punto
  flotante.

Mitigación: las métricas **continuas** (log10-MAE, MdAPE, δ<1,25, Spearman)
son las principales y las de banda son secundarias. Se reporta la tabla de
sensibilidad a ε (`reports/tables/eps_sensitivity_<set>.md`), el error por
banda real (`reports/figures/error_by_band_<set>.png`) y la matriz de
confusión. Nunca se mueven los bordes para acomodar los datos.

### 7.2 Confusión dataset ↔ banda

Con solo RESIDE (< 100 m) y Foggy Cityscapes (≥ 150 m), reconocer el dataset
(estilo de la foto, proporción, ciudad alemana) basta para acertar la banda.
Se mitiga con:

1. CS-Fog, que pone las cuatro bandas en el mismo dominio;
2. el baseline `mean_by_source`, que mide cuánto se gana solo sabiendo la
   fuente;
3. la sonda lineal de fuente sobre features congeladas
   (`camanchaca probes`);
4. la selección de modelo por la **media de log10-MAE por fuente** en val;
5. el Spearman dentro de cada fuente.

### 7.3 Naturaleza ordinal

Los datasets oficiales solo tienen ~10 valores distintos de V (7 de RESIDE y
3 de Foggy Cityscapes), así que la regresión sobre ellos se parece a una
clasificación ordinal. Por eso:

- CS-Fog aporta V continua;
- ambos modelos llevan la **misma** cabeza ordinal **CORN** (Shi et al., 2023): 3
  logits condicionales `P(banda > k | banda > k−1)`, consistentes en rango, sin
  pesos por clase, junto a la regresión de log10 V;
- se reportan las bandas derivadas de la regresión (`band_pred_reg`) y las de
  la cabeza ordinal (`band_pred_ord`), con macro-F1 sobre una lista fija de
  bandas, F1 por clase, matriz de confusión y MAE ordinal.

## 8. CS-Fog: niebla propia con distancia métrica

Implementación: `camanchaca prep cs-fog` (`src/camanchaca/data/manifests/cs_fog.py`,
`src/camanchaca/data/cityscapes_geometry.py`). Requiere `leftImg8bit`, `disparity`, `camera`
y `gtFine` de Cityscapes (Cordts et al., 2016).

1. **Disparidad.** `p` = PNG de 16 bits; `p ≤ 1` es inválido;
   `d = (p − 1)/256` px.
2. **Profundidad y distancia.** `Z = B·f_x/d` (B = baseline del
   `camera.json` de cada imagen, en m). Distancia euclídea cámara–punto, que
   es la longitud de aire que atraviesa la luz:
   `l = Z·√(1 + ((u − u0)/f_x)² + ((v − v0)/f_y)²)`.
3. **Cielo y huecos.** Píxeles con `labelId = 23` (cielo): `l = ∞`, así que
   `t = 0` y el píxel toma el color A. Los demás huecos de disparidad se
   rellenan con el valor del vecino válido más cercano, usando la transformada
   de distancia euclídea (`scipy.ndimage.distance_transform_edt`).
4. **Geometría.** Se recorta el 20 % inferior (capó del vehículo propio, igual
   que en Foggy Cityscapes) y se redimensiona a lado corto 288 px: imagen
   bicúbica, distancia y labels por vecino más cercano. La distancia se guarda
   en float16 (`aux/cs/{stem}_dist.npy`) y la niebla se renderiza con esa
   distancia ya redondeada, para que el archivo auxiliar sea exactamente el
   usado.
5. **Muestreo.** K = 2 visibilidades por frame, `V ~ log-U[15, 1000]` m y
   `A ~ U[0,75; 0,95]` (escalar, gris), con un generador sembrado por
   `sha256(semilla, stem)`. Es determinista y no depende del orden de
   procesamiento.
6. **Render.** `β = 2,996/V`; `I = J·t + A·(1 − t)`, `t = exp(−β·l)`
   (`fog_synthesis.render_fog`).

Límites conocidos:

- El error de la distancia estéreo crece con `Z²`. Con `B·f_x ≈ 480 px·m`, una
  disparidad de 4,8 px corresponde a ~100 m y una de 1 px a ~480 m, así que
  las regiones lejanas son ruidosas.
- La niebla es homogénea y A es un escalar.
- El render es en sRGB (no en RGB lineal).

Todo esto se declara como parte de la brecha sintético → real.

## 9. Validación de las etiquetas

| Chequeo | Qué muestra | Cómo |
|---|---|---|
| Versión de RESIDE | que los nombres siguen `id_A_beta` (≥ 90 %), los β/A únicos y ~35 variantes por escena; aborta ante reempaques tipo RESIDE-6K, que perdieron β | `camanchaca prep reside --raw … --check-only` |
| Consistencia perceptual | el canal oscuro medio (He et al., 2011) debe crecer con β **dentro de cada fuente** (Spearman > 0). Es **consistencia, no prueba**: el canal oscuro también depende del contenido y de A | columna `dc_mean` de `runs/<exp>/baselines/features_<set>.csv` (lo genera `camanchaca baselines`) vs `beta` |
| Exactitud del render propio | renderizar y luego invertir el ASM con `J`, `A` y `d` conocidos recupera β | `tests/test_fog_synthesis.py` |
| Orden en niebla real | V̂(hazy) < V̂(GT) en los 45 pares de O-HAZE | `reports/tables/ohaze_pairs.md` (`camanchaca evaluate --final`) |
| Test métrico externo | error en HazeRD, nunca usado para entrenar ni para elegir | `reports/tables/metrics_ext.md` |

## 10. Limitaciones

- **Sintético → real.** Toda la niebla con etiqueta es sintética y homogénea.
  La camanchaca es advectiva, heterogénea, marina y con gradiente vertical.
  O-HAZE es niebla real de máquina, sin visibilidad medida. AIR-VIEW
  (Mourning et al., 2025) sí tiene visibilidad medida, pero casi todas sus
  etiquetas superan 1,6 km y requiere suscripción. Queda como trabajo
  relacionado y limitación.
- **RESIDE es equivalente**, no métrico.
- **Convención de HazeRD por confirmar** (sección 5). Se reportan ambas
  lecturas si no se puede confirmar.
- **Pocos niveles discretos** en los datasets oficiales y **etiquetas en los
  bordes** (sección 7).
- **Sin datos chilenos con etiqueta.** Las fotos de La Pólvora son solo
  cualitativas.

## 11. Referencias

Claves de [`docs/informe/refs.bib`](informe/refs.bib): `koschmieder1924theorie`,
`middleton1952vision`, `wmo2018guide`, `he2011single`, `li2019benchmarking`
(RESIDE), `sakaridis2018semantic` (Foggy Cityscapes), `cordts2016cityscapes`,
`zhang2017hazerd`, `ancuti2018ohaze`, `mourning2025airview`, `shi2023corn`,
`aashto2018policy`.
