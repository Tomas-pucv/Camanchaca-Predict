# `data/manifests/` — manifiestos por fuente

Un CSV por fuente de datos, generado por `camanchaca prep <fuente>`. Cada fila es
una imagen con el **esquema canónico** de `src/camanchaca/data/schema.py`
(`relpath, source, group, frame, city, beta, A, v_mor_m, v_eps002_m, log10_v,
band, band_idx, label_quality, is_clear, split`) más algunas columnas extra
según la fuente. `camanchaca split` los combina en `data/splits/v1/`.

| Archivo | Comando | Contenido |
|---|---|---|
| `reside_ots.csv` | `camanchaca prep reside --raw <OTS>` | RESIDE-OTS oficial (`id_A_beta`), una A por (escena, beta); `label_quality = equivalente` |
| `foggy_cs.csv` | `camanchaca prep foggy-cityscapes --zip <foggy.zip>` | Foggy Cityscapes oficial, beta 0.005/0.01/0.02 (V MOR 600/300/150 m); extra `refined` si se da `--refined-list` |
| `cs_fog.csv` | `camanchaca prep cs-fog --cs-root <cityscapes>` | niebla propia, V MOR log-uniforme 15-1000 m; extras `aux_dist`, `aux_labels`, `clear_relpath` |
| `hazerd.csv` | `camanchaca prep hazerd --raw <HazeRD>` | test externo métrico; extras `v_nominal_m`, `hazerd_eps`, `hazerd_mode` |
| `ohaze.csv` | `camanchaca prep ohaze --raw <O-HAZE.zip>` | pares reales hazy/GT sin etiqueta (GT con `is_clear = True`) |
| `folder_<NOMBRE>.csv` | `camanchaca prep folder --raw <fotos> --name <NOMBRE>` | fotos propias (p. ej. La Pólvora); etiqueta opcional con `--labels` |
| `cars_gt.csv`, `cars_images.csv` | `camanchaca prep cars --cs-root <cityscapes>` | análisis de autos (NO son manifiestos de imágenes: `split` los excluye) |

Defaults de cada fuente: `configs/data/<fuente>.yaml` (las banderas tienen prioridad).

## Reglas

- **Rutas relativas** a la raíz de datos (`$CAMANCHACA_DATA`, por defecto `data/`):
  los mismos CSV funcionan en Colab, local y CI.
- La etiqueta es la **V MOR** (umbral de contraste 5 %): `v_mor_m = 2.996 / beta`.
  `beta` es el ancla común; `v_eps002_m` (2 %) se guarda solo como sensibilidad.
- Los CSV son deterministas (orden por `relpath`, semillas por frame/escena):
  regenerarlos con los mismos datos y parámetros produce los mismos bytes.

## Licencias: qué se versiona

Estos CSV son **derivados abstractos** (nombres de archivo, coeficientes y
etiquetas numéricas; ningún píxel), compatibles con los términos de Cityscapes,
RESIDE, HazeRD y O-HAZE. **Nunca** se suben imágenes, recortes, cachés
redimensionados, mapas de distancia (`aux/`), tars ni la lista "refined" de
Foggy Cityscapes (su licencia prohíbe redistribuirla: se descarga en tiempo de
ejecución). `.gitignore` ya excluye `data/images/`, `data/aux/`, `*.tar`, `*.zip`
y `*.npy`.
