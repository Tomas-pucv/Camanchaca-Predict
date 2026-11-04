# `data/splits/v1/` — particiones versionadas

Generadas por:

```bash
camanchaca split --manifests "data/manifests/*.csv" --out data/splits/v1
```

| Archivo | Contenido |
|---|---|
| `train.csv`, `val.csv`, `test.csv` | imágenes de entrenamiento, validación (selección de modelo y LR) y test (bloqueado: solo con `--final`) |
| `ext.csv` | evaluación externa: HazeRD, O-HAZE y fotos propias (nunca se entrena con ellas) |
| `SPLIT_INFO.json` | parámetros (semilla, fracciones, ciudades), conteos por split x fuente x banda, sha256 de cada CSV y de los manifiestos de entrada |
| `interpret_examples.csv` | conjunto FIJO de ejemplos para interpretabilidad (lo crea `camanchaca interpret`) |

Además, `split` escribe `reports/tables/data_counts.{csv,md}` (n imágenes y n
grupos por conjunto x fuente x banda) para la data card y el informe.

## Reglas de partición (anti-fuga)

- **Cityscapes** (`foggy_cs`, `cs_fog`): por CIUDAD. Test = ciudades del val
  oficial (frankfurt, lindau, munster); val = 3 ciudades fijas del train oficial
  (hanover, jena, tubingen, ~15 %); train = el resto. Todas las variantes de un
  frame (3 beta oficiales + K beta propios) quedan en el mismo split. Las
  ciudades se leen de `configs/data/foggy_cityscapes.yaml` (o `--val-cities` /
  `--test-cities`).
- **RESIDE-OTS** (y el sintético de las pruebas): 70/15/15 agrupado por escena
  (`group`), eligiendo entre varias permutaciones la de menor sesgo de bandas.
- **HazeRD, O-HAZE, carpetas propias**: `ext`.
- `check_no_leakage` falla si un `group` o un `frame` aparece en más de un split.

Las rutas son relativas a `$CAMANCHACA_DATA`; las imágenes se restauran en
Colab con `camanchaca unpack --tar <Drive>/<fuente>.tar`. Los CSV y el JSON se
versionan (derivados abstractos, sin píxeles); las imágenes **nunca**.
Si cambian los datos o los parámetros, crear `v2/` en vez de sobrescribir `v1/`
cuando ya haya resultados reportados con `v1`.
