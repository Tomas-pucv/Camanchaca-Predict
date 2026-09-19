# Data card

## Origen

| Dataset | Fuente | Contenido | Licencia / acceso |
|---|---|---|---|
| RESIDE-OTS | Li et al. 2018 (espejos en Kaggle/HuggingFace) | ~7k–30k imágenes outdoor con niebla sintética + escenas limpias | Solo investigación |
| SOTS-OUT | RESIDE (test oficial) | 500 imágenes de test | Solo investigación |
| O-HAZE | NTIRE 2018 (registro con correo institucional) | 45 escenas reales: niebla real generada con máquina de humo + referencia limpia | Solo investigación |

## Estructura local esperada

```
data/
├── raw/                      # NO se versiona en git
│   ├── reside_ots/           # descargado por notebook 01
│   ├── ohaze/{hazy,clear}/   # organizado por scripts/download_ohaze.py
│   └── demo_clear/           # fotos propias para el modo demo (opcional)
├── processed/
│   ├── reside_manifest.csv   # notebook 01: rutas + beta/A/escena
│   └── labels.csv            # notebook 02: + V, log10_v, banda
└── splits/                   # SÍ se versiona (CSV pequeños)
    ├── train.csv / val.csv / test.csv
    └── class_weights.json
```

## Reglas

- **Nunca** subir imágenes a git (`.gitignore` lo bloquea).
- Los CSV de `data/splits/` y `class_weights.json` sí se commitean para
  reproducibilidad exacta de experimentos.
- Toda métrica y figura generada va a `reports/` y también a Drive
  (`/content/drive/MyDrive/camanchaca/`) desde los notebooks.

## Reproducibilidad

- Semilla global 42 (notebooks y `configs/`).
- Splits regenerables con `make splits` (misma semilla → mismo split).
- `scripts/verify_env.py` documenta versiones del entorno de cada corrida.
