# Barrido de LR — experimento `main`

Misma grilla y mismo presupuesto para todos los modelos (semilla 0, 1200 pasos). Criterio: `val_log10_mae_macro_source` en val (menor es mejor; empate -> menor LR).

| Modelo | LR backbone | LR cabeza | Mejor valor | Paso | Pasos | Elegido |
|---|---|---|---|---|---|---|
| resnet50 | 3e-05 | 0.0003 | 0.1221 | 900 | 1200 |  |
| resnet50 | 0.0001 | 0.001 | 0.0893 | 900 | 1200 |  |
| resnet50 | 0.0003 | 0.003 | 0.0726 | 1200 | 1200 | **sí** |
| deit_s | 3e-05 | 0.0003 | 0.0566 | 1200 | 1200 |  |
| deit_s | 0.0001 | 0.001 | 0.0522 | 1200 | 1200 | **sí** |
| deit_s | 0.0003 | 0.003 | 0.0523 | 1200 | 1200 |  |
