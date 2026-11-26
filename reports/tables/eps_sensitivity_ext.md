# Sensibilidad al umbral de contraste — conjunto `ext`


Convención principal ε = 0,05 (MOR, V = 2,996/β) vs. ε = 0,02 (V = 3,912/β, usada en el avance). Se convierten la verdad y la predicción con el mismo factor (×1,306), por lo que el log10-MAE no cambia y solo cambian las métricas de bandas (los cortes 50/100/200 m quedan fijos en metros). Media ± sd entre semillas.

| Modelo | Macro-F1 ε=0,05 | Macro-F1 ε=0,02 | Exactitud ε=0,05 | Exactitud ε=0,02 | Falso-seguro ε=0,05 | Falso-seguro ε=0,02 | n peligrosos ε=0,05 / 0,02 | log10-MAE (décadas, igual) |
|---|---|---|---|---|---|---|---|---|
| **resnet50** | 0.0992 ± 0.0133 | 0.129 ± 0.00254 | 0.204 ± 0.0077 | 0.2 ± 3.40e-17 | 0 ± 0 | 0 ± 0 | 30 / 15 | 0.817 ± 0.0228 |
| **deit_s** | 0.0897 ± 0.00123 | 0.134 ± 0.0111 | 0.2 ± 3.40e-17 | 0.204 ± 0.0077 | 0 ± 0 | 0 ± 0 | 30 / 15 | 0.839 ± 0.0212 |
| **majority_band** | 0.0833 | 0.111 | 0.2 | 0.2 | 0 | 0 | 30 / 15 | 0.791 |
| **mean_by_source** | 0.0833 | 0.19 | 0.2 | 0.4 | 1 | 1 | 30 / 15 | 0.429 |
| **mean_global** | 0.0833 | 0.19 | 0.2 | 0.4 | 1 | 1 | 30 / 15 | 0.429 |
| **physics_hgb** | 0.193 | 0.306 | 0.267 | 0.28 | 0 | 0 | 30 / 15 | 0.592 |
| **physics_ridge** | 0.361 | 0.543 | 0.387 | 0.467 | 0 | 0 | 30 / 15 | 0.422 |
