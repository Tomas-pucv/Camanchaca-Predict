# Sensibilidad al umbral de contraste — conjunto `val`


Convención principal ε = 0,05 (MOR, V = 2,996/β) vs. ε = 0,02 (V = 3,912/β, usada en el avance). Se convierten la verdad y la predicción con el mismo factor (×1,306), por lo que el log10-MAE no cambia y solo cambian las métricas de bandas (los cortes 50/100/200 m quedan fijos en metros). Media ± sd entre semillas.

| Modelo | Macro-F1 ε=0,05 | Macro-F1 ε=0,02 | Exactitud ε=0,05 | Exactitud ε=0,02 | Falso-seguro ε=0,05 | Falso-seguro ε=0,02 | n peligrosos ε=0,05 / 0,02 | log10-MAE (décadas, igual) |
|---|---|---|---|---|---|---|---|---|
| **resnet50** | 0.898 ± 0.00343 | 0.814 ± 0.0103 | 0.924 ± 0.00275 | 0.858 ± 0.00555 | 0.00694 ± 0.000772 | 0.054 ± 0.00237 | 2592 / 2536 | 0.0656 ± 0.00137 |
| **deit_s** | 0.912 ± 0.00223 | 0.865 ± 0.00431 | 0.934 ± 0.00242 | 0.892 ± 0.00334 | 0.00502 ± 0.00204 | 0.0539 ± 0.00254 | 2592 / 2536 | 0.0498 ± 0.000687 |
| **majority_band** | 0.161 | 0.141 | 0.474 | 0.393 | 0 | 0 | 2592 / 2536 | 0.556 |
| **mean_by_source** | 0.44 | 0.418 | 0.656 | 0.587 | 0.163 | 0.144 | 2592 / 2536 | 0.246 |
| **mean_global** | 0.0601 | 0.0603 | 0.137 | 0.137 | 1 | 1 | 2592 / 2536 | 0.492 |
| **physics_hgb** | 0.727 | 0.709 | 0.793 | 0.759 | 0.0529 | 0.0797 | 2592 / 2536 | 0.137 |
| **physics_ridge** | 0.565 | 0.583 | 0.642 | 0.628 | 0.117 | 0.194 | 2592 / 2536 | 0.222 |
