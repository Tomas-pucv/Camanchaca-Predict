# Sensibilidad al umbral de contraste — conjunto `test`


Convención principal ε = 0,05 (MOR, V = 2,996/β) vs. ε = 0,02 (V = 3,912/β, usada en el avance). Se convierten la verdad y la predicción con el mismo factor (×1,306), por lo que el log10-MAE no cambia y solo cambian las métricas de bandas (los cortes 50/100/200 m quedan fijos en metros). Media ± sd entre semillas.

| Modelo | Macro-F1 ε=0,05 | Macro-F1 ε=0,02 | Exactitud ε=0,05 | Exactitud ε=0,02 | Falso-seguro ε=0,05 | Falso-seguro ε=0,02 | n peligrosos ε=0,05 / 0,02 | log10-MAE (décadas, igual) |
|---|---|---|---|---|---|---|---|---|
| **resnet50** | 0.895 ± 0.0019 | 0.806 ± 0.00946 | 0.924 ± 0.00138 | 0.854 ± 0.00514 | 0.00862 ± 0.00135 | 0.0542 ± 0.00335 | 2592 / 2536 | 0.0619 ± 0.000846 |
| **deit_s** | 0.915 ± 0.00274 | 0.863 ± 0.0028 | 0.939 ± 0.00227 | 0.894 ± 0.00217 | 0.00604 ± 0.000589 | 0.0557 ± 0.00299 | 2592 / 2536 | 0.0456 ± 0.000705 |
| **majority_band** | 0.156 | 0.137 | 0.455 | 0.378 | 0 | 0 | 2592 / 2536 | 0.578 |
| **mean_by_source** | 0.442 | 0.413 | 0.65 | 0.58 | 0.163 | 0.144 | 2592 / 2536 | 0.251 |
| **mean_global** | 0.0633 | 0.0617 | 0.145 | 0.141 | 1 | 1 | 2592 / 2536 | 0.492 |
| **physics_hgb** | 0.736 | 0.721 | 0.801 | 0.772 | 0.039 | 0.0591 | 2592 / 2536 | 0.137 |
| **physics_ridge** | 0.588 | 0.612 | 0.665 | 0.659 | 0.0883 | 0.161 | 2592 / 2536 | 0.21 |
