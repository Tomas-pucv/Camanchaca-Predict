# Diferencia pareada — conjunto `ext`


Δ = **resnet50 − deit_s** del modelo promediado entre semillas, con los MISMOS grupos remuestreados para ambos (bootstrap agrupado pareado, B = 2000). IC 95 % percentil; p bilateral bootstrap = 2·min(P(Δ*≤0), P(Δ*≥0)). En log10-MAE y falso-seguro un Δ negativo favorece a resnet50; en macro-F1, un Δ positivo favorece a resnet50. 'Sorteos indefinidos' = remuestreos sin casos peligrosos (falso-seguro no definido).

| Subconjunto | Métrica | Δ (resnet50 − deit_s) | IC 95 % | p | Grupos | Sorteos indefinidos |
|---|---|---|---|---|---|---|
| Global | log10-MAE (décadas) | -0.0228 | [-0.0434, -0.00327] | 0.022 | 15 | 0/2000 |
| Global | Tasa falso-seguro (fracción) | 0 | [0, 0] | 1 | 15 | 0/2000 |
| Global | Macro-F1 bandas desde V̂ | 0.00952 | [0, 0.0209] | 0.052 | 15 | 0/2000 |
| Fuente: hazerd | log10-MAE (décadas) | -0.0228 | [-0.0452, -0.00252] | 0.026 | 15 | 0/2000 |
| Fuente: hazerd | Tasa falso-seguro (fracción) | 0 | [0, 0] | 1 | 15 | 0/2000 |
| Fuente: hazerd | Macro-F1 bandas desde V̂ | 0.00952 | [0.000319, 0.0209] | 0.047 | 15 | 0/2000 |
| Calidad de etiqueta: metrica | log10-MAE (décadas) | -0.0228 | [-0.044, -0.00308] | 0.016 | 15 | 0/2000 |
| Calidad de etiqueta: metrica | Tasa falso-seguro (fracción) | 0 | [0, 0] | 1 | 15 | 0/2000 |
| Calidad de etiqueta: metrica | Macro-F1 bandas desde V̂ | 0.00952 | [0, 0.0215] | 0.053 | 15 | 0/2000 |
