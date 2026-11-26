# Diferencia pareada — conjunto `test`


Δ = **resnet50 − deit_s** del modelo promediado entre semillas, con los MISMOS grupos remuestreados para ambos (bootstrap agrupado pareado, B = 2000). IC 95 % percentil; p bilateral bootstrap = 2·min(P(Δ*≤0), P(Δ*≥0)). En log10-MAE y falso-seguro un Δ negativo favorece a resnet50; en macro-F1, un Δ positivo favorece a resnet50. 'Sorteos indefinidos' = remuestreos sin casos peligrosos (falso-seguro no definido).

| Subconjunto | Métrica | Δ (resnet50 − deit_s) | IC 95 % | p | Grupos | Sorteos indefinidos |
|---|---|---|---|---|---|---|
| Global | log10-MAE (décadas) | 0.0163 | [0.0119, 0.0193] | < 0.0005 | 545 | 0/2000 |
| Global | Tasa falso-seguro (fracción) | 0.00257 | [0.000571, 0.0051] | 0.017 | 545 | 0/2000 |
| Global | Macro-F1 bandas desde V̂ | -0.0207 | [-0.0267, -0.012] | < 0.0005 | 545 | 0/2000 |
| Fuente: cs_fog | log10-MAE (décadas) | 0.00682 | [0.00265, 0.00877] | 0.001 | 235 | 0/2000 |
| Fuente: cs_fog | Tasa falso-seguro (fracción) | 0.00474 | [-0.0023, 0.0243] | 0.254 | 235 | 0/2000 |
| Fuente: cs_fog | Macro-F1 bandas desde V̂ | -0.0276 | [-0.0376, -0.00128] | 0.039 | 235 | 0/2000 |
| Fuente: foggy_cs | log10-MAE (décadas) | 0.0298 | [0.0242, 0.032] | < 0.0005 | 235 | 0/2000 |
| Fuente: foggy_cs | Tasa falso-seguro (fracción) | — | — | — | 235 | 2000/2000 |
| Fuente: foggy_cs | Macro-F1 bandas desde V̂ | -0.0101 | [-0.0132, -0.0021] | 0.01 | 235 | 0/2000 |
| Fuente: reside_ots | log10-MAE (décadas) | 0.0113 | [0.00753, 0.0149] | < 0.0005 | 310 | 0/2000 |
| Fuente: reside_ots | Tasa falso-seguro (fracción) | 0.00215 | [0.000307, 0.0043] | 0.023 | 310 | 0/2000 |
| Fuente: reside_ots | Macro-F1 bandas desde V̂ | -0.0238 | [-0.0362, -0.0108] | 0.001 | 310 | 0/2000 |
| Calidad de etiqueta: equivalente | log10-MAE (décadas) | 0.0113 | [0.00747, 0.0152] | < 0.0005 | 310 | 0/2000 |
| Calidad de etiqueta: equivalente | Tasa falso-seguro (fracción) | 0.00215 | [0.000307, 0.00445] | 0.02 | 310 | 0/2000 |
| Calidad de etiqueta: equivalente | Macro-F1 bandas desde V̂ | -0.0238 | [-0.0365, -0.0108] | < 0.0005 | 310 | 0/2000 |
| Calidad de etiqueta: metrica | log10-MAE (décadas) | 0.0206 | [0.0159, 0.0224] | < 0.0005 | 235 | 0/2000 |
| Calidad de etiqueta: metrica | Tasa falso-seguro (fracción) | 0.00474 | [-0.00246, 0.0255] | 0.289 | 235 | 0/2000 |
| Calidad de etiqueta: metrica | Macro-F1 bandas desde V̂ | -0.017 | [-0.025, 0.00381] | 0.119 | 235 | 0/2000 |
