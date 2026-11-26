# Diferencia pareada — conjunto `val`


Δ = **resnet50 − deit_s** del modelo promediado entre semillas, con los MISMOS grupos remuestreados para ambos (bootstrap agrupado pareado, B = 2000). IC 95 % percentil; p bilateral bootstrap = 2·min(P(Δ*≤0), P(Δ*≥0)). En log10-MAE y falso-seguro un Δ negativo favorece a resnet50; en macro-F1, un Δ positivo favorece a resnet50. 'Sorteos indefinidos' = remuestreos sin casos peligrosos (falso-seguro no definido).

| Subconjunto | Métrica | Δ (resnet50 − deit_s) | IC 95 % | p | Grupos | Sorteos indefinidos |
|---|---|---|---|---|---|---|
| Global | log10-MAE (décadas) | 0.0158 | [0.0124, 0.0183] | < 0.0005 | 574 | 0/2000 |
| Global | Tasa falso-seguro (fracción) | 0.00193 | [-0.000521, 0.00435] | 0.133 | 574 | 0/2000 |
| Global | Macro-F1 bandas desde V̂ | -0.014 | [-0.0224, -0.00762] | 0.001 | 574 | 0/2000 |
| Fuente: cs_fog | log10-MAE (décadas) | 0.00314 | [0.000115, 0.00551] | 0.047 | 264 | 0/2000 |
| Fuente: cs_fog | Tasa falso-seguro (fracción) | 0.00079 | [-0.0103, 0.00855] | 0.889 | 264 | 0/2000 |
| Fuente: cs_fog | Macro-F1 bandas desde V̂ | -0.00539 | [-0.0184, 0.0145] | 0.448 | 264 | 0/2000 |
| Fuente: foggy_cs | log10-MAE (décadas) | 0.0329 | [0.0301, 0.0367] | < 0.0005 | 264 | 0/2000 |
| Fuente: foggy_cs | Tasa falso-seguro (fracción) | — | — | — | 264 | 2000/2000 |
| Fuente: foggy_cs | Macro-F1 bandas desde V̂ | -0.0116 | [-0.0217, -0.00635] | 0.002 | 264 | 0/2000 |
| Fuente: reside_ots | log10-MAE (décadas) | 0.0103 | [0.00664, 0.014] | < 0.0005 | 310 | 0/2000 |
| Fuente: reside_ots | Tasa falso-seguro (fracción) | 0.00215 | [-0.000307, 0.00476] | 0.095 | 310 | 0/2000 |
| Fuente: reside_ots | Macro-F1 bandas desde V̂ | -0.0187 | [-0.0312, -0.00609] | 0.003 | 310 | 0/2000 |
| Calidad de etiqueta: equivalente | log10-MAE (décadas) | 0.0103 | [0.00678, 0.0143] | < 0.0005 | 310 | 0/2000 |
| Calidad de etiqueta: equivalente | Tasa falso-seguro (fracción) | 0.00215 | [-0.000461, 0.00492] | 0.111 | 310 | 0/2000 |
| Calidad de etiqueta: equivalente | Macro-F1 bandas desde V̂ | -0.0187 | [-0.0319, -0.00648] | 0.004 | 310 | 0/2000 |
| Calidad de etiqueta: metrica | log10-MAE (décadas) | 0.021 | [0.0188, 0.0238] | < 0.0005 | 264 | 0/2000 |
| Calidad de etiqueta: metrica | Tasa falso-seguro (fracción) | 0.00079 | [-0.0111, 0.00772] | 0.973 | 264 | 0/2000 |
| Calidad de etiqueta: metrica | Macro-F1 bandas desde V̂ | -0.00664 | [-0.018, 0.00494] | 0.196 | 264 | 0/2000 |
