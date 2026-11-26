# Distancia de detección V_det vs. visibilidad MOR

V_det = primera distancia donde el recall relativo a 'clear' cae bajo 0.5 (interpolación lineal entre centros de bin). IC 95 % bootstrap agrupado por secuencia (218 grupos, 1000 remuestreos). Solo entran los bins con ≥ 20 autos GT (en la estimación y en cada remuestreo; 'Bins' = bins con recall relativo definido). Censura: '> x' = el recall relativo nunca cae bajo el umbral en el rango; '≤ x' = ya está bajo en el primer bin válido.

Detector: Faster R-CNN R50-FPN v2 (pesos `coco`), umbral de puntaje 0.5, IoU ≥ 0.5, clases car+truck+bus; GT: altura ≥ 15 px (resolución completa) y profundidad válida; 475 frames × 6 condiciones.

| Condición | β (1/m) | V MOR (m) | V_det | IC V_det | V_det / V MOR | Vel. segura con V_det (km/h) | Vel. segura con V MOR (km/h) | detectados/GT | Bins |
|---|---|---|---|---|---|---|---|---|---|
| foggy_b0.005 | 0.005 | 599 | > 145 m | [137 m, > 145 m] | > 0.24 | > 91 | 214 | 2953/3904 | 15 |
| foggy_b0.01 | 0.01 | 300 | 115 m | [97 m, 120 m] | 0.39 | 79 | 143 | 2695/3904 | 15 |
| foggy_b0.02 | 0.02 | 150 | 71 m | [63 m, 74 m] | 0.47 | 57 | 93 | 2185/3904 | 15 |
| csfog_v100 | 0.03 | 100 | 50 m | [47 m, 51 m] | 0.5 | 44 | 72 | 1677/3904 | 15 |
| csfog_v50 | 0.0599 | 50 | 33 m | [32 m, 35 m] | 0.66 | 33 | 44 | 1054/3904 | 15 |

Distancia de detención de referencia (reacción 2,5 s, pavimento mojado μ = 0,4): 30 km/h → 30 m, 50 km/h → 59 m, 80 km/h → 118 m, 100 km/h → 168 m.
La velocidad 'segura' es la que hace que la distancia de detención quepa en la distancia indicada; es ILUSTRATIVA, no una recomendación operativa.

Lectura: si V_det < V MOR, el detector deja de ver autos ANTES de la visibilidad meteorológica (la MOR describe un objeto negro grande y contrastado; un auto real es más chico y de menor contraste).
