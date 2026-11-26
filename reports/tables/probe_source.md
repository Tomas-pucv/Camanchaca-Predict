# Sonda lineal de FUENTE sobre features congeladas


¿El backbone codifica de qué dataset viene la imagen? Regresión logística (features estandarizadas) que predice la `source`, con validación cruzada GroupKFold por `group` (escenas completas fuera del entrenamiento de la sonda). Datos: submuestra estratificada por fuente de val; checkpoint de la semilla 0. Exactitud balanceada muy por encima del azar (1/k) = la representación separa los datasets; combinado con una banda predecible desde la fuente, es riesgo de ATAJO (acertar la visibilidad reconociendo el dataset).

| Representación | Fuentes (k) | n (imágenes) | Grupos | Pliegues | Exactitud balanceada | Azar (1/k) | Exactitud |
|---|---|---|---|---|---|---|---|
| resnet50 (backbone, semilla 0) | 3 | 900 | 425 | 5 | 0.92 | 0.333 | 0.92 |
| deit_s (backbone, semilla 0) | 3 | 900 | 425 | 5 | 0.959 | 0.333 | 0.959 |
| features físicas (baseline, 12 dim) | 3 | 900 | 425 | 5 | 0.84 | 0.333 | 0.84 |

## Banda desde la fuente sola

Clasificador trivial: la banda mayoritaria de cada fuente en los pliegues de entrenamiento (misma validación cruzada agrupada, solo filas con etiqueta). Si supera claramente a la banda mayoritaria global, el dataset delata la banda.

| n (con etiqueta) | Bandas presentes | Exactitud (fuente → banda) | Exactitud balanceada | Exactitud banda mayoritaria global | Azar balanceado (1/k) |
|---|---|---|---|---|---|
| 900 | 4 | 0.617 | 0.43 | 0.293 | 0.25 |
