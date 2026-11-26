# Interpretabilidad cuantitativa: ¿dónde mira cada modelo?

Conjunto: `test`, 16 ejemplos fijos (estratificados por fuente × banda; los mismos para todos los modelos), corrida semilla 0.

- **Grad-CAM (−ŷ)**: evidencia de MENOR visibilidad; **(+ŷ)**: de MAYOR visibilidad. **Rollout**: flujo de atención del token CLS (sin signo).
- Cada celda: **razón masa/área** (fracción de la atribución en la región dividida por la fracción de píxeles; > 1 = la región recibe más atribución que su tamaño) y, entre paréntesis, la fracción de masa en %.
- Terciles de distancia calculados por imagen sobre la distancia finita; `cielo` = distancia infinita. Mapas auxiliares alineados con la misma geometría de evaluación (resize cuadrado + recorte central).

## Por distancia

| Modelo | Mapa | cerca | medio | lejos | cielo | n imágenes |
|---|---|---|---|---|---|---|
| ResNet-50 | Grad-CAM (−ŷ) | 0.47 (15 %) | 0.97 (31 %) | 1.5 (50 %) | 1.5 (3 %) | 7 |
| ResNet-50 | Grad-CAM (+ŷ) | 1.5 (49 %) | 0.88 (29 %) | 0.69 (22 %) | 0.013 (0 %) | 8 |
| DeiT-S | Grad-CAM (−ŷ) | 1.3 (44 %) | 0.61 (20 %) | 0.97 (32 %) | 2 (5 %) | 8 |
| DeiT-S | Grad-CAM (+ŷ) | 0.94 (31 %) | 1.2 (40 %) | 0.86 (28 %) | 0.42 (1 %) | 8 |
| DeiT-S | rollout (sin signo) | 1 (34 %) | 0.96 (31 %) | 0.99 (32 %) | 1.2 (3 %) | 8 |
| (área media) |  | 33 % | 33 % | 33 % | 2 % |  |

## Por clase gtFine

| Modelo | Mapa | cielo | calzada | vehiculo | edificio | vegetacion | otros | n imágenes |
|---|---|---|---|---|---|---|---|---|
| ResNet-50 | Grad-CAM (−ŷ) | 1.5 (3 %) | 0.57 (18 %) | 0.67 (5 %) | 1.5 (43 %) | 1.1 (20 %) | 0.89 (10 %) | 7 |
| ResNet-50 | Grad-CAM (+ŷ) | 0.013 (0 %) | 1.7 (53 %) | 0.58 (4 %) | 0.79 (24 %) | 0.56 (10 %) | 0.8 (9 %) | 8 |
| DeiT-S | Grad-CAM (−ŷ) | 2 (5 %) | 1.6 (49 %) | 0.34 (2 %) | 0.76 (23 %) | 0.81 (15 %) | 0.58 (7 %) | 8 |
| DeiT-S | Grad-CAM (+ŷ) | 0.42 (1 %) | 0.86 (27 %) | 1.3 (9 %) | 1.1 (32 %) | 0.92 (17 %) | 1.2 (14 %) | 8 |
| DeiT-S | rollout (sin signo) | 1.2 (3 %) | 1 (33 %) | 1.1 (8 %) | 0.96 (29 %) | 0.98 (18 %) | 0.91 (11 %) | 8 |
| (área media) |  | 2 % | 31 % | 7 % | 30 % | 18 % | 12 % |  |

Lectura: si la hipótesis física es correcta, el mapa −ŷ debería concentrarse en las regiones LEJANAS y el cielo (donde la niebla borra el contraste), y el +ŷ en estructuras cercanas y nítidas. Una razón ≈ 1 en todas las regiones indica una atribución difusa (señal global).
