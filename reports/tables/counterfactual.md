# Contrafactual: niebla dependiente de la distancia vs. niebla uniforme emparejada

Conjunto `test`: 200 pares (imagen limpia, V) con distancia por píxel; corrida semilla 0. Ambas versiones se re-renderizan desde la imagen limpia con el mismo A y pasan por el mismo JPEG; la uniforme usa una transmisión constante t₀ con el MISMO canal oscuro medio que la versión física.

- **|Δlog10 V̂|**: cambio medio de la predicción al quitar la estructura de profundidad (0,1 ≈ factor 1,26).
- **Spearman / pendiente**: relación de V̂ con la V verdadera en cada versión (pendiente de log10 V̂ vs log10 V; 1 = ideal).

| Modelo | n | \|Δlog10 V̂\| medio | Δ medio (unif − prof.) | Spearman (prof.) | Spearman (unif.) | Pendiente (prof.) | Pendiente (unif.) | MAE log10 (prof.) | MAE log10 (unif.) |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | 200 | 0.14 | -0.108 | 0.992 | 0.975 | 0.985 | 1.1 | 0.0495 | 0.146 |
| DeiT-S | 200 | 0.257 | -0.237 | 0.994 | 0.911 | 1 | 1.13 | 0.0434 | 0.263 |

## Interpretación

- **ResNet-50**: sensibilidad moderada a la estructura de profundidad (|Δ| medio ≈ factor 1.38); la relación con V se conserva con niebla uniforme (pendiente similar).
- **DeiT-S**: cambia claramente su predicción sin la estructura de profundidad (|Δ| medio ≈ factor 1.81): usa la atenuación dependiente de la distancia; la relación con V se conserva con niebla uniforme (pendiente similar).

Umbrales orientativos de la lectura automática: |Δ| < 0,05 (factor < 1,12) = insensible; 0,05–0,15 = moderado; > 0,15 (factor > 1,41) = sensible. Un modelo "físico" debería ser sensible y perder la relación con V en la versión uniforme.
