# Riesgos y mitigación

Convención: **señal** = indicio observable que activa la mitigación;
**responsable** según roles A (datos), B (entrenamiento), C (ViT/evaluación).

| ID | Riesgo | Prob. | Impacto | Señal de alarma | Mitigación | Resp. | Estado |
|----|--------|-------|---------|-----------------|------------|-------|--------|
| R1 | Datos sintéticos ≠ camanchaca real (niebla advectiva, heterogénea, marina) | Alta | Alto | Predicciones "planas" en O-HAZE; errores sistemáticos | O-HAZE como test externo cualitativo; análisis de error por escena; roadmap de captura de datos propios (cámaras DTV / fotografías en ruta); reporte honesto de la brecha | C | Abierto |
| R2 | El β del filename podría no estar documentado oficialmente (unidades/convención) | Media | Alto | ρ de Spearman bajo en la validación del canal oscuro; betas fuera de rango plausible (V > 5 km o < 5 m) | Validación física ASM+depth (ITS); consistencia perceptual; sensibilidad con ε=0.02/0.05; descartar outliers | A | En curso |
| R3 | Desbalance de bandas (p. ej. muy pocas imágenes con V < 50 m) | Alta | Medio | Histograma de V por banda tras el etiquetado | Class weights en la pérdida; re-síntesis con t̂^k (re-densificación físicamente consistente); reportar F1 por banda | A+B | Abierto |
| R4 | Colab free: desconexiones, límites de GPU y de disco | Alta | Medio | Sesión interrumpida; "no backend available" | Checkpoints + history en Drive al final de cada época; entrenamientos ≤ 40 min; reanudar desde checkpoint; subset controlado de imágenes | B | Mitigado |
| R5 | ViT-B/16 no cabe / es lento en T4 (16 GB) | Media | Medio | CUDA OOM; > 2 s por iteración | AMP fp16; fase A con tronco congelado; batch 16 + grad accumulation; fallback ViT-S/16 | C | Mitigado |
| R6 | Fuga de datos por variantes de la misma escena en train/test | Media | Alto | Métricas sospechosamente altas; mismas escenas en dos conjuntos | Split agrupado por escena + tests automáticos anti-fuga + CSV versionados | A | Mitigado |
| R7 | Plazos: avance 21-09 y entrega 28-09/01-10 | Alta | Medio | Tareas de la checklist atrasadas ≥ 1 día | MVP priorizado (pipeline completo con ResNet aunque el ViT no termine); roles claros; daily de 10 min | todos | En curso |
| R8 | Sobreajuste del ViT con dataset reducido | Media | Medio | Brecha train/val creciente | Preentrenamiento 21k; aumentación geométrica (sin jitter de color); early stopping; weight decay 0.05 | C | Abierto |
| R9 | "¿Y la camanchaca de La Pólvora?" — generalización cuestionada en la evaluación | Alta | Alto | Pregunta del profesor :) | Estrategia explícita: (1) física del etiquetado, (2) test externo real O-HAZE, (3) análisis de error, (4) plan de datos locales; no sobre-vender | C | Abierto |

## Detalle de los tres riesgos críticos

**R1 — Brecha sintético/real.** Es la limitación central del proyecto y se
asume como tal. La física del etiquetado (Koschmieder) es aplicable a
cualquier medio dispersor, pero la *apariencia* de la niebla sintética difiere
de la camanchaca (textura, gradiente vertical, no-homogeneidad). El modelo
ViT fue elegido justamente por su capacidad de atender a patrones globales no
locales, que es donde la niebla heterogénea se manifiesta. Aún así, el
resultado honesto esperado es "buena correlación con visibilidad física en
datos sintéticos y comportamiento razonable en niebla real", no "sistema
listo para La Pólvora".

**R2 — Unidades del β.** Toda la cadena de etiquetas depende del parseo del
filename. La validación por canal oscuro (Spearman) y la reproducción del ASM
con profundidades conocidas dan evidencia suficiente para el avance; si la
correlación fuera débil, se re-etiquetaría usando la re-síntesis controlada
(generar nosotros la niebla con β conocido, quedando las etiquetas garantizadas
por construcción).

**R4/R5 — Recursos de cómputo.** Todo el pipeline está diseñado para Colab
free con GPU T4: subset controlado de imágenes (≤ ~12k), checkpoints por época
en Drive, AMP en ambos modelos, fase de congelado en el ViT y batch/accum
configurables. El notebook 04 (ResNet) cabe en una sesión de ~30 min; el 05
(ViT) en una o dos sesiones.
