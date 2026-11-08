# Riesgos y mitigación (entrega final)

Actualizado al 3-nov-2026. Reemplaza la tabla de la primera versión del
proyecto.

**Responsables.**

- **Tomás** (Tomás Ignacio Moraga): datos, baselines, análisis de autos, demo.
- **Nelson** (Nelson Mauricio Fuentes): entrenamiento, evaluación,
  interpretabilidad, latencia.

**Convenciones.** *Señal* = indicio observable que activa la mitigación.
*Estado*:

- *Mitigado*: la mitigación ya está en el código o en los docs.
- *En curso*: depende de correr en Colab o de terceros.
- *Abierto*: es una limitación que se declara.

## Tabla de riesgos

| ID | Riesgo | Prob. | Impacto | Señal de alarma | Mitigación | Resp. | Estado |
|---|---|---|---|---|---|---|---|
| R1 | **Brecha sintético → real**: toda la niebla con etiqueta es sintética y homogénea; la camanchaca es advectiva, heterogénea y marina | Alta | Alto | Errores sistemáticos en O-HAZE o en las fotos de La Pólvora; V̂(hazy) ≥ V̂(GT) en muchos pares | Declararla como limitación central. Evidencia: O-HAZE (orden por pares + cualitativo), HazeRD (test métrico externo), fotos de La Pólvora (cualitativo). No vender un "sistema listo". Trabajo futuro: datos medidos | Ambos | Abierto (se declara) |
| R2 | **Confusión dataset ↔ banda**: RESIDE solo < 75 m y Foggy Cityscapes solo ≥ 150 m, así que reconocer la fuente predice la banda | Alta | Alto | `mean_by_source` cerca de los modelos; sonda de fuente con exactitud ≫ azar; Spearman bajo dentro de cada fuente | CS-Fog (4 bandas en un mismo dominio, en train y test); selección por media de log10-MAE **por fuente**; baseline `mean_by_source`; `camanchaca probes`; métricas por fuente | Tomás (datos), Nelson (probes) | Mitigado en código; evidencia pendiente de correr |
| R3 | **Etiquetas en el borde de banda**: RESIDE β = 0,06 → 49,9 m; HazeRD 50/100/200 m si su ε es 0,05 | Alta | Medio | Falsos-seguros concentrados en un solo nivel de β; resultados de banda que cambian con ε | Métricas continuas como principales; error por banda real (`error_by_band_*.png`) y matriz de confusión; tabla `eps_sensitivity_*.md`; bordes fijos por la SSD (nunca ajustados a los datos) | Nelson | Mitigado |
| R4 | **Convención de HazeRD** (ε de su script: 0,02 o 0,05) desconocida | Media | Medio | `demo_simu_haze.m` usa otro umbral o el paquete trae otra estructura | Leer `demo_simu_haze.m` al descargar; `--eps` configurable; si no se confirma, reportar ambas lecturas | Tomás | En curso |
| R5 | **Versión de RESIDE**: la primera versión pudo usar un reempaque (RESIDE-6K) sin β | Media | Alto | `prep reside --check-only` con < 90 % de nombres `id_A_beta` | El comando aborta ante reempaques; usar el OTS oficial o un espejo que conserve los nombres | Tomás | En curso |
| R6 | **Retrasos de acceso a datos** (aprobación de Cityscapes, formulario de HazeRD, O-HAZE) | Media | Alto | Sin cuenta aprobada al 5-nov | Pedir acceso **de inmediato** (si aún no se hizo). **Go/no-go el 6-nov**: sin Cityscapes se entrena solo con RESIDE (+ HazeRD como test), sin CS-Fog ni autos, y se declara la confusión. Sin HazeRD al 12-nov: test métrico solo con CS-Fog test | Tomás | En curso |
| R7 | **Límites de Colab/Drive**: Drive gratis de 15 GB frente a ~45 GB de zips de Cityscapes; desconexiones; cuota de GPU | Alta | Medio | "Disk quota exceeded"; sesión cortada; sin GPU disponible | Preprocesar en runtime CPU leyendo del zip **sin extraer**; tars reducidos (~1–3 GB) en Drive; borrar cada zip antes del siguiente; vaciar la papelera; corridas idempotentes y reanudables (`DONE`, `last.pt`); usar las 2 cuentas; respaldo en Kaggle Notebooks (solo RESIDE) | Nelson (cómputo), Tomás (datos) | Mitigado |
| R8 | **Licencias**: Cityscapes y RESIDE prohíben redistribuir; detectores AGPL | Media | Alto | Imágenes de datasets en git, en la demo o en un Space | `.gitignore` de imágenes, aux y tars; solo CSV derivados; demo con fotos propias o CC (`app/examples/README.md`); detector de torchvision (BSD-3), no Ultralytics (AGPL-3.0) | Tomás | Mitigado |
| R9 | **Falla de la demo en vivo** (red de la sala, pesos, portátil) | Media | Alto | La app no carga los checkpoints en el ensayo | Correr en el portátil por CPU con pesos ya descargados (sha256); sin depender de Colab ni del enlace `share`; **video de respaldo de 2–3 min** listo el 19-nov; ensayo en la sala | Tomás | En curso |
| R10 | **Comparación injusta ResNet vs ViT** | Baja | Alto | Diferencias en cabeza, pasos, LR o normalización | Par IN-1k de tamaño similar (23,5 M vs 21,7 M sin la capa de ImageNet); misma cabeza, pérdida, pasos, batch, aumentación y criterio de selección; barrido de LR con el mismo presupuesto; `assert_same_protocol` con tests | Nelson | Mitigado |
| R11 | **Sobreajuste / pocos niveles de β** (la regresión se parece a una clasificación) | Media | Medio | Brecha train/val creciente; predicciones pegadas a los niveles de entrenamiento | Early stopping por val; weight decay 0,05; CS-Fog continuo; cabeza ordinal CORN; HazeRD (100/200 m) como prueba de interpolación | Nelson | Mitigado |
| R12 | **Tiempo** (final el 23 o el 26-nov; 2 personas) | Alta | Alto | Hitos atrasados más de 2 días | Cronograma con gates (abajo); MVP primero; el DeiT-B queda como *stretch* de 1 semilla con corte el 13-nov | Ambos | En curso |
| R13 | **Presentación**: no usar los 15 min o que no presenten ambos integrantes | Media | Alto | Ensayo fuera de tiempo | Confirmar fecha y nómina del grupo; guion de 15 min con ~7 min por persona; 2 ensayos cronometrados | Ambos | En curso |
| R14 | **Historial git poco incremental**: 14 commits del 19 al 22-sep con 3 nombres de una misma persona y 0 commits de Nelson | Alta | Medio | Un solo commit grande al final | `.mailmap`; commits por grupos lógicos de cada autor con su identidad; commits de resultados a medida que se corren; tag `v1.0-final` | Ambos | En curso |

## Cronograma con gates

| Fecha | Hito (gate) | Resp. |
|---|---|---|
| 3–4 nov (ya) | Pedir acceso a Cityscapes y HazeRD si aún no se hizo; descargar O-HAZE; RESIDE: `prep reside --check-only`, `prep`, `split` y 1 corrida corta de punta a punta | Tomás (datos), Nelson (corrida) |
| **6-nov** | **Go/no-go Cityscapes** (ver R6) | Tomás |
| 8-nov | Datos congelados: `data/splits/v1` y `SPLIT_INFO.json` commiteados; baselines en val | Tomás |
| 12-nov | Barrido de LR + 3 semillas × 2 modelos (~5 GPU-h repartidas en las 2 cuentas); tablas del MVP en val | Nelson |
| 13-nov | Corte del *stretch* DeiT-B (1 semilla) | Nelson |
| 15-nov | **Congelar resultados**: evaluación `--final` una sola vez, interpretabilidad, autos, latencia, `camanchaca report` | Ambos |
| 19-nov | Informe, data card, láminas y video de la demo listos | Ambos |
| 20–22 nov | 2 ensayos cronometrados de 15 min con los dos presentando | Ambos |
| 23 o 26-nov | Presentación final (confirmar la fecha) | Ambos |

## Detalle de los riesgos críticos

**R1. Brecha sintético → real.** La física de la etiqueta (Koschmieder + ASM)
vale para cualquier medio dispersor homogéneo. Lo que no se transfiere es la
*apariencia*: textura, gradiente vertical, no homogeneidad y luz marina. El
resultado honesto esperado es "buena estimación de la visibilidad física en
niebla sintética con distancia métrica, con un comportamiento a verificar en
niebla real". Sin datos medidos de niebla real, el proyecto no puede afirmar
precisión en La Pólvora.

**R2. Confusión dataset ↔ banda.** Es el atajo más probable en una mezcla de
RESIDE + Foggy Cityscapes. CS-Fog lo ataca en origen. Las sondas lo miden:
baseline por fuente, sonda lineal de fuente y contrafactual de niebla
uniforme vs dependiente de la distancia. Si el modelo apenas supera a
`mean_by_source`, se reporta así.

**R6/R7. Datos y cómputo.** El presupuesto del MVP es de ~4–5 GPU-h en una T4
(≈ 9 con margen), así que la GPU no es el cuello de botella. Los cuellos de
botella son el acceso a los datos y el disco. La regla es preprocesar una vez,
entrenar desde `/content` y que cada corrida (modelo, semilla) sea
idempotente y no supere unos 30 minutos.
