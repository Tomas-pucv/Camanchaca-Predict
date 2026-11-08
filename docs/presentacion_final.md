# Presentación final — guion, checklist y FAQ

**Fecha:** 23 o 26-nov-2026 (por confirmar). **Duración:**
15:00 ± 0:30, con los dos integrantes presentando (~7,5 min cada uno).
**Carpeta:** `presentations/2026-11-23_final/` (fuente + PDF + video de
respaldo de la demo).

**Mensaje central (una frase):** "Estimamos la visibilidad MOR en metros con
ResNet-50 y DeiT-S bajo un protocolo idéntico, con incertidumbre y pruebas de
atajo; [COMPLETAR: resultado principal en una frase, con IC]."

## 1. Guion cronometrado

Tiempos acumulados para un total de 15:00. Columna "Evidencia" = archivo del
que salen las cifras o figuras de la lámina.

| # | Lámina | Quién | Dur. | Acum. | Contenido (≤ 3 ideas) | Evidencia |
|---|---|---|---|---|---|---|
| 1 | Título + mensaje central | Tomás | 0:20 | 0:20 | Nombre, equipo, una frase | — |
| 2 | Problema: camanchaca en La Pólvora | Tomás | 1:10 | 1:30 | Foto real de camanchaca (propia); SSD(80) ≈ 119 m y SSD(100) ≈ 168 m; "¿cuántos metros veo?" | `docs/labeling_criteria.md` §6 |
| 3 | Qué cambió desde el avance | Nelson | 0:40 | 2:10 | Datos reales con etiqueta métrica, comparación justa, IC + baselines + interpretabilidad | — |
| 4 | Etiqueta: β → V MOR y bandas | Tomás | 1:00 | 3:10 | V = 2,996/β (OMM); ε = 0,02 sobreestimaba V en ~31 %; bandas 50/100/200 m con velocidad segura ≈ 44/71/111 km/h; equivalente vs métrica | `labeling_criteria.md` §2–6 |
| 5 | Datos: tabla final + ejemplos por banda | Tomás | 1:10 | 4:20 | n por fuente × split × banda; 1 ejemplo por banda y fuente (figura propia, citada) | `reports/tables/data_counts.md` |
| 6 | Atajo dataset ↔ banda, CS-Fog y brecha de dominio | Tomás | 0:50 | 5:10 | RESIDE < 75 m y Foggy ≥ 150 m; CS-Fog cubre las 4 bandas; sintético homogéneo vs camanchaca real | `data_card.md` §7 |
| 7 | Método: par justo y protocolo idéntico | Nelson | 1:00 | 6:10 | ResNet-50 (23,5 M) vs DeiT-S (21,7 M), IN-1k; misma cabeza, pasos y selección; barrido de LR igual; 3 semillas | `configs/protocol/base.yaml` |
| 8 | Resultados con IC y baselines | Nelson | 1:30 | 7:40 | Tabla con media ± d.e. e IC 95 %; baselines en la misma tabla; diferencia pareada y regla de decisión | `metrics_test.md`, `paired_test.md` |
| 9 | Aciertos y errores + por fuente/banda + externos | Nelson | 1:00 | 8:40 | 3 aciertos y 3 errores por modelo (V real vs V̂ en m); HazeRD; pares O-HAZE | `scatter_test.png`, `metrics_ext.md`, `ohaze_pairs.md` |
| 10 | ¿Atajos? Interpretabilidad comparada | Nelson | 1:15 | 9:55 | Grad-CAM (ambos) + rollout (DeiT) sobre los mismos ejemplos; atribución lejos/cielo; contrafactual y sonda de fuente; H1–H3 | `interpret_examples_neg.png`, `attribution.md`, `counterfactual.md`, `probe_source.md` |
| 11 | Autos: ¿hasta dónde se ven? | Tomás | 1:00 | 10:55 | Recall relativo vs distancia por β; V_det vs MOR vs SSD; "objetos a distancia conocida" | `cars_recall_vs_distance.png`, `cars_vdet.md` |
| 12 | Velocidad vs precisión: decisión | Nelson | 0:45 | 11:40 | ms/img en CPU y T4, GMACs; Pareto; decisión explícita para una cámara vial en CPU | `latency.md`, `pareto_speed_accuracy.png` |
| 13 | **Demo en vivo** (≤ 2:00) | Tomás opera (1:15) + Nelson comenta el mapa y la latencia (0:45) | 2:00 | 13:40 | Foto nueva de La Pólvora → V̂, banda, incertidumbre, Grad-CAM; pestaña "Simular niebla" | `app/app.py` |
| 14 | Conclusiones, limitaciones y trabajo futuro | Nelson (0:40) + Tomás (0:40) | 1:20 | 15:00 | Qué podemos afirmar y qué no; sin validación medida; cámara fija con hitos a distancia conocida | `informe.md` §5 |

**Reparto:** Tomás ≈ 7:25 y Nelson ≈ 7:35.

**Plan B de la demo:** si la app no responde en 20 s, pasar al **video de
respaldo** (2–3 min, grabado a más tardar el 19-nov) sin improvisar ni
reiniciar en vivo.

## 2. Checklist de láminas

- [ ] Letra ≥ 20 pt en el texto y ≥ 16 pt en tablas y ejes. Ninguna tabla
      con más de 6 columnas o 6 filas; el detalle va al informe.
- [ ] Una idea por lámina, con **≤ 3 viñetas**.
- [ ] Toda cifra con **unidades** (m, km/h, ms) y, si es una métrica, con
      **IC 95 %** o ± d.e. y el número de semillas.
- [ ] Ejes en m o en log10 m, con el nombre de la magnitud. Figuras a
      ≥ 150 dpi (las de `reports/figures/` ya salen así).
- [ ] Ejemplos de **cada banda** (lámina 5) y **aciertos y errores** (lámina
      9).
- [ ] Usar siempre "**visibilidad equivalente**" para RESIDE y "métrica" para
      Cityscapes y HazeRD.
- [ ] **Ortografía:** revisar con el corrector y con la lectura del otro
      integrante. Revisar tildes, paréntesis y comillas balanceados.
- [ ] Citar las fuentes en el pie de cada figura de dataset. No mostrar
      caras ni patentes legibles.
- [ ] Numerar las láminas para las preguntas.
- [ ] Exportar a PDF y probarlo en el proyector o en otro computador.
- [ ] **Ambos presentan**, y cada uno puede responder sobre todo el código.

## 3. Ensayos

| Ensayo | Fecha | Duración medida | Correcciones |
|---|---|---|---|
| 1 (completo, con demo en vivo) | 20-nov | [COMPLETAR] | [COMPLETAR] |
| 2 (completo, en la sala o red similar, con el video de respaldo probado) | 21–22-nov | [COMPLETAR] | [COMPLETAR] |

La meta es 14:45–15:15. Si un ensayo pasa de 15:30, recortar la lámina 6 o la
12 (nunca la demo ni los resultados).

## 4. Preguntas frecuentes (respuestas preparadas)

**¿Por qué ε = 0,05 y no 0,02 como en el avance?** Porque 0,05 es el MOR de
la OMM y es la convención de Foggy Cityscapes. Además, 0,02 da una V un 31 %
mayor, es decir, sobreestima la visibilidad, y eso no es conservador. β es el
ancla común y 0,02 se reporta como sensibilidad (`eps_sensitivity_*.md`).

**¿Qué significa "visibilidad equivalente"?** En RESIDE la niebla se
sintetizó con una profundidad estimada por una red y sin escala métrica. Su V
es la que tendría una escena con esa profundidad, no metros medidos. Por eso
reportamos aparte las fuentes métricas: Foggy Cityscapes, CS-Fog y HazeRD.

**¿El modelo no aprende simplemente qué dataset es?** Ese era el riesgo:
RESIDE < 75 m y Foggy ≥ 150 m. Lo atacamos con CS-Fog, que tiene las cuatro
bandas en el mismo dominio, con selección por media de log10-MAE por fuente,
con el baseline "media por fuente", con una sonda lineal de fuente y con un
contrafactual de niebla uniforme. [COMPLETAR con los resultados de
`probe_source.md` y `counterfactual.md`.]

**¿Por qué no usaron AIR-VIEW, que tiene visibilidad medida?** Casi todas sus
etiquetas superan 1,6 km (80 % son de 9–10+ millas), no cubre nuestras bandas
de 0–200 m, requiere suscripción y sus cámaras son meteorológicas. Lo citamos
como trabajo relacionado y como limitación.

**¿Por qué no estimar directamente la distancia a los autos?** La pregunta
del proyecto es la visibilidad, que resume la niebla en una cifra comparable
con la distancia de detención. La detectabilidad de autos entra como análisis
complementario: el recall del detector según la distancia
métrica por β y la $V_{det}$ comparada con el MOR y la SSD. Es el enfoque de
"objetos a distancia conocida". Usamos Cityscapes en vez de KITTI porque ya
teníamos distancia estéreo, instancias de autos y niebla oficial con β
conocido, sin otra descarga. KITTI, Multifog y DENSE quedan como trabajo
futuro.

**¿Por qué DeiT-S y no ViT-B?** Para comparar modelos de tamaño y
preentrenamiento similares: 21,7 M vs 23,5 M parámetros, ambos IN-1k. El
avance comparaba un ViT-B IN-21k de 87 M con una ResNet de 25 M, una
comparación injusta. DeiT-B se usa solo como punto de velocidad, con una
semilla.

**¿Cómo saben que una diferencia es real?** La regla, fijada antes de ver los
datos, es que solo hay diferencia si el IC 95 % del bootstrap pareado agrupado
(por escena o secuencia) excluye 0. Además usamos 3 semillas, y el test se
evaluó una sola vez con `--final`.

**¿Funciona con la camanchaca real?** No lo podemos afirmar: no hay datos
reales con visibilidad medida. Mostramos O-HAZE (orden por pares) y fotos de
La Pólvora de forma cualitativa. La validación requiere una cámara fija con
hitos a distancia conocida o un visibilímetro.

**¿Y de noche, o con lluvia?** Está fuera de distribución: todos los datos
son diurnos. La demo advierte una posible noche.

**¿Por qué log10 y una cabeza ordinal?** El log10 hace que un error de ×2 pese
igual a 30 m que a 600 m. La cabeza ordinal CORN existe porque hay pocos
niveles de β y la decisión es por banda. Reportamos
las bandas de la regresión y las de CORN.

**¿Cómo eligieron el learning rate sin favorecer a un modelo?** Con la misma
grilla {3e-5, 1e-4, 3e-4}, los mismos pasos y la misma semilla para ambos,
eligiendo solo con val (`selected_lr.json`).

**¿Qué hizo cada integrante?** Tomás se encargó de los datos, los baselines,
el análisis de autos y la demo. Nelson, del entrenamiento, la evaluación, la
interpretabilidad y la latencia. Se ve en el historial de git, consolidado con
`.mailmap`.

**¿Usaron IA?** Sí, un asistente de IA para código y documentación, bajo
nuestra revisión. Está declarado en el README y en el informe. Las cifras
salen solo del código.
