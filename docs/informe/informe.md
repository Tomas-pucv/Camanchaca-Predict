---
title: "Camanchaca-Predict: estimación de la visibilidad bajo niebla con ResNet-50 y DeiT-S"
subtitle: "Informe técnico final — Aprendizaje Automático II, PUCV 2026-2 (Grupo 5)"
author:
  - Tomás Ignacio Moraga
  - Nelson Mauricio Fuentes
date: "Noviembre de 2026"
lang: es
bibliography: refs.bib
link-citations: true
geometry: margin=2cm
fontsize: 11pt
---

<!--
ESTADO (26-nov-2026): secciones 1 a 5 redactadas. Las cifras de la sección 4
salen de reports/ (evaluación final en Colab; ver reports/RESULTADOS.md). Las
hipótesis y reglas de decisión de 4.2 se fijaron ANTES de ver los resultados y
se aplican tal como están escritas. No escribir ninguna cifra que no salga de
reports/. Presupuesto de páginas: ver docs/informe/README.md.
-->

# 1. Motivación

La **camanchaca** es la niebla costera advectiva de la zona central y norte de
Chile: aire húmedo del Pacífico que se enfría sobre aguas frías, forma
estratos bajos y entra a la costa por quebradas y pasos bajos. En el Gran
Valparaíso afecta al **Camino La Pólvora**, principal acceso de los camiones
al puerto, con tramos de alta velocidad expuestos a la niebla. La
niebla es peligrosa cuando la **visibilidad** V cae bajo la **distancia de
detención** (SSD) del vehículo. Con un tiempo de percepción-reacción de
2,5 s [@aashto2018policy] y fricción de piso mojado μ = 0,4,

$$\mathrm{SSD}(v) = v\,t_r + \frac{v^2}{2\mu g},$$

lo que da SSD ≈ 119 m a 80 km/h y ≈ 168 m a 100 km/h. Hoy no existe sobre esa
ruta una medición continua de la visibilidad en metros. Una cámara vial y un
modelo de visión serían una alternativa barata a un visibilímetro, siempre que
el modelo estime una magnitud física y no solo "cuánta niebla se ve".

**Formulación.** A partir de una sola imagen estimamos la visibilidad
meteorológica **MOR** en metros, una magnitud continua, y la traducimos a
cuatro **bandas de seguridad**: crítico < 50 m, alto riesgo 50–100 m,
precaución 100–200 m y aceptable ≥ 200 m. Los bordes vienen de la SSD: con
V = 50, 100 y 200 m, la mayor velocidad que permite detenerse dentro de lo
visible es ≈ 44, 71 y 111 km/h, respectivamente (máximo entero con
SSD ≤ V). El error más costoso es el
**falso-seguro**: decir V ≥ 100 m cuando en realidad V < 100 m.

**Pregunta de aprendizaje automático.** El curso pide comparar una CNN y un
Vision Transformer. La hipótesis de fondo es que la niebla es una señal
**global y dependiente de la distancia**: atenúa más lo lejano y lleva la
imagen hacia la luz atmosférica. La auto-atención global de un ViT podría
aprovecharla mejor que los campos receptivos locales de una CNN. Para que la
comparación sea válida, ambos modelos deben tener tamaño y preentrenamiento
similares y un protocolo idéntico.

**Alcance.** Estimamos la **visibilidad** y agregamos un **análisis
complementario de autos**: el recall de un detector en función de la distancia
métrica, por nivel de niebla. Ese análisis sigue la idea de "objetos a
distancia conocida" y conecta la visibilidad con la detectabilidad de lo que
importa en la vía.

# 2. Datos

**Etiqueta.** Todas las fuentes se anclan en el coeficiente de extinción β
[1/m] del modelo atmosférico de dispersión (ASM) [@koschmieder1924theorie;
@middleton1952vision]:

$$I = J\,t + A\,(1-t),\qquad t = e^{-\beta d}.$$

La etiqueta es el MOR de la OMM, con umbral de contraste ε = 0,05
[@wmo2018guide]:

$$V = -\ln(0{,}05)/\beta \approx 2{,}996/\beta.$$

En el avance usamos ε = 0,02 y lo llamamos "conservador", lo cual era un
error: con ε = 0,02, V es un 31 % mayor para el mismo β, es decir,
**sobreestima** la visibilidad. Ahora ε = 0,02 se reporta solo como
sensibilidad. El modelo aprende log10 V, y cada imagen lleva una calidad de
etiqueta: *métrica* si la distancia usada para sintetizar la niebla está en
metros, *equivalente* si es una profundidad estimada y relativa, y *ninguna*
si no hay etiqueta.

| Fuente | Niebla | Etiqueta | V MOR | Uso |
|---|---|---|---|---|
| RESIDE-OTS [@li2019benchmarking] | ASM sobre fotos reales, profundidad monocular | equivalente | 15–75 m (7 β) | train/val/test por escena |
| Foggy Cityscapes [@sakaridis2018semantic] | ASM con distancia estéreo | métrica | 150/300/600 m | train/val/test por ciudad |
| CS-Fog (propia, sobre Cityscapes [@cordts2016cityscapes]) | ASM con distancia estéreo | métrica | 15–1000 m continua | train/val/test por ciudad |
| HazeRD [@zhang2017hazerd; @zhang2019hazerddata] | ASM con profundidad métrica | métrica | 5 niveles nominales; ≈ 38–766 m si ε~H~ = 0,02 | solo test externo |
| O-HAZE [@ancuti2018ohaze] | **real** (máquina de niebla), pares | ninguna | — | externo: cualitativo + orden por pares |
| Fotos propias (La Pólvora) | real / despejada | ninguna | — | externo cualitativo, demo |

: Fuentes de datos. Conteos por split, fuente y banda en `reports/tables/data_counts.md`: 36.542 imágenes en 5.292 grupos; el conjunto externo tiene 75 imágenes de HazeRD y 90 de O-HAZE.

**RESIDE-OTS.** Usamos el OTS oficial (2.061 escenas × 7 β × 5 A). El
preprocesamiento verifica que los nombres sigan el patrón `id_A_beta` y aborta
ante reempaques sin β, como RESIDE-6K, que circulan en espejos públicos.
Conservamos **una A por
(escena, β)**, ≈ 14,4 mil imágenes, para no inflar el conjunto con variantes
casi idénticas. Como la profundidad de RESIDE es estimada, su V es una
*visibilidad equivalente*.

**Foggy Cityscapes y CS-Fog.** Foggy Cityscapes oficial solo cubre
150–600 m, y RESIDE solo < 75 m. Con esas dos fuentes, **la identidad del
dataset predice la banda**, un atajo evidente. Por eso generamos niebla
propia con profundidad real y construimos **CS-Fog**:

1. la distancia euclídea por píxel sale de la disparidad estéreo de
   Cityscapes, $Z = B f_x/d$ y
   $l = Z\sqrt{1+((u-u_0)/f_x)^2+((v-v_0)/f_y)^2}$;
2. el cielo se toma como distancia infinita y los huecos se rellenan con el
   vecino válido más cercano;
3. se renderiza el ASM con V MOR log-uniforme entre 15 y 1000 m y A uniforme
   en [0,75; 0,95], dos visibilidades por frame, con semilla determinista.

Así las cuatro bandas aparecen en un mismo dominio visual. Cityscapes se
particiona **por ciudad**: test = frankfurt, lindau y munster (el val
oficial); val = hanover, jena y tubingen; train = el resto. Se recorta el capó
en todas sus imágenes.

**Externos.** HazeRD es el único test con etiqueta métrica fuera del
entrenamiento. Su script define la niebla por visibilidad nominal con un
umbral propio ε~H~, así que $\beta = -\ln(\varepsilon_H)/V_{nom}$ y la
etiqueta es $V = 2{,}996/\beta$. Con el valor por defecto ε~H~ = 0,02 (por
confirmar en `demo_simu_haze.m`), los niveles nominales de 50, 100, 200, 500 y
1000 m equivalen a ≈ 38, 77, 153, 383 y 766 m MOR. Si el umbral no se
confirma, se reportan ambas lecturas.
O-HAZE es niebla real sin medición: se usa cualitativamente y para verificar,
en sus 45 pares, que $\hat V(\text{con niebla}) < \hat V(\text{limpia})$.
AIR-VIEW [@mourning2025airview] aporta visibilidad medida en cámaras de la
FAA, pero casi todas sus etiquetas superan 1,6 km y requiere suscripción, así
que queda como trabajo relacionado.

**Licencias y sesgos.** Las imágenes de Cityscapes, Foggy Cityscapes y RESIDE
no se pueden redistribuir: el repositorio versiona solo CSV derivados, sin
píxeles. Toda la niebla etiquetada es sintética, homogénea y diurna, sobre
ciudades europeas o fotos de internet; no hay datos chilenos con etiqueta.
Detalle en `docs/data_card.md` [@gebru2021datasheets]; la ficha de los
modelos está en `docs/model_card.md` [@mitchell2019model].

# 3. Arquitectura y método

**Par justo.** Comparamos `resnet50.a1_in1k` [@he2016deep; @wightman2021resnet]
con `deit_small_patch16_224.fb_in1k` [@touvron2021training;
@dosovitskiy2021image], ambos de timm [@wightman2019timm]. Los dos están
preentrenados solo en ImageNet-1k, a 224 px, con la misma normalización.
Tienen 23,5 M y 21,7 M parámetros sin la capa de clasificación de ImageNet
(25,6 M y 22,1 M con ella) y ≈ 4,1 y 4,6 GMACs a 224 px, respectivamente.

| Elemento | ResNet-50 | DeiT-S |
|---|---|---|
| Cabeza | Dropout 0,1 → regresión de log10 V estandarizado + 3 logits ordinales CORN [@shi2023corn] | idéntica |
| Pérdida | SmoothL1 + 0,5 · CORN | idéntica |
| Optimización | AdamW (wd 0,05, sin decay en normas y sesgos), LR de la cabeza = 10 × LR del backbone | idéntica |
| LR del backbone | barrido {3e-5, 1e-4, 3e-4}, 1.200 pasos, semilla 0, criterio en val | mismo barrido y presupuesto |
| Schedule | warmup 300 + coseno, **máximo 3.600 pasos**, early stopping tras 4 evaluaciones sin mejora, batch 64, AMP, clip 1,0 | idéntico |
| Aumentación | Resize 256 → crop 224 + flip, **sin** jitter de color | idéntica |
| Selección | media de log10-MAE por fuente en val, cada 300 pasos | idéntica |
| Semillas | 0, 1, 2 | 0, 1, 2 |

: Protocolo compartido (`configs/protocol/base.yaml`). Una función `assert_same_protocol` verifica que solo cambie el backbone.

La cabeza ordinal CORN responde a que los datasets
oficiales tienen pocos niveles de β, así que la regresión se parece a una
clasificación ordinal. Reportamos las bandas derivadas de la regresión y las
de la cabeza ordinal. Un DeiT-B (86 M) con una sola semilla se entrena solo
como punto extra de la curva velocidad vs precisión, fuera de la comparación
principal.

**Métricas e incertidumbre.**

- *Principales:* log10-MAE (por fuente y su media), MdAPE, δ < 1,25 y Spearman
  dentro de cada fuente.
- *Por banda:* macro-F1 sobre una lista fija de bandas, F1 por clase y matriz
  de confusión.
- *Seguridad:* falso-seguro (indefinido, y no 0, si no hay casos peligrosos) y
  AUC-PR de V < 100 m.

Reportamos media ± desviación sobre 3 semillas e IC 95 % por **bootstrap
agrupado** por escena o secuencia [@efron1993introduction]. La diferencia
**pareada** ResNet − DeiT usa la misma matriz de remuestreo para ambos
modelos. El test y los externos están **bloqueados**: solo se evalúan con la
bandera `--final`, una vez, después de fijar todo en val.

**Baselines y sondas de atajo.**

- *Triviales:* media global, media por fuente (mide el atajo de la fuente) y
  banda mayoritaria.
- *Físico sin deep learning:* Ridge y HistGradientBoosting sobre features del
  canal oscuro [@he2011single], contraste, saturación, bordes y entropía. Si
  las redes no lo superan, podrían estar aprendiendo solo contraste global.
- *Sonda lineal de fuente* sobre las features congeladas: ¿el backbone
  codifica el dataset?
- *Contrafactual:* niebla uniforme (t constante) emparejada en canal oscuro
  medio con la niebla física sobre imágenes limpias de test. Si las
  predicciones no cambian, el modelo no usa la atenuación con la distancia.

**Interpretabilidad.** Usamos Grad-CAM [@selvaraju2017gradcam] con signo
explícito (−ŷ = evidencia de menor visibilidad) en ambos modelos: en ResNet
sobre `layer4[-1]` y en DeiT sobre `blocks[-1].norm1`. En DeiT agregamos
*attention rollout* [@abnar2020quantifying]. Todo se calcula sobre un
**conjunto fijo y versionado** de ejemplos estratificado por fuente × banda
(`data/splits/v1/interpret_examples.csv`), el mismo para ambos modelos. Cuantificamos la fracción de atribución por
tercil de distancia (y cielo) y por clase semántica de gtFine (cielo,
calzada, vehículo, edificio, vegetación).

**Análisis de autos.** Sobre las 500 imágenes del val de Cityscapes, en
condición limpia, Foggy Cityscapes (3 β) y CS-Fog con V = 50 y 100 m, corremos
Faster R-CNN v2 de torchvision preentrenado en COCO [@ren2015faster;
@li2021benchmarking; @lin2014coco]. La distancia de cada auto es la mediana de
la distancia estéreo sobre su máscara erosionada. Se emparejan detecciones
de la clase *car* de COCO con IoU ≥ 0,5 y score ≥ 0,5, sobre autos de al
menos 15 px de alto con distancia válida. Medimos el recall por bins de 10 m, normalizado
por el recall en limpio. Definimos $V_{det}$ como la distancia donde el recall
relativo cae bajo 0,5, con censura cuando nunca cae, y comparamos con el MOR
y con la SSD. Los IC son bootstrap por secuencia.

**Velocidad.** Medimos parámetros, GMACs, latencia con batch 1 y throughput
con batch 64, en T4 (fp16) y en CPU (fp32).

# 4. Resultados

## 4.1 Estado y datos evaluados

Las cifras de esta sección salen de la evaluación final sobre:

- **RESIDE-OTS** oficial: niebla sintética sobre fotos reales; visibilidad
  equivalente.
- **Foggy Cityscapes y CS-Fog**: niebla sintética sobre escenas de conducción
  reales con distancia métrica.
- **HazeRD**: test métrico externo.
- **O-HAZE**: la única niebla real, sin visibilidad medida, así que solo
  permite evaluar el orden por pares y hacer inspección cualitativa.

Ningún conjunto tiene **niebla real con visibilidad medida**, y no
reportaremos precisión en niebla real como si la hubiera.
La evaluación final se ejecutó una sola vez en Google Colab, con los
splits congelados de `data/splits/v1` (sha256 de `test.csv` y `ext.csv` en el
README). Las tablas y figuras están en `reports/RESULTADOS.md` y se
versionaron el 25 y el 26-nov-2026 (commits `e5b8106`, notebooks 03–04, y
`4352a36`, notebook 05).

## 4.2 Hipótesis y reglas de decisión (fijadas antes de ver los resultados)

- **Criterio principal:** log10-MAE medio por fuente en el test interno.
  ResNet-50 y DeiT-S **difieren** solo si el IC 95 % del bootstrap pareado
  agrupado de la diferencia excluye 0. Si lo incluye, diremos que no
  difieren. Se informa además la media ± d.e. de cada modelo entre semillas.
- **H1 (contexto global).** DeiT-S usa más las claves del campo lejano
  (cielo y horizonte, donde se acumula la luz atmosférica), mientras que
  ResNet-50 se apoya más en los bordes locales cercanos. *Prueba:* razón
  masa/área de la atribución en el tercil lejano + cielo frente al tercil
  cercano (`attribution.md`), sobre los mismos ejemplos para ambos modelos.
- **H2 (atajo compartido).** Ambos modelos usan en parte el contraste global.
  *Se apoya* si (a) en el contrafactual la predicción apenas cambia al pasar
  a niebla uniforme (|Δlog10 V̂| medio < 0,05, es decir, factor < 1,12) y la
  pendiente de log10 V̂ frente a log10 V con niebla uniforme es similar a la
  de la niebla física (`counterfactual.md`), o (b) la sonda de fuente supera
  claramente el azar (1/k) y el baseline por fuente queda cerca de las
  redes.
- **H3 (robustez).** DeiT-S degrada menos que ResNet-50 con niebla densa y
  entre datasets. *Prueba:* error en la banda crítica
  (`error_by_band_test.png`) y diferencia pareada por fuente en el test
  interno (`paired_test.md`) y en HazeRD (`paired_ext.md`). Con 14–15
  escenas en HazeRD esperamos IC anchos; un IC que incluye 0 se reporta como
  no concluyente.
- **Velocidad vs precisión.** Escenario: cámara vial con CPU que procesa una
  imagen cada varios segundos, donde la latencia no es restrictiva. Se elige
  el modelo con menor log10-MAE y menor falso-seguro. Si sus IC se solapan,
  se elige el de menor latencia en CPU.

## 4.3 Comparación principal

| Modelo | log10-MAE [IC 95 %] | MdAPE (%) | δ<1,25 | Spearman | macro-F1 (reg / CORN) | Falso-seguro [IC 95 %] | AUC-PR V<100 |
|---|---|---|---|---|---|---|---|
| Media global / por fuente / banda mayoritaria | 0,492 / 0,251 / 0,578 | 81,7 / 50,0 / 78,7 | 0,022 / 0,263 / 0,159 | — / 0,782 / — | 0,063 / 0,442 / 0,156 | 1 / 0,163 / 0 | 0,555 / 0,97 / 0,555 |
| Físico (Ridge / HGB) | 0,210 / 0,137 | 38,9 / 21,3 | 0,274 / 0,510 | 0,865 / 0,921 | 0,588 / 0,736 | 0,088 / 0,039 | 0,967 / 0,984 |
| ResNet-50 (3 semillas) | 0,0619 ± 0,0008 [0,0586; 0,0669] | 10,9 | 0,793 | 0,982 | 0,895 / 0,911 | 0,0086 [0,0049; 0,0124] | 1,000 |
| DeiT-S (3 semillas) | 0,0456 ± 0,0007 [0,0401; 0,0535] | 6,5 | 0,866 | 0,986 | 0,915 / 0,929 | 0,0060 [0,0025; 0,0099] | 1,000 |
| Diferencia pareada ResNet − DeiT [IC 95 %] | 0,0163 [0,0119; 0,0193] | | | | reg: −0,0207 [−0,0267; −0,0120] | 0,0026 [0,0006; 0,0051] | |

: Test interno (4.670 imágenes, 545 grupos). Media ± d.e. entre 3 semillas e IC 95 % por bootstrap agrupado (B = 2.000); los baselines tienen una sola corrida. log10-MAE y Spearman globales. Fuente: `reports/tables/metrics_test.md` y `paired_test.md`.

Figuras:

- dispersión log-log de V̂ vs V (`reports/figures/scatter_test.png`);
- error por banda (`error_by_band_test.png`);
- matrices de confusión (`confusion_test.png`);
- desempeño por fuente (`per_source_test.png`);
- curvas de aprendizaje (`learning_curves.png`).

Ambas redes superan con holgura a todos los baselines: el mejor, el físico
HGB, tiene un log10-MAE de 0,137, frente a 0,062 de ResNet-50 y 0,046 de
DeiT-S. DeiT-S tiene menor error (error multiplicativo típico ×1,11 frente a
×1,15), menor falso-seguro (0,60 % frente a 0,86 %) y mayor macro-F1 de
bandas. El criterio de 4.2 es la media del log10-MAE por fuente; los reportes
entregan el log10-MAE global y el de cada fuente, y **en los cuatro la
diferencia pareada excluye 0** a favor de DeiT-S (§4.4). Según la regla, los
modelos **difieren**. En términos absolutos la diferencia es pequeña: los dos
tienen δ<1,25 sobre 0,79 y Spearman sobre 0,98.

## 4.4 Por fuente, por banda y externos

**Por fuente (test).** DeiT-S tiene menor log10-MAE que ResNet-50 en las
tres fuentes: CS-Fog 0,045 frente a 0,052, Foggy Cityscapes 0,020 frente a
0,050 y RESIDE 0,064 frente a 0,075. Las diferencias pareadas excluyen 0 en
las tres (CS-Fog 0,0068 [0,0027; 0,0088]; Foggy 0,0298 [0,0242; 0,0320];
RESIDE 0,0113 [0,0075; 0,0149]) y en el subconjunto de etiqueta métrica
(0,0206 [0,0159; 0,0224]). Dentro de cada fuente ambos ordenan bien la
visibilidad (Spearman 0,99 en CS-Fog, 0,94 en Foggy y 0,92–0,94 en RESIDE),
así que no se limitan a reconocer el dataset. **Por banda**
(`error_by_band_test.png`), en la banda crítica los dos modelos tienen errores
parecidos; la ventaja de DeiT-S se concentra en Precaución y Aceptable.

**HazeRD (externo métrico, 75 imágenes de 15 escenas).** Ambos modelos
fallan: log10-MAE 0,817 ± 0,023 (ResNet-50) y 0,839 ± 0,021 (DeiT-S), con
error multiplicativo ×6,6 y ×6,9. Son peores que predecir la media (0,429) y
que el baseline físico Ridge (0,422). Conservan parte del orden (Spearman
0,60 y 0,61, frente a 0,90 del Ridge), pero casi todas sus predicciones
quedan bajo 100 m para visibilidades reales de 38 a 766 m
(`scatter_ext.png`): subestiman, y por eso el falso-seguro es 0. En HazeRD la
diferencia pareada favorece a ResNet-50 (−0,0228 [−0,0434; −0,0033]).
**H3 no se apoya:** con niebla densa (banda crítica) los dos modelos tienen
errores parecidos, y entre datasets DeiT-S solo es mejor dentro de las
fuentes de entrenamiento; en HazeRD el IC excluye 0 a favor de ResNet-50.
Pares de O-HAZE en los que $\hat V(\text{con niebla}) < \hat V(\text{limpia})$:
0,993 ± 0,013 de los 45 pares para ambos modelos (baseline físico: 0,956),
con una razón media de −0,63 décadas (`ohaze_pairs.md`).
Sensibilidad a ε = 0,02 (verdad y predicción se escalan por el mismo factor
1,306, así que el log10-MAE no cambia y solo cambian las métricas de bandas):
el macro-F1 baja de 0,895 a 0,806 (ResNet-50) y de 0,915 a 0,863 (DeiT-S), y
el falso-seguro sube a 0,054 y 0,056. El orden entre modelos se mantiene
(`eps_sensitivity_test.md`).
Galería con V real y V̂ en metros para ambos modelos: los 16 ejemplos fijos
de `reports/figures/interpret_examples_neg.png`.

## 4.5 Atajos e interpretabilidad

| Prueba | ResNet-50 | DeiT-S | Lectura prevista |
|---|---|---|---|
| Sonda de fuente (exactitud balanceada vs azar) | 0,92 (azar 0,33) | 0,96 (azar 0,33) | ≫ azar: el backbone codifica el dataset |
| Contrafactual: \|Δlog10 V̂\| medio y pendiente log10 V̂ vs log10 V (física / uniforme) | 0,14; 0,985 / 1,10 | 0,257; 1,00 / 1,13 | \|Δ\| < 0,05 y pendiente uniforme ≈ física: usa contraste global |
| Atribución −ŷ en tercil lejano / cielo / tercil cercano (masa/área) | 1,5 / 1,5 / 0,47 | 0,97 / 2,0 / 1,3 | H1 |
| Atribución −ŷ en calzada / vehículos / edificios | 0,57 / 0,67 / 1,5 | 1,6 / 0,34 / 0,76 | qué región usa cada modelo |

: Pruebas de atajo e interpretabilidad. Sonda: 900 imágenes de val, k = 3 fuentes. Contrafactual: 200 pares de test, semilla 0. Atribución: 16 ejemplos fijos de test (7–8 por mapa), semilla 0. Fuente: `probe_source.md`, `counterfactual.md` y `attribution.md`.

**H1 no se apoya.** El mapa −ŷ de ResNet-50 se concentra más en el tercil
lejano (razón 1,5) que el de DeiT-S (0,97). DeiT-S solo pone más peso relativo
en el cielo (2,0 frente a 1,5), que ocupa ~2 % del área, y en la calzada
(1,6). Con 7–8 imágenes por mapa, la evidencia es débil. **H2 no se apoya con
la regla de 4.2.** Al pasar a niebla uniforme la predicción cambia en
promedio ×1,38 (ResNet-50) y ×1,81 (DeiT-S), por encima del umbral de 0,05
décadas. La sonda identifica la fuente muy por encima del azar, pero el
baseline por fuente (0,251) queda lejos de las redes. Ambos modelos usan la
atenuación según la distancia, DeiT-S más que ResNet-50; aun así, con niebla
uniforme siguen ordenando bien (Spearman 0,975 y 0,911), así que también
usan el contraste global.

Figura: Grad-CAM de ambos modelos y rollout de DeiT sobre los
mismos ejemplos (`reports/figures/interpret_examples_neg.png`).

## 4.6 Autos: detectabilidad frente a distancia

Con 475 frames de Cityscapes val × 6 condiciones (3.904 autos GT), el
detector deja de ver autos bastante antes de la visibilidad MOR
(`cars_vdet.md`, `cars_recall_vs_distance.png`): $V_{det}$ = 115 m
[97; 120] con MOR de 300 m (β = 0,01), 71 m [63; 74] con 150 m (β = 0,02),
50 m [47; 51] con 100 m y 33 m [32; 35] con 50 m (CS-Fog). La razón
$V_{det}$/MOR va de 0,39 a 0,66: la MOR describe un objeto negro grande y
contrastado, y un auto real es más chico y de menor contraste. Con la SSD de
§1, la velocidad cuya distancia de detención cabe en $V_{det}$ es de 79, 57,
44 y 33 km/h, frente a 143, 93, 72 y 44 km/h si se usara la MOR. Se esperaba
que $V_{det}$ quedara censurada por encima del rango de los bins (hasta
150 m) con β = 0,005 y 0,01; ocurrió solo con β = 0,005 (> 145 m).

## 4.7 Velocidad vs precisión

En CPU (fp32, lote 1) DeiT-S tarda 29,9 ms por imagen y ResNet-50, 54,7 ms;
en GPU (fp16, lote 1), 5,1 y 7,6 ms. Con lote 64 en GPU procesan 2.740 y
1.445 imágenes/s (`latency.md`). Aplicando la regla de 4.2: DeiT-S tiene
menor log10-MAE (sus IC no se solapan) y menor falso-seguro (IC solapados), y
además es más rápido en CPU. **Se elige DeiT-S**
(`pareto_speed_accuracy.png`).

## 4.8 Por qué ResNet y ViT difieren (o no)

Interpretamos el resultado con el siguiente marco, definido de antemano:

1. **Sesgo inductivo.** La ResNet agrega información local con campos
   receptivos crecientes y la promedia al final (GAP). DeiT mezcla todos los
   parches desde la primera capa. La niebla es una señal de **baja
   frecuencia**: velo hacia A y pérdida de contraste que crece con la
   distancia. En principio, ambos pueden capturarla: la ResNet mediante
   estadísticas globales tras el GAP y DeiT mediante la atención. Una ventaja
   del ViT debería verse en H1, con más atribución lejana, y en H3, con mejor
   transferencia a otro render.
2. **Datos y preentrenamiento.** Con ~10⁴ imágenes y preentrenamiento solo en
   IN-1k, el sesgo de localidad de la CNN suele ser una ventaja en eficiencia
   de datos. Si la CNN gana, este es el primer candidato a explicación.
3. **Atajos.** Si ambos modelos quedan cerca del baseline físico o del
   baseline por fuente, o si el contrafactual muestra que responden igual a
   niebla uniforme, la comparación de arquitecturas está dominada por el
   atajo. En ese caso la conclusión correcta es que el problema, tal como lo
   definen estos datos, no discrimina entre arquitecturas.
4. **Por fuente y por banda.** Una diferencia concentrada en una sola fuente,
   por ejemplo en RESIDE con etiqueta equivalente, pesa menos que una en las
   fuentes métricas.

**Aplicación del marco.** En la distribución de entrenamiento DeiT-S es
mejor, y la diferencia excluye 0 en las tres fuentes, también en las de
etiqueta métrica, así que no depende de RESIDE (punto 4). La ventaja de
eficiencia de datos de la CNN que anticipaba el punto 2 no apareció: con la
misma receta y los mismos datos, el ViT preentrenado en IN-1k aprendió mejor.
Sobre el punto 1, el contrafactual sugiere un mecanismo: DeiT-S es más
sensible a la estructura de la niebla según la distancia (×1,81 frente a
×1,38), algo consistente con que la atención global integre la atenuación en
toda la imagen. La atribución por terciles, en cambio, no muestra más
atención lejana en DeiT-S (H1 no se apoya). El punto 3 limita el alcance: las
redes superan por lejos a los baselines en el test interno, pero la sonda
muestra que codifican el dataset, y en HazeRD ambas caen por debajo de la
media, con ResNet-50 apenas menos mal. Concluimos que DeiT-S es mejor **en
niebla sintética del mismo dominio**; fuera de él, el problema tal como lo
definen estos datos no discrimina entre arquitecturas, porque ambas fallan.
Una salvedad: el LR elegido para ResNet-50 quedó en el borde de la grilla
(3 × 10⁻⁴, con el mejor paso en el último del barrido), por lo que ResNet-50
podría estar algo subajustada.

# 5. Trabajo futuro

1. **Validación con visibilidad medida.** Instalar una cámara fija en La
   Pólvora con hitos (postes o señales) a distancias conocidas de 50, 100 y
   200 m, o junto a un visibilímetro. Así operan las cámaras de visibilidad,
   y es la versión operativa de "objetos a distancia conocida".
   Como alternativa, usar el subconjunto de baja visibilidad de AIR-VIEW con
   split por estación.
2. **Distancia a los autos.** Regresión de la distancia por auto con ResNet vs ViT como backbone en
   KITTI 3D, evaluación con niebla en Multifog KITTI y test en niebla real en
   Seeing Through Fog (DENSE). Agregar una comparación CNN vs ViT entre
   detectores (p. ej. YOLOS) bajo el mismo protocolo de recall frente a
   distancia.
3. **Niebla más realista.** Usar niebla heterogénea (β con ruido espacial),
   gradiente vertical, render en RGB lineal e iluminación marina. Además,
   invertir el ASM en O-HAZE con profundidad monocular métrica para obtener
   una V aproximada con incertidumbre.
4. **Incertidumbre calibrada** (p. ej. predicción conforme), para que la demo
   entregue intervalos en vez de la dispersión entre semillas, y **detección
   de fuera de distribución** (noche, lluvia).
5. **Escala.** Agregar DeiT-B o pares IN-21k emparejados, y destilación a un
   modelo para CPU embebida.

# Limitaciones

- **Sintético → real.** Toda la niebla etiquetada es sintética y homogénea;
  la camanchaca no lo es.
- **Sin validación con visibilidad medida** en niebla real.
- **RESIDE aporta visibilidad equivalente**, no metros.
- **HazeRD** tiene 15 escenas y una convención de umbral por confirmar.
- **Fuera del dominio de entrenamiento los modelos no sirven:** en HazeRD son
  peores que predecir la media y subestiman la visibilidad.
- **Etiquetas en los bordes de banda:** RESIDE β = 0,06 → 49,9 m.
- **Confusión dataset ↔ banda** en las fuentes oficiales. Se mitiga con
  CS-Fog, pero no se elimina por completo, porque RESIDE sigue siendo < 75 m.
- **Distancia estéreo imprecisa más allá de ~100 m**, lo que censura $V_{det}$
  para la niebla leve.
- **Solo imágenes diurnas.**
- **Cómputo limitado** a una GPU T4 de Colab: 3 semillas y un barrido de 3
  valores de LR.

# Referencias

::: {#refs}
:::

# Declaración de uso de IA

Parte del código (paquete `camanchaca`, pruebas, notebooks y demo), de la
documentación y del borrador de este informe se escribió con apoyo de
asistentes de IA, bajo instrucciones y revisión del equipo, conforme a las
normas del curso y de la PUCV. El diseño experimental lo decidió el equipo: datos, etiqueta MOR, par
de modelos, protocolo, hipótesis y reglas de decisión. Los experimentos los
ejecutó el equipo en Google Colab, y **todas las cifras provienen de las
tablas generadas por el código** en `reports/`. Ninguna fue escrita a mano. Ambos integrantes revisaron y pueden explicar cada
componente. La responsabilidad por el contenido es del equipo.
