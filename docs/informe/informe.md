---
title: "Camanchaca-Predict: estimación de la visibilidad bajo niebla con ResNet-50 y DeiT-S"
subtitle: "Informe técnico final — Aprendizaje Automático II, PUCV 2026-2 (Grupo 5)"
author:
  - Tomás Ignacio Moraga
  - Nelson Mauricio Fuentes
date: "Noviembre de 2026 (borrador del 3-nov-2026: resultados pendientes)"
lang: es
bibliography: refs.bib
link-citations: true
geometry: margin=2cm
fontsize: 11pt
---

<!--
ESTADO: las secciones 1, 2, 3 y 5 están redactadas. La sección 4 contiene el
diseño de resultados (tablas, figuras, hipótesis y reglas de decisión fijadas
ANTES de ver los datos) con marcadores [COMPLETAR con reports/...] que se
llenan desde las tablas generadas por `camanchaca report` tras la evaluación
final en Colab. No escribir ninguna cifra que no salga de reports/.
Presupuesto de páginas: ver docs/informe/README.md.
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

: Fuentes de datos. Conteos por split, fuente y banda: [COMPLETAR con reports/tables/data_counts.md].

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
[COMPLETAR con el estado final: fechas de corrida, commit, `reports/RESULTADOS.md`].

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

| Modelo | log10-MAE macro | MdAPE | δ<1,25 | Spearman (por fuente) | macro-F1 (reg / CORN) | Falso-seguro | AUC-PR V<100 |
|---|---|---|---|---|---|---|---|
| Media global / por fuente / banda mayoritaria | [COMPLETAR] | | | | | | |
| Físico (Ridge / HGB) | [COMPLETAR] | | | | | | |
| ResNet-50 (3 semillas) | [COMPLETAR] | | | | | | |
| DeiT-S (3 semillas) | [COMPLETAR] | | | | | | |
| Diferencia pareada ResNet − DeiT [IC 95 %] | [COMPLETAR con reports/tables/paired_test.md] | | | | | | |

: Test interno, media ± d.e. entre semillas e IC 95 % por bootstrap agrupado. [COMPLETAR con reports/tables/metrics_test.md].

Figuras previstas:

- dispersión log-log de V̂ vs V (`reports/figures/scatter_test.png`);
- error por banda (`error_by_band_test.png`);
- matrices de confusión (`confusion_test.png`);
- desempeño por fuente (`per_source_test.png`);
- curvas de aprendizaje (`learning_curves.png`).

[COMPLETAR: 3–4 frases de lectura de la tabla, sin adjetivos que la tabla no
respalde.]

## 4.4 Por fuente, por banda y externos

[COMPLETAR con reports/tables/metrics_test.md (desglose por fuente y por
`label_quality`) y reports/tables/metrics_ext.md (HazeRD)].
Pares de O-HAZE en los que $\hat V(\text{con niebla}) < \hat V(\text{limpia})$:
[COMPLETAR con reports/tables/ohaze_pairs.md].
Sensibilidad a ε = 0,02 (verdad y predicción se escalan por el mismo factor
1,306, así que el log10-MAE no cambia y solo cambian las métricas de bandas):
[COMPLETAR con reports/tables/eps_sensitivity_test.md].
Galería de aciertos y errores, con V real y V̂ en metros, para ambos modelos:
[COMPLETAR con figura].

## 4.5 Atajos e interpretabilidad

| Prueba | ResNet-50 | DeiT-S | Lectura prevista |
|---|---|---|---|
| Sonda de fuente (exactitud balanceada vs azar) | [COMPLETAR] | [COMPLETAR] | ≫ azar: el backbone codifica el dataset |
| Contrafactual: \|Δlog10 V̂\| medio y pendiente log10 V̂ vs log10 V (física / uniforme) | [COMPLETAR] | [COMPLETAR] | \|Δ\| < 0,05 y pendiente uniforme ≈ física: usa contraste global |
| Atribución en tercil lejano + cielo (masa/área) | [COMPLETAR] | [COMPLETAR] | H1 |
| Atribución en calzada / vehículos / edificios | [COMPLETAR] | [COMPLETAR] | qué región usa cada modelo |

: Pruebas de atajo e interpretabilidad. [COMPLETAR con reports/tables/probe_source.md, counterfactual.md y attribution.md].

Figura prevista: Grad-CAM de ambos modelos y rollout de DeiT sobre los
mismos ejemplos (`reports/figures/interpret_examples_neg.png`).

## 4.6 Autos: detectabilidad frente a distancia

[COMPLETAR con reports/tables/cars_recall.md, cars_vdet.md y
reports/figures/cars_recall_vs_distance.png]. Con β = 0,005 y 0,01 (MOR de
600 y 300 m), se espera que $V_{det}$ quede censurada por encima del rango de
los bins (hasta 150 m), donde la distancia estéreo es poco precisa. La
comparación informativa es con β = 0,02 y con CS-Fog a 50 y 100 m.

## 4.7 Velocidad vs precisión

[COMPLETAR con reports/tables/latency.md y
reports/figures/pareto_speed_accuracy.png] y aplicar la regla de 4.2.

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

[COMPLETAR: 1 párrafo que aplique el marco a los resultados. Si el IC incluye
0, decir explícitamente "no encontramos diferencia" y explicar por qué es
plausible.]

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
- **HazeRD** tiene 14–15 escenas y una convención de umbral por confirmar.
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
