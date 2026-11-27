# Pendientes de la presentación final (26-nov-2026)

Fuente: `index.html`. Para encontrar lo que falta: `grep -n "COMPLETAR" index.html`.
Los bloques grandes son un `<div class="pending">…</div>`, y cada uno se
reemplaza **entero** por la figura.

Las cifras vienen de `reports/RESULTADOS.md`:

- notebooks 01–04: commit `e5b8106`;
- notebook 05 (autos): commit `4352a36`.

Si se vuelven a correr, hay que revisar las láminas 05–12 y el anexo.

## Reemplazo tipo para una figura

```html
<figure class="fig reveal" style="--d:.2s;align-self:start">
    <img src="assets/NOMBRE.png" alt="Descripción breve">
    <figcaption>Qué muestra</figcaption>
</figure>
```

La imagen va en `presentations/2026-11-26_final/assets/` y se enlaza con ruta
relativa.

## Falta

| # | Lámina | Qué falta | De dónde sale |
|---|---|---|---|
| 6 | 13 Demo | Si la app se reinicia, actualizar el enlace gradio.live (aparece 2 veces en `index.html`) y, si se quiere, una captura con un resultado | `assets/demo_app.jpg` |
| 7 | 01 y 02 | Decidir si se da crédito a las fotos de prensa | — |

## Respaldo

La versión anterior a los cambios estéticos quedó solo en la copia local
(`presentations/2026-11-26_final_respaldo_v1/`); no se subió al repositorio.

## Hecho

- Fotos: portada (`foto_puerto_niebla.jpg`) y lámina 02 (`foto_camanchaca_valparaiso.jpg`, `foto_la_polvora.jpg`).
- Lámina 11, autos: figura `cars_recall_rel.png` (panel derecho de `cars_recall_vs_distance.png`), tabla de V<sub>det</sub> y lectura.
- Lámina 14: viñeta de trabajo futuro sobre autos.
- Portada: frase central revisada.
- Lámina 13: captura de la app con el resultado de la foto 04 (`assets/demo_app.jpg`) y enlace clicable a la demo, actualizado el 26-nov.
- Lámina 05b (nueva): `ejemplos_bandas.png`, un ejemplo de test por banda y por fuente.
- Lámina 09b: `aciertos_errores.png`, 3 aciertos y 3 errores con los mismos ejemplos para ambos modelos (cifras en `reports/tables/aciertos_errores.csv`).
  Ambas figuras se generaron con `celda_colab_fotos.py`, usando las predicciones de `runs/main` del Drive (coinciden con `metrics_test.md`) y las imágenes de test de los `.tar`.
- Gráficos con el estilo de las láminas (`assets/g_*.png`), generados con `graficos_deck.py --runs <Drive>/runs/main`: curvas (07), confusión (08b), error por dataset y HazeRD (09), autos (11), velocidad vs precisión (12).
- Tipografías locales en `assets/fonts/` (Barlow, OFL): el deck ya no depende de internet.

## Antes de presentar

- [x] `grep -n "COMPLETAR\|class=\"review\"" index.html` no devuelve nada.
- [ ] Recorrer las láminas a pantalla completa (tecla F) en el computador de la sala.
- [x] Exportar a PDF y dejar `final.pdf` en esta carpeta.
- [ ] Ensayos cronometrados (meta 14:45–15:15) → `ensayos.md`.

## Atajos

- Flechas, espacio, PgUp/PgDn: navegar. `index.html#11` abre la lámina 11 y `#A1` la primera del anexo.
- **N**: notas del orador. **E**: editar texto; **Ctrl+S** descarga el HTML editado. **F**: pantalla completa.
