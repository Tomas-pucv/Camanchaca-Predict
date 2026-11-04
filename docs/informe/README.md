# Informe técnico final

Fuente: [`informe.md`](informe.md) (Markdown + pandoc). Bibliografía:
[`refs.bib`](refs.bib). PDF de entrega: `docs/informe/informe.pdf`, que se
genera con el comando de abajo y se commitea junto a la fuente en la entrega
final.

## Compilar

Requiere [pandoc](https://pandoc.org/) ≥ 2.11 (incluye `--citeproc`) y una
distribución LaTeX con `xelatex` (TeX Live o MiKTeX; en Colab:
`apt-get install -y pandoc texlive-xetex texlive-lang-spanish`).

```bash
cd docs/informe
pandoc informe.md --citeproc --pdf-engine=xelatex -o informe.pdf
# estilo IEEE opcional (descargar ieee.csl del repositorio de estilos de Zotero/CSL):
pandoc informe.md --citeproc --csl ieee.csl --pdf-engine=xelatex -o informe.pdf
# vista previa rápida sin LaTeX:
pandoc informe.md --citeproc -s --mathjax -o informe.html
```

El encabezado YAML de `informe.md` ya fija `bibliography: refs.bib`,
`lang: es`, márgenes de 2 cm y letra de 11 pt. Las citas usan la sintaxis
`[@clave]`. Si pandoc avisa `citation … not found`, falta la clave en
`refs.bib`. Las entradas marcadas `VERIFICAR` en `refs.bib` se deben confirmar
con la fuente original antes de entregar.

## Estructura y presupuesto de páginas (4–6 páginas sin referencias)

La rúbrica exige exactamente estas 5 secciones. Además se incluyen
Limitaciones, Referencias y la Declaración de uso de IA.

| Sección | Páginas | Responsable | Contenido mínimo |
|---|---|---|---|
| 1. Motivación | ~0,75 | Tomás | camanchaca y La Pólvora, SSD y bandas, formulación, alcance |
| 2. Datos | ~1,25 | Tomás | tabla de fuentes, etiqueta MOR, equivalente vs métrica, CS-Fog, splits, externos, licencias y sesgos |
| 3. Arquitectura y método | ~1 | Nelson | par justo + tabla de protocolo, métricas, bootstrap, baselines, atajos, interpretabilidad, autos, latencia |
| 4. Resultados | ~2 | Nelson (autos: Tomás) | hipótesis y reglas de decisión, tabla principal con IC, por fuente/banda/externos, atajos e interpretabilidad, autos, velocidad vs precisión, **por qué ResNet y ViT difieren (o no)** |
| 5. Trabajo futuro | ~0,5 | Ambos | validación medida, distancia a autos (KITTI/DENSE), niebla realista, incertidumbre |
| Limitaciones + Declaración de IA | ~0,5 | Ambos | lista honesta; uso de IA |

Si el PDF supera las 6 páginas, recortar en este orden:

1. reducir la tabla de protocolo de §3 a sus filas distintivas (el resto está
   en `docs/model_card.md`);
2. pasar a la §4 solo las figuras principales (dispersión, atribución, autos)
   y dejar el resto en `reports/RESULTADOS.md`;
3. acortar §1.

## Cómo completar la sección 4

1. En Colab: evaluación final una sola vez
   (`camanchaca predict|baselines|evaluate … --final`, luego `probes`,
   `latency`, `interpret`, `counterfactual`, `cars` y `report`). Ver la
   secuencia completa en el README principal.
2. Reemplazar cada `[COMPLETAR con reports/…]` copiando las cifras de la tabla
   indicada. **No escribir ninguna cifra que no esté en `reports/`.**
3. Aplicar las reglas de decisión de §4.2 tal como están escritas. No se
   cambian después de ver los resultados.
4. Incluir las figuras desde `reports/figures/` (≥ 150 dpi), con la sintaxis
   `![Leyenda con unidades.](../../reports/figures/scatter_test.png){width=48%}`.
5. Revisión cruzada: el otro integrante lee la sección completa antes de
   cerrar.
