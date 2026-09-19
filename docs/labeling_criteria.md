# Criterio de etiquetado: de imágenes con niebla a visibilidad en metros

## 1. El problema: ningún dataset trae visibilidad en metros

Los benchmarks de niebla (RESIDE, O-HAZE, HazeRD, Foggy Cityscapes) fueron
diseñados para **dehazing**: pares (imagen con niebla, imagen limpia). Ninguno
incluye la variable que la seguridad vial necesita: **a cuántos metros de
distancia deja de verse un objeto**. Por eso el proyecto debe **definir y
validar su propio criterio de etiquetado** — este documento es ese criterio.

## 2. Marco físico

### Ley de Koschmieder

Un objeto oscuro observado contra el fondo del cielo a distancia `d` tiene
contraste aparente:

```
C(d) = exp(-beta * d)
```

donde `beta` [1/m] es el coeficiente de extinción atmosférica (dispersión +
absorción). La **visibilidad meteorológica** se define como la distancia a la
que el contraste cae a un umbral `epsilon`:

```
V = -ln(epsilon) / beta
```

Con `epsilon = 0.02` (umbral conservador, orientado a seguridad vial):

```
V ≈ 3.912 / beta      [metros]
```

**Sensibilidad del umbral:** con `epsilon = 0.05` (estándar MOR de aviación)
la fórmula es `V = 3.0 / beta`, es decir visibilidades ~23 % menores. El
proyecto usa `epsilon = 0.02` por ser conservador (subestimar la visibilidad
es más seguro que sobreestimarla) y reporta la sensibilidad.

### Modelo atmosférico de dispersión (ASM)

RESIDE sintetiza sus imágenes con niebla a partir de escenas limpias `J`,
mapas de profundidad `d` y luz atmosférica `A`:

```
I(x) = J(x) * t(x) + A * (1 - t(x)),     t(x) = exp(-beta * d(x))
```

Cada imagen con niebla de RESIDE-OTS lleva en su nombre de archivo el `beta`
usado en la síntesis (patrón `escena_A_beta.jpg`, p. ej. `0001_0.85_0.2.jpg`).
Ese `beta` es la **ancla física**: aplicando Koschmieder obtenemos una
**etiqueta en metros** con significado físico directo.

## 3. Construcción de la etiqueta

Para cada imagen `escena_A_beta.jpg` de RESIDE-OTS:

1. Parsear `beta` del nombre de archivo (notebook 02).
2. Etiqueta de regresión: `V = 3.912 / beta` [m]; el modelo aprende `log10(V)`
   (estabiliza numéricamente el rango ~10–2000 m).
3. Etiqueta de clasificación: banda de seguridad según la tabla siguiente.

## 4. Bandas de seguridad y distancia de detención (SSD)

La distancia de detención en piso mojado (μ = 0.4, reacción 2.5 s) es:

```
SSD = v * 2.5 + v^2 / (2 * 0.4 * 9.81)      [v en m/s]
```

| Velocidad | SSD (m) |
|---|---|
| 60 km/h | ≈ 77 |
| 80 km/h | ≈ 119 |
| 100 km/h | ≈ 168 |
| 120 km/h | ≈ 225 |

De ahí las bandas (el Camino La Pólvora se conduce a 80–110 km/h):

| Banda | Rango | Justificación operativa |
|---|---|---|
| `critico` | V < 50 m | Ni frenando a 60 km/h se detiene a tiempo |
| `alto_riesgo` | 50–100 m | Seguro solo a ≤ 60 km/h |
| `precaucion` | 100–200 m | Precaución hasta ~100 km/h |
| `aceptable` | ≥ 200 m | Margen incluso a 120 km/h |

## 5. Validación de las etiquetas

Las etiquetas dependen de que el `beta` del filename sea realmente el
coeficiente del ASM en unidades 1/m. Tres validaciones (notebook 02):

1. **Consistencia perceptual (siempre):** el brillo medio del canal oscuro
   (proxy de densidad de niebla, He et al. 2011) debe correlacionar
   positivamente con `beta`. Se reporta el coeficiente de Spearman sobre una
   muestra; valores altos (ρ > 0.7) indican etiquetas consistentes con la
   percepción.
2. **Reproducción del ASM (si hay profundidad):** en el subconjunto indoor
   (ITS, escenas con depth maps de NYU) se puede re-renderizar la niebla con
   `t = exp(-beta*d)` y comparar contra la imagen entregada; confirma unidades
   y fórmula del generador.
3. **Sanity check de rangos:** distribución de V por banda; una banda casi
   vacía (p. ej. < 50 m) activa la mitigación R3 (re-síntesis).

## 6. Split train / val / test (anti-fuga)

RESIDE-OTS contiene **múltiples versiones con niebla por escena limpia**
(mismo `scene_id`, distinto `beta`). Una imagen de test "hereda" el contenido
de la escena vista en train → inflación de métricas. Por eso:

- Unidad de split = **escena** (`scene_id`), nunca la imagen.
- Proporciones 70 / 15 / 15, semilla 42.
- Entre 50 permutaciones se elige la de menor sesgo en la distribución de
  bandas entre conjuntos.
- Tests automáticos (`tests/test_splits.py`) verifican cero escenas
  compartidas; los CSV de splits se versionan en `data/splits/`.

## 7. Limitaciones y honestidad del enfoque

- **Brecha sintético → real:** RESIDE es niebla homogénea sintetizada sobre
  escenas urbanas/exteriores de Asia; la camanchaca es niebla advectiva,
  heterogénea y marina. O-HAZE (niebla real, 45 escenas) se usa como test
  externo **cualitativo** para inspeccionar el comportamiento, no como
  benchmark métrico (no tiene etiqueta en metros).
- **Asumimos** que el `beta` del filename está en 1/m y corresponde al ASM
  estándar; la validación de la sección 5 da evidencia, no prueba definitiva.
- **Sin datos chilenos** en esta fase: la generalización al Camino La Pólvora
  se plantea como roadmap (capturas propias / cámaras DTV) y como análisis de
  riesgo R1/R9, no como resultado.
- La conversión a banda usa cortes fijos (50/100/200 m); una futura
  calibración con velocidades reales de la ruta podría refinarlos.

## 8. Referencias

- Koschmieder, H. (1924). *Theorie der horizontalen Sichtweite*.
- McCartney, E. J. (1976). *Optics of the Atmosphere: Scattering by Molecules and Particles*. Wiley.
- He, K., Sun, J., Tang, X. (2011). *Single Image Haze Removal Using Dark Channel Prior*. IEEE TPAMI.
- Li, B. et al. (2018). *RESIDE: A Benchmark for Single Image Dehazing*. arXiv:1712.04143.
- Ancuti, C. et al. (2018). *NTIRE 2018 Image Dehazing Challenge Report* (dataset O-HAZE).
- WMO No. 8. *Guide to Instruments and Methods of Observation* (definición de visibilidad meteorológica).
