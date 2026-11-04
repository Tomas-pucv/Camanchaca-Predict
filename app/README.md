# Demo Gradio — Camanchaca-Predict

Interfaz web que estima la **visibilidad (MOR, metros)** de una foto con niebla
usando el ensamble de semillas de ResNet-50 o DeiT-S entrenado en este
repositorio. Corre en la **CPU de un portátil** (no necesita GPU).

| Salida | Qué es |
|---|---|
| Visibilidad estimada (MOR) | media geométrica de las semillas, en metros (V = 2,996/β) |
| Banda de seguridad | probabilidad media (cabeza ordinal CORN) de Crítico <50 m / Alto riesgo 50–100 / Precaución 100–200 / Aceptable ≥200 m |
| Incertidumbre | dispersión entre semillas como factor multiplicativo (×); no es un IC calibrado |
| Velocidad segura | **ILUSTRATIVA**: velocidad cuya distancia de detención (t = 2,5 s, μ = 0,4) cabe en V̂ |
| Mapa | Grad-CAM de −ŷ (evidencia de MENOR visibilidad; si ninguna región la aporta se avisa "VACÍO" y se muestra la imagen sin colorear) o attention rollout (solo DeiT, escala min–max) |
| Avisos | posible noche, escala de grises, imagen diminuta, proporción extrema, transparencia |

Pestañas: **Estimar visibilidad**, **Simular niebla** (niebla sintética sobre
una foto despejada con profundidad *aproximada* de plano-suelo, y comparación V
fijada vs V̂) y **Acerca de** (supuestos, datos y limitaciones).
**No apto para uso operativo.**

## 1. Ejecutar localmente

```bash
pip install -e ".[demo]"                 # desde la raíz del repo (Python >= 3.10)
python scripts/fetch_weights.py --exp main   # o entrenar: camanchaca train-all ...
python app/app.py --exp main             # abre http://127.0.0.1:7860
python -m camanchaca.demo --exp main     # equivalente, con el paquete instalado
```

Opciones de `app/app.py`:

| Bandera | Uso |
|---|---|
| `--exp main` | descubre `runs/main/<modelo>/seed*/best.pt` (nombre o YAML del experimento) |
| `--ckpt RUTA` | checkpoint explícito, repetible (`--ckpt runs/main/deit_s/seed0/best.pt --ckpt ...`) |
| `--runs-root DIR` | raíz de corridas (por defecto `$CAMANCHACA_RUNS` o `runs/`) |
| `--device auto\|cpu\|cuda` | dispositivo (por defecto `auto`) |
| `--port 7860`, `--server-name 127.0.0.1` | dirección local (`0.0.0.0` para exponerla en la red) |
| `--share` | enlace público temporal de Gradio |

Si no encuentra checkpoints, la interfaz **arranca igual** y explica cómo
obtenerlos (la estimación muestra el error hasta cargarlos). Si el paquete no
está instalado, `app/app.py` usa `src/` directamente (igual requiere torch,
timm y gradio).

Latencia: del orden de décimas de segundo por semilla en una CPU de portátil;
medir con `camanchaca latency --exp configs/experiment/main.yaml --device cpu`.

## 2. Ejecutar en Colab

Abrir `notebooks/06_demo.ipynb`: tras la celda de configuración común (que fija
`CAMANCHACA_RUNS` en Drive) lanza la demo en la celda con
`camanchaca.demo.app.launch_demo(..., share=True)`. Alternativa por CLI (bloquea la
celda hasta detenerla):

```python
!python app/app.py --exp main --share
```

El enlace `*.gradio.live` dura hasta una semana (best effort) y depende de que
la sesión siga viva. Para la presentación conviene correr la demo **en el
portátil** con los pesos descargados y tener grabado un video de respaldo de
2–3 minutos.

## 3. Publicar y descargar los pesos

Los checkpoints (`*.pt`) no se versionan en git. Se publican como assets de un
**GitHub Release** con su sha256 en `app/weights.json`:

```bash
# quien entrenó (con runs/main/<modelo>/seed<k>/best.pt):
python scripts/fetch_weights.py --make-manifest --exp main --release-dir dist/weights \
    --url-template "https://github.com/Tomas-pucv/Camanchaca-Predict/releases/download/v1.0-final/{file}"
gh release create v1.0-final dist/weights/*.pt --title "Pesos v1.0" --notes "ResNet-50 y DeiT-S, 3 semillas"
git add app/weights.json && git commit -m "Manifiesto de pesos v1.0"

# cualquiera (laptop o Colab):
python scripts/fetch_weights.py --exp main
# alternativa sin red pública: copiar desde una carpeta de Drive
python scripts/fetch_weights.py --exp main --drive-dir /content/drive/MyDrive/camanchaca/weights
```

`fetch_weights.py` deja cada archivo en `runs/main/<modelo>/seed<k>/best.pt`,
verifica el sha256 (si no coincide, borra la descarga y termina con error) y
omite lo que ya está. Los checkpoints se cargan con `torch.load`: usar solo
pesos de fuentes confiables y con sha256 verificado.

Licencias: los modelos se entrenan con datasets de uso académico/no comercial
(Cityscapes, RESIDE); revisar sus términos antes de publicar pesos y no subir
nunca imágenes de esos datasets (ver `app/examples/README.md`).

## 4. Pruebas

```bash
pytest tests/test_app.py -q                 # incluye tests lentos que levantan la app
pytest tests/test_app.py -q -m "not slow"   # solo los rápidos
```

Los tests usan checkpoints con pesos aleatorios: verifican que la demo no se cae
con entradas nuevas (RGBA, escala de grises, 8×8, imágenes enormes, EXIF), que
el mapa sale de `camanchaca.interpret` (Grad-CAM para ResNet-50 y DeiT-S, rollout
para DeiT-S), que `python app/app.py` arranca con y sin checkpoints y responde a
`gradio_client`, y que `fetch_weights.py` verifica el sha256; no la calidad de la
predicción.
