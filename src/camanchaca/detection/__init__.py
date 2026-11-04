"""Análisis complementario: detección de autos bajo niebla vs distancia.

- `detector`: Faster R-CNN v2 de torchvision (COCO) en lote, con umbral bajo.
- `matching`: IoU y emparejamiento voraz uno-a-uno por puntaje.
- `recall_vs_distance`: recall por bin de distancia y condición, recall
  relativo a 'clear', distancia de detección V_det con censura e IC bootstrap
  agrupado por secuencia, y comparación con V MOR y la distancia de detención.
- `commands`: subcomando `camanchaca cars`.
"""
