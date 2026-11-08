"""Interpretabilidad de los regresores de visibilidad.

- `gradcam`: Grad-CAM nativo para REGRESIÓN con signo explícito
  (−ŷ = evidencia de MENOR visibilidad; +ŷ = evidencia de MAYOR visibilidad).
- `rollout`: attention rollout (Abnar & Zuidema, ACL 2020) para DeiT/ViT de timm.
- `examples`: conjunto FIJO de ejemplos estratificado por fuente × banda.
- `analysis`: fracción de atribución por tercil de distancia y por clase gtFine,
  y paneles de figuras.
- `counterfactual`: niebla dependiente de la distancia vs niebla uniforme
  emparejada en canal oscuro (¿el modelo usa la profundidad o solo el contraste?).
- `commands`: subcomandos `camanchaca interpret` y `camanchaca counterfactual`.
"""
