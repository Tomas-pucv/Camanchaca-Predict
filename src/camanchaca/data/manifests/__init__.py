"""Constructores de manifiestos por fuente (uno por dataset).

Cada módulo lee el dataset ORIGINAL (directorio o zip), guarda una copia
redimensionada en `<data_root>/images/<fuente>/...` (JPEG, lado corto 288) y
devuelve un DataFrame con el esquema canónico de `camanchaca.data.schema`
(rutas RELATIVAS a la raíz de datos). Los CSV resultantes van a
`data/manifests/<fuente>.csv` y luego `camanchaca split` los combina.

Módulos
-------
reside_ots         RESIDE-OTS oficial ('id_A_beta'); aborta ante reempaques.
foggy_cityscapes   Foggy Cityscapes oficial (beta 0.005/0.01/0.02), desde zip o carpeta.
cs_fog             Niebla propia sobre Cityscapes con distancia métrica (estéreo).
cars               Autos de gtFine (caja + distancia) e imágenes por condición.
hazerd             HazeRD (profundidad métrica; renderizado en Python si hace falta).
ohaze              O-HAZE (pares hazy/GT reales, sin etiqueta).
folder             Carpeta cualquiera (fotos propias), con CSV opcional de V medida.
"""
