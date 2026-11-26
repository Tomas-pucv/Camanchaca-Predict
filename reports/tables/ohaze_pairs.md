# Pares O-HAZE: ¿V̂(niebla) < V̂(limpia)?


O-HAZE no tiene etiqueta métrica de visibilidad; la prueba mínima es ORDINAL: para cada escena, la imagen con niebla real debe recibir una visibilidad predicha MENOR que su imagen limpia (GT). Se informa la fracción de pares que cumplen (media ± sd entre semillas; un empate cuenta 0,5; 0,5 = azar) y la razón media log10(V̂ niebla / V̂ limpia) en décadas (negativa = correcto). 'Válidos' = pares con ambas predicciones finitas y positivas (los demás se excluyen).

| Conjunto | Modelo | Semillas | Pares | Válidos | Fracción V̂(niebla) < V̂(limpia) | log10(V̂ niebla / V̂ limpia) (décadas) |
|---|---|---|---|---|---|---|
| ext | resnet50 | 3 | 45 | 45 | 0.993 ± 0.0128 | -0.626 ± 0.00729 |
| ext | deit_s | 3 | 45 | 45 | 0.993 ± 0.0128 | -0.629 ± 0.0395 |
| ext | majority_band | 1 | 45 | 45 | 0.5 | 0 |
| ext | mean_by_source | 1 | 45 | 45 | 0.5 | 0 |
| ext | mean_global | 1 | 45 | 45 | 0.5 | 0 |
| ext | physics_hgb | 1 | 45 | 45 | 0.956 | -0.66 |
| ext | physics_ridge | 1 | 45 | 45 | 0.956 | -0.631 |
