# Costo computacional y latencia

Pesos aleatorios (la latencia no depende de los valores). GMACs con `torch.utils.flop_counter` a 224×224 y lote 1 (MACs = FLOPs/2). Latencia = mediana por lote tras calentamiento; CPU en fp32, GPU en fp16. `physics_features` = extracción de las features del baseline físico (una imagen 256×256, CPU). Máquina: `x86_64`.

| Modelo | Dispositivo | Precisión | Lote | Parámetros (M) | GMACs (224×224) | ms/lote (mediana) | ms/imagen | imágenes/s | Repeticiones |
|---|---|---|---|---|---|---|---|---|---|
| resnet50 | cpu | fp32 | 1 | 23.5 | 4.09 | 54.7 | 54.7 | 18.3 | 20 |
| resnet50 | cpu | fp32 | 64 | 23.5 | 4.09 | 2815 | 44 | 22.7 | 4 |
| resnet50 | cuda | fp16 | 1 | 23.5 | 4.09 | 7.64 | 7.64 | 131 | 20 |
| resnet50 | cuda | fp16 | 64 | 23.5 | 4.09 | 44.3 | 0.692 | 1445 | 20 |
| deit_s | cpu | fp32 | 1 | 21.7 | 4.6 | 29.9 | 29.9 | 33.5 | 20 |
| deit_s | cpu | fp32 | 64 | 21.7 | 4.6 | 1376 | 21.5 | 46.5 | 7 |
| deit_s | cuda | fp16 | 1 | 21.7 | 4.6 | 5.12 | 5.12 | 195 | 20 |
| deit_s | cuda | fp16 | 64 | 21.7 | 4.6 | 23.4 | 0.365 | 2740 | 20 |
| physics_features | cpu | fp64 | 1 | 0 | — | 22.9 | 22.9 | 43.6 | 20 |
