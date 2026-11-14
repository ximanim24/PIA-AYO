# Dataset — Hiper-heurística de Selección para TSP (Proyecto Final AYO)

Instancias generadas por `generate_dataset.py`. Tres distribuciones de puntos
(`uniform`, `clustered`, `border`), tamaños n ∈ {5, 10, 15, 20, 30, 40, 60, 80, 100},
≈18-19 réplicas por combinación (n, distribución) = 500 instancias totales
(27 grupos no dividen 500 exacto, así que las réplicas se reparten lo más parejo posible: 14 grupos con 19, 13 grupos con 18).

**Los estudiantes generan las etiquetas (`best_algorithm`) ellos mismos**, corriendo
sus propios tres algoritmos de Tarea 2 sobre las instancias provistas — este
dataset NO incluye costos ni ganador precalculado para el alumno.

## Archivos que reciben los estudiantes

- `data/train.csv` (351 filas) y `data/test.csv` (149 filas): `instance_id`, `n`,
  `distribution`, y 4 características numéricas (`mean_pairwise_dist`, `std_coords`,
  `mean_centroid_dist`, `mean_nn_dist`) — **sin** columnas de costo ni de etiqueta.
- `data/instances/inst_NNNN.csv`: coordenadas (x, y) de cada instancia — los
  estudiantes corren sus tres algoritmos de Tarea 2 sobre cada archivo para
  obtener `cost_2opt_hill_climbing`, `cost_sa_2opt`, `cost_mst_2opt`, y derivan
  `best_algorithm` (el de menor costo) ellos mismos.

## Archivos solo para el instructor

- `data/instructor_reference/train_with_labels.csv` y `test_with_labels.csv`:
  misma partición, con costos y `best_algorithm` calculados con las
  implementaciones de referencia de `generate_dataset.py` — **no distribuir**,
  sirve para verificar que las etiquetas que reporten los estudiantes sean
  razonables (sus implementaciones de Tarea 2 pueden diferir ligeramente de la
  referencia, así que no se espera un match exacto, solo que el patrón general
  — qué algoritmo domina en qué distribución/tamaño — sea consistente).

## Balance de clases de referencia (instructor)

```
Total:  2opt_hill_climbing: 341 (68%)   mst_2opt: 97 (19%)   sa_2opt: 62 (12%)
Train:  2opt_hill_climbing: 240         mst_2opt: 70         sa_2opt: 41
Test:   2opt_hill_climbing: 101         mst_2opt: 27         sa_2opt: 21
```

Con las implementaciones de los estudiantes, estos números pueden variar algo —
es esperado y parte del análisis (sus algoritmos de Tarea 2 no son
necesariamente idénticos a la referencia).

## Reproducibilidad

- `python3 generate_dataset.py` regenera instancias + `data/instructor_reference/*_with_labels.csv`
  completos, con semilla fija (`SEED = 42`), stdlib puro (sin numpy/scipy).
- `python3 make_student_csvs.py` deriva `data/train.csv` / `data/test.csv`
  (versión pública, sin costos/etiqueta) a partir de los archivos de referencia.
