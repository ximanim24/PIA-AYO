# Persona 4 – Machine Learning y análisis del modelo (Hiper-heurística TSP)

Clasificador que predice `best_algorithm` (`2opt_hill_climbing`, `sa_2opt`, `mst_2opt`)
a partir de las características de la instancia. Resultados de la corrida única con las
etiquetas reales de P3 en `report/`.

## Estructura

```
ml/features.py           features, carga de datos y chequeos de las etiquetas
ml/train_eval.py         baseline, CV, selección, test (una vez), regret, importancias,
                         correlación, ablación por grupos y análisis de colapso
ml/extra_analysis.py     análisis complementario: IC del regret, empates, concentración del regret
ml/make_mock_labels.py   etiquetas FALSAS solo para depurar (sus resultados nunca se reportan)
hh_tsp_dataset/data/     train.csv y test.csv del curso
data/labeled/            train_labeled.csv y test_labeled.csv (etiquetas reales de P3)
report/                  ml_results.json, extra_analysis.json, figuras PNG, model_tree.joblib
                         y Seccion_P4_Modelo_y_Analisis.docx (secciones 5, 6, 7.1 y 7.2)
```

## Cómo correrlo

Requiere Python 3, numpy, pandas, scikit-learn, matplotlib y joblib. Ejecutar desde la raíz.

```bash
python -m ml.train_eval        # corrida final con etiquetas reales -> report/
python -m ml.extra_analysis    # análisis complementario -> report/extra_analysis.json
# Depuración (opcional): python ml/make_mock_labels.py y luego
#   python -m ml.train_eval --suffix _mock --outdir <carpeta_temporal>
```

## Notas

- Features: `n`, `distribution` y 4 numéricas. Excluidas por fuga: `instance_id`, `cost_*`,
  `best_algorithm`, `n_tied` (esta última solo se usa para describir resultados).
- Selección de modelo solo por CV (5 pliegues estratificados) sobre train; test se evalúa una vez.
- Semilla fija (`SEED = 0`) en CV, árbol, permutación y bootstraps.
- Si P3 cambia las etiquetas (parámetros de SA o regla de desempate) hay que repetir las corridas.
