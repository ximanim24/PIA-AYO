# ml/make_mock_labels.py
# Etiquetas FALSAS para depurar el código. Sus resultados no se reportan nunca.
import os
import numpy as np
import pandas as pd

DATA = 'hh_tsp_dataset/data'
OUT = 'data/labeled'
CLASSES = ['2opt_hill_climbing', 'sa_2opt', 'mst_2opt']
rng = np.random.default_rng(0)


def make(split):
    df = pd.read_csv(f'{DATA}/{split}.csv')
    base = df['n'] * df['mean_nn_dist'] + 0.1
    clustered = df['distribution'].astype(str).str.lower().str.contains('clus|norm')
    adj_sa = np.where(df['n'] <= 20, -0.02, 0.03)
    adj_mst = np.where(clustered & (df['n'] >= 40), -0.02, 0.02)
    noise = lambda: 1 + 0.01 * rng.standard_normal(len(df))
    df['cost_2opt_hill_climbing'] = base * noise()
    df['cost_sa_2opt'] = base * (1 + adj_sa) * noise()
    df['cost_mst_2opt'] = base * (1 + adj_mst) * noise()
    costs = df[[f'cost_{c}' for c in CLASSES]].to_numpy()
    df['best_algorithm'] = [CLASSES[i] for i in costs.argmin(axis=1)]
    df['n_tied'] = 1
    df.to_csv(f'{OUT}/{split}_labeled_mock.csv', index=False)


os.makedirs(OUT, exist_ok=True)
for s in ('train', 'test'):
    make(s)
print('Listo: train_labeled_mock.csv y test_labeled_mock.csv')
