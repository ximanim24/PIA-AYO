"""Balance de clases, baseline trivial y empates por n.   python -m label_generation.class_balance [--tag _stub]"""
import argparse

import pandas as pd

from label_generation.run_labels import OUT_DIR, PRIORITY

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='')
tag = ap.parse_args().tag

tr = pd.read_csv(f'{OUT_DIR}/train_labeled{tag}.csv')
te = pd.read_csv(f'{OUT_DIR}/test_labeled{tag}.csv')
al = pd.concat([tr, te])


def pct(s):
    return (s.value_counts(normalize=True) * 100).reindex(PRIORITY, fill_value=0)


bal = pd.DataFrame({'global_n': al['best_algorithm'].value_counts().reindex(PRIORITY, fill_value=0),
                    'global_pct': pct(al['best_algorithm']),
                    'train_pct': pct(tr['best_algorithm']),
                    'test_pct': pct(te['best_algorithm'])}).round(2)
print(bal)
bal.to_csv(f'{OUT_DIR}/class_balance{tag}.csv')

major_train = tr['best_algorithm'].value_counts().idxmax()
base = (te['best_algorithm'] == major_train).mean() * 100
print(f'\nClase mayoritaria global: {al["best_algorithm"].value_counts().idxmax()}')
print(f'Baseline en test (siempre "{major_train}", mayoritaria de TRAIN): {base:.2f}%')

ties = al.assign(empate=al['n_tied'] >= 2).groupby('n')['empate'].agg(['mean', 'sum']).round(3)
ties.columns = ['frac_empates', 'n_empates']
print('\nInstancias con empate (n_tied >= 2) por n:')
print(ties)
ties.to_csv(f'{OUT_DIR}/ties_by_n{tag}.csv')
print(f'Empates totales: {int((al["n_tied"] >= 2).sum())} de {len(al)}')
