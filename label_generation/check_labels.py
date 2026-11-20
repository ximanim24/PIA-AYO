"""Verificación del Paso 5/6 sobre los CSV etiquetados.

    python -m label_generation.check_labels [--tag _stub]
    python -m label_generation.check_labels --repro cacheA.csv cacheB.csv   # reproducibilidad
"""
import argparse
import sys

import pandas as pd

from label_generation.run_labels import OUT_DIR, PRIORITY, TOL
from tsp_utils import DATA_DIR


def check(tag):
    errors = []
    seen = set()
    for split in ('train', 'test'):
        orig = pd.read_csv(f'{DATA_DIR}/{split}.csv')
        lab = pd.read_csv(f'{OUT_DIR}/{split}_labeled{tag}.csv')
        costs = [f'cost_{a}' for a in PRIORITY]
        if len(lab) != len(orig):
            errors.append(f'{split}: {len(lab)} filas, se esperaban {len(orig)}')
        if list(lab['instance_id']) != list(orig['instance_id']):
            errors.append(f'{split}: instance_id distintos u ordenados diferente al original')
        missing = [c for c in orig.columns if c not in lab.columns]
        if missing:
            errors.append(f'{split}: faltan columnas originales {missing}')
        if lab[costs].isna().any().any():
            errors.append(f'{split}: hay NaN en costos')
        if (lab[costs] <= 0).any().any():
            errors.append(f'{split}: hay costos <= 0')
        if not set(lab['best_algorithm']) <= set(PRIORITY):
            errors.append(f'{split}: clases inválidas {set(lab["best_algorithm"]) - set(PRIORITY)}')
        m = lab[costs].min(axis=1)
        ntied = sum((lab[f'cost_{a}'] <= m + TOL).astype(int) for a in PRIORITY)
        first = lab[costs].apply(lambda r: next(a for a in PRIORITY if r[f'cost_{a}'] <= r.min() + TOL), axis=1)
        if (first != lab['best_algorithm']).any():
            errors.append(f'{split}: best_algorithm no coincide con el mínimo + regla de desempate')
        if (ntied != lab['n_tied']).any():
            errors.append(f'{split}: n_tied inconsistente')
        if seen & set(lab['instance_id']):
            errors.append('instancias repetidas entre train y test')
        seen |= set(lab['instance_id'])
        print(f'{split}: {len(lab)} filas, clases {dict(lab["best_algorithm"].value_counts())}')
    return errors


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='')
    ap.add_argument('--repro', nargs=2, metavar=('CACHE_A', 'CACHE_B'))
    args = ap.parse_args()
    if args.repro:
        a, b = (pd.read_csv(p).sort_values('instance_id') for p in args.repro)
        same = list(a['instance_id']) == list(b['instance_id']) and (a['cost'].values == b['cost'].values).all()
        print('REPRODUCIBLE: costos idénticos' if same else 'NO reproducible: hay estado aleatorio global')
        sys.exit(0 if same else 1)
    errs = check(args.tag)
    print('OK' if not errs else 'ERRORES:\n- ' + '\n- '.join(errs))
    sys.exit(1 if errs else 0)
