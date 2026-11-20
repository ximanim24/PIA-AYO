"""Pipeline de etiquetado (Persona 3).

Corre cada algoritmo sobre las 500 instancias, guarda un caché por algoritmo
(costo + segundos) y, cuando están los tres, escribe train/test etiquetados.

    python -m label_generation.run_labels --stub            # prueba con algoritmos falsos
    python -m label_generation.run_labels --tarea2 RUTA     # algoritmos de Tarea2-Algoritmia (ver tarea2_adapter.py)
    python -m label_generation.run_labels --limit 20        # muestra para estimar tiempo
    python -m label_generation.run_labels --only sa_2opt    # un solo algoritmo
"""
import argparse
import os
import random
import time

import pandas as pd
from joblib import Parallel, delayed

from tsp_utils import load_instance, seed_from_id, DATA_DIR

OUT_DIR = 'data/labeled'
PRIORITY = ['2opt_hill_climbing', 'mst_2opt', 'sa_2opt']  # orden de desempate
TOL = 1e-9


def get_algos(use_stub, tarea2_dir=None):
    if tarea2_dir:
        from label_generation.tarea2_adapter import T2HC, T2SA, T2MST
        return {'2opt_hill_climbing': T2HC(tarea2_dir), 'sa_2opt': T2SA(tarea2_dir), 'mst_2opt': T2MST(tarea2_dir)}
    if use_stub:
        from label_generation.stubs import StubHC, StubSA, StubMST
        return {'2opt_hill_climbing': StubHC, 'sa_2opt': StubSA, 'mst_2opt': StubMST}
    from algorithms import hill_climbing_2opt, sa_2opt, mst_2opt
    return {'2opt_hill_climbing': hill_climbing_2opt, 'sa_2opt': sa_2opt, 'mst_2opt': mst_2opt}


def cache_path(name, tag):
    return f'{OUT_DIR}/cache_{name}{tag}.csv'


def run_one(algo, instance_id):
    coords = load_instance(instance_id)
    t0 = time.perf_counter()
    _, cost = algo.solve(coords, seed=seed_from_id(instance_id))
    return {'instance_id': instance_id, 'cost': float(cost),
            'seconds': time.perf_counter() - t0}


def run_algorithm(name, algo, ids, n_jobs, tag):
    path = cache_path(name, tag)
    if os.path.exists(path):
        print(f'[{name}] caché encontrado, se reutiliza: {path}')
        return pd.read_csv(path)
    print(f'[{name}] corriendo {len(ids)} instancias...')
    rows = Parallel(n_jobs=n_jobs)(delayed(run_one)(algo, i) for i in ids)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)  # se escribe al terminar: un caché existente siempre está completo
    return df


def pick_best(costs):
    """costs: {algoritmo: costo}. Devuelve (ganador, cuántos empataron en el mínimo)."""
    m = min(costs.values())
    tied = [a for a in PRIORITY if costs[a] <= m + TOL]
    return tied[0], len(tied)


def label(split_df, caches):
    df = split_df.copy()
    for name, cdf in caches.items():
        df = df.merge(cdf[['instance_id', 'cost']].rename(columns={'cost': f'cost_{name}'}),
                      on='instance_id', how='left', validate='one_to_one')
    res = [pick_best({a: r[f'cost_{a}'] for a in PRIORITY}) for _, r in df.iterrows()]
    df['best_algorithm'] = [b for b, _ in res]
    df['n_tied'] = [k for _, k in res]  # solo descriptivo: NO usar como feature (fuga)
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stub', action='store_true', help='usar algoritmos falsos')
    ap.add_argument('--tarea2', metavar='DIR', help='usar los archivos de Tarea2-Algoritmia en DIR (sufijo _t2)')
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--only', nargs='*', choices=PRIORITY, help='correr solo estos algoritmos')
    ap.add_argument('--limit', type=int, help='muestra aleatoria (fija) de N instancias, solo para estimar tiempos')
    args = ap.parse_args()

    tag = ('_stub' if args.stub else '') + ('_t2' if args.tarea2 else '') + (f'_lim{args.limit}' if args.limit else '')
    os.makedirs(OUT_DIR, exist_ok=True)
    train = pd.read_csv(f'{DATA_DIR}/train.csv')
    test = pd.read_csv(f'{DATA_DIR}/test.csv')
    ids = list(pd.concat([train, test])['instance_id'])
    if args.limit:  # las primeras N serían todas n=5: se muestrea para que haya todos los tamaños
        ids = random.Random(0).sample(ids, args.limit)

    for name, algo in get_algos(args.stub, args.tarea2).items():
        if args.only and name not in args.only:
            continue
        df = run_algorithm(name, algo, ids, args.n_jobs, tag)
        print(f'[{name}] total {df["seconds"].sum():.1f}s de CPU, máx por instancia {df["seconds"].max():.2f}s')

    if args.limit:
        print('Muestra terminada: no se generan CSV etiquetados.')
        raise SystemExit

    caches = {n: pd.read_csv(cache_path(n, tag)) for n in PRIORITY if os.path.exists(cache_path(n, tag))}
    if len(caches) < 3:
        print(f'Faltan cachés de: {sorted(set(PRIORITY) - set(caches))}. Aún no se etiqueta.')
        raise SystemExit
    for n, c in caches.items():
        if set(ids) - set(c['instance_id']):
            raise SystemExit(f'El caché de {n} no cubre todas las instancias: bórralo y vuelve a correr.')
    label(train, caches).to_csv(f'{OUT_DIR}/train_labeled{tag}.csv', index=False)
    label(test, caches).to_csv(f'{OUT_DIR}/test_labeled{tag}.csv', index=False)
    print('Listo: CSV etiquetados generados.')
