# label_generation/check_algos.py
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tsp_utils import (load_instance, dist_matrix, tour_cost,
                       is_valid_tour, seed_from_id, DATA_DIR)
from algorithms import hill_climbing_2opt as hc, mst_2opt as mst

ids = pd.concat([pd.read_csv(f'{DATA_DIR}/train.csv'),
                 pd.read_csv(f'{DATA_DIR}/test.csv')])['instance_id']

rows = []
for iid in ids:
    coords = load_instance(iid)
    D = dist_matrix(coords)
    for name, algo in [('hc', hc), ('mst', mst)]:
        t0 = time.perf_counter()
        tour, cost = algo.solve(coords, seed=seed_from_id(iid))
        dt = time.perf_counter() - t0
        assert is_valid_tour(tour, len(coords)), (iid, name)
        assert abs(cost - tour_cost(tour, D)) < 1e-9, (iid, name)
        rows.append({'instance_id': iid, 'algo': name, 'n': len(coords),
                     'cost': cost, 'seconds': dt})

out = pd.DataFrame(rows)
os.makedirs('data', exist_ok=True)
out.to_csv('data/check_hc_mst.csv', index=False)
print(out.groupby(['algo', 'n'])['seconds'].agg(['mean', 'max']))
print('Listo: todas las instancias pasaron la validación.')
