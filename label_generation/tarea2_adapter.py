"""Adaptador OPCIONAL: expone los algoritmos de Tarea2-Algoritmia (3_BusquedaLocal.py,
4_SimulatedAnnealing.py, 5_MSTApproximation.py) con el contrato solve(coords, seed) -> (tour, costo).

No modifica los archivos de la Tarea 2. Fija random.seed(seed) antes de cada corrida, porque esos
archivos usan el `random` global (sin esto el etiquetado no sería reproducible).
Uso:  python -m label_generation.run_labels --tarea2 RUTA/A/Tarea2-Algoritmia
"""
import importlib.util
import os
import random

os.environ.setdefault('MPLBACKEND', 'Agg')  # los archivos de Tarea 2 importan pyplot
import numpy as np

# Parámetros del bloque main de 4_SimulatedAnnealing.py (esquema exponencial)
SA_PARAMS = dict(cooling_schedule='exponential', initial_temp=1000.0, cooling_rate=0.99, max_iter=3000)

_CACHE = {}


def _module(directory, filename):
    key = (directory, filename)
    if key not in _CACHE:
        spec = importlib.util.spec_from_file_location(filename[:-3].replace('.', '_'),
                                                      os.path.join(directory, filename))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CACHE[key] = mod
    return _CACHE[key]


def _dist(coords):
    c = np.asarray(coords, dtype=float)
    return np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)


class _Base:
    def __init__(self, directory):
        self.directory = directory


class T2HC(_Base):
    def solve(self, coords, seed=0):
        m = _module(self.directory, '3_BusquedaLocal.py')
        D = _dist(coords)
        random.seed(seed)
        tour, cost, _ = m.two_opt(m.nearest_neighbor(D), D)  # vecino más cercano, inicio aleatorio con semilla
        return tour, float(cost)


class T2SA(_Base):
    def solve(self, coords, seed=0):
        m = _module(self.directory, '4_SimulatedAnnealing.py')
        D = _dist(coords)
        random.seed(seed)
        tour, cost, _, _ = m.simulated_annealing(D, **SA_PARAMS)
        return tour, float(cost)


class T2MST(_Base):
    def solve(self, coords, seed=0):
        m = _module(self.directory, '5_MSTApproximation.py')
        D = _dist(coords)
        base = m.dfs_preorder(m.prim_mst(D), root=0)
        tour, cost = m.two_opt_single(base, D)
        return tour, float(cost)
