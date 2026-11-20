"""Algoritmos falsos con la misma interfaz que los reales: solve(coords, seed) -> (tour, costo).
Solo sirven para probar el pipeline; el ganador es casi aleatorio."""
import numpy as np
from tsp_utils import dist_matrix, tour_cost


def _stub(coords, seed, factor):
    rng = np.random.default_rng(seed)
    D = dist_matrix(coords)
    tour = [int(x) for x in rng.permutation(len(coords))]
    return tour, tour_cost(tour, D) * factor


class StubHC:
    @staticmethod
    def solve(coords, seed=0):
        return _stub(coords, seed, 0.60)


class StubSA:
    @staticmethod
    def solve(coords, seed=0):
        return _stub(coords, seed + 1, 0.62)


class StubMST:
    @staticmethod
    def solve(coords, seed=0):
        return _stub(coords, seed + 2, 0.61)
