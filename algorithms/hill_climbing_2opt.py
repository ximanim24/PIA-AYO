"""algorithms/hill_climbing_2opt.py"""
import numpy as np
from tsp_utils import dist_matrix, tour_cost
from algorithms.two_opt import two_opt


def solve(coords: np.ndarray, seed: int = 0):
    """Hill climbing con 2-opt desde un tour inicial aleatorio.
    Devuelve (tour, costo)."""
    rng = np.random.default_rng(seed)
    D = dist_matrix(coords)
    n = len(coords)
    start = [int(x) for x in rng.permutation(n)]
    tour = two_opt(start, D)
    return tour, tour_cost(tour, D)
