import itertools
import numpy as np
import pytest

from tsp_utils import dist_matrix, tour_cost, is_valid_tour
from algorithms import hill_climbing_2opt as hc
from algorithms import mst_2opt as mst
from algorithms.two_opt import two_opt

ALGOS = [hc, mst]
SIZES = [5, 10, 40, 100]


def random_coords(n, seed=0):
    return np.random.default_rng(seed).random((n, 2))


@pytest.mark.parametrize("algo", ALGOS)
@pytest.mark.parametrize("n", SIZES)
def test_tour_valido_y_costo_consistente(algo, n):
    coords = random_coords(n)
    tour, cost = algo.solve(coords, seed=1)
    assert is_valid_tour(tour, n)
    assert cost == pytest.approx(tour_cost(tour, dist_matrix(coords)))


@pytest.mark.parametrize("algo", ALGOS)
def test_reproducible_con_semilla(algo):
    coords = random_coords(30)
    assert algo.solve(coords, seed=7) == algo.solve(coords, seed=7)


def test_hc_cambia_con_semilla_o_al_menos_no_falla():
    coords = random_coords(30)
    t1, _ = hc.solve(coords, seed=1)
    t2, _ = hc.solve(coords, seed=2)
    assert is_valid_tour(t1, 30) and is_valid_tour(t2, 30)


def test_two_opt_nunca_empeora():
    coords = random_coords(40)
    D = dist_matrix(coords)
    start = list(range(40))
    assert tour_cost(two_opt(start, D), D) <= tour_cost(start, D) + 1e-9


def optimo_fuerza_bruta(coords):
    D = dist_matrix(coords)
    n = len(coords)
    return min(tour_cost([0] + list(p), D)
               for p in itertools.permutations(range(1, n)))


@pytest.mark.parametrize("algo", ALGOS)
def test_nunca_mejor_que_el_optimo(algo):
    coords = random_coords(8, seed=3)
    _, cost = algo.solve(coords, seed=0)
    assert cost >= optimo_fuerza_bruta(coords) - 1e-9


@pytest.mark.parametrize("algo", ALGOS)
def test_puntos_duplicados(algo):
    coords = random_coords(10)
    coords[3] = coords[4]  # dos puntos idénticos
    tour, _ = algo.solve(coords, seed=0)
    assert is_valid_tour(tour, 10)
