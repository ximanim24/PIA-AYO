import itertools
import numpy as np
import pytest

from algorithms import sa_2opt
from tsp_utils import dist_matrix, tour_cost


def _is_valid_tour(tour: list[int], n: int) -> bool:
    return (
        isinstance(tour, list)
        and len(tour) == n
        and all(isinstance(x, (int, np.integer)) for x in tour)
        and set(int(x) for x in tour) == set(range(n))
    )


def _random_coords(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((n, 2))


@pytest.mark.parametrize("n", [5, 10, 40, 100])
def test_tour_valido_y_costo_consistente(n: int) -> None:
    coords = _random_coords(n, seed=100 + n)
    tour, cost = sa_2opt.solve(coords, seed=1)
    assert _is_valid_tour(tour, n)
    expected = tour_cost(tour, dist_matrix(coords))
    assert cost == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_reproducible_con_semilla() -> None:
    coords = _random_coords(30, seed=7)
    res_a = sa_2opt.solve(coords, seed=7)
    res_b = sa_2opt.solve(coords, seed=7)
    assert res_a[0] == res_b[0]
    assert res_a[1] == pytest.approx(res_b[1], rel=1e-12, abs=1e-12)


def test_semillas_distintas_dan_tours_validos() -> None:
    coords = _random_coords(30, seed=12)
    for s in (1, 2, 3):
        tour, cost = sa_2opt.solve(coords, seed=s)
        assert _is_valid_tour(tour, 30)
        assert cost > 0.0


def test_nunca_mejor_que_el_optimo_exacto() -> None:
    coords = _random_coords(8, seed=3)
    d_mat = dist_matrix(coords)
    opt_cost = min(
        tour_cost([0] + list(perm), d_mat)
        for perm in itertools.permutations(range(1, 8))
    )
    _, cost = sa_2opt.solve(coords, seed=0)
    assert cost >= opt_cost - 1e-9


def test_puntos_duplicados() -> None:
    coords = _random_coords(10, seed=21)
    coords[3] = coords[4]
    tour, cost = sa_2opt.solve(coords, seed=0)
    assert _is_valid_tour(tour, 10)
    assert cost == pytest.approx(tour_cost(tour, dist_matrix(coords)), rel=1e-9, abs=1e-9)


def test_override_de_parametros() -> None:
    coords = _random_coords(20, seed=9)
    tour, cost = sa_2opt.solve(coords, seed=0, params={"min_iters": 200, "cooling": "linear"})
    assert _is_valid_tour(tour, 20)
    assert cost == pytest.approx(tour_cost(tour, dist_matrix(coords)), rel=1e-9, abs=1e-9)
