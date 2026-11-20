import json
import math
import os
import random

from tsp_utils import dist_matrix, tour_cost

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "sa_params.json")

DEFAULT_PARAMS = {
    "scale": 100.0,
    "initial_temp": 10.0,
    "final_temp": 0.01,
    "cooling": "exponential",
    "cooling_rate": None,  # None => factor calculado para llegar a final_temp en max_iter
    "iters_per_city": 0,
    "min_iters": 5000,
    "n_runs": 1,
}


def load_params() -> dict:
    params = dict(DEFAULT_PARAMS)
    if os.path.exists(_PARAMS_PATH):
        with open(_PARAMS_PATH, encoding="utf-8") as f:
            params.update(json.load(f))
    return params


def _temperature(
    step: int,
    max_iter: int,
    initial_temp: float,
    final_temp: float,
    cooling: str,
    cooling_rate: float | None,
) -> float:
    if max_iter <= 1:
        return final_temp
    progress = step / (max_iter - 1)
    if cooling == "linear":
        return initial_temp + (final_temp - initial_temp) * progress
    if cooling == "exponential":
        if cooling_rate is not None:
            return initial_temp * (cooling_rate ** step)
        if initial_temp > 0 and final_temp > 0:
            return initial_temp * ((final_temp / initial_temp) ** progress)
        return final_temp
    raise ValueError(f"Esquema de enfriamiento desconocido: {cooling}")


def _single_run(ds: list[list[float]], n: int, rng: random.Random, p: dict) -> tuple[list[int], float]:
    tour = list(range(n))
    rng.shuffle(tour)

    current_cost = sum(ds[tour[i]][tour[(i + 1) % n]] for i in range(n))
    best_tour = list(tour)
    best_cost = current_cost

    iters_per_city = p.get("iters_per_city", 0)
    min_iters = p.get("min_iters", 5000)
    max_iter = max(int(min_iters), int(iters_per_city * n))

    t0 = float(p.get("initial_temp", 10.0))
    tf = float(p.get("final_temp", 0.01))
    cooling = p.get("cooling", "exponential")
    cooling_rate = p.get("cooling_rate", None)

    for step in range(max_iter):
        temp = _temperature(step, max_iter, t0, tf, cooling, cooling_rate)

        i = rng.randrange(n - 1)
        k = rng.randrange(i + 1, n)
        if i == 0 and k == n - 1:
            continue

        a = tour[i]
        b = tour[(i + 1) % n]
        c = tour[k]
        d = tour[(k + 1) % n]

        delta = (ds[a][c] + ds[b][d]) - (ds[a][b] + ds[c][d])

        if delta < 0.0:
            accept = True
        else:
            prob = math.exp(-delta / max(temp, 1e-12))
            accept = rng.random() < prob

        if accept:
            tour[i + 1 : k + 1] = reversed(tour[i + 1 : k + 1])
            current_cost += delta
            if current_cost < best_cost - 1e-12:
                best_cost = current_cost
                best_tour = list(tour)

    return best_tour, best_cost


def solve(coords, seed: int = 0, params: dict | None = None) -> tuple[list[int], float]:
    """Devuelve (tour, costo) calculado con tsp_utils.tour_cost sobre distancias originales."""
    p = load_params()
    if params:
        p.update(params)

    d_mat = dist_matrix(coords)
    n = len(d_mat)
    if n < 4:
        tour = list(range(n))
        return tour, tour_cost(tour, d_mat)

    scale = float(p.get("scale", 100.0))
    ds = [[float(d_mat[r][c]) * scale for c in range(n)] for r in range(n)]

    rng = random.Random(seed)
    best_tour: list[int] | None = None
    best_scaled_cost = float("inf")

    n_runs = max(1, int(p.get("n_runs", 1)))
    for _ in range(n_runs):
        candidate_tour, candidate_cost = _single_run(ds, n, rng, p)
        if candidate_cost < best_scaled_cost:
            best_scaled_cost = candidate_cost
            best_tour = candidate_tour

    assert best_tour is not None
    final_tour = [int(x) for x in best_tour]
    return final_tour, tour_cost(final_tour, d_mat)
