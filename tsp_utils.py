"""tsp_utils.py – utilidades comunes del proyecto."""
import numpy as np
import pandas as pd

DATA_DIR = "hh_tsp_dataset/data"


def load_instance(instance_id: str) -> np.ndarray:
    """Devuelve un arreglo (n, 2) con las coordenadas de la instancia."""
    df = pd.read_csv(f"{DATA_DIR}/instances/{instance_id}.csv")
    return df.iloc[:, :2].to_numpy(dtype=float)


def dist_matrix(coords: np.ndarray) -> np.ndarray:
    """Matriz de distancias euclidianas (n, n)."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def tour_cost(tour, D: np.ndarray) -> float:
    """Costo del ciclo CERRADO (regresa al punto inicial)."""
    n = len(tour)
    return float(sum(D[tour[i], tour[(i + 1) % n]] for i in range(n)))


def is_valid_tour(tour, n: int) -> bool:
    """True si el tour es una permutación de 0..n-1."""
    return sorted(tour) == list(range(n))


def seed_from_id(instance_id: str) -> int:
    """Semilla reproducible a partir de 'inst_0042' -> 42."""
    return int(instance_id.split("_")[1])
