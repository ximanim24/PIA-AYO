"""algorithms/two_opt.py – búsqueda local 2-opt (primera mejora)."""


def two_opt(tour, D, eps: float = 1e-12):
    """Mejora el tour con intercambios 2-opt hasta no encontrar mejora."""
    tour = [int(x) for x in tour]
    n = len(tour)
    if n < 4:
        return tour
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # mismas aristas, no es un movimiento válido
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < -eps:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
    return tour
