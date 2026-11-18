"""algorithms/mst_2opt.py"""
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from tsp_utils import dist_matrix, tour_cost
from algorithms.two_opt import two_opt


def mst_preorder_tour(D: np.ndarray, root: int = 0):
    """Tour inicial: recorrido en preorden del MST."""
    n = len(D)
    T = minimum_spanning_tree(D).toarray()
    adj = [[] for _ in range(n)]
    for i, j in zip(*np.nonzero(T)):
        adj[i].append(int(j))
        adj[j].append(int(i))
    visited = [False] * n
    order = []
    stack = [root]
    while stack:
        u = stack.pop()
        if visited[u]:
            continue
        visited[u] = True
        order.append(u)
        for v in sorted(adj[u], reverse=True):  # determinista
            if not visited[v]:
                stack.append(v)
    return order


def solve(coords: np.ndarray, seed: int = 0):
    """MST + preorden + 2-opt. Es determinista: 'seed' se ignora."""
    D = dist_matrix(coords)
    tour = two_opt(mst_preorder_tour(D), D)
    return tour, tour_cost(tour, D)
