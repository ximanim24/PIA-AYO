"""Generate the TSP Selection-Hyper-Heuristic training/test dataset for the AYO final project.

Three instance distributions (uniform, clustered/normal, border-concentrated),
three TSP algorithms (2-opt hill climbing, Simulated Annealing + 2-opt, MST + 2-opt),
sizes n = 2..15, K replicate instances per (n, distribution).

Pure stdlib, no third-party dependencies (instances are tiny, no need for numpy).
"""
import csv
import math
import os
import random

SEED = 42
SIZES = [5, 10, 15, 20, 30, 40, 60, 80, 100]
DISTRIBUTIONS = ["uniform", "clustered", "border"]
TOTAL_INSTANCES = 500
TRAIN_FRACTION = 0.7


def replica_counts(n_groups, total):
    """Spread `total` as evenly as possible across `n_groups` groups."""
    base, remainder = divmod(total, n_groups)
    return [base + 1 if i < remainder else base for i in range(n_groups)]

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
INSTANCES_DIR = os.path.join(OUT_DIR, "instances")
REF_DIR = os.path.join(OUT_DIR, "instructor_reference")


# ---------------------------------------------------------------------------
# Instance generators
# ---------------------------------------------------------------------------
def gen_uniform(n, rng):
    return [(rng.random(), rng.random()) for _ in range(n)]


def gen_clustered(n, rng, sigma=0.07):
    num_clusters = rng.randint(1, 3)
    centers = [(rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)) for _ in range(num_clusters)]
    points = []
    for _ in range(n):
        cx, cy = rng.choice(centers)
        x = min(max(rng.gauss(cx, sigma), 0.0), 1.0)
        y = min(max(rng.gauss(cy, sigma), 0.0), 1.0)
        points.append((x, y))
    return points


def gen_border(n, rng, jitter=0.05):
    points = []
    for _ in range(n):
        side = rng.randint(0, 3)
        t = rng.random()
        j = abs(rng.gauss(0.0, jitter))
        if side == 0:  # left edge, x = 0
            x, y = j, t
        elif side == 1:  # right edge, x = 1
            x, y = 1.0 - j, t
        elif side == 2:  # bottom edge, y = 0
            x, y = t, j
        else:  # top edge, y = 1
            x, y = t, 1.0 - j
        points.append((min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)))
    return points


GENERATORS = {"uniform": gen_uniform, "clustered": gen_clustered, "border": gen_border}


# ---------------------------------------------------------------------------
# TSP utilities
# ---------------------------------------------------------------------------
def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def tour_cost(tour, points):
    n = len(tour)
    return sum(dist(points[tour[i]], points[tour[(i + 1) % n]]) for i in range(n))


def nearest_neighbor_tour(points):
    n = len(points)
    unvisited = set(range(n))
    current = 0
    unvisited.remove(current)
    tour = [current]
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist(points[current], points[j]))
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return tour


def two_opt(tour, points):
    """Hill-climbing 2-opt local search: repeat until no improving move exists."""
    n = len(tour)
    if n < 4:
        return tour[:]
    tour = tour[:]
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dist(points[a], points[c]) + dist(points[b], points[d])) - (
                    dist(points[a], points[b]) + dist(points[c], points[d])
                )
                if delta < -1e-12:
                    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                    improved = True
    return tour


def two_opt_hill_climbing(points, rng):
    initial = nearest_neighbor_tour(points)
    return two_opt(initial, points)


def simulated_annealing_2opt(points, rng, t0=15.0, alpha=0.995, iters=None):
    n = len(points)
    if iters is None:
        iters = max(500, 150 * n)
    tour = nearest_neighbor_tour(points)
    cost = tour_cost(tour, points)
    best_tour, best_cost = tour[:], cost
    t = t0
    for _ in range(iters):
        if n < 4:
            break
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        if i == 0 and j == n - 1:
            continue
        a, b = tour[i], tour[(i + 1) % n]
        c, d = tour[j], tour[(j + 1) % n]
        delta = (dist(points[a], points[c]) + dist(points[b], points[d])) - (
            dist(points[a], points[b]) + dist(points[c], points[d])
        )
        if delta < 0 or rng.random() < math.exp(-delta / max(t, 1e-9)):
            tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
            cost += delta
            if cost < best_cost:
                best_tour, best_cost = tour[:], cost
        t *= alpha
    return best_tour


def mst_tour(points):
    """Prim's MST, then DFS preorder traversal (classic 2-approximation)."""
    n = len(points)
    in_tree = [False] * n
    min_edge = [math.inf] * n
    parent = [-1] * n
    min_edge[0] = 0.0
    adj = [[] for _ in range(n)]
    for _ in range(n):
        u = min((v for v in range(n) if not in_tree[v]), key=lambda v: min_edge[v])
        in_tree[u] = True
        if parent[u] != -1:
            adj[u].append(parent[u])
            adj[parent[u]].append(u)
        for v in range(n):
            if not in_tree[v]:
                d = dist(points[u], points[v])
                if d < min_edge[v]:
                    min_edge[v] = d
                    parent[v] = u
    tour, visited = [], [False] * n

    def dfs(u):
        visited[u] = True
        tour.append(u)
        for v in adj[u]:
            if not visited[v]:
                dfs(v)

    dfs(0)
    return tour


def mst_two_opt(points, rng):
    base = mst_tour(points)
    return two_opt(base, points)


ALGORITHMS = {
    "2opt_hill_climbing": two_opt_hill_climbing,
    "sa_2opt": simulated_annealing_2opt,
    "mst_2opt": mst_two_opt,
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(points):
    n = len(points)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pairwise = [dist(points[i], points[j]) for i, j in pairs] or [0.0]
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    centroid_dists = [dist(p, (cx, cy)) for p in points]
    nn_dists = []
    for i in range(n):
        others = [dist(points[i], points[j]) for j in range(n) if j != i]
        nn_dists.append(min(others) if others else 0.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    def stdev(vals):
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

    return {
        "n": n,
        "mean_pairwise_dist": sum(pairwise) / len(pairwise),
        "std_coords": (stdev(xs) + stdev(ys)) / 2,
        "mean_centroid_dist": sum(centroid_dists) / n,
        "mean_nn_dist": sum(nn_dists) / n,
    }


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def main():
    os.makedirs(INSTANCES_DIR, exist_ok=True)
    rng = random.Random(SEED)
    rows = []
    instance_id = 0

    group_keys = [(n, d) for n in SIZES for d in DISTRIBUTIONS]
    counts = dict(zip(group_keys, replica_counts(len(group_keys), TOTAL_INSTANCES)))

    for n in SIZES:
        for dist_name in DISTRIBUTIONS:
            for _rep in range(counts[(n, dist_name)]):
                instance_id += 1
                inst_id = f"inst_{instance_id:04d}"
                points = GENERATORS[dist_name](n, rng)

                with open(os.path.join(INSTANCES_DIR, f"{inst_id}.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["x", "y"])
                    w.writerows(points)

                costs = {}
                for algo_name, algo_fn in ALGORITHMS.items():
                    tour = algo_fn(points, rng)
                    costs[algo_name] = tour_cost(tour, points)

                best_algo = min(costs, key=costs.get)
                feats = extract_features(points)

                row = {
                    "instance_id": inst_id,
                    "n": n,
                    "distribution": dist_name,
                    **feats,
                    "cost_2opt_hill_climbing": costs["2opt_hill_climbing"],
                    "cost_sa_2opt": costs["sa_2opt"],
                    "cost_mst_2opt": costs["mst_2opt"],
                    "best_algorithm": best_algo,
                }
                rows.append(row)

    # Stratified split: within each (n, distribution) group, TRAIN_FRACTION to train.
    groups = {}
    for row in rows:
        key = (row["n"], row["distribution"])
        groups.setdefault(key, []).append(row)

    train_rows, test_rows = [], []
    for key, group_rows in groups.items():
        rng.shuffle(group_rows)
        n_train = round(len(group_rows) * TRAIN_FRACTION)
        train_rows.extend(group_rows[:n_train])
        test_rows.extend(group_rows[n_train:])

    fieldnames = [
        "instance_id", "n", "distribution",
        "mean_pairwise_dist", "std_coords", "mean_centroid_dist", "mean_nn_dist",
        "cost_2opt_hill_climbing", "cost_sa_2opt", "cost_mst_2opt", "best_algorithm",
    ]

    def write_csv(path, data_rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in data_rows:
                w.writerow({k: r[k] for k in fieldnames})

    os.makedirs(REF_DIR, exist_ok=True)
    write_csv(os.path.join(REF_DIR, "train_with_labels.csv"), train_rows)
    write_csv(os.path.join(REF_DIR, "test_with_labels.csv"), test_rows)

    print(f"Total instances: {len(rows)}")
    print(f"Train: {len(train_rows)}  Test: {len(test_rows)}")
    from collections import Counter
    print("Winner counts (overall):", Counter(r["best_algorithm"] for r in rows))
    print("Train winner counts:", Counter(r["best_algorithm"] for r in train_rows))
    print("Test winner counts:", Counter(r["best_algorithm"] for r in test_rows))


if __name__ == "__main__":
    main()
