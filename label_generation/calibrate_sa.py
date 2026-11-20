import os
import time
import pandas as pd

from algorithms import hill_climbing_2opt as hc
from algorithms import mst_2opt as mst
from algorithms import sa_2opt
import tsp_utils

CONFIGS = {
    "tarea2_exp_5000": {
        "scale": 100.0,
        "initial_temp": 10.0,
        "final_temp": 0.01,
        "cooling": "exponential",
        "cooling_rate": None,
        "iters_per_city": 0,
        "min_iters": 5000,
        "n_runs": 1,
    },
    "tarea2_lineal_5000": {
        "scale": 100.0,
        "initial_temp": 10.0,
        "final_temp": 0.01,
        "cooling": "linear",
        "cooling_rate": None,
        "iters_per_city": 0,
        "min_iters": 5000,
        "n_runs": 1,
    },
    "iters_con_n": {
        "scale": 100.0,
        "initial_temp": 10.0,
        "final_temp": 0.01,
        "cooling": "exponential",
        "cooling_rate": None,
        "iters_per_city": 150,
        "min_iters": 5000,
        "n_runs": 1,
    },
    "iters_con_n_x2": {
        "scale": 100.0,
        "initial_temp": 10.0,
        "final_temp": 0.01,
        "cooling": "exponential",
        "cooling_rate": None,
        "iters_per_city": 150,
        "min_iters": 5000,
        "n_runs": 2,
    },
}


def _seed_from_instance_id(instance_id: str) -> int:
    if hasattr(tsp_utils, "seed_from_id"):
        return int(tsp_utils.seed_from_id(instance_id))
    return int(str(instance_id).split("_")[-1])


def main(per_n: int = 3) -> None:
    data_dir = getattr(tsp_utils, "DATA_DIR", "hh_tsp_dataset/data")
    train_path = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_path)

    sample_df = train_df.groupby("n", group_keys=False).head(per_n)
    rows = []

    for _, row in sample_df.iterrows():
        iid = str(row["instance_id"])
        n_val = int(row["n"])
        dist_type = row.get("distribution", "")
        coords = tsp_utils.load_instance(iid)
        seed = _seed_from_instance_id(iid)

        _, c_hc = hc.solve(coords, seed=seed)
        _, c_mst = mst.solve(coords, seed=seed)
        ref_cost = min(c_hc, c_mst)

        for cfg_name, cfg in CONFIGS.items():
            t0 = time.perf_counter()
            _, c_sa = sa_2opt.solve(coords, seed=seed, params=cfg)
            elapsed = time.perf_counter() - t0

            sa_wins = (c_sa < ref_cost - 1e-9) and (abs(c_sa - c_hc) > 1e-9) and (abs(c_sa - c_mst) > 1e-9)
            gap_rel = (c_sa - ref_cost) / ref_cost if ref_cost > 0 else 0.0

            rows.append(
                {
                    "config": cfg_name,
                    "instance_id": iid,
                    "n": n_val,
                    "distribution": dist_type,
                    "cost_sa": c_sa,
                    "cost_hc": c_hc,
                    "cost_mst": c_mst,
                    "ref_best": ref_cost,
                    "sa_wins": int(sa_wins),
                    "gap_rel": gap_rel,
                    "seconds": elapsed,
                }
            )

    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "calibracion_sa.csv")
    res_df = pd.DataFrame(rows)
    res_df.to_csv(out_path, index=False)

    print(f"Calibración guardada en {out_path} ({len(res_df)} filas).")
    print("\n--- Resumen por configuración ---")
    summary = res_df.groupby("config")[["sa_wins", "gap_rel", "seconds"]].mean()
    print(summary.round(4).to_string())

    print("\n--- Detalle por configuración y n ---")
    by_n = res_df.groupby(["config", "n"])[["sa_wins", "gap_rel", "seconds"]].mean()
    print(by_n.round(4).to_string())


if __name__ == "__main__":
    main()
