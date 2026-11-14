"""Strip cost/label columns from the instructor reference CSVs to produce the
student-facing train.csv/test.csv (instances + features only, no answers).

Run after generate_dataset.py, once per instructor_reference refresh.
"""
import csv
import os

BASE = os.path.dirname(__file__)
REF_DIR = os.path.join(BASE, "data", "instructor_reference")
OUT_DIR = os.path.join(BASE, "data")

PUBLIC_FIELDS = [
    "instance_id", "n", "distribution",
    "mean_pairwise_dist", "std_coords", "mean_centroid_dist", "mean_nn_dist",
]


def strip(src_name, dst_name):
    with open(os.path.join(REF_DIR, src_name), newline="") as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(OUT_DIR, dst_name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PUBLIC_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in PUBLIC_FIELDS})


if __name__ == "__main__":
    strip("train_with_labels.csv", "train.csv")
    strip("test_with_labels.csv", "test.csv")
    print("Wrote student-facing train.csv / test.csv (no cost/label columns).")
