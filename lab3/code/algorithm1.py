"""Algorithm 1 (Ben-Hur et al., 2002): stability of k-means across subsamples.

For each k from 2 to k_max, repeat B times: draw two random subsamples of a
fraction m of the data, cluster each with k-means, and compute the Jaccard
similarity of the two clusterings on the points the subsamples share.

Usage (run from the code folder):
    python algorithm1.py --m 0.8 --B 100 --k-max 10 --n-jobs 8 --loop inner

Writes ../results/S.csv (B rows, one column per k) and
../results/run_info.csv (settings and elapsed time).
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
from joblib import Parallel, delayed, parallel_config
from sklearn.cluster import KMeans

from similarity import similarity

CODE_DIR = Path(__file__).resolve().parent
LAB_DIR = CODE_DIR.parent


def load_binary_data(path):
    """Load lingBinary.Rdata and keep only the binary question columns."""
    frame = pyreadr.read_r(str(path))["lingBinary"]
    question_columns = [c for c in frame.columns if c.startswith("Q")]
    return frame[question_columns].to_numpy(dtype=np.float32)


def one_repetition(data, k, m, seed_sequence, kmeans_params):
    """Run the body of Algorithm 1 once and return the similarity score."""
    rng = np.random.default_rng(seed_sequence)
    n_points = data.shape[0]
    n_sub = int(round(m * n_points))
    kmeans_seeds = seed_sequence.spawn(2)

    labels_on_full = []
    for seed in kmeans_seeds:
        rows = rng.choice(n_points, size=n_sub, replace=False)
        model = KMeans(
            n_clusters=k,
            random_state=np.random.default_rng(seed).integers(2**31 - 1),
            **kmeans_params,
        )
        labels = np.full(n_points, -1, dtype=np.int64)
        labels[rows] = model.fit_predict(data[rows])
        labels_on_full.append(labels)

    shared = (labels_on_full[0] >= 0) & (labels_on_full[1] >= 0)
    return similarity(labels_on_full[0][shared], labels_on_full[1][shared])


def run_k(data, k, m, n_repeats, seed_sequences, kmeans_params, n_jobs):
    """Run all B repetitions for one k, in parallel over the repetitions."""
    scores = Parallel(n_jobs=n_jobs)(
        delayed(one_repetition)(data, k, m, seed_sequences[i], kmeans_params)
        for i in range(n_repeats)
    )
    return np.array(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=LAB_DIR / "data" / "lingBinary.Rdata")
    parser.add_argument("--out-dir", default=LAB_DIR / "results")
    parser.add_argument("--m", type=float, default=0.8)
    parser.add_argument("--B", type=int, default=100)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--loop", choices=["inner", "outer"], default="inner",
                        help="parallelize over repetitions (inner) or over k (outer)")
    parser.add_argument("--n-jobs", type=int, default=int(
        os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
    parser.add_argument("--seed", type=int, default=215)
    args = parser.parse_args()

    kmeans_params = dict(init="k-means++", n_init=1, max_iter=300, tol=1e-4)
    data = load_binary_data(args.data)
    k_values = list(range(2, args.k_max + 1))
    root = np.random.SeedSequence(args.seed)
    seeds = {k: root.spawn(1)[0].spawn(args.B) for k in k_values}

    start = time.perf_counter()
    # One BLAS/OpenMP thread per worker so workers do not compete.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        if args.loop == "inner":
            results = {k: run_k(data, k, args.m, args.B, seeds[k], kmeans_params,
                                args.n_jobs) for k in k_values}
        else:
            outer = Parallel(n_jobs=args.n_jobs)(
                delayed(run_k)(data, k, args.m, args.B, seeds[k], kmeans_params, 1)
                for k in k_values)
            results = dict(zip(k_values, outer))
    elapsed = time.perf_counter() - start

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame({f"k{k}": results[k] for k in k_values})
    scores.to_csv(out_dir / "S.csv", index=False)
    info = dict(m=args.m, B=args.B, k_max=args.k_max, loop=args.loop,
                n_jobs=args.n_jobs, n_init=kmeans_params["n_init"],
                seed=args.seed, elapsed_seconds=round(elapsed, 1))
    pd.DataFrame([info]).to_csv(out_dir / "run_info.csv", index=False)
    print(f"Done in {elapsed:.1f}s with {args.n_jobs} jobs. Mean similarity by k:")
    print(scores.mean().round(3).to_string())


if __name__ == "__main__":
    main()
