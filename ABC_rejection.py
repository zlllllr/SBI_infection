"""Rejection ABC for adaptive-network SIR parameter inference.

This script estimates (beta, gamma, rho) from the observed dataset using
basic rejection ABC:
1) sample parameters from the prior,
2) simulate synthetic datasets,
3) compute summary statistics,
4) accept samples with small summary-distance.

Usage example:
    python ABC_rejection.py --n-pilot 20 --n-samples 200 --accept-quantile 0.05
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from simulator import simulate


SUMMARY_STAT_NAMES = [
    "infected_extinction_t",
    "degree_std",
    "rewire_peak",
    "infected_auc",
    "degree_tail_low",
    "infected_final",
    "degree_tail_hi",
    "infected_peak",
    "infected_t_peak",
]


@dataclass(frozen=True)
class PriorBounds:
    beta: Tuple[float, float] = (0.05, 0.50)
    gamma: Tuple[float, float] = (0.02, 0.20)
    rho: Tuple[float, float] = (0.0, 0.8)


def extinction_time_after_peak(infected_curve: np.ndarray, t_horizon: int, threshold: float = 0.01) -> float:
    """Return normalized first time at/after peak where infection drops below threshold.

    If the trajectory never drops below the threshold, return 1.0.
    """
    peak_idx = int(np.argmax(infected_curve))
    after_peak = infected_curve[peak_idx:]
    below = np.where(after_peak <= threshold)[0]
    if len(below) == 0:
        return 1.0
    return float((peak_idx + below[0]) / max(t_horizon, 1))


def load_observed_dataset(data_dir: str) -> Dict[str, np.ndarray]:
    """Load observed CSV files into dense arrays indexed by replicate and time/degree."""
    infected_path = os.path.join(data_dir, "infected_timeseries.csv")
    rewiring_path = os.path.join(data_dir, "rewiring_timeseries.csv")
    degree_path = os.path.join(data_dir, "final_degree_histograms.csv")

    infected_raw = np.genfromtxt(infected_path, delimiter=",", names=True)
    rewiring_raw = np.genfromtxt(rewiring_path, delimiter=",", names=True)
    degree_raw = np.genfromtxt(degree_path, delimiter=",", names=True)

    n_reps = int(max(np.max(infected_raw["replicate_id"]), np.max(rewiring_raw["replicate_id"]), np.max(degree_raw["replicate_id"]))) + 1
    n_time = int(max(np.max(infected_raw["time"]), np.max(rewiring_raw["time"]))) + 1
    n_degree_bins = int(np.max(degree_raw["degree"])) + 1

    infected = np.zeros((n_reps, n_time), dtype=float)
    rewiring = np.zeros((n_reps, n_time), dtype=float)
    degree_hist = np.zeros((n_reps, n_degree_bins), dtype=float)

    for row in infected_raw:
        rid = int(row["replicate_id"])
        t = int(row["time"])
        infected[rid, t] = float(row["infected_fraction"])

    for row in rewiring_raw:
        rid = int(row["replicate_id"])
        t = int(row["time"])
        rewiring[rid, t] = float(row["rewire_count"])

    for row in degree_raw:
        rid = int(row["replicate_id"])
        d = int(row["degree"])
        degree_hist[rid, d] = float(row["count"])

    return {
        "infected": infected,
        "rewiring": rewiring,
        "degree_hist": degree_hist,
    }


def compute_replicate_features(
    infected_curve: np.ndarray,
    rewiring_curve: np.ndarray,
    degree_hist: np.ndarray,
) -> np.ndarray:
    """Compute refined low-overlap features for one replicate."""
    t_max = max(len(infected_curve) - 1, 1)

    infected_peak = float(np.max(infected_curve))
    infected_t_peak = float(np.argmax(infected_curve) / t_max)
    infected_auc = float(np.trapezoid(infected_curve, dx=1.0) / t_max)
    infected_final = float(infected_curve[-1])
    infected_extinction_t = extinction_time_after_peak(infected_curve, t_horizon=t_max)

    rewire_peak = float(np.max(rewiring_curve))

    counts = np.asarray(degree_hist, dtype=float)
    total_nodes = max(float(np.sum(counts)), 1.0)
    degrees = np.arange(len(counts), dtype=float)
    probs = counts / total_nodes
    mean_degree = float(np.sum(degrees * probs))
    var_degree = float(np.sum(((degrees - mean_degree) ** 2) * probs))
    degree_std = float(np.sqrt(max(var_degree, 0.0)))
    degree_tail_low = float(np.sum(probs[:5]))
    degree_tail_hi = float(np.sum(probs[15:]))

    return np.array(
        [
            infected_extinction_t,
            degree_std,
            rewire_peak,
            infected_auc,
            degree_tail_low,
            infected_final,
            degree_tail_hi,
            infected_peak,
            infected_t_peak,
        ],
        dtype=float,
    )


def summarize_dataset(
    infected_matrix: np.ndarray,
    rewiring_matrix: np.ndarray,
    degree_hist_matrix: np.ndarray,
) -> np.ndarray:
    """Convert a multi-replicate dataset into refined summary vector.

    Use means across replicates to reduce redundant variability summaries.
    """
    n_reps = infected_matrix.shape[0]
    rep_features = np.zeros((n_reps, len(SUMMARY_STAT_NAMES)), dtype=float)

    for r in range(n_reps):
        rep_features[r] = compute_replicate_features(
            infected_curve=infected_matrix[r],
            rewiring_curve=rewiring_matrix[r],
            degree_hist=degree_hist_matrix[r],
        )

    feature_means = np.mean(rep_features, axis=0)
    return feature_means


def sample_from_prior(rng: np.random.Generator, bounds: PriorBounds) -> np.ndarray:
    """Draw one parameter vector theta=(beta, gamma, rho) from the prior."""
    beta = rng.uniform(bounds.beta[0], bounds.beta[1])
    gamma = rng.uniform(bounds.gamma[0], bounds.gamma[1])
    rho = rng.uniform(bounds.rho[0], bounds.rho[1])
    return np.array([beta, gamma, rho], dtype=float)


def simulate_dataset_summary(
    theta: np.ndarray,
    n_reps: int,
    n_steps: int,
    n_nodes: int,
    p_edge: float,
    n_infected0: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate n_reps runs at parameter theta and return summary statistics."""
    infected = np.zeros((n_reps, n_steps + 1), dtype=float)
    rewiring = np.zeros((n_reps, n_steps + 1), dtype=float)
    degree_hist = np.zeros((n_reps, 31), dtype=float)

    for r in range(n_reps):
        inf_r, rew_r, deg_r = simulate(
            beta=float(theta[0]),
            gamma=float(theta[1]),
            rho=float(theta[2]),
            N=n_nodes,
            p_edge=p_edge,
            n_infected0=n_infected0,
            T=n_steps,
            rng=rng,
        )
        infected[r] = inf_r
        rewiring[r] = rew_r
        degree_hist[r] = deg_r

    return summarize_dataset(infected, rewiring, degree_hist)


def standardized_distance(s_obs: np.ndarray, s_sim: np.ndarray, scales: np.ndarray) -> float:
    """Euclidean distance on standardized summary components."""
    z = (s_sim - s_obs) / scales
    return float(np.sqrt(np.sum(z * z)))


def fit_summary_scales(
    n_pilot: int,
    s_obs: np.ndarray,
    n_reps: int,
    n_steps: int,
    n_nodes: int,
    p_edge: float,
    n_infected0: int,
    bounds: PriorBounds,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate per-summary scales from pilot simulations for distance normalization."""
    pilot_summaries = np.zeros((n_pilot, len(s_obs)), dtype=float)

    for i in range(n_pilot):
        theta = sample_from_prior(rng, bounds)
        pilot_summaries[i] = simulate_dataset_summary(
            theta=theta,
            n_reps=n_reps,
            n_steps=n_steps,
            n_nodes=n_nodes,
            p_edge=p_edge,
            n_infected0=n_infected0,
            rng=rng,
        )

    scales = np.std(pilot_summaries, axis=0)
    scales = np.where(scales < 1e-8, 1.0, scales)
    return scales


def run_rejection_abc(
    s_obs: np.ndarray,
    n_samples: int,
    accept_quantile: float,
    scales: np.ndarray,
    n_reps: int,
    n_steps: int,
    n_nodes: int,
    p_edge: float,
    n_infected0: int,
    bounds: PriorBounds,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Run rejection ABC and return accepted samples and diagnostics."""
    thetas = np.zeros((n_samples, 3), dtype=float)
    distances = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        theta = sample_from_prior(rng, bounds)
        s_sim = simulate_dataset_summary(
            theta=theta,
            n_reps=n_reps,
            n_steps=n_steps,
            n_nodes=n_nodes,
            p_edge=p_edge,
            n_infected0=n_infected0,
            rng=rng,
        )
        d = standardized_distance(s_obs=s_obs, s_sim=s_sim, scales=scales)

        thetas[i] = theta
        distances[i] = d

    epsilon = float(np.quantile(distances, accept_quantile))
    accepted_mask = distances <= epsilon
    accepted_thetas = thetas[accepted_mask]
    return accepted_thetas, distances, epsilon, accepted_mask


def summarize_posterior(accepted_thetas: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute posterior summary statistics for accepted samples."""
    if accepted_thetas.shape[0] == 0:
        raise ValueError("No accepted samples. Increase accept_quantile or n_samples.")

    means = np.mean(accepted_thetas, axis=0)
    stds = np.std(accepted_thetas, axis=0)
    q05 = np.quantile(accepted_thetas, 0.05, axis=0)
    q50 = np.quantile(accepted_thetas, 0.50, axis=0)
    q95 = np.quantile(accepted_thetas, 0.95, axis=0)

    return {
        "mean": means,
        "std": stds,
        "q05": q05,
        "q50": q50,
        "q95": q95,
    }


def save_results(
    out_prefix: str,
    accepted_thetas: np.ndarray,
    all_distances: np.ndarray,
    accepted_mask: np.ndarray,
) -> None:
    """Save accepted parameters and all distances to CSV files."""
    accepted_path = f"{out_prefix}_accepted.csv"
    distances_path = f"{out_prefix}_distances.csv"

    np.savetxt(
        accepted_path,
        accepted_thetas,
        delimiter=",",
        header="beta,gamma,rho",
        comments="",
    )

    distance_table = np.column_stack(
        [
            np.arange(len(all_distances), dtype=int),
            all_distances,
            accepted_mask.astype(int),
        ]
    )
    np.savetxt(
        distances_path,
        distance_table,
        delimiter=",",
        header="sample_id,distance,accepted",
        comments="",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run rejection ABC for (beta, gamma, rho).")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing observed CSV files.")
    parser.add_argument("--n-pilot", type=int, default=20, help="Number of pilot simulations to scale summary distances.")
    parser.add_argument("--n-samples", type=int, default=200, help="Number of prior samples for rejection ABC.")
    parser.add_argument("--accept-quantile", type=float, default=0.05, help="Acceptance quantile epsilon (e.g., 0.05 keeps best 5%).")
    parser.add_argument("--n-reps", type=int, default=None, help="Replicates per simulated dataset (default: match observed replicates).")
    parser.add_argument("--n-steps", type=int, default=200, help="Simulation time steps T.")
    parser.add_argument("--n-nodes", type=int, default=200, help="Number of agents N.")
    parser.add_argument("--p-edge", type=float, default=0.05, help="Initial Erdos-Renyi edge probability.")
    parser.add_argument("--n-infected0", type=int, default=5, help="Initially infected agents.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for reproducibility.")
    parser.add_argument("--out-prefix", type=str, default="abc_rejection", help="Output filename prefix.")
    return parser.parse_args()


def main() -> None:
    """Entry point for running rejection ABC."""
    args = parse_args()
    if not (0.0 < args.accept_quantile < 1.0):
        raise ValueError("--accept-quantile must be strictly between 0 and 1.")
    if args.n_pilot <= 1:
        raise ValueError("--n-pilot must be >= 2.")
    if args.n_samples <= 1:
        raise ValueError("--n-samples must be >= 2.")

    start = time.time()
    rng = np.random.default_rng(args.seed)
    bounds = PriorBounds()

    observed = load_observed_dataset(args.data_dir)
    observed_reps = observed["infected"].shape[0]
    n_reps = observed_reps if args.n_reps is None else int(args.n_reps)

    s_obs = summarize_dataset(
        infected_matrix=observed["infected"],
        rewiring_matrix=observed["rewiring"],
        degree_hist_matrix=observed["degree_hist"],
    )

    scales = fit_summary_scales(
        n_pilot=args.n_pilot,
        s_obs=s_obs,
        n_reps=n_reps,
        n_steps=args.n_steps,
        n_nodes=args.n_nodes,
        p_edge=args.p_edge,
        n_infected0=args.n_infected0,
        bounds=bounds,
        rng=rng,
    )

    accepted_thetas, all_distances, epsilon, accepted_mask = run_rejection_abc(
        s_obs=s_obs,
        n_samples=args.n_samples,
        accept_quantile=args.accept_quantile,
        scales=scales,
        n_reps=n_reps,
        n_steps=args.n_steps,
        n_nodes=args.n_nodes,
        p_edge=args.p_edge,
        n_infected0=args.n_infected0,
        bounds=bounds,
        rng=rng,
    )

    posterior = summarize_posterior(accepted_thetas)
    save_results(args.out_prefix, accepted_thetas, all_distances, accepted_mask)

    n_acc = accepted_thetas.shape[0]
    acc_rate = n_acc / args.n_samples

    print("Rejection ABC complete")
    print(f"Observed replicates: {observed_reps}")
    print(f"Simulated replicates per sample: {n_reps}")
    print(f"Pilot simulations: {args.n_pilot}")
    print(f"Total samples: {args.n_samples}")
    print(f"Accepted samples: {n_acc}")
    print(f"Acceptance rate: {acc_rate:.4f}")
    print(f"Epsilon (distance threshold): {epsilon:.6f}")
    print()

    names = ["beta", "gamma", "rho"]
    for i, name in enumerate(names):
        print(
            f"{name}: mean={posterior['mean'][i]:.4f}, std={posterior['std'][i]:.4f}, "
            f"q05={posterior['q05'][i]:.4f}, median={posterior['q50'][i]:.4f}, q95={posterior['q95'][i]:.4f}"
        )

    elapsed = time.time() - start
    print()
    print(f"Saved: {args.out_prefix}_accepted.csv")
    print(f"Saved: {args.out_prefix}_distances.csv")
    print(f"Elapsed seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
