"""SMC-ABC for adaptive-network SIR parameter inference.

This script estimates (beta, gamma, rho) from the observed dataset using
Sequential Monte Carlo Approximate Bayesian Computation (SMC-ABC):
1) initialize a particle population from prior simulations,
2) run a sequence of decreasing tolerances,
3) resample and perturb particles at each stage,
4) compute weighted posterior summaries.

Usage example:
    python SMC_ABC.py --n-particles 80 --n-stages 4 --stage-quantile 0.60
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ABC_rejection import (
    PriorBounds,
    load_observed_dataset,
    sample_from_prior,
    simulate_dataset_summary,
    standardized_distance,
    summarize_dataset,
)


@dataclass(frozen=True)
class SMCConfig:
    n_particles: int = 80
    n_init_samples: int = 800
    n_stages: int = 4
    stage_quantile: float = 0.60
    kernel_scale: float = 0.60


def within_bounds(theta: np.ndarray, bounds: PriorBounds) -> bool:
    """Check if theta is inside prior support."""
    return (
        bounds.beta[0] <= float(theta[0]) <= bounds.beta[1]
        and bounds.gamma[0] <= float(theta[1]) <= bounds.gamma[1]
        and bounds.rho[0] <= float(theta[2]) <= bounds.rho[1]
    )


def weighted_covariance(thetas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return weighted covariance matrix with numerical jitter."""
    w = weights / np.sum(weights)
    mean = np.sum(thetas * w[:, None], axis=0)
    centered = thetas - mean
    cov = (centered * w[:, None]).T @ centered
    cov += np.eye(cov.shape[0]) * 1e-6
    return cov


def mvn_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Evaluate multivariate normal density N(mean, cov) at x."""
    d = mean.shape[0]
    diff = x - mean
    try:
        sign, log_det = np.linalg.slogdet(cov)
        if sign <= 0:
            return 0.0
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return 0.0

    quad = float(diff.T @ inv_cov @ diff)
    log_norm = -0.5 * (d * math.log(2.0 * math.pi) + log_det)
    log_val = log_norm - 0.5 * quad
    return float(math.exp(log_val))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Compute weighted quantile for q in [0, 1]."""
    idx = np.argsort(values)
    v_sorted = values[idx]
    w_sorted = weights[idx]
    cdf = np.cumsum(w_sorted)
    cdf = cdf / cdf[-1]
    pos = int(np.searchsorted(cdf, q, side="left"))
    pos = min(max(pos, 0), len(v_sorted) - 1)
    return float(v_sorted[pos])


def weighted_summary(thetas: np.ndarray, weights: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute weighted posterior summary statistics."""
    w = weights / np.sum(weights)
    mean = np.sum(thetas * w[:, None], axis=0)
    var = np.sum(((thetas - mean) ** 2) * w[:, None], axis=0)
    std = np.sqrt(np.maximum(var, 0.0))

    q05 = np.array([weighted_quantile(thetas[:, i], w, 0.05) for i in range(3)], dtype=float)
    q50 = np.array([weighted_quantile(thetas[:, i], w, 0.50) for i in range(3)], dtype=float)
    q95 = np.array([weighted_quantile(thetas[:, i], w, 0.95) for i in range(3)], dtype=float)

    return {
        "mean": mean,
        "std": std,
        "q05": q05,
        "q50": q50,
        "q95": q95,
    }


def initialize_particles(
    s_obs: np.ndarray,
    scales: np.ndarray,
    cfg: SMCConfig,
    n_reps: int,
    n_steps: int,
    n_nodes: int,
    p_edge: float,
    n_infected0: int,
    bounds: PriorBounds,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Initialize particles from prior by selecting the best distances."""
    thetas = np.zeros((cfg.n_init_samples, 3), dtype=float)
    distances = np.zeros(cfg.n_init_samples, dtype=float)

    for i in range(cfg.n_init_samples):
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
        thetas[i] = theta
        distances[i] = standardized_distance(s_obs=s_obs, s_sim=s_sim, scales=scales)

    keep_idx = np.argsort(distances)[: cfg.n_particles]
    particles = thetas[keep_idx]
    particle_distances = distances[keep_idx]
    weights = np.ones(cfg.n_particles, dtype=float) / cfg.n_particles
    epsilon = float(np.max(particle_distances))
    return particles, weights, particle_distances, epsilon, cfg.n_init_samples


def resample_index(weights: np.ndarray, rng: np.random.Generator) -> int:
    """Draw one ancestor index according to normalized weights."""
    return int(rng.choice(len(weights), p=weights))


def propose_particle(
    particles: np.ndarray,
    weights: np.ndarray,
    kernel_cov: np.ndarray,
    bounds: PriorBounds,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a bounded perturbed particle from the weighted particle set."""
    while True:
        idx = resample_index(weights, rng)
        proposal = rng.multivariate_normal(mean=particles[idx], cov=kernel_cov)
        if within_bounds(proposal, bounds):
            return proposal


def denominator_mixture_density(
    theta: np.ndarray,
    prev_particles: np.ndarray,
    prev_weights: np.ndarray,
    kernel_cov: np.ndarray,
) -> float:
    """Compute denominator mixture density in SMC-ABC weight update."""
    dens = 0.0
    for j in range(prev_particles.shape[0]):
        dens += float(prev_weights[j]) * mvn_pdf(theta, prev_particles[j], kernel_cov)
    return dens


def run_smc_abc(
    s_obs: np.ndarray,
    scales: np.ndarray,
    cfg: SMCConfig,
    n_reps: int,
    n_steps: int,
    n_nodes: int,
    p_edge: float,
    n_infected0: int,
    bounds: PriorBounds,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]], int]:
    """Run SMC-ABC and return final particles, weights, distances, and diagnostics."""
    particles, weights, distances, epsilon, sim_count = initialize_particles(
        s_obs=s_obs,
        scales=scales,
        cfg=cfg,
        n_reps=n_reps,
        n_steps=n_steps,
        n_nodes=n_nodes,
        p_edge=p_edge,
        n_infected0=n_infected0,
        bounds=bounds,
        rng=rng,
    )

    stage_info: List[Dict[str, float]] = []
    ess = float(1.0 / np.sum(weights * weights))
    stage_info.append(
        {
            "stage": 0,
            "epsilon": epsilon,
            "ess": ess,
            "accept_rate": cfg.n_particles / max(cfg.n_init_samples, 1),
            "simulations": float(sim_count),
        }
    )

    for stage in range(1, cfg.n_stages):
        target_epsilon = float(np.quantile(distances, cfg.stage_quantile))
        epsilon_new = min(target_epsilon, epsilon * 0.999)

        prev_particles = particles.copy()
        prev_weights = weights.copy()

        cov = weighted_covariance(prev_particles, prev_weights)
        kernel_cov = (cfg.kernel_scale ** 2) * cov + np.eye(3) * 1e-6

        new_particles = np.zeros_like(prev_particles)
        new_distances = np.zeros_like(distances)
        raw_weights = np.zeros_like(prev_weights)

        accepted = 0
        proposals = 0
        while accepted < cfg.n_particles:
            theta_prop = propose_particle(
                particles=prev_particles,
                weights=prev_weights,
                kernel_cov=kernel_cov,
                bounds=bounds,
                rng=rng,
            )
            proposals += 1
            sim_count += 1

            s_sim = simulate_dataset_summary(
                theta=theta_prop,
                n_reps=n_reps,
                n_steps=n_steps,
                n_nodes=n_nodes,
                p_edge=p_edge,
                n_infected0=n_infected0,
                rng=rng,
            )
            d = standardized_distance(s_obs=s_obs, s_sim=s_sim, scales=scales)

            if d <= epsilon_new:
                denom = denominator_mixture_density(
                    theta=theta_prop,
                    prev_particles=prev_particles,
                    prev_weights=prev_weights,
                    kernel_cov=kernel_cov,
                )
                if denom <= 1e-300:
                    continue

                new_particles[accepted] = theta_prop
                new_distances[accepted] = d
                raw_weights[accepted] = 1.0 / denom
                accepted += 1

        weights = raw_weights / np.sum(raw_weights)
        particles = new_particles
        distances = new_distances
        epsilon = float(np.max(distances))
        ess = float(1.0 / np.sum(weights * weights))

        stage_info.append(
            {
                "stage": float(stage),
                "epsilon": epsilon,
                "ess": ess,
                "accept_rate": cfg.n_particles / max(proposals, 1),
                "simulations": float(sim_count),
            }
        )

    return particles, weights, distances, stage_info, sim_count


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
    """Estimate per-summary scales from pilot simulations."""
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


def save_results(
    out_prefix: str,
    particles: np.ndarray,
    weights: np.ndarray,
    distances: np.ndarray,
    stage_info: List[Dict[str, float]],
) -> None:
    """Save SMC particle and stage diagnostics tables."""
    particles_path = f"{out_prefix}_particles.csv"
    stages_path = f"{out_prefix}_stages.csv"

    particle_table = np.column_stack([particles, weights, distances])
    np.savetxt(
        particles_path,
        particle_table,
        delimiter=",",
        header="beta,gamma,rho,weight,distance",
        comments="",
    )

    stage_table = np.array(
        [
            [row["stage"], row["epsilon"], row["ess"], row["accept_rate"], row["simulations"]]
            for row in stage_info
        ],
        dtype=float,
    )
    np.savetxt(
        stages_path,
        stage_table,
        delimiter=",",
        header="stage,epsilon,ess,accept_rate,cumulative_simulations",
        comments="",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for SMC-ABC run."""
    parser = argparse.ArgumentParser(description="Run SMC-ABC for (beta, gamma, rho).")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing observed CSV files.")
    parser.add_argument("--n-pilot", type=int, default=100, help="Number of pilot simulations for summary scaling.")
    parser.add_argument("--n-particles", type=int, default=80, help="Number of SMC particles.")
    parser.add_argument("--n-init-samples", type=int, default=800, help="Initial prior simulations used to seed particles.")
    parser.add_argument("--n-stages", type=int, default=4, help="Number of SMC tolerance stages (including stage 0).")
    parser.add_argument("--stage-quantile", type=float, default=0.60, help="Per-stage quantile for tightening tolerance.")
    parser.add_argument("--kernel-scale", type=float, default=0.60, help="Scale multiplier for particle perturbation covariance.")
    parser.add_argument("--n-reps", type=int, default=None, help="Replicates per simulated dataset (default: observed replicates).")
    parser.add_argument("--n-steps", type=int, default=200, help="Simulation time steps T.")
    parser.add_argument("--n-nodes", type=int, default=200, help="Number of agents N.")
    parser.add_argument("--p-edge", type=float, default=0.05, help="Initial Erdos-Renyi edge probability.")
    parser.add_argument("--n-infected0", type=int, default=5, help="Initially infected agents.")
    parser.add_argument("--seed", type=int, default=24680, help="Random seed for reproducibility.")
    parser.add_argument("--out-prefix", type=str, default="abc_smc", help="Output filename prefix.")
    return parser.parse_args()


def main() -> None:
    """Entry point for running SMC-ABC."""
    args = parse_args()
    if args.n_pilot <= 1:
        raise ValueError("--n-pilot must be >= 2.")
    if args.n_particles <= 4:
        raise ValueError("--n-particles must be >= 5.")
    if args.n_init_samples < args.n_particles:
        raise ValueError("--n-init-samples must be >= --n-particles.")
    if args.n_stages < 2:
        raise ValueError("--n-stages must be >= 2.")
    if not (0.0 < args.stage_quantile < 1.0):
        raise ValueError("--stage-quantile must be strictly between 0 and 1.")

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

    cfg = SMCConfig(
        n_particles=args.n_particles,
        n_init_samples=args.n_init_samples,
        n_stages=args.n_stages,
        stage_quantile=args.stage_quantile,
        kernel_scale=args.kernel_scale,
    )

    particles, weights, distances, stage_info, total_sims = run_smc_abc(
        s_obs=s_obs,
        scales=scales,
        cfg=cfg,
        n_reps=n_reps,
        n_steps=args.n_steps,
        n_nodes=args.n_nodes,
        p_edge=args.p_edge,
        n_infected0=args.n_infected0,
        bounds=bounds,
        rng=rng,
    )

    posterior = weighted_summary(particles, weights)
    save_results(
        out_prefix=args.out_prefix,
        particles=particles,
        weights=weights,
        distances=distances,
        stage_info=stage_info,
    )

    print("SMC-ABC complete")
    print(f"Observed replicates: {observed_reps}")
    print(f"Simulated replicates per proposal: {n_reps}")
    print(f"Pilot simulations: {args.n_pilot}")
    print(f"Particles: {args.n_particles}")
    print(f"Initial simulations: {args.n_init_samples}")
    print(f"Stages: {args.n_stages}")
    print(f"Total simulator calls: {total_sims}")
    print()

    for row in stage_info:
        print(
            f"stage={int(row['stage'])}, epsilon={row['epsilon']:.6f}, "
            f"ESS={row['ess']:.2f}, stage_accept_rate={row['accept_rate']:.4f}"
        )

    print()
    names = ["beta", "gamma", "rho"]
    for i, name in enumerate(names):
        print(
            f"{name}: mean={posterior['mean'][i]:.4f}, std={posterior['std'][i]:.4f}, "
            f"q05={posterior['q05'][i]:.4f}, median={posterior['q50'][i]:.4f}, q95={posterior['q95'][i]:.4f}"
        )

    elapsed = time.time() - start
    print()
    print(f"Saved: {args.out_prefix}_particles.csv")
    print(f"Saved: {args.out_prefix}_stages.csv")
    print(f"Elapsed seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
