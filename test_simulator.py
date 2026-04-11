"""Run the adaptive-network SIR simulator and print output for x steps."""

import argparse
import numpy as np

from simulator import simulate


def run_simulation(steps):
    """Run one simulation and print the per-step outputs."""
    n_nodes = 200
    rng = np.random.default_rng(123)

    infected_fraction, rewire_counts, degree_histogram = simulate(
        beta=0.2,
        gamma=0.1,
        rho=0.3,
        N=n_nodes,
        T=steps,
        rng=rng,
    )

    print(f"Simulation run for {steps} steps")
    print("time, infected_fraction, rewire_count")
    for t in range(steps + 1):
        print(f"{t}, {infected_fraction[t]:.4f}, {int(rewire_counts[t])}")

    print("\nFinal degree histogram (bin 30 = degree >= 30):")
    print(degree_histogram)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulator for x steps and print output.")
    parser.add_argument("--steps", type=int, default=10, help="Number of simulation steps.")
    args = parser.parse_args()
    run_simulation(args.steps)
