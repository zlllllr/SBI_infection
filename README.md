# SBI for Epidemic Parameter Inference on Adaptive Networks

This repository contains code, data, notebooks, and report material for simulation-based inference (SBI) on a stochastic adaptive-network SIR model.

Goal: infer unknown parameters $(\beta,\gamma,\rho)$ from aggregate observations when the likelihood is intractable.

## Model summary

- Population: 200 agents on an undirected Erdos-Renyi contact network ($p=0.05$).
- States: Susceptible (S), Infected (I), Recovered (R).
- Initial condition: 5 infected agents at time 0.
- Time horizon: 200 steps.

At each step:

1. Infection: each S-I edge transmits with probability $\beta$.
2. Recovery: each infected node recovers with probability $\gamma$.
3. Rewiring: each S-I edge rewires away from infected contact with probability $\rho$.

Parameter priors used by ABC methods:

| Parameter | Meaning | Prior |
|---|---|---|
| $\beta$ | Infection probability per S-I edge per step | Uniform(0.05, 0.50) |
| $\gamma$ | Recovery probability per infected agent per step | Uniform(0.02, 0.20) |
| $\rho$ | Rewiring probability per S-I edge per step | Uniform(0.0, 0.8) |

## Project structure

Core code:

- `simulator.py`: adaptive-network SIR simulator.
- `ABC_rejection.py`: rejection ABC with pilot scaling and quantile acceptance.
- `SMC_ABC.py`: SMC-ABC with weighted particles, stage-wise tolerance tightening, and diagnostics.

Observed dataset:

- `data/infected_timeseries.csv`
- `data/rewiring_timeseries.csv`
- `data/final_degree_histograms.csv`

Analysis notebooks:

- `parameter_sweep_visualization.ipynb`: parameter-sweep diagnostics and refined summary-statistic selection.
- `abc_rejection_analysis.ipynb`: rejection ABC workflow, posterior plots, diagnostics.
- `abc_smc_analysis.ipynb`: SMC-ABC workflow, stage diagnostics, weighted posterior plots, rejection-vs-SMC comparison.

## Setup

Recommended environment:

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install numpy pandas matplotlib jupyter
```

## Quick usage

Run rejection ABC:

```bash
python ABC_rejection.py --n-pilot 100 --n-samples 2000 --accept-quantile 0.05 --n-reps 20 --out-prefix abc_rejection_analysis
```

Run SMC-ABC:

```bash
python SMC_ABC.py --n-pilot 100 --n-particles 100 --n-init-samples 1000 --n-stages 4 --stage-quantile 0.60 --kernel-scale 0.60 --n-reps 20 --out-prefix abc_smc_analysis
```

Open notebooks:

```bash
jupyter notebook
```

## Summary statistics used in ABC runs

The refined 9-dimensional summary vector is:

- infected_extinction_t
- degree_std
- rewire_peak
- infected_auc
- degree_tail_low
- infected_final
- degree_tail_hi
- infected_peak
- infected_t_peak

These were selected from parameter-sweep diagnostics to balance:

1. sensitivity to parameter changes,
2. complementarity (low overlap),
3. joint informativeness for $(\beta,\gamma,\rho)$,
4. compact dimensionality.

## Notes

- SMC-ABC generally improves efficiency and posterior concentration relative to basic rejection ABC in this project.
- A residual $\beta$-$\rho$ trade-off remains and should be interpreted as a structural identifiability limitation rather than a pure sampling artifact.
