# ISAC-MIMO-DRL

Deep reinforcement learning for beamforming in mmWave ISAC (Integrated Sensing and Communication) systems. A DDPG agent learns to steer a MIMO beamformer to balance communication throughput against sensing accuracy — no closed-form solution required.

Research project completed during work at the German Aerospace Centre (DLR).

## Background

Built on two papers:

- **Lillicrap et al. (2015)** — DDPG for continuous action spaces. The policy maps raw channel observations to beamforming weights and learns the communication/sensing tradeoff end-to-end.
- **Liu et al., IEEE ComST (2022)** — ISAC survey. The Pareto curve evaluation follows their formulation of the sensing-communication tradeoff.

Channel model: Saleh-Valenzuela clustered mmWave propagation. Mobility: 3GPP Release 16 NR-V2X.

## Stack

- **DRL:** PyTorch, Stable-Baselines3, Gymnasium
- **Optimisation:** CVXPY (convex baselines)
- **Experimentation:** TensorBoard, Jupyter
- **Python:** 3.11+, uv

## Run

```bash
pip install uv
uv sync
jupyter notebook
```

See `notebooks/` for training runs and Pareto curve plots.
