# ISAC-MIMO-DRL

Deep reinforcement learning for beamforming in mmWave ISAC (Integrated Sensing and Communication) systems. The agent learns to steer a MIMO beamformer to balance communication rate and sensing accuracy — no closed-form solution needed.

## Background

This project builds on two papers:

**[Lillicrap et al. (2015)](https://arxiv.org/abs/1509.02971)** — DDPG maps raw channel observations to continuous beamforming weights. The policy learns the tradeoff between pointing energy at the communication receiver versus the sensing target.

**[Liu et al. "Integrated Sensing and Communications" (IEEE 2022)](https://doi.org/10.1109/COMST.2022.3145772)** — this survey frames the fundamental tradeoff between sensing and communication in shared-spectrum systems. The Pareto curve evaluation in this project directly follows their formulation.

Other building blocks: the Saleh-Valenzuela clustered channel model for mmWave propagation, 3GPP Release 16 NR-V2X for vehicle mobility, and the Wymeersch et al. 6G ISAC framework as a longer-term reference.

## Quick start

```bash
uv sync
uv run python training/classical_baseline.py   # DFT steering-vector baseline
uv run python training/train_ddpg.py           # train the DRL agent
uv run pytest -v                               # 6 tests, should all pass
```

## Project structure

```
environment/     # Gymnasium env — ISACEnv, channel model, MIMO system, V2X scenario
training/        # DDPG training, classical DFT baseline
evaluation/      # evaluation scripts, Pareto curves, plotting
models/          # saved SB3 checkpoints (gitignored)
logs/            # .npy result files and .png plots
docs/            # architecture docs, future reports
tests/           # pytest suite (channel physics, reward behavior)
```

The main components live in `environment/`:
- `isac_env.py` — the Gymnasium environment. Wraps everything.
- `mimo_system.py` — ULA steering vectors, array gain calculations
- `channel_model.py` — Saleh-Valenzuela clustered mmWave channel, supports both random and temporally coherent modes
- `v2x_scenario.py` — 1D vehicle motion model

## Reward design

The agent gets a weighted sum of two normalised terms:

- **Communication** — Shannon rate normalised by the theoretical maximum: `log2(1 + SNR) / log2(1 + SNR_max)`
- **Sensing** — squared cosine similarity between the beamforming vector `w` and the steering vector at the target AoA: `|w^H · a(AoA)|^2 / (||w||^2 · Nt)`. This is 0 when the beam misses the target and 1 when it's perfectly aligned. No calibration needed.

The balance is controlled by `alpha` and `beta` in `RewardConfig`.

## Phased roadmap

The project is being developed in five phases. See [`.hermes/plans/2026-05-19_145500-isac-mimo-drl-reorg-and-phases.md`](.hermes/plans/2026-05-19_145500-isac-mimo-drl-reorg-and-phases.md) for details.

| Phase | What | Status |
|-------|------|--------|
| P1 | Reward redesign, channel physics fixes, verification tests, repo cleanup | **In progress** |
| P2 | 3GPP channel model, UPA support, urban/highway V2X profiles | Planned |
| P3 | PPO / SAC / TD3 benchmark suite | Planned |
| P4 | WMMSE, MMSE-ISAC baselines, Pareto analysis | Planned |
| P5 | Full write-up, ablation studies, reproducibility pack | Planned |

## Why this matters for IDC

This project is built for the research topic *"Intelligent Decision-making for Cognitive ISAC Networks"* (IDC, FAU, supervised by Dr. Lahmeri). The current setup is a single-node ISAC baseline. The natural next step is extending it to cognitive, network-level decision-making — adaptive coverage, interference-aware resource allocation, and environment-aware beamforming. That's exactly what the IDC project calls for.

## References

- Lillicrap et al. *Continuous control with deep reinforcement learning*. arXiv:1509.02971, 2015.
- Liu et al. *A survey on fundamental limits of integrated sensing and communication*. IEEE Commun. Surv. Tutor., 2022.
- Saleh & Valenzuela. *A statistical model for indoor multipath propagation*. IEEE JSAC, 1987.
- 3GPP TR 37.885 — *Study on NR V2X*, Release 16.
- Wymeersch et al. *Integration of communication and sensing in 6G*. IEEE ComMag, 2021.