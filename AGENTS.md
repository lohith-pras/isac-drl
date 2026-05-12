# AGENTS.md — ISAC-MIMO-DRL

## Project overview

DRL-based beamforming for ISAC (Integrated Sensing and Communication) in mmWave MIMO. Custom Gymnasium env (`ISACEnv`), trained with SB3 DDPG, evaluated against a DFT steering-vector baseline.

## Key commands

```bash
uv sync                               # install deps (add --group dev for lint/test tools)
uv run python training/train_ddpg.py  # train DDPG (→ models/ddpg_isac_best, models/ddpg_isac_final)
uv run python evaluation/evaluate_agent.py   # eval trained agent (→ logs/ddpg_eval.npy)
uv run python training/classical_baseline.py # DFT baseline (→ logs/classical_baseline.npy)
uv run python evaluation/plot_results.py     # generate reward/beampattern/scatter plots
uv run python evaluation/pareto_curve.py     # generate Pareto frontier plot
```

Lint and test:
```bash
uv pip install ruff pytest   # one-time
uv run ruff check .          # lint
uv run pytest                # 1 quick integration test in tests/
```

## Architecture

- **`environment/`** — the custom Gymnasium env and physics models
  - `isac_env.py` — `ISACEnv(gym.Env)`: action = 2×Nt beamforming weights in `[-1,1]`, observation = flattened `H` (real+imag) + `(dist, vel, aoa)`
  - `mimo_system.py` — `MIMOSystem`: ULA steering vectors, array gain
  - `channel_model.py` — `SVChannelModel`: Saleh-Valenzuela clustered mmWave channel
  - `v2x_scenario.py` — `V2XScenario`: 1D vehicle motion model
- **`training/`** — training scripts
  - `train_ddpg.py` — SB3 DDPG with `Monitor → DummyVecEnv → VecNormalize` wrapper stack
- **`evaluation/`** — evaluation and visualization scripts
- **`utils/`** — `RewardConfig` dataclass (alpha/beta weights, power budgets), NVIDIA API client
- **`models/`** — saved SB3 model checkpoints and VecNormalize stats
- **`logs/`** — `.npy` result files and `.png` plots
- **`notebooks/`** — Colab training notebook
- **`main.py`** — unused stub

## Important conventions and gotchas

### Model loading + VecNormalize

The DDPG model was trained inside a `VecNormalize` wrapper. When loading for evaluation you MUST:

1. Build the **identical wrapper stack**: `Monitor → DummyVecEnv → VecNormalize.load(stats_path, vec_env)`
2. Set `vec_env.training = False` and `vec_env.norm_reward = False` after loading
3. Load the model with `DDPG.load(model_path, env=vec_env)` — the env must be the VecNormalize-wrapped one

See `evaluation/evaluate_agent.py:108-121` for the canonical pattern.

### Training wrapper order matters

Training uses: `Monitor(env) → DummyVecEnv([λ: env]) → VecNormalize(env, norm_obs=True, norm_reward=False)`.

The saved model is relative to this wrapper, so evaluation must replicate it exactly.

### Colab `sys.path` artifacts

`evaluation/evaluate_agent.py` has a `sys.path.insert(0, '/content/isac-mimo-drl')` line left from Colab usage. This is harmless locally (the directory won't exist) but don't cargo-cult it into new files — use the project root as the working directory instead.

### pyproject.toml `packages` is stale

`pyproject.toml` lists `packages = ["src/isac_mimo_drl"]` but no `src/` directory exists. The actual package layout is flat directories (`environment/`, `training/`, etc.) at the repo root. Building a wheel may fail; running scripts directly with `uv run python <path>` works fine.

### Lint: ruff only checks E/F/I/W

Config in `pyproject.toml` selects just `["E", "F", "I", "W"]`. There's no typecheck step configured.

### Test file is manual

`tests/test_environment.py` is not a pytest-style test class — it's a script with manual `assert` calls and a `test_environment()` function. It runs correctly under pytest but is essentially a smoke test.