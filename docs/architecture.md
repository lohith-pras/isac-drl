# Architecture Guide — ISAC-MIMO-DRL

## Overview

This project implements a Deep Reinforcement Learning (DRL) approach to beamforming for Integrated Sensing and Communication (ISAC) in mmWave MIMO systems. It uses a custom Gymnasium environment to model the physical layer and motion dynamics.

## System Components

### 1. Environment (`environment/`)
- **`isac_env.py`**: The core Gymnasium environment (`ISACEnv`).
  - **Action Space**: `Box(-1.0, 1.0, (2*Nt,))` representing real and imaginary parts of beamforming weights.
  - **Observation Space**: Flattened complex channel matrix `H` (real + imag) plus vehicle state `(distance, velocity, AoA)`.
  - **Reward**: Weighted sum of Communication Rate and Sensing Gain: `alpha * N_comm + beta * N_sens`. Sensing gain uses squared cosine similarity toward the vehicle's AoA.
- **`mimo_system.py`**: Models the physical antenna array (ULA). Provides steering vectors and array gain calculations.
- **`channel_model.py`**: Saleh-Valenzuela clustered mmWave channel model. Supports pathloss, Doppler shifts, and a **coherent mode** for temporal evolution.
- **`v2x_scenario.py`**: 1D motion model for a vehicle moving along a road relative to a fixed base station.

### 2. Training (`training/`)
- **`train_ddpg.py`**: Training script using Stable-Baselines3 (SB3) DDPG implementation.
- **Wrapper Stack**: To ensure stable training, the environment is wrapped: `Monitor → DummyVecEnv → VecNormalize`.

### 3. Evaluation (`evaluation/`)
- Scripts for evaluating trained agents, comparing against a DFT-based classical baseline, and generating Pareto curves for the ISAC trade-off.

### 4. Utilities (`utils/`)
- **`reward_config.py`**: Configuration for reward weights and power budgets.
- **`config.py`**: General project configuration.

## Data Flow

1. **Reset**: `ISACEnv.reset()` initializes the `V2XScenario` and generates the initial `SVChannelModel` matrix.
2. **Action**: The DRL agent provides beamforming weights.
3. **Step**:
   - Weights are mapped to a complex beamformer.
   - Communication rate is computed via log-det formula.
   - Sensing reward is computed via squared cosine similarity to the AoA.
   - Scenario advances vehicle position; distance and AoA are updated.
   - New channel matrix is generated for the next step.
4. **Reward**: Returned as a scalar to the agent for optimization.

## Important Note on Model Loading

Trained models must be loaded with the same `VecNormalize` statistics used during training. See `evaluation/evaluate_agent.py` for the correct loading procedure.
