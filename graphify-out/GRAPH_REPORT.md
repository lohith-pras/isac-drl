# Graph Report - .  (2026-05-12)

## Corpus Check
- Corpus is ~9,159 words - fits in a single context window. You may not need a graph.

## Summary
- 172 nodes · 249 edges · 19 communities (17 shown, 2 thin omitted)
- Extraction: 79% EXTRACTED · 21% INFERRED · 0% AMBIGUOUS · INFERRED: 52 edges (avg confidence: 0.69)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_ISAC Gymnasium Environment|ISAC Gymnasium Environment]]
- [[_COMMUNITY_DDPG Training & Callbacks|DDPG Training & Callbacks]]
- [[_COMMUNITY_Pareto & Evaluation Utilities|Pareto & Evaluation Utilities]]
- [[_COMMUNITY_Plotting & DFT Base|Plotting & DFT Base]]
- [[_COMMUNITY_MIMO System Physics|MIMO System Physics]]
- [[_COMMUNITY_Channel Model & Integration Tests|Channel Model & Integration Tests]]
- [[_COMMUNITY_V2X Motion Scenario|V2X Motion Scenario]]
- [[_COMMUNITY_DDPG Evaluation Script|DDPG Evaluation Script]]
- [[_COMMUNITY_Classical Baseline (DFT)|Classical Baseline (DFT)]]
- [[_COMMUNITY_Semantic Docs (Project Overview)|Semantic Docs (Project Overview)]]
- [[_COMMUNITY_NVIDIA API & Config|NVIDIA API & Config]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 17|Community 17]]

## God Nodes (most connected - your core abstractions)
1. `ISACEnv` - 25 edges
2. `MIMOSystem` - 19 edges
3. `V2XScenario` - 17 edges
4. `SVChannelModel` - 17 edges
5. `RewardConfig` - 14 edges
6. `ProgressCallback` - 11 edges
7. `NoiseDecayCallback` - 11 edges
8. `EvalPoint` - 9 edges
9. `_make_env()` - 9 edges
10. `make_isac_env()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `ProgressCallback` --uses--> `SVChannelModel`  [INFERRED]
  training/train_ddpg.py → environment/channel_model.py
- `ProgressCallback` --uses--> `ISACEnv`  [INFERRED]
  training/train_ddpg.py → environment/isac_env.py
- `ProgressCallback` --uses--> `MIMOSystem`  [INFERRED]
  training/train_ddpg.py → environment/mimo_system.py
- `ProgressCallback` --uses--> `V2XScenario`  [INFERRED]
  training/train_ddpg.py → environment/v2x_scenario.py
- `NoiseDecayCallback` --uses--> `SVChannelModel`  [INFERRED]
  training/train_ddpg.py → environment/channel_model.py

## Communities (19 total, 2 thin omitted)

### Community 0 - "ISAC Gymnasium Environment"
Cohesion: 0.13
Nodes (12): ISACEnv, Custom Gymnasium environment for ISAC-MIMO beamforming.  The agent controls a co, Estimate an empirical upper bound for the raw sensing gain.          Bootstraps, Map a flat action vector to a complex beamforming vector.          Parameters, Build the observation vector.          Returns         -------         np.ndarra, Gymnasium environment for ISAC-MIMO beamforming.      Observation     ----------, Compute the achievable communication rate using the log-det formula.          Pa, Compute the raw array-gain toward the current AoA.          Parameters         - (+4 more)

### Community 1 - "DDPG Training & Callbacks"
Cohesion: 0.13
Nodes (14): BaseCallback, main(), make_isac_env(), NoiseDecayCallback, ProgressCallback, Training script: DDPG for ISAC-MIMO beamforming.  Uses stable-baselines3 DDPG wi, Train DDPG on the ISAC-MIMO environment., Prints a summary every ``print_freq`` environment steps. (+6 more)

### Community 2 - "Pareto & Evaluation Utilities"
Cohesion: 0.14
Nodes (20): _build_plot(), _calibrate_max_sensing_gain(), EvalPoint, main(), _make_env(), _normalise_loaded_data(), _pareto_frontier(), Pareto frontier analysis for ISAC-MIMO beamforming.  Loads pre-computed evaluati (+12 more)

### Community 3 - "Plotting & DFT Base"
Cohesion: 0.14
Nodes (20): _build_fixed_observation(), _compute_array_gain_pattern(), _ddpg_beamformer(), _dft_beamformer(), _load_ddpg_with_vecnorm(), main(), _make_raw_env(), plot_beampattern() (+12 more)

### Community 4 - "MIMO System Physics"
Cohesion: 0.16
Nodes (8): MIMOSystem, MIMO system model for mmWave ISAC beamforming., Initialize a transmit beamforming matrix.          Parameters         ----------, Compute the array gain for a given beamforming vector and angle.          Parame, Compute the ULA steering vector.          Parameters         ----------, Compute the transmit ULA steering vector.          Parameters         ----------, MIMO system model for millimeter-wave (mmWave) ISAC scenarios.      Parameters, Compute the receive ULA steering vector.          Parameters         ----------

### Community 5 - "Channel Model & Integration Tests"
Cohesion: 0.15
Nodes (9): Clustered mmWave channel model based on the Saleh-Valenzuela model.  This module, Generate the complex channel matrix **H**.          The matrix includes path-los, Saleh-Valenzuela (SV) clustered mmWave channel model.      The channel matrix is, Return the transmit steering vector for *angle_deg*., Return the receive steering vector for *angle_deg*., SVChannelModel, Manual test for the ISAC-MIMO environment.  Runs a short rollout with random act, Instantiate ISACEnv and run 10 random steps.      Prints observation shape, acti (+1 more)

### Community 6 - "V2X Motion Scenario"
Cohesion: 0.16
Nodes (8): current_state(), Basic V2X scenario with a fixed base station and a moving vehicle.  State -----, Advance the simulation by one time-step.          The vehicle moves forward alon, Reset the vehicle to the initial position and time to zero.          Returns, One-dimensional V2X motion model.      A single vehicle moves along a straight r, Return the current 2-D Cartesian position of the vehicle.          Returns, Compute Euclidean distance and AoA from BS to vehicle.          Returns, V2XScenario

### Community 7 - "DDPG Evaluation Script"
Cohesion: 0.23
Nodes (11): _debug_policy_probe(), evaluate(), main(), _make_isac_env(), _print_summary(), Evaluation script: DDPG agent on ISAC-MIMO environment.  Loads the best saved DD, Run ``n_episodes`` deterministic evaluation episodes.      Parameters     ------, Print mean ± std summary for each recorded metric. (+3 more)

### Community 8 - "Classical Baseline (DFT)"
Cohesion: 0.27
Nodes (9): dft_action_for_aoa(), main(), make_env(), Classical beamforming baseline for ISAC-MIMO.  Runs a fixed DFT (steering-vector, Execute the classical baseline benchmark and save results., Create a fresh ISACEnv instance with default parameters., Return a simple DFT (steering-vector) beamformer aligned with *aoa*.      The be, Run the classical DFT beamforming baseline.      Calibration of the sensing norm (+1 more)

### Community 9 - "Semantic Docs (Project Overview)"
Cohesion: 0.33
Nodes (6): ISACEnv (Gymnasium Env), MIMOSystem (Physics Model), SVChannelModel (Saleh-Valenzuela), DDPG Training Script, V2XScenario (Vehicle Motion), VecNormalize Loading Pattern

### Community 10 - "NVIDIA API & Config"
Cohesion: 0.5
Nodes (3): get_nvidia_client(), Configuration module for ISAC-MIMO-DRL.  Loads environment variables from .env a, Return an OpenAI-compatible client configured for the NVIDIA Build API.

## Knowledge Gaps
- **79 isolated node(s):** `Training script: DDPG for ISAC-MIMO beamforming.  Uses stable-baselines3 DDPG wi`, `Prints a summary every ``print_freq`` environment steps.`, `Linearly decays the OU action-noise sigma from ``sigma_start``     to ``sigma_en`, `Build and return a plain ISACEnv (will be wrapped later).`, `Train DDPG on the ISAC-MIMO environment.` (+74 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ISACEnv` connect `ISAC Gymnasium Environment` to `DDPG Training & Callbacks`, `Pareto & Evaluation Utilities`, `Plotting & DFT Base`, `MIMO System Physics`, `Channel Model & Integration Tests`, `V2X Motion Scenario`, `DDPG Evaluation Script`, `Classical Baseline (DFT)`?**
  _High betweenness centrality (0.246) - this node is a cross-community bridge._
- **Why does `_make_raw_env()` connect `Plotting & DFT Base` to `ISAC Gymnasium Environment`, `DDPG Training & Callbacks`, `MIMO System Physics`, `Channel Model & Integration Tests`, `V2X Motion Scenario`?**
  _High betweenness centrality (0.185) - this node is a cross-community bridge._
- **Why does `MIMOSystem` connect `MIMO System Physics` to `ISAC Gymnasium Environment`, `DDPG Training & Callbacks`, `Pareto & Evaluation Utilities`, `Plotting & DFT Base`, `Channel Model & Integration Tests`, `DDPG Evaluation Script`, `Classical Baseline (DFT)`?**
  _High betweenness centrality (0.172) - this node is a cross-community bridge._
- **Are the 13 inferred relationships involving `ISACEnv` (e.g. with `ProgressCallback` and `NoiseDecayCallback`) actually correct?**
  _`ISACEnv` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `MIMOSystem` (e.g. with `ProgressCallback` and `NoiseDecayCallback`) actually correct?**
  _`MIMOSystem` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `V2XScenario` (e.g. with `ProgressCallback` and `NoiseDecayCallback`) actually correct?**
  _`V2XScenario` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `SVChannelModel` (e.g. with `ProgressCallback` and `NoiseDecayCallback`) actually correct?**
  _`SVChannelModel` has 11 INFERRED edges - model-reasoned connections that need verification._