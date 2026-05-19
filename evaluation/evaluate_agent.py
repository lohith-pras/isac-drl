"""
Evaluation script: DDPG agent on ISAC-MIMO environment.

Loads the best saved DDPG model and its VecNormalize statistics,
runs 200 evaluation episodes on ISACEnv with Nt=4, Nr=4, and records
per-episode metrics (mean reward, mean communication rate, mean sensing
gain).  Results are saved to logs/ddpg_eval.npy and summary statistics
are printed at the end.

Usage
-----
    python -m evaluation.evaluate_agent
or
    python evaluation/evaluate_agent.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.channel_model import SVChannelModel
from environment.isac_env import ISACEnv
from environment.mimo_system import MIMOSystem
from environment.v2x_scenario import V2XScenario
from utils.reward_config import RewardConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = _ROOT / "models" / "ddpg_isac_best"
_VECNORM_PATH = _MODEL_DIR / "vecnormalize.pkl"
_LOG_DIR = _ROOT / "logs"
_RESULTS_PATH = _LOG_DIR / "ddpg_eval.npy"

# Number of evaluation episodes
N_EVAL_EPISODES = 200


# ---------------------------------------------------------------------------
# Environment factory (mirrors training/train_ddpg.py)
# ---------------------------------------------------------------------------
def _make_isac_env() -> ISACEnv:
    """Instantiate ISACEnv with Nt=4, Nr=4 and the default scenario."""
    mimo = MIMOSystem(Nt=4, Nr=4, carrier_freq=28e9)
    scenario = V2XScenario(speed_kmh=60.0, update_interval=0.1)
    channel = SVChannelModel(
        mimo=mimo,
        distance=100.0,
        num_clusters=3,
        rays_per_cluster=10,
        azimuth_spread_deg=10.0,
    )
    reward_cfg = RewardConfig(alpha=0.5, beta=0.5)
    return ISACEnv(
        mimo=mimo,
        scenario=scenario,
        channel=channel,
        reward_config=reward_cfg,
        max_steps=200,
    )


# ---------------------------------------------------------------------------
# Policy probe (debug)
# ---------------------------------------------------------------------------
def _debug_policy_probe(model: DDPG, vec_env: VecNormalize, n_steps: int = 5) -> None:
    """
    Run ``n_steps`` deterministic steps and report the raw action outputs.

    Prints the beamforming weight vector (real + imag parts, shape 2×Nt) at
    each step, then computes the L2 distance between consecutive actions so
    it is immediately clear whether the agent is producing a *static* policy
    (same action regardless of observation) or a *dynamic* one (action varies
    with the changing channel / scenario state).

    Parameters
    ----------
    model : DDPG
        Loaded stable-baselines3 DDPG agent.
    vec_env : VecNormalize
        The wrapped, frozen evaluation environment.
    n_steps : int
        Number of steps to probe (default 5).
    """
    print("=" * 60)
    print(f"DEBUG: Policy probe — {n_steps} deterministic steps")
    print("=" * 60)

    obs = vec_env.reset()
    prev_action: np.ndarray | None = None

    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        raw_action = action[0]   # shape (2*Nt,); strip the batch dimension

        # Split into real / imag halves for readability
        n_half = len(raw_action) // 2
        real_part = raw_action[:n_half]
        imag_part = raw_action[n_half:]

        print(f"\n  Step {step + 1}/{n_steps}")
        print(f"    real parts : [{', '.join(f'{v:+.5f}' for v in real_part)}]")
        print(f"    imag parts : [{', '.join(f'{v:+.5f}' for v in imag_part)}]")

        if prev_action is not None:
            delta = np.linalg.norm(raw_action - prev_action)
            changed = "CHANGED  ✓" if delta > 1e-6 else "STATIC   ✗"
            print(f"    Δ vs prev  : {delta:.6f}  →  {changed}")
        else:
            print("    Δ vs prev  : — (first step)")

        prev_action = raw_action.copy()
        obs, _, done_arr, _ = vec_env.step(action)
        if bool(done_arr[0]):
            print("    (episode ended early during probe)")
            obs = vec_env.reset()
            prev_action = None

    # Overall verdict
    print()
    if prev_action is not None:
        # Collect all actions again for a quick all-same check
        print("  Verdict: see Δ values above.")
        print("    • All Δ ≈ 0  →  STATIC policy (agent ignores observation)")
        print("    • Any Δ > 0  →  DYNAMIC policy (action adapts to state)")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------
def evaluate(n_episodes: int = N_EVAL_EPISODES) -> dict[str, np.ndarray]:
    """
    Run ``n_episodes`` deterministic evaluation episodes.

    Parameters
    ----------
    n_episodes : int
        Number of episodes to evaluate (default 200).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys:
        * ``"episode_rewards"``   – shape (n_episodes,)
        * ``"episode_comm_rates"`` – shape (n_episodes,)
        * ``"episode_sens_gains"`` – shape (n_episodes,)
    """
    # ------------------------------------------------------------------
    # 1. Validate model artifacts exist
    # ------------------------------------------------------------------
    model_zip = _MODEL_DIR / "best_model.zip"
    if not model_zip.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at: {model_zip}\n"
            "Run training/train_ddpg.py first, or update _MODEL_DIR."
        )
    if not _VECNORM_PATH.exists():
        raise FileNotFoundError(
            f"VecNormalize stats not found at: {_VECNORM_PATH}\n"
            "Ensure the training script saved 'vecnormalize.pkl' alongside the model."
        )

    # ------------------------------------------------------------------
    # 2. Build the evaluation environment (same wrapper stack as training)
    # ------------------------------------------------------------------
    raw_env = _make_isac_env()
    monitored = Monitor(raw_env)
    vec_env = DummyVecEnv([lambda: monitored])  # noqa: B023

    # Load saved normalisation statistics and lock them (no further updates)
    vec_env = VecNormalize.load(str(_VECNORM_PATH), vec_env)
    vec_env.training = False        # freeze running mean / std
    vec_env.norm_reward = False     # do NOT normalise rewards during eval

    # ------------------------------------------------------------------
    # 3. Load the DDPG model
    # ------------------------------------------------------------------
    print(f"Loading model from: {model_zip}")
    model = DDPG.load(str(_MODEL_DIR / "best_model"), env=vec_env)
    print("Model loaded successfully.\n")

    # ------------------------------------------------------------------
    # 3b. DEBUG: static vs. dynamic policy probe (5 steps)
    # ------------------------------------------------------------------
    _debug_policy_probe(model, vec_env)

    # ------------------------------------------------------------------
    # 4. Run evaluation episodes
    # ------------------------------------------------------------------
    episode_rewards: list[float] = []
    episode_comm_rates: list[float] = []
    episode_sens_gains: list[float] = []

    print(f"Running {n_episodes} evaluation episodes …")
    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False

        step_rewards: list[float] = []
        step_comm_rates: list[float] = []
        step_sens_gains: list[float] = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_arr = vec_env.step(action)
            done = bool(done_arr[0])

            step_rewards.append(float(reward[0]))
            info = info_arr[0]
            step_comm_rates.append(float(info.get("comm_rate", np.nan)))
            step_sens_gains.append(float(info.get("sensing_gain", np.nan)))

        ep_mean_reward = float(np.mean(step_rewards))
        ep_mean_comm = float(np.nanmean(step_comm_rates))
        ep_mean_sens = float(np.nanmean(step_sens_gains))

        episode_rewards.append(ep_mean_reward)
        episode_comm_rates.append(ep_mean_comm)
        episode_sens_gains.append(ep_mean_sens)

        if (ep + 1) % 20 == 0:
            print(
                f"  Episode {ep + 1:>3}/{n_episodes}  |  "
                f"reward={ep_mean_reward:.4f}  |  "
                f"comm={ep_mean_comm:.4f} bps/Hz  |  "
                f"sens={ep_mean_sens:.4f}"
            )

    vec_env.close()

    return {
        "episode_rewards": np.array(episode_rewards),
        "episode_comm_rates": np.array(episode_comm_rates),
        "episode_sens_gains": np.array(episode_sens_gains),
    }


def _print_summary(results: dict[str, np.ndarray]) -> None:
    """Print mean ± std summary for each recorded metric."""
    rewards = results["episode_rewards"]
    comm = results["episode_comm_rates"]
    sens = results["episode_sens_gains"]

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY  (200 episodes, Nt=4, Nr=4)")
    print("=" * 60)
    print(
        f"  Mean Episode Reward  : {rewards.mean():.4f} "
        f"± {rewards.std():.4f}  "
        f"[min={rewards.min():.4f}, max={rewards.max():.4f}]"
    )
    print(
        f"  Mean Comm Rate       : {comm.mean():.4f} "
        f"± {comm.std():.4f} bps/Hz  "
        f"[min={comm.min():.4f}, max={comm.max():.4f}]"
    )
    print(
        f"  Mean Sensing Gain    : {sens.mean():.4f} "
        f"± {sens.std():.4f}  "
        f"[min={sens.min():.4f}, max={sens.max():.4f}]"
    )
    print("=" * 60)


def main() -> None:
    """Entry point: evaluate, save results, and print summary."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    results = evaluate(n_episodes=N_EVAL_EPISODES)

    # Save structured results
    np.save(str(_RESULTS_PATH), results)
    print(f"\nResults saved to: {_RESULTS_PATH}")

    _print_summary(results)


if __name__ == "__main__":
    main()
