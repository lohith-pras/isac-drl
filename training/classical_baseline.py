"""
Classical beamforming baseline for ISAC-MIMO.

Runs a fixed DFT (steering-vector) beamformer pointed at the current
vehicle AoA for 200 episodes and logs the results for later
comparison with the RL agent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from environment.channel_model import SVChannelModel
from environment.isac_env import ISACEnv
from environment.mimo_system import MIMOSystem
from environment.v2x_scenario import V2XScenario
from utils.reward_config import RewardConfig


def make_env() -> ISACEnv:
    """Create a fresh ISACEnv instance with default parameters."""
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
    env = ISACEnv(
        mimo=mimo,
        scenario=scenario,
        channel=channel,
        reward_config=reward_cfg,
        max_steps=200,
    )
    return env


def dft_action_for_aoa(env: ISACEnv, aoa: float) -> np.ndarray:
    """
    Return a simple DFT (steering-vector) beamformer aligned with *aoa*.

    The beamforming vector is the transmit steering vector at the
    given AoA, normalised to unit norm.  The action returned is the
    flattened concatenation of its real and imaginary parts, clipped
    to the [-1, 1] range expected by the environment.
    """
    # Transmit steering vector at the vehicle's AoA
    w_complex = env.mimo.tx_steering_vector(aoa)
    w_complex = w_complex / np.linalg.norm(w_complex)

    # Flatten real / imag and clip to [-1, 1]
    action = np.concatenate([w_complex.real, w_complex.imag])
    action = np.clip(action, -1.0, 1.0)
    return action


def run_baseline(
    num_eval_episodes: int = 200,
    max_steps: int = 200,
) -> dict:
    """
    Run the classical DFT beamforming baseline.

    Calibration of the sensing normalisation bound is handled
    automatically by ``ISACEnv.__init__()``; no warmup phase is needed.

    Parameters
    ----------
    num_eval_episodes : int, optional
        Number of evaluation episodes to run (default 200).
    max_steps : int, optional
        Maximum steps per episode (default 200).

    Returns
    -------
    dict
        Dictionary with arrays of per-episode metrics.
    """
    env = make_env()

    # ------------------------------------------------------------------
    # Evaluation phase: fixed DFT beamformer
    # ------------------------------------------------------------------
    print(f"Running {num_eval_episodes} evaluation episodes ...")
    episode_rewards = []
    episode_comm_rates = []
    episode_sensing_gains = []

    for ep in range(num_eval_episodes):
        env.reset()
        total_reward = 0.0
        total_comm = 0.0
        total_sensing = 0.0
        steps = 0

        while True:
            aoa = env._state["angle_of_arrival"]
            action = dft_action_for_aoa(env, aoa)

            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_comm += info["comm_rate"]
            total_sensing += info["sensing_gain"]
            steps += 1

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_comm_rates.append(total_comm / max(steps, 1))
        episode_sensing_gains.append(total_sensing / max(steps, 1))

        if (ep + 1) % 10 == 0:
            print(
                f"Episode {ep + 1}/{num_eval_episodes}  |  "
                f"Avg Reward: {np.mean(episode_rewards[-10:]):.4f}  |  "
                f"Avg Comm: {np.mean(episode_comm_rates[-10:]):.4f}  |  "
                f"Avg Sensing: {np.mean(episode_sensing_gains[-10:]):.4f}"
            )

    env.close()

    results = {
        "episode_rewards": np.array(episode_rewards),
        "episode_comm_rates": np.array(episode_comm_rates),
        "episode_sensing_gains": np.array(episode_sensing_gains),
    }
    return results


def main() -> None:
    """Execute the classical baseline benchmark and save results."""
    print("=" * 60)
    print("Running Classical DFT Baseline")
    print("=" * 60)

    results = run_baseline(
        num_eval_episodes=200,
        max_steps=200,
    )

    # Save to logs/classical_baseline.npy
    root = Path(__file__).resolve().parent.parent
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_path = log_dir / "classical_baseline.npy"

    np.save(save_path, results)
    print(f"\nResults saved to: {save_path}")

    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Mean Reward:        {np.mean(results['episode_rewards']):.4f}")
    print(f"Std  Reward:        {np.std(results['episode_rewards']):.4f}")
    print(f"Mean Comm Rate:     {np.mean(results['episode_comm_rates']):.4f}")
    print(f"Mean Sensing Gain:  {np.mean(results['episode_sensing_gains']):.4f}")


if __name__ == "__main__":
    main()
