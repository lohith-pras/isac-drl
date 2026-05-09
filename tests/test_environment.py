"""
Manual test for the ISAC-MIMO environment.

Runs a short rollout with random actions and prints per-step
information.  No external test framework is used.
"""

import sys
from pathlib import Path

import numpy as np

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.channel_model import SVChannelModel
from environment.isac_env import ISACEnv
from environment.mimo_system import MIMOSystem
from environment.v2x_scenario import V2XScenario
from utils.reward_config import RewardConfig


def test_environment() -> None:
    """
    Instantiate ISACEnv and run 10 random steps.

    Prints observation shape, action shape, and reward at each step,
    and asserts that the shapes are consistent with the constructor
    parameters.
    """
    print("=" * 60)
    print("Testing ISAC-MIMO Environment")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Build components
    # ------------------------------------------------------------------
    mimo = MIMOSystem(Nt=4, Nr=4, carrier_freq=28e9)
    scenario = V2XScenario(speed_kmh=60.0, update_interval=0.1)
    channel = SVChannelModel(
        mimo=mimo,
        distance=100.0,
        num_clusters=3,
        rays_per_cluster=10,
        azimuth_spread_deg=10.0,
    )
    reward_cfg = RewardConfig()

    env = ISACEnv(
        mimo=mimo,
        scenario=scenario,
        channel=channel,
        reward_config=reward_cfg,
        max_steps=200,
    )

    # Reset the environment
    obs, info = env.reset(seed=42)

    # Expected shapes
    expected_obs_dim = 2 * mimo.Nr * mimo.Nt + 3
    expected_act_dim = 2 * mimo.Nt

    print(f"Initial observation shape: {obs.shape}")
    print(f"Expected observation size: {expected_obs_dim}")
    print(f"Expected action size:      {expected_act_dim}")
    print("-" * 60)

    # ------------------------------------------------------------------
    # 2. Run 10 random steps
    # ------------------------------------------------------------------
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Assertions
        assert obs.shape == (expected_obs_dim,), (
            f"Step {step}: observation shape {obs.shape} does not match "
            f"expected ({expected_obs_dim},)"
        )
        assert action.shape == (expected_act_dim,), (
            f"Step {step}: action shape {action.shape} does not match "
            f"expected ({expected_act_dim},)"
        )

        print(
            f"Step {step:2d} | "
            f"Obs: {obs.shape} | "
            f"Act: {action.shape} | "
            f"Reward: {reward:+.4f} | "
            f"CommRate: {info['comm_rate']:.4f} | "
            f"SensingGain: {info['sensing_gain']:.4f}"
        )

        if terminated or truncated:
            print("Episode finished early.")
            break

    print("-" * 60)
    print("All assertions passed. Environment test complete.")
    print("=" * 60)


if __name__ == "__main__":
    test_environment()
