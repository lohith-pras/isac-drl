"""
Training script: DDPG for ISAC-MIMO beamforming.

Uses stable-baselines3 DDPG with observation normalisation via
VecNormalize, and logs to TensorBoard.
"""

import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.channel_model import SVChannelModel
from environment.isac_env import ISACEnv
from environment.mimo_system import MIMOSystem
from environment.v2x_scenario import V2XScenario
from utils.reward_config import RewardConfig


# ------------------------------------------------------------------
# Progress callback (prints every N steps)
# ------------------------------------------------------------------
class ProgressCallback(BaseCallback):
    """Prints a summary every ``print_freq`` environment steps."""

    def __init__(self, print_freq: int = 5000) -> None:
        super().__init__()
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        # num_timesteps is updated by the base callback
        if self.num_timesteps % self.print_freq == 0:
            # Info dict available in the last rollout
            ep_info = self.locals.get("infos", [{}])[-1]
            ep_reward = ep_info.get("episode", {}).get("r", np.nan)
            print(
                f"[Step {self.num_timesteps:>7}]  "
                f"Episode reward: {ep_reward:>10.4f}"
            )
        return True


# ------------------------------------------------------------------
# Action-noise decay callback
# ------------------------------------------------------------------
class NoiseDecayCallback(BaseCallback):
    """Linearly decays the OU action-noise sigma from ``sigma_start``
    to ``sigma_end`` over ``decay_steps`` environment steps.

    After ``decay_steps`` the sigma is clamped to ``sigma_end``.
    """

    def __init__(
        self,
        sigma_start: float = 0.3,
        sigma_end: float = 0.05,
        decay_steps: int = 150_000,
    ) -> None:
        super().__init__()
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.decay_steps = decay_steps

    def _on_step(self) -> bool:
        frac = min(self.num_timesteps / self.decay_steps, 1.0)
        new_sigma = self.sigma_start + frac * (self.sigma_end - self.sigma_start)
        # model.action_noise is an OrnsteinUhlenbeckActionNoise instance
        noise = self.model.action_noise
        if noise is not None:
            noise.sigma = new_sigma * np.ones_like(noise.sigma)
        return True


# ------------------------------------------------------------------
# Environment factory
# ------------------------------------------------------------------
def make_isac_env() -> gym.Env:
    """Build and return a plain ISACEnv (will be wrapped later)."""
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


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------
def main() -> None:
    """Train DDPG on the ISAC-MIMO environment."""
    # Paths
    root = Path(__file__).resolve().parent.parent
    log_dir = root / "logs" / "ddpg_isac"
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create and wrap the environment ----------------------------
    # Monitor → DummyVecEnv → VecNormalize
    env = make_isac_env()
    env = Monitor(env)                     # Tracks episode rewards
    env = DummyVecEnv([lambda: env])      # Vectorised wrapper
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    # 2. DDPG exploration noise (Ornstein-Uhlenbeck) -----------------
    n_actions = env.action_space.shape[-1]  # type: ignore[attr-defined]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.3 * np.ones(n_actions),
    )

    # 3. Build the DDPG agent ---------------------------------------
    policy_kwargs = dict(net_arch=[256, 256])
    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=100_000,
        action_noise=action_noise,
        tau=0.005,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        verbose=1,
    )

    # 4. Callbacks -------------------------------------------------
    progress_cb = ProgressCallback(print_freq=5000)
    noise_decay_cb = NoiseDecayCallback(
        sigma_start=0.3,
        sigma_end=0.05,
        decay_steps=150_000,
    )
    eval_cb = EvalCallback(
        env,
        best_model_save_path=str(model_dir / "ddpg_isac_best"),
        log_path=str(log_dir),
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    # 5. Train -------------------------------------------------------
    total_timesteps = 300_000
    print(f"Starting DDPG training for {total_timesteps:,} timesteps …")
    t_start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=[progress_cb, noise_decay_cb, eval_cb])
    elapsed = time.time() - t_start
    print(f"Training finished in {elapsed:.1f} s")

    # 6. Save final model + VecNormalize stats -----------------------
    final_path = model_dir / "ddpg_isac_final"
    model.save(str(final_path))
    env.save(str(final_path / "vecnormalize.pkl"))
    print(f"Final model saved to: {final_path}")
    print(f"VecNormalize stats saved to: {final_path / 'vecnormalize.pkl'}")

    env.close()


if __name__ == "__main__":
    main()
