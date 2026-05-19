"""
Post-training visualisation for ISAC-MIMO beamforming.

Generates three publication-quality plots and saves them to ``logs/``:

1. ``reward_curve.png``
   DDPG training reward over environment timesteps, loaded from
   the ``evaluations.npz`` file written by SB3's ``EvalCallback``.

2. ``sensing_vs_comm.png``
   Scatter plot comparing per-episode sensing gain vs. communication
   rate for the DDPG agent (``ddpg_eval.npy``) and the DFT classical
   baseline (``classical_baseline.npy``).

3. ``beampattern.png``
   Polar plot of the transmit array gain pattern for:
     - The trained DDPG agent's deterministic action at a fixed
       observation (AoA = 30°, distance = 100 m, velocity = 16.67 m/s).
     - The ideal DFT steering-vector beamformer at the same AoA.

Usage
-----
    python -m evaluation.plot_results
or
    python evaluation/plot_results.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, '/content/isac-mimo-drl')

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless – no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from stable_baselines3 import DDPG
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
_LOG_DIR = _ROOT / "logs"

# SB3 EvalCallback writes evaluations.npz inside the log directory
_EVAL_NPZ = _LOG_DIR / "ddpg_isac" / "evaluations.npz"

_DDPG_NPY = _LOG_DIR / "ddpg_eval.npy"
_BASELINE_NPY = _LOG_DIR / "classical_baseline.npy"

_MODEL_DIR = _ROOT / "models" / "ddpg_isac_best"
_VECNORM_PKL = _MODEL_DIR / "vecnormalize.pkl"

# Fixed evaluation AoA for the beam-pattern plot
_FIXED_AOA_DEG: float = 30.0
_FIXED_DISTANCE_M: float = 100.0
_FIXED_VELOCITY_MS: float = 60.0 / 3.6   # 60 km/h → m/s


# ---------------------------------------------------------------------------
# Shared dark-mode plot style
# ---------------------------------------------------------------------------
_DARK = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "text.color": "#c9d1d9",
    "font.family": "DejaVu Sans",
}


# ---------------------------------------------------------------------------
# Environment / model helpers
# ---------------------------------------------------------------------------
def _make_raw_env(alpha: float = 0.5, beta: float = 0.5) -> ISACEnv:
    mimo = MIMOSystem(Nt=4, Nr=4, carrier_freq=28e9)
    scenario = V2XScenario(speed_kmh=60.0, update_interval=0.1)
    channel = SVChannelModel(
        mimo=mimo,
        distance=_FIXED_DISTANCE_M,
        num_clusters=3,
        rays_per_cluster=10,
        azimuth_spread_deg=10.0,
    )
    reward_cfg = RewardConfig(alpha=alpha, beta=beta)
    return ISACEnv(
        mimo=mimo,
        scenario=scenario,
        channel=channel,
        reward_config=reward_cfg,
        max_steps=200,
    )


def _load_ddpg_with_vecnorm() -> tuple[DDPG, VecNormalize]:
    """
    Load the best DDPG checkpoint together with its frozen VecNormalize stats.

    Returns
    -------
    tuple[DDPG, VecNormalize]
    """
    if not (_MODEL_DIR / "best_model.zip").exists():
        raise FileNotFoundError(
            f"Model not found at {_MODEL_DIR / 'best_model.zip'}.\n"
            "Run training/train_ddpg.py first."
        )
    if not _VECNORM_PKL.exists():
        raise FileNotFoundError(
            f"VecNormalize stats not found at {_VECNORM_PKL}.\n"
            "Ensure training saved vecnormalize.pkl to models/ddpg_isac_best/."
        )

    from stable_baselines3.common.monitor import Monitor  # local import

    raw_env = _make_raw_env()
    vec_env = DummyVecEnv([lambda: Monitor(raw_env)])  # noqa: B023
    vec_env = VecNormalize.load(str(_VECNORM_PKL), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = DDPG.load(str(_MODEL_DIR / "best_model"), env=vec_env)
    return model, vec_env


def _build_fixed_observation(env: ISACEnv) -> np.ndarray:
    """
    Construct a deterministic observation at (AoA=30°, d=100 m, v=60 km/h).

    We reset the underlying environment and manually override its internal
    state to get a reproducible channel snapshot at the fixed AoA.
    """
    env.reset(seed=0)

    # Override scenario state with the fixed values
    env._state = {
        "distance": _FIXED_DISTANCE_M,
        "velocity": _FIXED_VELOCITY_MS,
        "angle_of_arrival": _FIXED_AOA_DEG,
    }
    env._H = env.channel.generate(velocity_ms=_FIXED_VELOCITY_MS)
    return env._get_observation()


# ---------------------------------------------------------------------------
# Plot 1: Reward curve
# ---------------------------------------------------------------------------
def plot_reward_curve() -> None:
    """
    Load SB3 EvalCallback's ``evaluations.npz`` and plot mean ± std reward
    vs. training timesteps.

    The ``evaluations.npz`` file contains:
      - ``timesteps`` : 1-D int array
      - ``results``   : 2-D float array of shape (n_evals, n_eval_episodes)
      - ``ep_lengths``: 2-D int array of same shape
    """
    if not _EVAL_NPZ.exists():
        raise FileNotFoundError(
            f"evaluations.npz not found at {_EVAL_NPZ}.\n"
            "Run training/train_ddpg.py and ensure log_path is set in EvalCallback."
        )

    data = np.load(str(_EVAL_NPZ))
    timesteps: np.ndarray = data["timesteps"]          # shape (n_evals,)
    results: np.ndarray = data["results"]              # shape (n_evals, n_eps)

    mean_r = results.mean(axis=1)
    std_r = results.std(axis=1)

    with plt.rc_context(_DARK):
        fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
        ax.grid(True, zorder=0)

        # Shaded ±1 std band
        ax.fill_between(
            timesteps,
            mean_r - std_r,
            mean_r + std_r,
            alpha=0.18,
            color="#58a6ff",
            zorder=1,
        )
        # Mean line
        ax.plot(
            timesteps,
            mean_r,
            color="#58a6ff",
            linewidth=2.0,
            label="DDPG mean eval reward ± 1 std",
            zorder=3,
        )

        # Smoothed trend (exponential moving average, span ≈ 5 eval points)
        if len(mean_r) >= 5:
            alpha_ema = 0.3
            ema = np.array(mean_r, dtype=float)
            for i in range(1, len(ema)):
                ema[i] = alpha_ema * mean_r[i] + (1 - alpha_ema) * ema[i - 1]
            ax.plot(
                timesteps,
                ema,
                color="#3fb950",
                linewidth=1.5,
                linestyle="-",
                alpha=0.85,
                label="EMA trend",
                zorder=4,
            )

        ax.set_xlabel("Environment Timesteps", fontsize=12, labelpad=8)
        ax.set_ylabel("Episode Reward", fontsize=12, labelpad=8)
        ax.set_title(
            "DDPG Training Reward Curve\n"
            "ISAC-MIMO  ·  Nt=4, Nr=4  ·  28 GHz mmWave",
            fontsize=13, pad=12,
        )

        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}k")
        )

        legend = ax.legend(fontsize=9, framealpha=0.3,
                           facecolor="#161b22", edgecolor="#30363d")
        for t in legend.get_texts():
            t.set_color("#c9d1d9")

        fig.tight_layout()
        out = _LOG_DIR / "reward_curve.png"
        fig.savefig(str(out), bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Plot 2: Sensing gain vs. comm rate scatter
# ---------------------------------------------------------------------------
def plot_sensing_vs_comm() -> None:
    """
    Scatter plot of per-episode sensing gain vs. communication rate for the
    DDPG agent and the DFT classical baseline.

    Both ``.npy`` files are expected to contain dicts with keys:
      - ``episode_comm_rates``
      - ``episode_sens_gains``  (DDPG) / ``episode_sensing_gains`` (baseline)
    """
    if not _DDPG_NPY.exists():
        raise FileNotFoundError(
            f"DDPG results not found at {_DDPG_NPY}.\n"
            "Run evaluation/evaluate_agent.py first."
        )
    if not _BASELINE_NPY.exists():
        raise FileNotFoundError(
            f"Classical baseline not found at {_BASELINE_NPY}.\n"
            "Run training/classical_baseline.py first."
        )

    ddpg = np.load(str(_DDPG_NPY), allow_pickle=True).item()
    bl = np.load(str(_BASELINE_NPY), allow_pickle=True).item()

    ddpg_comm = ddpg["episode_comm_rates"]
    ddpg_sens = ddpg["episode_sens_gains"]
    bl_comm = bl["episode_comm_rates"]
    bl_sens = bl["episode_sensing_gains"]

    with plt.rc_context(_DARK):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
        ax.grid(True, zorder=0)

        # DFT baseline – orange diamonds
        ax.scatter(
            bl_comm, bl_sens,
            marker="D", s=45, alpha=0.55,
            color="#f0883e", edgecolors="none",
            zorder=3,
            label=f"DFT baseline  (n={len(bl_comm)})",
        )
        # DFT mean cross-hair
        ax.axvline(bl_comm.mean(), color="#f0883e", linewidth=1.0,
                   linestyle=":", alpha=0.6, zorder=2)
        ax.axhline(bl_sens.mean(), color="#f0883e", linewidth=1.0,
                   linestyle=":", alpha=0.6, zorder=2)

        # DDPG – teal circles
        ax.scatter(
            ddpg_comm, ddpg_sens,
            marker="o", s=45, alpha=0.55,
            color="#58a6ff", edgecolors="none",
            zorder=3,
            label=f"DDPG agent  (n={len(ddpg_comm)})",
        )
        # DDPG mean cross-hair
        ax.axvline(ddpg_comm.mean(), color="#58a6ff", linewidth=1.0,
                   linestyle=":", alpha=0.6, zorder=2)
        ax.axhline(ddpg_sens.mean(), color="#58a6ff", linewidth=1.0,
                   linestyle=":", alpha=0.6, zorder=2)

        # Mean markers (larger, solid)
        ax.scatter(
            [ddpg_comm.mean()], [ddpg_sens.mean()],
            marker="*", s=260, color="#58a6ff",
            edgecolors="#e6edf3", linewidths=0.8, zorder=5,
            label=f"DDPG mean  ({ddpg_comm.mean():.2f}, {ddpg_sens.mean():.2f})",
        )
        ax.scatter(
            [bl_comm.mean()], [bl_sens.mean()],
            marker="P", s=200, color="#f0883e",
            edgecolors="#e6edf3", linewidths=0.8, zorder=5,
            label=f"DFT mean  ({bl_comm.mean():.2f}, {bl_sens.mean():.2f})",
        )

        ax.set_xlabel("Mean Episode Comm Rate (bps/Hz)", fontsize=12, labelpad=8)
        ax.set_ylabel("Mean Episode Sensing Gain (raw array gain)", fontsize=12, labelpad=8)
        ax.set_title(
            "Sensing Gain vs. Communication Rate\n"
            "DDPG Agent vs. DFT Classical Baseline  ·  200 Episodes each",
            fontsize=13, pad=12,
        )

        legend = ax.legend(fontsize=9, framealpha=0.3,
                           facecolor="#161b22", edgecolor="#30363d")
        for t in legend.get_texts():
            t.set_color("#c9d1d9")

        fig.tight_layout()
        out = _LOG_DIR / "sensing_vs_comm.png"
        fig.savefig(str(out), bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Plot 3: Beam pattern polar plot
# ---------------------------------------------------------------------------
def _compute_array_gain_pattern(
    w: np.ndarray,
    mimo: MIMOSystem,
    theta_deg: np.ndarray,
) -> np.ndarray:
    """
    Compute the transmit array gain (linear scale) over all angles.

    Parameters
    ----------
    w : np.ndarray
        Complex beamforming vector of shape (Nt,).
    mimo : MIMOSystem
        MIMO system for steering-vector computation.
    theta_deg : np.ndarray
        Angles in degrees at which to evaluate gain.

    Returns
    -------
    np.ndarray
        Array gain values (linear, shape matches theta_deg).
    """
    gains = np.array([mimo.array_gain(w, float(th)) for th in theta_deg])
    return gains


def _dft_beamformer(mimo: MIMOSystem, aoa_deg: float) -> np.ndarray:
    """Unit-norm DFT steering-vector beamformer at *aoa_deg*."""
    w = mimo.tx_steering_vector(aoa_deg)
    return w / np.linalg.norm(w)


def _ddpg_beamformer(
    model: DDPG,
    vec_env: VecNormalize,
    raw_env: ISACEnv,
) -> np.ndarray:
    """
    Obtain the DDPG agent's deterministic beamformer at the fixed observation.

    The raw action from the model is converted to the complex beamforming
    vector using the same ``_action_to_beamformer`` path used inside ISACEnv,
    then normalised to unit norm for a fair pattern comparison.
    """
    obs_raw = _build_fixed_observation(raw_env)  # (obs_dim,)
    # VecNormalize expects (n_envs, obs_dim); wrap and normalise
    obs_norm = vec_env.normalize_obs(obs_raw[np.newaxis, :])
    action, _ = model.predict(obs_norm, deterministic=True)

    # Convert to complex beamformer via ISACEnv helper
    w = raw_env._action_to_beamformer(action[0])
    norm = np.linalg.norm(w)
    return w / max(norm, 1e-12)   # unit norm for pattern comparison


def plot_beampattern() -> None:
    """
    Polar plot of the DDPG agent beam pattern vs. the DFT baseline at AoA=30°.

    The array-gain pattern is swept over θ ∈ [−90°, 90°] and plotted on
    a half-space polar axes (top hemisphere only, matching ULA convention).
    Gain is converted to dB and plotted in the range [−30, 0] dB (normalised
    to the peak of each pattern).
    """
    model, vec_env = _load_ddpg_with_vecnorm()

    # Use the underlying ISACEnv from the VecNormalize stack
    raw_env: ISACEnv = vec_env.venv.envs[0].env  # Monitor → ISACEnv
    mimo: MIMOSystem = raw_env.mimo

    # Angular sweep
    theta_deg = np.linspace(-90.0, 90.0, 361)
    theta_rad = np.deg2rad(theta_deg)

    # DFT beamformer
    w_dft = _dft_beamformer(mimo, _FIXED_AOA_DEG)
    gain_dft = _compute_array_gain_pattern(w_dft, mimo, theta_deg)

    # DDPG beamformer
    w_ddpg = _ddpg_beamformer(model, vec_env, raw_env)
    gain_ddpg = _compute_array_gain_pattern(w_ddpg, mimo, theta_deg)

    # Normalise each pattern to its own peak and convert to dB
    _db_floor = -30.0   # display floor in dB

    def _to_db_norm(gain: np.ndarray) -> np.ndarray:
        peak = gain.max() if gain.max() > 0 else 1.0
        with np.errstate(divide="ignore"):
            db = 10.0 * np.log10(gain / peak)
        return np.clip(db, _db_floor, 0.0)

    db_dft = _to_db_norm(gain_dft)
    db_ddpg = _to_db_norm(gain_ddpg)

    # Shift so the floor maps to 0 radius (polar plots need non-negative r)
    r_dft = db_dft - _db_floor      # range [0, 30]
    r_ddpg = db_ddpg - _db_floor    # range [0, 30]

    with plt.rc_context(_DARK):
        fig = plt.figure(figsize=(8, 7), dpi=140)
        fig.patch.set_facecolor("#0d1117")

        ax = fig.add_subplot(111, projection="polar")
        ax.set_facecolor("#161b22")

        # Polar axes: θ=0 at top (North), clockwise positive
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        # Only show the frontal hemisphere [−90°, 90°]
        ax.set_thetamin(-90)
        ax.set_thetamax(90)

        # Plot patterns
        ax.plot(theta_rad, r_ddpg,
                color="#58a6ff", linewidth=2.2,
                label="DDPG agent", zorder=4)
        ax.fill(theta_rad, r_ddpg,
                alpha=0.12, color="#58a6ff", zorder=3)

        ax.plot(theta_rad, r_dft,
                color="#f0883e", linewidth=2.0,
                linestyle="--", label="DFT baseline", zorder=4)
        ax.fill(theta_rad, r_dft,
                alpha=0.10, color="#f0883e", zorder=3)

        # Mark the target AoA
        aoa_rad = math.radians(_FIXED_AOA_DEG)
        ax.axvline(aoa_rad, color="#3fb950", linewidth=1.3,
                   linestyle=":", alpha=0.8, zorder=5)
        ax.text(
            aoa_rad, r_dft.max() * 1.05,
            f"AoA = {_FIXED_AOA_DEG:.0f}°",
            color="#3fb950", fontsize=9, ha="center", zorder=6,
        )

        # Radial ticks → map back to dB labels
        db_ticks = [0, 10, 20, 30]                     # offsets from floor
        db_labels = [f"{_db_floor + d:.0f} dB" for d in db_ticks]
        ax.set_rticks(db_ticks)
        ax.set_yticklabels(db_labels, fontsize=7.5, color="#8b949e")
        ax.set_rlabel_position(80)

        # Angular ticks
        ax.set_thetagrids(
            np.arange(-90, 91, 30),
            labels=[f"{a}°" for a in range(-90, 91, 30)],
            color="#8b949e", fontsize=8,
        )

        ax.tick_params(colors="#8b949e")
        ax.spines["polar"].set_color("#30363d")
        ax.grid(color="#21262d", linestyle="--", linewidth=0.5)

        ax.set_title(
            f"Transmit Beam Pattern  ·  AoA = {_FIXED_AOA_DEG:.0f}°\n"
            "DDPG Agent vs. DFT Baseline  ·  Nt=4  ·  28 GHz ULA",
            fontsize=12, pad=18, color="#e6edf3",
        )

        legend = ax.legend(
            loc="lower left", bbox_to_anchor=(-0.18, -0.12),
            fontsize=9, framealpha=0.3,
            facecolor="#161b22", edgecolor="#30363d",
        )
        for t in legend.get_texts():
            t.set_color("#c9d1d9")

        fig.tight_layout()
        out = _LOG_DIR / "beampattern.png"
        fig.savefig(str(out), bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        vec_env.close()
        print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Generate all three result plots."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 56)
    print("Generating ISAC-MIMO result plots")
    print("=" * 56)

    print("\n[1/3] Training reward curve …")
    plot_reward_curve()

    print("\n[2/3] Sensing gain vs. comm rate scatter …")
    plot_sensing_vs_comm()

    print("\n[3/3] Beam pattern polar plot …")
    plot_beampattern()

    print("\nAll plots saved to:", _LOG_DIR)


if __name__ == "__main__":
    main()
