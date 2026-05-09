"""
Pareto frontier analysis for ISAC-MIMO beamforming.

Loads pre-computed evaluation results for the DDPG agent and the DFT
classical baseline, then generates 5 additional random-policy evaluation
runs — one for each of the five alpha/beta trade-off points — to map
out the achievable communication-rate vs. sensing-gain trade-off space.

A Pareto frontier curve is fitted through the alpha/beta sweep points and
the DDPG / classical-baseline operating points are overlaid.

The final plot is saved to ``logs/pareto_curve.png``.

Usage
-----
    python -m evaluation.pareto_curve
or
    python evaluation/pareto_curve.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")  # headless rendering – no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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
_DDPG_PATH = _LOG_DIR / "ddpg_eval.npy"
_BASELINE_PATH = _LOG_DIR / "classical_baseline.npy"
_OUTPUT_PATH = _LOG_DIR / "pareto_curve.png"

# Evaluation settings
_N_EPISODES = 200   # random-policy episodes per alpha/beta point
_MAX_STEPS = 200    # steps per episode (matches training config)
_SEED = 42

# Alpha/beta trade-off combinations to sweep
_TRADEOFFS: list[tuple[float, float]] = [
    (0.2, 0.8),
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.8, 0.2),
]


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
class EvalPoint(NamedTuple):
    """Mean normalised comm-rate and sensing-gain for one operating point."""
    label: str
    comm_norm: float    # normalised to [0, 1]
    sens_norm: float    # normalised to [0, 1]
    alpha: float | None = None
    beta: float | None = None


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def _make_env(alpha: float = 0.5, beta: float = 0.5) -> ISACEnv:
    """Build an ISACEnv with the requested alpha/beta weighting."""
    mimo = MIMOSystem(Nt=4, Nr=4, carrier_freq=28e9)
    scenario = V2XScenario(speed_kmh=60.0, update_interval=0.1)
    channel = SVChannelModel(
        mimo=mimo,
        distance=100.0,
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
        max_steps=_MAX_STEPS,
    )


# ---------------------------------------------------------------------------
# Random-policy rollout
# ---------------------------------------------------------------------------
def _run_random_policy(
    alpha: float,
    beta: float,
    n_episodes: int = _N_EPISODES,
    seed: int = _SEED,
) -> tuple[float, float]:
    """
    Roll out a purely random policy and return mean normalised metrics.

    Parameters
    ----------
    alpha, beta : float
        ISAC reward weights.
    n_episodes : int
        Number of evaluation episodes.
    seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    tuple[float, float]
        (mean_comm_norm, mean_sens_norm) averaged over all steps across
        all episodes.
    """
    rng = np.random.default_rng(seed)
    env = _make_env(alpha=alpha, beta=beta)

    all_comm_norm: list[float] = []
    all_sens_norm: list[float] = []

    for _ in range(n_episodes):
        env.reset()
        done = False
        while not done:
            action = rng.uniform(-1.0, 1.0, size=env.action_space.shape)
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            all_comm_norm.append(float(info["comm_norm"]))
            all_sens_norm.append(float(info["sens_norm"]))

    env.close()
    return float(np.mean(all_comm_norm)), float(np.mean(all_sens_norm))


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def _theoretical_max_comm_rate() -> float:
    """
    Compute the theoretical maximum communication rate used by ISACEnv.

    Replicates the formula in ISACEnv.__init__ so we can normalise raw
    ``comm_rate`` values loaded from pre-computed .npy files.
    """
    reward_cfg = RewardConfig(alpha=0.5, beta=0.5)
    tx_power = reward_cfg.dbm_to_linear(reward_cfg.max_tx_power_dbm)
    noise = reward_cfg.dbm_to_linear(reward_cfg.noise_power_dbm)
    max_snr = tx_power * 4 * 4 / noise   # Nt=4, Nr=4
    return math.log2(1.0 + max_snr)


def _calibrate_max_sensing_gain() -> float:
    """
    Return the empirical sensing-gain bound from a fresh ISACEnv.

    Instantiating ISACEnv runs the 500-sample calibration internally,
    so we can read ``_max_sensing_gain`` directly.
    """
    env = _make_env()
    bound = env._max_sensing_gain
    env.close()
    return float(bound)


def _normalise_loaded_data(
    raw_comm: np.ndarray,
    raw_sens: np.ndarray,
    max_comm: float,
    max_sens: float,
) -> tuple[float, float]:
    """Normalise raw arrays to [0, 1] and return their scalar means."""
    comm_norm = float(np.clip(raw_comm / max_comm, 0.0, 1.0).mean())
    sens_norm = float(np.clip(raw_sens / max_sens, 0.0, 1.0).mean())
    return comm_norm, sens_norm


# ---------------------------------------------------------------------------
# Pareto frontier helpers
# ---------------------------------------------------------------------------
def _pareto_frontier(
    comm_vals: np.ndarray,
    sens_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract and sort the upper-right Pareto frontier from 2-D points.

    A point dominates another if it has *both* higher comm and higher
    sensing gain.  Returns the non-dominated points sorted by comm_norm.

    Parameters
    ----------
    comm_vals, sens_vals : np.ndarray
        1-D arrays of the same length.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (comm_sorted, sens_sorted) for the Pareto-optimal subset.
    """
    points = np.column_stack([comm_vals, sens_vals])
    # Sort by comm ascending
    idx = np.argsort(points[:, 0])
    points = points[idx]

    pareto: list[int] = []
    max_sens = -np.inf
    # Sweep right-to-left so we keep the point with highest sensing for
    # each comm level
    for i in range(len(points) - 1, -1, -1):
        if points[i, 1] >= max_sens:
            pareto.append(i)
            max_sens = points[i, 1]

    pareto = sorted(pareto)
    return points[pareto, 0], points[pareto, 1]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
_STYLE = {
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


def _build_plot(
    sweep_points: list[EvalPoint],
    ddpg_point: EvalPoint,
    baseline_point: EvalPoint,
) -> plt.Figure:
    """
    Assemble the Pareto frontier figure.

    Parameters
    ----------
    sweep_points : list[EvalPoint]
        Alpha/beta sweep operating points (random policy).
    ddpg_point : EvalPoint
        DDPG agent operating point.
    baseline_point : EvalPoint
        Classical DFT baseline operating point.

    Returns
    -------
    matplotlib.figure.Figure
    """
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 7), dpi=140)
        ax.grid(True, zorder=0)

        # ------------------------------------------------------------------
        # 1. Pareto frontier curve through the alpha/beta sweep points
        # ------------------------------------------------------------------
        sweep_comm = np.array([p.comm_norm for p in sweep_points])
        sweep_sens = np.array([p.sens_norm for p in sweep_points])

        pf_comm, pf_sens = _pareto_frontier(sweep_comm, sweep_sens)

        ax.plot(
            pf_comm,
            pf_sens,
            color="#58a6ff",
            linewidth=2.2,
            linestyle="--",
            alpha=0.75,
            zorder=2,
            label="Pareto frontier (random-policy sweep)",
        )

        # Shaded region under the Pareto frontier
        ax.fill_between(
            pf_comm,
            pf_sens,
            alpha=0.08,
            color="#58a6ff",
            zorder=1,
        )

        # ------------------------------------------------------------------
        # 2. Alpha/beta sweep scatter (colour-coded by alpha)
        # ------------------------------------------------------------------
        cmap = plt.cm.plasma  # type: ignore[attr-defined]
        alphas_arr = np.array([p.alpha for p in sweep_points], dtype=float)
        colours = cmap((alphas_arr - alphas_arr.min()) / (alphas_arr.ptp() + 1e-9))

        for pt, col in zip(sweep_points, colours):
            ax.scatter(
                pt.comm_norm,
                pt.sens_norm,
                color=col,
                s=100,
                zorder=5,
                edgecolors="#e6edf3",
                linewidths=0.8,
            )
            ax.annotate(
                f"α={pt.alpha:.1f}/β={pt.beta:.1f}",
                xy=(pt.comm_norm, pt.sens_norm),
                xytext=(8, 6),
                textcoords="offset points",
                fontsize=8.5,
                color="#c9d1d9",
                zorder=6,
            )

        # ------------------------------------------------------------------
        # 3. Classical DFT baseline
        # ------------------------------------------------------------------
        ax.scatter(
            baseline_point.comm_norm,
            baseline_point.sens_norm,
            marker="D",
            s=160,
            color="#f0883e",
            edgecolors="#e6edf3",
            linewidths=1.0,
            zorder=7,
            label=f"DFT baseline  (comm={baseline_point.comm_norm:.3f}, "
                  f"sens={baseline_point.sens_norm:.3f})",
        )
        ax.annotate(
            "DFT baseline",
            xy=(baseline_point.comm_norm, baseline_point.sens_norm),
            xytext=(10, -14),
            textcoords="offset points",
            fontsize=9,
            color="#f0883e",
            zorder=8,
            fontweight="bold",
        )

        # ------------------------------------------------------------------
        # 4. DDPG agent
        # ------------------------------------------------------------------
        ax.scatter(
            ddpg_point.comm_norm,
            ddpg_point.sens_norm,
            marker="*",
            s=340,
            color="#3fb950",
            edgecolors="#e6edf3",
            linewidths=0.8,
            zorder=7,
            label=f"DDPG agent  (comm={ddpg_point.comm_norm:.3f}, "
                  f"sens={ddpg_point.sens_norm:.3f})",
        )
        ax.annotate(
            "DDPG agent",
            xy=(ddpg_point.comm_norm, ddpg_point.sens_norm),
            xytext=(10, 8),
            textcoords="offset points",
            fontsize=9,
            color="#3fb950",
            zorder=8,
            fontweight="bold",
        )

        # ------------------------------------------------------------------
        # 5. Colour-bar legend for alpha values
        # ------------------------------------------------------------------
        sm = plt.cm.ScalarMappable(  # type: ignore[attr-defined]
            cmap=cmap,
            norm=plt.Normalize(vmin=alphas_arr.min(), vmax=alphas_arr.max()),  # type: ignore[attr-defined]
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.035)
        cbar.set_label("α (comm weight)", color="#c9d1d9", labelpad=8)
        cbar.ax.yaxis.set_tick_params(color="#8b949e")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")

        # ------------------------------------------------------------------
        # 6. Labels, legend, title
        # ------------------------------------------------------------------
        ax.set_xlabel("Normalised Communication Rate", fontsize=12, labelpad=8)
        ax.set_ylabel("Normalised Sensing Gain", fontsize=12, labelpad=8)
        ax.set_title(
            "ISAC-MIMO  ·  Communication-Sensing Pareto Frontier\n"
            r"Nt=4, Nr=4  ·  28 GHz mmWave  ·  V2X scenario",
            fontsize=13,
            pad=14,
            color="#e6edf3",
        )

        ax.set_xlim(-0.02, 1.06)
        ax.set_ylim(-0.02, 1.06)

        legend = ax.legend(
            loc="lower left",
            fontsize=8.5,
            framealpha=0.3,
            facecolor="#161b22",
            edgecolor="#30363d",
        )
        for text in legend.get_texts():
            text.set_color("#c9d1d9")

        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Load data, run random-policy sweep, build Pareto plot, save."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Compute normalisation bounds from a fresh environment
    # ------------------------------------------------------------------
    print("Calibrating normalisation bounds …")
    max_comm = _theoretical_max_comm_rate()
    max_sens = _calibrate_max_sensing_gain()
    print(f"  max_comm_rate   = {max_comm:.4f} bps/Hz")
    print(f"  max_sensing_gain = {max_sens:.4f}\n")

    # ------------------------------------------------------------------
    # 2. Load DDPG evaluation results
    # ------------------------------------------------------------------
    if not _DDPG_PATH.exists():
        raise FileNotFoundError(
            f"DDPG results not found at {_DDPG_PATH}.\n"
            "Run evaluation/evaluate_agent.py first."
        )
    ddpg_data = np.load(str(_DDPG_PATH), allow_pickle=True).item()
    ddpg_comm, ddpg_sens = _normalise_loaded_data(
        ddpg_data["episode_comm_rates"],
        ddpg_data["episode_sens_gains"],
        max_comm,
        max_sens,
    )
    ddpg_point = EvalPoint("DDPG", ddpg_comm, ddpg_sens)
    print(f"DDPG agent    →  comm_norm={ddpg_comm:.4f}  sens_norm={ddpg_sens:.4f}")

    # ------------------------------------------------------------------
    # 3. Load classical DFT baseline results
    # ------------------------------------------------------------------
    if not _BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Classical baseline not found at {_BASELINE_PATH}.\n"
            "Run training/classical_baseline.py first."
        )
    bl_data = np.load(str(_BASELINE_PATH), allow_pickle=True).item()
    bl_comm, bl_sens = _normalise_loaded_data(
        bl_data["episode_comm_rates"],
        bl_data["episode_sensing_gains"],
        max_comm,
        max_sens,
    )
    baseline_point = EvalPoint("DFT baseline", bl_comm, bl_sens)
    print(f"DFT baseline  →  comm_norm={bl_comm:.4f}  sens_norm={bl_sens:.4f}")

    # ------------------------------------------------------------------
    # 4. Random-policy sweep across alpha/beta combinations
    # ------------------------------------------------------------------
    print(f"\nRunning random-policy sweep ({len(_TRADEOFFS)} configs × "
          f"{_N_EPISODES} episodes) …")
    sweep_points: list[EvalPoint] = []
    for idx, (alpha, beta) in enumerate(_TRADEOFFS):
        print(f"  [{idx + 1}/{len(_TRADEOFFS)}]  α={alpha:.1f}  β={beta:.1f} …", end="", flush=True)
        c_norm, s_norm = _run_random_policy(alpha=alpha, beta=beta, seed=_SEED + idx)
        sweep_points.append(
            EvalPoint(
                label=f"α={alpha:.1f}/β={beta:.1f}",
                comm_norm=c_norm,
                sens_norm=s_norm,
                alpha=alpha,
                beta=beta,
            )
        )
        print(f"  comm={c_norm:.4f}  sens={s_norm:.4f}")

    # ------------------------------------------------------------------
    # 5. Build and save the plot
    # ------------------------------------------------------------------
    print("\nRendering Pareto frontier plot …")
    fig = _build_plot(sweep_points, ddpg_point, baseline_point)
    fig.savefig(str(_OUTPUT_PATH), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Plot saved to: {_OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # 6. Console summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{'Label':<22} {'comm_norm':>10} {'sens_norm':>10}")
    print("-" * 44)
    for pt in sweep_points:
        print(f"  {pt.label:<20} {pt.comm_norm:>10.4f} {pt.sens_norm:>10.4f}")
    print("-" * 44)
    print(f"  {'DDPG agent':<20} {ddpg_point.comm_norm:>10.4f} {ddpg_point.sens_norm:>10.4f}")
    print(f"  {'DFT baseline':<20} {baseline_point.comm_norm:>10.4f} {baseline_point.sens_norm:>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
