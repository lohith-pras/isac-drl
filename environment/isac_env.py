"""
Custom Gymnasium environment for ISAC-MIMO beamforming.

The agent controls a continuous beamforming vector.  The environment
simulates a V2X link, generates the instantaneous mmWave channel, and
returns a scalar reward that balances communication rate against
sensing (array-gain) performance.
"""

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.channel_model import SVChannelModel
from environment.mimo_system import MIMOSystem
from environment.v2x_scenario import V2XScenario
from utils.reward_config import RewardConfig


class ISACEnv(gym.Env):
    """
    Gymnasium environment for ISAC-MIMO beamforming.

    Observation
    -----------
    Flattened real and imaginary parts of the current channel matrix
    ``H`` concatenated with the scalar V2X state
    (distance, velocity, angle_of_arrival).

    Action
    ------
    Continuous beamforming weights (real + imag parts) normalised to
    the range [-1, 1].  Internally they are mapped to a complex vector
    ``w`` with  ``||w|| = sqrt(max_tx_power)``.

    Reward
    ------
    Reward is computed using fixed theoretical-bound normalisation:
    ``alpha * N_comm + beta * N_sens``
    where both terms are clipped to [0, 1].

    * ``N_comm = clip(comm_rate / max_comm_rate, 0, 1)``
      ``max_comm_rate = log2(1 + SNR_max * Nt * Nr)``
      ``SNR_max       = tx_power_linear / noise_power_linear``
    * ``N_sens = clip(array_gain / _max_sensing_gain, 0, 1)``
      ``_max_sensing_gain`` is calibrated empirically at init time by
      sampling 500 random actions and taking the peak × 1.2 (20% headroom).

    Parameters
    ----------
    mimo : MIMOSystem
        MIMO system model (Nt, Nr, carrier frequency, etc.).
    scenario : V2XScenario
        V2X motion model.
    channel : SVChannelModel
        Saleh-Valenzuela channel generator.
    reward_config : RewardConfig
        Configuration for reward weights and power budgets.
    max_steps : int, optional
        Episode length limit (default 200).
    """

    def __init__(
        self,
        mimo: MIMOSystem,
        scenario: V2XScenario,
        channel: SVChannelModel,
        reward_config: RewardConfig,
        *,
        max_steps: int = 200,
    ):
        super().__init__()

        self.mimo = mimo
        self.scenario = scenario
        self.channel = channel
        self.reward_cfg = reward_config
        self.max_steps = max_steps

        self.Nt = mimo.Nt
        self.Nr = mimo.Nr

        # ------------------------------------------------------------------
        # Observation space
        #   H_real (Nr*Nt) + H_imag (Nr*Nt) + [dist, vel, aoa] (3)
        # ------------------------------------------------------------------
        obs_dim = 2 * self.Nr * self.Nt + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float64,
        )

        # ------------------------------------------------------------------
        # Action space: real + imag parts of the beamforming vector
        #   shape = 2 * Nt,  range = [-1, 1]
        # ------------------------------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * self.Nt,),
            dtype=np.float64,
        )

        # Internal state
        self._current_step = 0
        self._H: np.ndarray | None = None
        self._state: dict[str, float] = {}

        # ------------------------------------------------------------------
        # Fixed normalisation bounds (computed once at init)
        # ------------------------------------------------------------------
        tx_power_linear = reward_config.dbm_to_linear(reward_config.max_tx_power_dbm)
        noise_linear = reward_config.dbm_to_linear(reward_config.noise_power_dbm)
        # Shannon upper bound: all Nt transmit antennas focus perfectly on
        # all Nr receive antennas, giving max SNR = P * Nt * Nr / sigma^2
        max_snr = tx_power_linear * self.Nt * self.Nr / noise_linear
        self._max_comm_rate: float = math.log2(1.0 + max_snr)

        # ------------------------------------------------------------------
        # Calibrate sensing bound via 500 random-action rollout
        #
        # The theoretical Nt² bound ignores path-loss and channel power
        # embedded in the raw array_gain, so we measure the empirical peak
        # and add 20 % headroom to keep sens_norm in a useful [0, 1] range.
        # ------------------------------------------------------------------
        self._max_sensing_gain: float = self._calibrate_sensing_bound(
            n_samples=500, headroom=1.2
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _calibrate_sensing_bound(self, n_samples: int = 500, headroom: float = 1.2) -> float:
        """
        Estimate an empirical upper bound for the raw sensing gain.

        Bootstraps a temporary environment state, samples ``n_samples``
        random beamforming actions, records the maximum raw sensing gain
        observed, then returns that peak scaled by ``headroom``.

        Parameters
        ----------
        n_samples : int
            Number of random actions to evaluate (default 500).
        headroom : float
            Multiplicative safety margin applied to the observed peak
            (default 1.2, i.e. 20 %).

        Returns
        -------
        float
            Calibrated sensing bound with headroom applied.
        """
        rng = np.random.default_rng(seed=0)

        # Bootstrap temporary state so _sensing_gain() can be called
        self._state = self.scenario.reset()
        self._H = self.channel.generate(velocity_ms=self._state["velocity"])

        peak = 0.0
        for _ in range(n_samples):
            # Sample a random action in [-1, 1]^(2*Nt)
            action = rng.uniform(-1.0, 1.0, size=self.action_space.shape)
            w = self._action_to_beamformer(action)
            gain = self.mimo.array_gain(w, self._state["angle_of_arrival"])
            if gain > peak:
                peak = gain
            # Advance scenario so AoA varies across the calibration sweep
            self._state = self.scenario.step()
            self._H = self.channel.generate(velocity_ms=self._state["velocity"])

        # Tear down temporary state; reset() will re-initialise properly
        self._state = {}
        self._H = None
        self._current_step = 0

        return float(peak * headroom)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _action_to_beamformer(self, action: np.ndarray) -> np.ndarray:
        """
        Map a flat action vector to a complex beamforming vector.

        Parameters
        ----------
        action : np.ndarray
            Flat real array of length 2*Nt in [-1, 1].

        Returns
        -------
        np.ndarray
            Complex beamforming vector of shape (Nt,) with unit norm,
            scaled by sqrt(max_tx_power).
        """
        real_part = action[: self.Nt]
        imag_part = action[self.Nt :]
        w_raw = real_part + 1j * imag_part

        # Normalise to unit norm, then scale by transmit power
        norm = np.linalg.norm(w_raw)
        if norm < 1e-12:
            # Fallback: single antenna excitation to avoid zero
            w = np.zeros(self.Nt, dtype=complex)
            w[0] = 1.0
        else:
            w = w_raw / norm

        max_power_linear = self.reward_cfg.dbm_to_linear(
            self.reward_cfg.max_tx_power_dbm
        )
        return w * math.sqrt(max_power_linear)

    def _get_observation(self) -> np.ndarray:
        """
        Build the observation vector.

        Returns
        -------
        np.ndarray
            1-D array: [H_real, H_imag, distance, velocity, aoa].
        """
        assert self._H is not None and self._state is not None
        return np.concatenate(
            [
                self._H.real.flatten(),
                self._H.imag.flatten(),
                np.array(
                    [
                        self._state["distance"],
                        self._state["velocity"],
                        self._state["angle_of_arrival"],
                    ]
                ),
            ]
        ).astype(np.float64)

    def _communication_rate(self, w: np.ndarray) -> float:
        """
        Compute the achievable communication rate using the log-det formula.

        Parameters
        ----------
        w : np.ndarray
            Complex beamforming vector of shape (Nt,).

        Returns
        -------
        float
            Achievable rate in bits/s/Hz (natural log base).
        """
        assert self._H is not None

        # Effective channel after beamforming: h_eq = H @ w  (shape Nr,)
        h_eq = self._H @ w

        # Signal covariance = h_eq * h_eq^H  (rank-1)
        # Noise covariance  = σ^2 I
        noise_linear = self.reward_cfg.dbm_to_linear(
            self.reward_cfg.noise_power_dbm
        )

        # SINR for SIMO channel after beamforming:
        # R = log2(1 + ||H w||^2 / σ^2)
        sinr = np.linalg.norm(h_eq) ** 2 / noise_linear
        return math.log2(1.0 + sinr)

    def _sensing_gain(self, w: np.ndarray) -> float:
        """
        Compute the raw array-gain toward the current AoA.

        Parameters
        ----------
        w : np.ndarray
            Complex beamforming vector of shape (Nt,).

        Returns
        -------
        float
            Raw array gain (unnormalised).  Normalisation against the
            theoretical peak (``Nt²``) is performed in ``step()``.
        """
        assert self._state is not None
        aoa = self._state["angle_of_arrival"]
        return self.mimo.array_gain(w, aoa)

    def _normalise(self, value: float, bound: float) -> float:
        """
        Normalise a scalar to [0, 1] using a fixed theoretical upper bound.

        Parameters
        ----------
        value : float
            Current raw value.
        bound : float
            Fixed theoretical maximum (must be > 0).

        Returns
        -------
        float
            Normalised value clipped to [0, 1].
        """
        return float(np.clip(value / max(bound, 1e-12), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:  # type: ignore[override]
        """
        Reset the environment to an initial state.

        Normalisation bounds are fixed theoretical constants computed in
        ``__init__`` and do not change between episodes.

        Parameters
        ----------
        seed : int | None, optional
            RNG seed for reproducibility.
        options : dict | None, optional
            Unused, kept for Gymnasium compatibility.

        Returns
        -------
        tuple[np.ndarray, dict]
            (observation, info)
        """
        super().reset(seed=seed)

        # Reset scenario and step counter
        self._current_step = 0
        self._state = self.scenario.reset()

        # Generate channel for the initial state
        self._H = self.channel.generate(velocity_ms=self._state["velocity"])

        obs = self._get_observation()
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:  # type: ignore[override]
        """
        Execute one time-step with the given beamforming action.

        Parameters
        ----------
        action : np.ndarray
            Continuous beamforming weights (flattened real + imag parts).

        Returns
        -------
        tuple[np.ndarray, float, bool, bool, dict]
            (observation, reward, terminated, truncated, info)
        """
        # ------------------------------------------------------------------
        # 1. Map action -> complex beamformer
        # ------------------------------------------------------------------
        w = self._action_to_beamformer(action)

        # ------------------------------------------------------------------
        # 2. Compute reward with fixed theoretical-bound normalisation
        # ------------------------------------------------------------------
        comm_rate = self._communication_rate(w)
        sensing = self._sensing_gain(w)  # raw array gain

        # Normalise each term to [0, 1] using fixed theoretical bounds
        comm_norm = self._normalise(comm_rate, self._max_comm_rate)
        sens_norm = self._normalise(sensing, self._max_sensing_gain)


        reward = (
            self.reward_cfg.alpha * comm_norm
            + self.reward_cfg.beta * sens_norm
        )

        # ------------------------------------------------------------------
        # 3. Advance dynamics
        # ------------------------------------------------------------------
        self._current_step += 1
        self._state = self.scenario.step()
        self._H = self.channel.generate(velocity_ms=self._state["velocity"])

        # ------------------------------------------------------------------
        # 4. Termination / truncation
        # ------------------------------------------------------------------
        terminated = False
        truncated = self._current_step >= self.max_steps

        info: dict[str, Any] = {
            "comm_rate": comm_rate,
            "sensing_gain": sensing,
            "comm_norm": comm_norm,
            "sens_norm": sens_norm,
        }

        obs = self._get_observation()
        return obs, float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> None:  # noqa: ARG002
        """
        Render the current environment state (no-op for now).

        Parameters
        ----------
        mode : str, optional
            Rendering mode (default "human").
        """
        print("ISAC-MIMO Environment")
        if self._state:
            print(f"  Step   : {self._current_step}")
            print(f"  Distance:  {self._state['distance']:.2f} m")
            print(f"  Velocity:  {self._state['velocity']:.2f} m/s")
            print(f"  AoA:       {self._state['angle_of_arrival']:.2f}°")
