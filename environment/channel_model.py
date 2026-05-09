"""
Clustered mmWave channel model based on the Saleh-Valenzuela model.

This module implements a geometric stochastic channel model for mmWave
massive MIMO systems.  The channel matrix is constructed from a small
number of dominant clusters, each containing several sub-paths (rays).
It is suitable for V2X and ISAC simulations.
"""
import math
from typing import List, Tuple

import numpy as np

from environment.mimo_system import MIMOSystem


class SVChannelModel:
    """
    Saleh-Valenzuela (SV) clustered mmWave channel model.

    The channel matrix is:

        H = sqrt(Nt * Nr / L) * Σ_i Σ_j
            α_{ij} * a_r(θ_{ij}^r) * a_t(θ_{ij}^t)^H

    where L = num_clusters * rays_per_cluster is the total number of
    paths, α_{ij} are complex path gains, and a_r(·), a_t(·) are the
    receive/transmit steering vectors.

    Parameters
    ----------
    mimo : MIMOSystem
        MIMO system instance (provides Nt, Nr, carrier frequency, and
        steering-vector helpers).
    distance : float, optional
        Transmitter--receiver distance in metres (default 100.0).
    num_clusters : int, optional
        Number of scattering clusters (default 3).
    rays_per_cluster : int, optional
        Number of rays per cluster (default 10).
    azimuth_spread_deg : float, optional
        Root-mean-square angular spread of each cluster in degrees
        (default 10.0).
    rng : np.random.Generator | None, optional
        Optional NumPy RNG for reproducibility.

    Attributes
    ----------
    Nt, Nr : int
        Number of Tx / Rx antennas.
    num_paths : int
        Total number of sub-paths (clusters × rays).
    """

    def __init__(
        self,
        mimo: MIMOSystem,
        *,
        distance: float = 100.0,
        num_clusters: int = 3,
        rays_per_cluster: int = 10,
        azimuth_spread_deg: float = 10.0,
        rng: np.random.Generator | None = None,
    ):
        self.mimo = mimo
        self.distance = distance
        self.num_clusters = num_clusters
        self.rays_per_cluster = rays_per_cluster
        self.azimuth_spread_deg = azimuth_spread_deg
        self.rng = rng if rng is not None else np.random.default_rng()

        self.Nt = mimo.Nt
        self.Nr = mimo.Nr
        self.num_paths = num_clusters * rays_per_cluster

        # ------------------------------------------------------------------
        # Pre-compute small-scale parameters for each path
        # ------------------------------------------------------------------
        # One aoa / aod per cluster, each ray is a small perturbation
        self._cluster_aoa_deg = self.rng.uniform(-60.0, 60.0, size=num_clusters)
        self._cluster_aod_deg = self.rng.uniform(-60.0, 60.0, size=num_clusters)

        # Path complex gain α ~ CN(0, 1)
        self._path_gains = (
            (self.rng.standard_normal((num_clusters, rays_per_cluster))
             + 1j * self.rng.standard_normal((num_clusters, rays_per_cluster)))
            / math.sqrt(2.0)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _steering_tx(self, angle_deg: float) -> np.ndarray:
        """Return the transmit steering vector for *angle_deg*."""
        return self.mimo.tx_steering_vector(angle_deg)

    def _steering_rx(self, angle_deg: float) -> np.ndarray:
        """Return the receive steering vector for *angle_deg*."""
        return self.mimo.rx_steering_vector(angle_deg)

    # ------------------------------------------------------------------
    # Channel matrix generator
    # ------------------------------------------------------------------

    def generate(self, velocity_ms: float = 0.0) -> np.ndarray:
        """
        Generate the complex channel matrix **H**.

        The matrix includes path-loss, small-scale fading, and an
        optional Doppler term.

        Parameters
        ----------
        velocity_ms : float, optional
            Relative radial velocity in m/s.  A positive value means
            the transmitter and receiver are moving toward each other.
            Doppler shift is applied uniformly to all paths.

        Returns
        -------
        np.ndarray
            Channel matrix of shape (Nr, Nt).
        """
        # Start with zero matrix
        H = np.zeros((self.Nr, self.Nt), dtype=complex)

        # Loop over clusters and rays
        for c in range(self.num_clusters):
            # Mean angles for this cluster
            mean_aoa = self._cluster_aoa_deg[c]
            mean_aod = self._cluster_aod_deg[c]

            for r in range(self.rays_per_cluster):
                # ------------------------------------------------------------
                # 1. Per-ray angles (Laplace / Gaussian spread around mean)
                # ------------------------------------------------------------
                aoa = mean_aoa + self.rng.normal(0.0, self.azimuth_spread_deg)
                aod = mean_aod + self.rng.normal(0.0, self.azimuth_spread_deg)

                # ------------------------------------------------------------
                # 2. Steering vectors
                # ------------------------------------------------------------
                ar = self._steering_rx(aoa)   # shape (Nr,)
                at = self._steering_tx(aod)   # shape (Nt,)

                # ------------------------------------------------------------
                # 3. Outer product a_r * a_t^H  =>  (Nr, Nt) contribution
                # ------------------------------------------------------------
                path_gain = self._path_gains[c, r]
                H += path_gain * np.outer(ar, at.conj())

        # ------------------------------------------------------------------
        # 4. Normalise so that E[||H||_F^2] = Nt * Nr  (standard convention)
        # ------------------------------------------------------------------
        # The inner double sum has L independent CN(0,1) terms.
        normalisation = math.sqrt(self.Nt * self.Nr / self.num_paths)
        H *= normalisation

        # ------------------------------------------------------------------
        # 5. Large-scale path loss
        # ------------------------------------------------------------------
        # Free-space PL ~ (4πd/λ)^2  =>  amplitude attenuation 1/d
        # We scale H by 1 / distance to keep the model simple.
        H /= math.sqrt(self.distance)

        # ------------------------------------------------------------------
        # 6. Doppler shift (if velocity is non-zero)
        # ------------------------------------------------------------------
        if abs(velocity_ms) > 1e-6:
            # Maximum Doppler shift fd = v / c * fc
            # We apply a common phase rotation for simplicity.
            fd = velocity_ms / 3e8 * self.mimo.carrier_freq
            # Phase term for a single coherence time (t=1 here for simplicity)
            doppler_phase = 2j * math.pi * fd
            H *= np.exp(doppler_phase)

        return H
