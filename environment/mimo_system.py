"""MIMO system model for mmWave ISAC beamforming."""

import numpy as np


class MIMOSystem:
    """
    MIMO system model for millimeter-wave (mmWave) ISAC scenarios.

    Parameters
    ----------
    Nt : int
        Number of transmit antennas.
    Nr : int
        Number of receive antennas.
    carrier_freq : float
        Carrier frequency in Hz (default is 28e9 for mmWave).
    c : float
        Speed of light in m/s (default is 3e8).
    """

    def __init__(
        self,
        Nt: int,
        Nr: int,
        *,
        carrier_freq: float = 28e9,
        c: float = 3e8,
    ):
        self.Nt = Nt
        self.Nr = Nr
        self.carrier_freq = carrier_freq
        self.c = c

        # Wavelength (m)
        self._wavelength = self.c / self.carrier_freq
        # Inter-antenna spacing for ULA: half-wavelength
        self._d = self._wavelength / 2.0

    # ------------------------------------------------------------------ #
    # Steering vector
    # ------------------------------------------------------------------ #
    def _steering_vector(self, theta: float, N: int) -> np.ndarray:
        """
        Compute the ULA steering vector.

        Parameters
        ----------
        theta : float
            Angle of arrival/departure in degrees.
        N : int
            Number of antenna elements.

        Returns
        -------
        np.ndarray
            Complex-valued steering vector of shape (N,).
        """
        theta_rad = np.deg2rad(theta)
        n = np.arange(N)
        phase = 2 * np.pi * self._d * np.sin(theta_rad) / self._wavelength
        return np.exp(1j * n * phase)

    def tx_steering_vector(self, theta: float) -> np.ndarray:
        """
        Compute the transmit ULA steering vector.

        Parameters
        ----------
        theta : float
            Angle of departure in degrees.

        Returns
        -------
        np.ndarray
            Complex-valued steering vector of shape (Nt,).
        """
        return self._steering_vector(theta, self.Nt)

    def rx_steering_vector(self, theta: float) -> np.ndarray:
        """
        Compute the receive ULA steering vector.

        Parameters
        ----------
        theta : float
            Angle of arrival in degrees.

        Returns
        -------
        np.ndarray
            Complex-valued steering vector of shape (Nr,).
        """
        return self._steering_vector(theta, self.Nr)

    # ------------------------------------------------------------------ #
    # Beamforming matrix
    # ------------------------------------------------------------------ #
    def init_beamformer(self, num_streams: int | None = None) -> np.ndarray:
        """
        Initialize a transmit beamforming matrix.

        Parameters
        ----------
        num_streams : int | None
            Number of data streams. If None, defaults to min(Nt, Nr).

        Returns
        -------
        np.ndarray
            Complex-valued beamforming matrix of shape (Nt, num_streams).
        """
        if num_streams is None:
            num_streams = min(self.Nt, self.Nr)

        # Random complex Gaussian initialization
        bf = (np.random.randn(self.Nt, num_streams)
              + 1j * np.random.randn(self.Nt, num_streams))
        # Normalize columns to unit norm
        bf = bf / np.linalg.norm(bf, axis=0, keepdims=True)
        return bf

    # ------------------------------------------------------------------ #
    # Array gain
    # ------------------------------------------------------------------ #
    def array_gain(
        self,
        w: np.ndarray,
        theta: float,
    ) -> float:
        """
        Compute the array gain for a given beamforming vector and angle.

        Parameters
        ----------
        w : np.ndarray
            Beamforming vector of shape (Nt,) or (Nt, 1).
        theta : float
            Angle of arrival/departure in degrees.

        Returns
        -------
        float
            Array gain (squared magnitude of the response).
        """
        w = w.flatten()
        a = self.tx_steering_vector(theta)
        gain = np.abs(np.vdot(a, w)) ** 2
        return float(gain)
