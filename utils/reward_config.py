"""Reward configuration dataclass for ISAC-MIMO-DRL."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """
    Configuration for the ISAC reward function.

    Attributes
    ----------
    alpha : float
        Weight for the communication rate term (default 0.5).
    beta : float
        Weight for the sensing gain term (default 0.5).
    noise_power_dbm : float
        Noise power in dBm (default -90).
    max_tx_power_dbm : float
        Maximum allowable transmit power in dBm (default 30).
    """

    alpha: float = 0.5
    beta: float = 0.5
    noise_power_dbm: float = -90.0
    max_tx_power_dbm: float = 30.0

    def dbm_to_linear(self, dbm: float) -> float:
        """
        Convert a dBm value to linear scale (Watts).

        Parameters
        ----------
        dbm : float
            Power in dBm.

        Returns
        -------
        float
            Power in linear (Watts).
        """
        return 10.0 ** (dbm / 10.0)
