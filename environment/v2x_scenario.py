"""
Basic V2X scenario with a fixed base station and a moving vehicle.

State
-----
distance        : float  (meters, Euclidean distance from BS)
velocity        : float  (m/s, scalar along the line)
angle_of_arrival: float  (degrees, measured from +x axis to vehicle)
"""

import math

import numpy as np


class V2XScenario:
    """
    One-dimensional V2X motion model.

    A single vehicle moves along a straight road; a base station (BS) is
    placed at a known lateral offset from the road.  At each timestep the
    vehicle's position is updated, and the LoS distance / AoA are
    recomputed.

    Parameters
    ----------
    road_length : float, optional
        Length of the road segment in meters (default 1000.0).
    road_offset : float, optional
        Perpendicular distance of the BS from the road center in meters
        (default 50.0).
    initial_position : float, optional
        Vehicle starting x-coordinate in meters (default 0.0).
    speed_kmh : float, optional
        Vehicle speed in km/h (default 60.0).
    update_interval : float, optional
        Time between consecutive updates in seconds (default 0.1).
    """

    def __init__(
        self,
        *,
        road_length: float = 1000.0,
        road_offset: float = 50.0,
        initial_position: float = 0.0,
        speed_kmh: float = 60.0,
        update_interval: float = 0.1,
    ):
        self.road_length = road_length
        self.road_offset = road_offset
        self.update_interval = update_interval

        # Base-station is fixed at (0.0, road_offset) so that the road
        # runs along y = 0 and the BS looks down at it.
        self._bs_position = np.array([0.0, self.road_offset], dtype=float)

        # Vehicle state
        self._position = float(initial_position)
        self._velocity = (speed_kmh * 1000.0) / 3600.0  # m/s

        self._t = 0.0  # simulation time (s)

    # ------------------------------------------------------------------ #
    # Internal geometry helpers
    # ------------------------------------------------------------------ #

    def _vehicle_position(self) -> np.ndarray:
        """
        Return the current 2-D Cartesian position of the vehicle.

        Returns
        -------
        np.ndarray
            Array of shape (2,) with [x, y] coordinates.
        """
        return np.array([self._position, 0.0], dtype=float)

    def _recompute_geometry(self) -> tuple[float, float]:
        """
        Compute Euclidean distance and AoA from BS to vehicle.

        Returns
        -------
        tuple[float, float]
            (distance, angle_of_arrival) where distance is in meters and
            angle_of_arrival is in degrees measured from the +x axis to the
            line-of-sight vector.
        """
        veh = self._vehicle_position()
        diff = veh - self._bs_position
        distance = float(np.linalg.norm(diff))
        angle = float(np.degrees(np.arctan2(diff[1], diff[0])))
        return distance, angle

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def current_state(self) -> dict[str, float]:
        """
        Return the current V2X state.

        Returns
        -------
        dict[str, float]
            Dictionary with keys:
            - "distance"         : LoS distance in meters.
            - "velocity"         : Scalar velocity in m/s.
            - "angle_of_arrival" : AoA in degrees.
        """
        distance, aoa = self._recompute_geometry()
        return {
            "distance": distance,
            "velocity": self._velocity,
            "angle_of_arrival": aoa,
        }

    def step(self) -> dict[str, float]:
        """
        Advance the simulation by one time-step.

        The vehicle moves forward along the road by
        ``velocity * update_interval``.  If the vehicle reaches the road
        end it stops.

        Returns
        -------
        dict[str, float]
            The updated state after moving (same schema as
            :attr:`current_state`).
        """
        # Update position
        delta = self._velocity * self.update_interval
        self._position = min(self._position + delta, self.road_length)
        self._t += self.update_interval
        return self.current_state

    def reset(self) -> dict[str, float]:
        """
        Reset the vehicle to the initial position and time to zero.

        Returns
        -------
        dict[str, float]
            The initial state.
        """
        self._position = 0.0
        self._t = 0.0
        return self.current_state
