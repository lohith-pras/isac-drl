"""Verify channel physics are correct."""
import numpy as np

from environment.channel_model import SVChannelModel
from environment.mimo_system import MIMOSystem


def test_channel_distance_tracking():
    """Different distances produce different pathloss."""
    mimo = MIMOSystem(Nt=4, Nr=4)
    ch = SVChannelModel(mimo=mimo, num_clusters=3)
    H_near = ch.generate(distance=10.0)
    H_far = ch.generate(distance=1000.0)
    power_near = np.linalg.norm(H_near)**2
    power_far = np.linalg.norm(H_far)**2
    msg = f"Near ({power_near}) should have higher power than far ({power_far})"
    assert power_near > power_far, msg

def test_channel_temporal_coherence():
    """Coherent mode should evolve smoothly."""
    mimo = MIMOSystem(Nt=4, Nr=4)
    # Using a very small velocity and update_interval to ensure slow evolution
    ch = SVChannelModel(mimo=mimo, num_clusters=3, coherent=True, update_interval=1e-5)
    ch.reset_coherent()
    H_1 = ch.generate(velocity_ms=1.0)
    H_2 = ch.generate(velocity_ms=1.0)

    # Correlation should be very high for small time step
    corr = np.abs(np.sum(H_1.conj() * H_2)) / (np.linalg.norm(H_1) * np.linalg.norm(H_2))
    assert corr > 0.99, f"Consecutive coherent channels too decorrelated: {corr}"


    # After many steps, it should decorrelate
    for _ in range(1000):
        H_3 = ch.generate(velocity_ms=10.0)

    corr_long = np.abs(np.sum(H_1.conj() * H_3)) / (np.linalg.norm(H_1) * np.linalg.norm(H_3))
    msg = f"Long-term correlation ({corr_long}) should be lower than short-term ({corr})"
    assert corr_long < corr, msg
