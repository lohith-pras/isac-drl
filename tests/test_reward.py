"""Verify the cosine-similarity sensing reward behaves correctly."""
import numpy as np
import pytest

from environment.channel_model import SVChannelModel
from environment.isac_env import ISACEnv
from environment.mimo_system import MIMOSystem
from environment.v2x_scenario import V2XScenario
from utils.reward_config import RewardConfig


@pytest.fixture
def env():
    mimo = MIMOSystem(Nt=4, Nr=4)
    scenario = V2XScenario()
    channel = SVChannelModel(mimo=mimo)
    reward_config = RewardConfig(alpha=1.0, beta=1.0)
    return ISACEnv(mimo=mimo, scenario=scenario, channel=channel, reward_config=reward_config)

def test_sensing_reward_aligned():
    """Beam steered at AoA should give reward ~1."""
    mimo = MIMOSystem(Nt=4, Nr=4)
    scenario = V2XScenario()
    channel = SVChannelModel(mimo=mimo)
    # Force alpha=0, beta=1 to isolate sensing reward
    reward_config = RewardConfig(alpha=0.0, beta=1.0)
    env = ISACEnv(mimo=mimo, scenario=scenario, channel=channel, reward_config=reward_config)

    # Reset to get state
    env.reset()

    # Get AoA from current state
    aoa = env._state["angle_of_arrival"]
    # Steering vector for this AoA
    a = env.mimo.tx_steering_vector(aoa)
    # Action = steering vector (flattened real+imag)
    action = np.concatenate([a.real, a.imag])

    _, reward, _, _, info = env.step(action)
    # Sensing reward should be ~1
    assert reward > 0.95, f"Aligned reward too low: {reward}. Info: {info}"

def test_sensing_reward_orthogonal():
    """Beam orthogonal to AoA should give reward ~0."""
    mimo = MIMOSystem(Nt=4, Nr=4)
    scenario = V2XScenario()
    channel = SVChannelModel(mimo=mimo)
    # Force alpha=0, beta=1 to isolate sensing reward
    reward_config = RewardConfig(alpha=0.0, beta=1.0)
    env = ISACEnv(mimo=mimo, scenario=scenario, channel=channel, reward_config=reward_config)

    env.reset()

    aoa = env._state["angle_of_arrival"]
    # Large offset to ensure orthogonality
    a_orth = env.mimo.tx_steering_vector(aoa + 60.0)
    action = np.concatenate([a_orth.real, a_orth.imag])

    _, reward, _, _, info = env.step(action)
    # Sensing reward should be low
    assert reward < 0.2, f"Orthogonal reward too high: {reward}. Info: {info}"
def test_sensing_reward_zero_action(env):
    """Zero action should give stable reward, no NaN."""
    env.reset()
    action = np.zeros(env.action_space.shape)
    _, reward, _, _, _ = env.step(action)
    assert not np.isnan(reward), "NaN from zero action"
    assert np.isfinite(reward), "Infinite reward from zero action"
