import pytest

pytest.importorskip("crafter")

from src.environments.crafter_adapter import CrafterEnvironmentAdapter


def test_crafter_reset_step_and_contract_metadata():
    environment = CrafterEnvironmentAdapter(seed=0)
    observation = environment.reset()
    assert observation.shape == (64, 64, 3)
    assert observation.dtype.name == "uint8"
    next_observation, reward, done, info = environment.step(0)
    assert next_observation.shape == (64, 64, 3)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert environment.state()["step_count"] == 1
    environment.close()


def test_crafter_rejects_invalid_actions():
    environment = CrafterEnvironmentAdapter(seed=0)
    with pytest.raises(ValueError):
        environment.step(17)
    with pytest.raises(TypeError):
        environment.step(True)
    environment.close()
