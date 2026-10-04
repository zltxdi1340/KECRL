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
    assert set(info) == {"inventory"}
    assert "semantic" not in info
    assert "player_pos" not in info
    assert environment.state()["inventory"] == info["inventory"]
    info["inventory"]["wood"] = -999
    assert environment.state()["inventory"]["wood"] != -999
    assert environment.state()["step_count"] == 1
    environment.reset()
    assert environment.state()["inventory"] is None
    environment.close()


def test_adapter_filters_unapproved_info_fields():
    import numpy as np
    from types import SimpleNamespace

    class Environment:
        observation_space = SimpleNamespace(shape=(64, 64, 3))
        action_space = SimpleNamespace(n=17)

        def step(self, action):
            return np.zeros((64, 64, 3), dtype=np.uint8), 0, False, {
                "semantic": "hidden map", "player_pos": (1, 2),
                "achievements": {"collect_wood": 1}, "unexpected": "hidden",
            }

    environment = CrafterEnvironmentAdapter(environment=Environment())
    assert environment.step(0)[3] == {}
    assert environment.state()["inventory"] is None


def test_crafter_rejects_invalid_actions():
    environment = CrafterEnvironmentAdapter(seed=0)
    with pytest.raises(ValueError):
        environment.step(17)
    with pytest.raises(TypeError):
        environment.step(True)
    environment.close()
