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
    assert environment.state()["episode_done"] is False
    environment.reset()
    assert environment.state()["inventory"] is None
    environment.close()


def test_adapter_exposes_current_observation_for_persistent_execution():
    pytest.importorskip("crafter")
    environment = CrafterEnvironmentAdapter(seed=0, length=2)
    observation = environment.reset()
    assert environment.current_observation() is observation
    environment.step(0)
    assert environment.current_observation().shape == (64, 64, 3)
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


def test_adapter_confirms_world_setup_from_public_placement_delta():
    import numpy as np
    from types import SimpleNamespace

    class Environment:
        observation_space = SimpleNamespace(shape=(64, 64, 3))
        action_space = SimpleNamespace(n=17)
        action_names = ["noop"] * 17

        def __init__(self):
            self.action_names[8] = "place_table"
            self.inventory = {"wood": 2}
            self.steps = 0

        def step(self, action):
            self.steps += 1
            if self.steps == 1:
                assert action == 0
                self.inventory = {"wood": 2}
            else:
                assert action == 8
                self.inventory = {"wood": 0}
            return np.zeros((64, 64, 3), dtype=np.uint8), 0, False, {
                "inventory": dict(self.inventory), "semantic": "hidden"
            }

    environment = CrafterEnvironmentAdapter(environment=Environment())
    environment.step(0)
    _, _, _, info = environment.step(8)
    assert info["world_object_setup"] == ["table"]
    assert environment.state()["world_object_setup"] == ["table"]


def test_adapter_does_not_confirm_setup_without_matching_public_delta():
    import numpy as np
    from types import SimpleNamespace

    class Environment:
        observation_space = SimpleNamespace(shape=(64, 64, 3))
        action_space = SimpleNamespace(n=17)
        action_names = ["noop"] * 17

        def step(self, action):
            return np.zeros((64, 64, 3), dtype=np.uint8), 0, False, {"inventory": {"wood": 2}}

    environment = CrafterEnvironmentAdapter(environment=Environment())
    assert environment.step(8)[3] == {"inventory": {"wood": 2}}
    assert environment.state()["world_object_setup"] is None


def test_setup_boundary_diagnostic_is_non_formal():
    import json
    from pathlib import Path

    config = json.loads(Path("configs/crafter_setup_boundary_diagnostic_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["observation_interface"] == "rgb64_inventory_v1"


def test_crafter_rejects_invalid_actions():
    environment = CrafterEnvironmentAdapter(seed=0)
    with pytest.raises(ValueError):
        environment.step(17)
    with pytest.raises(TypeError):
        environment.step(True)
    environment.close()
