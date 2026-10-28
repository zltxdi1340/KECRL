import json

import numpy as np
import pytest
import torch

from experiments import run_libero_runtime_native_prefix_cross_task_eye_hand_translation_sensitivity_diagnostic as diagnostic


def test_config_pins_translation_schedule():
    config, *_ = diagnostic._load_protocol(
        diagnostic.DEFAULT_CONFIG,
        diagnostic.DEFAULT_CAMERA_RESULT,
        diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
        diagnostic.DEFAULT_PREFIX_RESULT,
    )
    assert config["task_ids"] == [1, 3, 4, 7]
    assert config["perturbation"]["radii_pixels"] == [1, 2, 4]
    assert config["perturbation"]["directions"] == ["left", "right", "up", "down"]


def test_config_rejects_changed_translation_radius(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["perturbation"]["radii_pixels"] = [2]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="perturbation"):
        diagnostic._load_protocol(
            path,
            diagnostic.DEFAULT_CAMERA_RESULT,
            diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
            diagnostic.DEFAULT_PREFIX_RESULT,
        )


def test_direction_offset_and_distance_are_explicit():
    assert diagnostic._direction_offset("left", 2) == (-2, 0)
    assert diagnostic._direction_offset("down", 4) == (0, 4)
    with pytest.raises(ValueError):
        diagnostic._direction_offset("diagonal", 2)
    row = diagnostic._distance(np.asarray([1, -2, 0, 0, 0, 0, 0.5]), 12.0)
    assert row["action_mae"] == pytest.approx(3.5 / 7.0)
    assert row["image_mae"] == 12.0


def test_translation_utility_preserves_image_contract():
    image = torch.zeros((1, 3, 8, 8), dtype=torch.uint8)
    image[:, :, 2:4, 2:4] = 255
    shifted = diagnostic.synthetic._translate_eye_hand_image(image, dx_pixels=1, dy_pixels=-1)
    assert shifted.shape == image.shape
    assert shifted.dtype == torch.uint8
    assert not torch.equal(shifted, image)
