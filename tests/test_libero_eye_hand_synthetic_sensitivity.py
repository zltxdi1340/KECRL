import json

import pytest
import torch

from experiments import run_libero_eye_hand_synthetic_sensitivity as diagnostic


def test_fixed_translation_preserves_shape_dtype_and_is_reproducible():
    image = torch.arange(3 * 8 * 8, dtype=torch.int64).reshape(1, 3, 8, 8).remainder(256).to(torch.uint8)
    first = diagnostic._translate_eye_hand_image(image, dx_pixels=2, dy_pixels=-1)
    second = diagnostic._translate_eye_hand_image(image, dx_pixels=2, dy_pixels=-1)
    assert first.shape == image.shape
    assert first.dtype == torch.uint8
    assert torch.equal(first, second)
    assert not torch.equal(first, image)


def test_fixed_translation_rejects_large_offsets():
    image = torch.zeros(1, 3, 8, 8, dtype=torch.uint8)
    with pytest.raises(ValueError, match="too large"):
        diagnostic._translate_eye_hand_image(image, dx_pixels=4, dy_pixels=0)


def test_sensitivity_config_pins_arms_seeds_and_hashes():
    loaded = diagnostic.load_sensitivity_config(diagnostic.DEFAULT_CONFIG)
    assert loaded[0]["formal_result"] is False
    assert tuple(loaded[0]["arms"]) == diagnostic.ARMS
    assert loaded[0]["perturbation"]["radii_pixels"] == [1, 2, 4]


def test_sensitivity_config_rejects_changed_direction(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["perturbation"]["directions"] = ["left"]
    path = tmp_path / "sensitivity.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="directions are invalid"):
        diagnostic.load_sensitivity_config(path)


def test_stats_include_gripper_and_translation_components():
    result = diagnostic._stats(
        [
            torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.float32).numpy(),
            torch.tensor([-1, 0, 0, 0, 0, 0, 0], dtype=torch.float32).numpy(),
        ],
        [2.0, 2.0],
        [
            torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32).numpy(),
            torch.tensor([0, 0, 0, 0, 0, 0, -1], dtype=torch.float32).numpy(),
        ],
    )
    assert result["sample_count"] == 2
    assert result["translation_mae"] == pytest.approx(1.0 / 3.0)
    assert result["gripper_mae"] == 0.0
