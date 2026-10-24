import json

import pytest
import torch

from experiments import run_libero_eye_hand_robustness_pilot as pilot
from experiments import run_libero_bc_causal_alignment_diagnostic as causal


def test_eye_hand_translation_is_deterministic_and_window_consistent():
    base = torch.arange(2 * 3 * 3 * 8 * 8, dtype=torch.int64).reshape(2, 3, 3, 8, 8)
    images = base.remainder(256).to(torch.uint8)
    images[:, 1] = images[:, 0]
    images[:, 2] = images[:, 0]

    first = causal._random_translate_eye_hand_sequences(
        images,
        max_translation_pixels=2,
        generator=torch.Generator().manual_seed(17),
    )
    second = causal._random_translate_eye_hand_sequences(
        images,
        max_translation_pixels=2,
        generator=torch.Generator().manual_seed(17),
    )

    assert torch.equal(first, second)
    assert torch.equal(first[:, 0], first[:, 1])
    assert torch.equal(first[:, 0], first[:, 2])
    assert first.dtype == torch.uint8


def test_eye_hand_translation_rejects_invalid_image_or_radius():
    with pytest.raises(ValueError, match=r"uint8 \[B, T, C, H, W\]"):
        causal._random_translate_eye_hand_sequences(
            torch.zeros(1, 3, 3, 8, 8),
            max_translation_pixels=2,
            generator=torch.Generator().manual_seed(1),
        )
    with pytest.raises(ValueError, match="too large"):
        causal._random_translate_eye_hand_sequences(
            torch.zeros(1, 1, 3, 8, 8, dtype=torch.uint8),
            max_translation_pixels=4,
            generator=torch.Generator().manual_seed(1),
        )


def test_eye_hand_pilot_config_is_fixed_and_nonformal():
    loaded = pilot.load_pilot_config(pilot.DEFAULT_CONFIG)
    assert loaded[0]["formal_result"] is False
    assert tuple(arm["name"] for arm in loaded[0]["arms"]) == pilot.PILOT_ARMS
    assert loaded[0]["training"]["optimizer_updates"] == 500


def test_eye_hand_pilot_config_rejects_changed_augmentation(tmp_path):
    config = json.loads(pilot.DEFAULT_CONFIG.read_text())
    config["eye_hand_augmentation"]["max_translation_pixels"] = 0
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="augmentation specification is invalid"):
        pilot.load_pilot_config(path)
