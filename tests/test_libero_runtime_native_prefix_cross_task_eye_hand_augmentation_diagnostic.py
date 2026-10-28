import json

import pytest

from experiments import run_libero_runtime_native_prefix_cross_task_eye_hand_augmentation_diagnostic as diagnostic


def test_config_pins_paired_augmentation_boundary():
    config, *_ = diagnostic._load_protocol(
        diagnostic.DEFAULT_CONFIG,
        diagnostic.DEFAULT_SENSITIVITY_RESULT,
        diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
        diagnostic.DEFAULT_PREFIX_RESULT,
    )
    assert config["augmentation_arms"] == ["baseline", "eye_hand_translation"]
    assert config["augmentation"]["max_translation_pixels"] == 4
    assert config["evaluation"]["offline_images_unaugmented"] is True


def test_config_rejects_evaluation_augmentation(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["augmentation"]["evaluation_augmentation_enabled"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="augmentation boundary"):
        diagnostic._load_protocol(
            path,
            diagnostic.DEFAULT_SENSITIVITY_RESULT,
            diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
            diagnostic.DEFAULT_PREFIX_RESULT,
        )


def test_config_rejects_checkpoint_selection(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["training"]["checkpoint_selection"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="training boundary"):
        diagnostic._load_protocol(
            path,
            diagnostic.DEFAULT_SENSITIVITY_RESULT,
            diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
            diagnostic.DEFAULT_PREFIX_RESULT,
        )
