import importlib.util
from pathlib import Path

import pytest
import torch


_PATH = Path(__file__).resolve().parents[1] / "scripts/check_libero_cuda_training.py"
_SPEC = importlib.util.spec_from_file_location("check_libero_cuda_training", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_SMOKE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SMOKE)


def _split():
    return {
        "formal_result": False,
        "task_roles": {
            "source_spi": [0],
            "spt_validation_spi": [1],
            "heldout_query_spi": [2],
        },
        "demonstration_indices_by_task_role": {
            "source_spi": {"meta_support": [0, 1], "meta_query": [2]},
        },
        "evaluation_initial_state_indices_by_task_role": {
            "source_spi": {"candidate_nonregression": [0], "qualification": [1]},
        },
    }


def test_scope_accepts_only_source_development_data():
    _SMOKE.validate_smoke_scope(
        _split(), task_id=0, demonstration_indices=[0, 1], initial_state_index=0
    )


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"task_id": 2, "demonstration_indices": [0], "initial_state_index": 0}, "source SPI"),
        ({"task_id": 0, "demonstration_indices": [2], "initial_state_index": 0}, "meta_support"),
        ({"task_id": 0, "demonstration_indices": [0], "initial_state_index": 1}, "candidate_nonregression"),
    ],
)
def test_scope_rejects_nondevelopment_roles(kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        _SMOKE.validate_smoke_scope(_split(), **kwargs)


def test_tiny_smoke_policy_has_bounded_seven_dimensional_output():
    model = _SMOKE.TinyLiberoSmokePolicy()
    output = model(
        torch.zeros(2, 3, 64, 64),
        torch.zeros(2, 3, 64, 64),
        torch.zeros(2, 7),
        torch.zeros(2, 2),
    )
    assert output.shape == (2, 7)
    assert torch.all(output >= -1)
    assert torch.all(output <= 1)
