import torch

from experiments.run_libero_task4_task7_public_state_jitter_augmentation_diagnostic import (
    PublicStateJitterDataset,
)


class _TinyDataset:
    def __init__(self):
        self.samples = [
            {
                "agentview_rgb": torch.zeros((2, 3, 4, 4), dtype=torch.uint8),
                "eye_in_hand_rgb": torch.ones((2, 3, 4, 4), dtype=torch.uint8),
                "joint_states": torch.zeros((2, 7), dtype=torch.float32),
                "gripper_states": torch.zeros((2, 2), dtype=torch.float32),
                "actions": torch.zeros((2, 7), dtype=torch.float32),
                "valid_mask": torch.ones((2,), dtype=torch.bool),
            }
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_public_state_jitter_is_deterministic_and_preserves_other_inputs():
    base = _TinyDataset()
    first = PublicStateJitterDataset(
        base, seed=123, joint_half_width=0.05, gripper_half_width=0.005
    )
    second = PublicStateJitterDataset(
        base, seed=123, joint_half_width=0.05, gripper_half_width=0.005
    )
    first_sample = first[0]
    second_sample = second[0]
    assert torch.equal(first_sample["joint_states"], second_sample["joint_states"])
    assert torch.equal(first_sample["gripper_states"], second_sample["gripper_states"])
    assert torch.equal(first_sample["agentview_rgb"], base.samples[0]["agentview_rgb"])
    assert torch.equal(first_sample["eye_in_hand_rgb"], base.samples[0]["eye_in_hand_rgb"])
    assert torch.equal(first_sample["actions"], base.samples[0]["actions"])
    assert torch.all(first_sample["joint_states"].abs() <= 0.05)
    assert torch.all(first_sample["gripper_states"].abs() <= 0.005)


def test_public_state_jitter_zero_width_is_an_exact_copy():
    base = _TinyDataset()
    dataset = PublicStateJitterDataset(
        base, seed=123, joint_half_width=0.0, gripper_half_width=0.0
    )
    sample = dataset[0]
    assert torch.equal(sample["joint_states"], base.samples[0]["joint_states"])
    assert torch.equal(sample["gripper_states"], base.samples[0]["gripper_states"])
