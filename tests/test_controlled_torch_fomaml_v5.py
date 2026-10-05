import torch

from experiments.train_controlled_torch_fomaml_v5 import _episode_seed, _merge_batches
from src.skills.torch_fomaml import PolicyEpisodeBatch


def _batch(value):
    return PolicyEpisodeBatch(
        observations=torch.tensor([[value, 0.0]], dtype=torch.float32),
        actions=torch.tensor([0], dtype=torch.long),
        rewards=(1.0,),
        legal_actions=(0, 1),
    )


def test_v5_episode_seed_is_stable_and_role_specific():
    first = _episode_seed(0, "train_support", "seed:0:train:0", "0")
    assert first == _episode_seed(0, "train_support", "seed:0:train:0", "0")
    assert first != _episode_seed(0, "train_query", "seed:0:train:0", "0")


def test_v5_merge_batches_preserves_episode_order_and_alignment():
    merged = _merge_batches([_batch(1.0), _batch(2.0)])
    assert merged.observations.shape == (2, 2)
    assert tuple(merged.observations[:, 0].tolist()) == (1.0, 2.0)
    assert len(merged.rewards) == len(merged.actions) == 2
