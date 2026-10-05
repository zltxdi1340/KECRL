from experiments.train_controlled_torch_fomaml_formal_v1 import _review_spt


def test_formal_runner_config_is_explicitly_diagnostic():
    import json

    config = json.load(open("configs/controlled_torch_fomaml_formal_v1.yaml", encoding="utf-8"))
    assert config["formal_result"] is False
    assert config["qualification"]["min_samples"] == 20
    assert config["spt"]["validation_batches"] == 2
    assert config["metrics"]["support_curve_checkpoints"] == [0, 50, 100, 200]


def test_formal_runner_module_exposes_spt_review_entrypoint():
    assert callable(_review_spt)
