import json

import numpy as np
import pytest

from experiments.audit_libero_gripper_action_mapping import ROOT, formatted, isolated_formatter, run


def formatter():
    config=json.loads((ROOT/'configs/libero-gripper-action-mapping-readonly-audit_v1.json').read_text())
    from pathlib import Path
    return isolated_formatter(Path(config['inputs']['formatter']['path']))


@pytest.mark.parametrize('previous',[(0.,0.),(-1.,1.),(1.,-1.)])
def test_actual_formatter_discards_same_sign_amplitude_at_same_history(previous):
    cls=formatter();p=np.array(previous)
    np.testing.assert_array_equal(formatted(cls,.9,p),formatted(cls,.1,p))
    np.testing.assert_array_equal(formatted(cls,-.9,p),formatted(cls,-.1,p))
    np.testing.assert_array_equal(formatted(cls,0.,p),p)


def test_same_sign_does_not_erase_different_accumulator_histories():
    cls=formatter()
    assert not np.array_equal(formatted(cls,.9,np.zeros(2)),formatted(cls,.9,np.array([-.5,.5])))


def test_mapping_audit_rejects_pin_drift_and_preserves_existing_output(tmp_path):
    path=tmp_path/'config.json'
    original=ROOT/'configs/libero-gripper-action-mapping-readonly-audit_v1.json'
    c=json.loads(original.read_text());c['inputs']['formatter']['sha256']='0'*64
    path.write_text(json.dumps(c));output=tmp_path/'result.json'
    with pytest.raises(ValueError,match='pinned input'):run(path,output)
    assert not output.exists()
    output.write_text('retained')
    with pytest.raises(FileExistsError):run(original,output)
    assert output.read_text()=='retained'
