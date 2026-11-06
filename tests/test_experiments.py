"""消融清单与预算约束。"""

import pytest

from ttt.experiments import batch1, moe_ablations


def test_twelve_arms_require_measured_flops():
    arms, pending = batch1("125m", tokens=1_000_000)
    assert len(arms) == 10 and len(pending) == 2
    for full in (2, 4):
        assert arms[f"C{full}"].model == arms[f"F{full}"].model
        assert not arms[f"C{full}"].inner.enabled and arms[f"F{full}"].inner.enabled
    records = {
        f"125m/{arm}{full}": {"method": "profiler", "training_flops_per_token": 2 if arm == "F" else 1}
        for full in (2, 4)
        for arm in ("C", "F")
    }
    complete, pending = batch1("125m", tokens=1_000_000, measurements=records)
    assert len(complete) == 12 and not pending
    assert complete["C2_isoflop"].training.steps == 2 * complete["F2"].training.steps
    records["125m/C2"]["method"] = "wall_time"
    with pytest.raises(ValueError, match="method"):
        batch1("125m", measurements=records)


def test_moe_loops_stay_outside_adaptive_suffix():
    arms = moe_ablations()
    assert len(arms) == 7
    assert arms["route_e2e"].model.adaptation_target == "router"
    for name, experiment in arms.items():
        if name.startswith("loop_") and experiment.model.loop_mode != "none":
            assert experiment.model.loop_end <= experiment.model.prefix_length
    early = arms["loop_early_ffn"].model
    assert early.loop_end <= early.first_k_dense_layers
