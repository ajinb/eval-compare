import json

import pytest

import eval_compare as ec


def test_mcnemar_matches_hand_computed_binomial():
    # 1 vs 8 disagreements: 2 * (C(9,0) + C(9,1)) / 2^9 = 20/512
    assert ec.mcnemar_exact(1, 8) == pytest.approx(20 / 512)


def test_mcnemar_is_symmetric_and_capped():
    assert ec.mcnemar_exact(3, 7) == ec.mcnemar_exact(7, 3)
    assert ec.mcnemar_exact(5, 5) == 1.0
    assert ec.mcnemar_exact(0, 0) == 1.0


def test_identical_runs_show_no_difference():
    run = {str(i): i % 2 == 0 for i in range(40)}
    result = ec.compare(run, dict(run))
    assert result["diff"] == 0
    assert result["verdict"] == "no_detectable_difference"


def test_clear_regression_is_detected():
    a = {str(i): True for i in range(30)}
    b = {str(i): i >= 12 for i in range(30)}
    result = ec.compare(a, b)
    assert result["verdict"] == "b_worse"
    assert result["ci_high"] < 0


def test_mismatched_cases_are_rejected():
    with pytest.raises(ValueError, match="different cases"):
        ec.compare({"a": True}, {"b": True})


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text('{"id":"a","pass":true}\n{"id":"a","pass":false}\n')
    with pytest.raises(ValueError, match="duplicate"):
        ec.load(str(path))


def test_required_cases_grows_as_effect_shrinks():
    assert ec.required_cases(0.02, 0.2) > 4 * ec.required_cases(0.05, 0.2) * 0.9


def test_required_cases_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        ec.required_cases(0.3, 0.2)


def test_cli_gate_exits_nonzero_only_on_regression(tmp_path):
    good, bad = tmp_path / "good.jsonl", tmp_path / "bad.jsonl"
    good.write_text("".join(json.dumps({"id": str(i), "pass": True}) + "\n" for i in range(30)))
    bad.write_text("".join(json.dumps({"id": str(i), "pass": i >= 12}) + "\n" for i in range(30)))
    assert ec.main([str(good), str(bad), "--fail-on-regression"]) == 1
    assert ec.main([str(bad), str(good), "--fail-on-regression"]) == 0
    assert ec.main([str(good), str(bad)]) == 0
