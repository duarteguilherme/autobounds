from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "balke_pearl_closed_form_ci_compare.py"
)
SPEC = importlib.util.spec_from_file_location("bp_compare", MODULE_PATH)
bp_compare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = bp_compare
SPEC.loader.exec_module(bp_compare)


def test_model_presets_have_valid_population_probabilities():
    for model in bp_compare.MODEL_PRESETS.values():
        probs = bp_compare.population_conditional_probs(model)
        assert probs.shape == (8,)
        assert np.all(probs >= 0.0)
        assert np.isclose(probs[:4].sum(), 1.0)
        assert np.isclose(probs[4:].sum(), 1.0)


def test_oracle_friendly_has_clear_active_gap():
    model = bp_compare.MODEL_PRESETS["oracle_friendly"]
    true_lb, oracle_idx, candidates = bp_compare.describe_model(model)
    ordered = np.sort(candidates)[::-1]
    gap = ordered[0] - ordered[1]
    assert oracle_idx == 2
    assert true_lb > -0.05
    assert gap > 0.4


def test_nearly_tied_model_has_small_active_gap():
    model = bp_compare.MODEL_PRESETS["nearly_tied"]
    true_lb, oracle_idx, candidates = bp_compare.describe_model(model)
    ordered = np.sort(candidates)[::-1]
    gap = ordered[0] - ordered[1]
    assert oracle_idx == 1
    assert true_lb < -0.3
    assert gap == 0.0


def test_subsample_rule_matches_requested_defaults():
    assert bp_compare.resolve_subsample_size(100, gamma=2.0 / 3.0, minimum=80) == 80
    assert bp_compare.resolve_subsample_size(2000, gamma=2.0 / 3.0, minimum=80) == 158
