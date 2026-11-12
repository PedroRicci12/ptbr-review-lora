import pytest

from ptbr_review_lora import protocol


def test_bonferroni_confidence():
    assert protocol.bonferroni_confidence(1) == pytest.approx(0.95)
    assert protocol.bonferroni_confidence(3) == pytest.approx(1 - 0.05 / 3)
    with pytest.raises(ValueError):
        protocol.bonferroni_confidence(0)


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [(0.01, 0.05, "B better"), (-0.05, -0.01, "B worse"), (-0.01, 0.02, "inconclusive"), (0.0, 0.03, "inconclusive")],
)
def test_verdict(low, high, expected):
    assert protocol.verdict(low, high) == expected


def test_protocol_names_existing_models():
    for _, b, a, _ in protocol.PRIMARY:
        assert {a, b} <= protocol.MODELS.keys()
    for b, a in protocol.DESCRIPTIVE:
        assert {a, b} <= protocol.MODELS.keys()


def test_three_preregistered_primary_comparisons():
    assert [(cid, b, a) for cid, b, a, _ in protocol.PRIMARY] == [
        ("H1", "qlora_human", "zero_shot"),
        ("H2", "qlora_stars", "zero_shot"),
        ("H3", "qlora_human", "qlora_stars"),
    ]


def test_run_primary_on_toy_data():
    y = ["negativo", "neutro", "positivo"] * 20
    good = list(y)
    bad = ["positivo"] * len(y)
    preds = {"qlora_human": good, "zero_shot": bad, "qlora_stars": bad}
    out = protocol.run_primary(y, preds, n_resamples=200)
    assert out["H1"]["verdict"] == "B better"
    assert out["H2"]["verdict"] == "inconclusive"  # identical models: difference 0 in every resample
    assert out["H3"]["verdict"] == "B better"
    assert out["H1"]["confidence"] == pytest.approx(1 - 0.05 / 3)
