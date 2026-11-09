import pytest

from ptbr_review_lora.metrics import (
    bootstrap_ci,
    compute_metrics,
    macro_f1,
    neutral_recall_by_group,
    paired_bootstrap,
)

LABELS = ["negativo", "neutro", "positivo"]


def noisy_predictions(n: int = 300) -> tuple[list[str], list[str]]:
    """Deterministic data: every third prediction is shifted to the next label (~67% correct)."""
    y_true = [LABELS[i % 3] for i in range(n)]
    y_pred = [LABELS[(i + (i % 9 == 0) + (i % 9 == 4) + (i % 9 == 8)) % 3] for i in range(n)]
    return y_true, y_pred


# --- metrics written for you ---


def test_macro_f1_counts_never_predicted_classes_as_zero():
    y_true = ["negativo", "neutro", "positivo", "positivo"]
    y_pred = ["positivo"] * 4
    # F1: negativo 0, neutro 0, positivo 2*0.5*1/(0.5+1) = 2/3.
    assert macro_f1(y_true, y_pred) == pytest.approx((0 + 0 + 2 / 3) / 3)


def test_compute_metrics_perfect_predictions():
    y = ["negativo", "neutro", "positivo"]
    m = compute_metrics(y, y)
    assert m["macro_f1"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_neutral_recall_by_group():
    groups = ["neutro", "neutro", "misto", "misto", "misto", "positivo"]
    y_pred = ["neutro", "positivo", "neutro", "negativo", "negativo", "neutro"]
    assert neutral_recall_by_group(groups, y_pred) == {"neutro": 0.5, "misto": pytest.approx(1 / 3)}


# --- exercise: bootstrap_ci ---


def test_bootstrap_ci_point_is_metric_on_full_data():
    y_true, y_pred = noisy_predictions()
    point, _, _ = bootstrap_ci(y_true, y_pred, n_resamples=200)
    assert point == pytest.approx(macro_f1(y_true, y_pred))


def test_bootstrap_ci_interval_contains_point():
    y_true, y_pred = noisy_predictions()
    point, low, high = bootstrap_ci(y_true, y_pred, n_resamples=500)
    assert low < point < high


def test_bootstrap_ci_perfect_predictions_have_no_uncertainty():
    y = ["negativo", "neutro", "positivo"] * 20
    assert bootstrap_ci(y, y, n_resamples=100) == (1.0, 1.0, 1.0)


def test_bootstrap_ci_is_deterministic_for_a_seed():
    y_true, y_pred = noisy_predictions()
    assert bootstrap_ci(y_true, y_pred, n_resamples=200, seed=1) == bootstrap_ci(
        y_true, y_pred, n_resamples=200, seed=1
    )


def test_bootstrap_ci_higher_confidence_gives_wider_interval():
    y_true, y_pred = noisy_predictions()
    _, low90, high90 = bootstrap_ci(y_true, y_pred, n_resamples=500, confidence=0.90)
    _, low99, high99 = bootstrap_ci(y_true, y_pred, n_resamples=500, confidence=0.99)
    assert low99 <= low90 and high90 <= high99


def test_bootstrap_ci_rejects_different_lengths():
    with pytest.raises(ValueError):
        bootstrap_ci(["neutro", "neutro"], ["neutro"])


# --- exercise: paired_bootstrap ---


def test_paired_bootstrap_identical_models_have_zero_difference():
    y_true, y_pred = noisy_predictions()
    result = paired_bootstrap(y_true, y_pred, y_pred, n_resamples=200)
    assert result == {"diff": 0.0, "low": 0.0, "high": 0.0, "p_b_not_better": 1.0}


def test_paired_bootstrap_clearly_better_model():
    y_true = LABELS * 50
    always_wrong = [LABELS[(LABELS.index(y) + 1) % 3] for y in y_true]
    result = paired_bootstrap(y_true, always_wrong, y_true, n_resamples=200)
    assert result["diff"] == pytest.approx(1.0)
    assert result["low"] > 0
    assert result["p_b_not_better"] == 0.0


def test_paired_bootstrap_swapping_models_negates_the_difference():
    y_true, y_pred = noisy_predictions()
    baseline = ["positivo"] * len(y_true)
    ab = paired_bootstrap(y_true, baseline, y_pred, n_resamples=300, seed=3)
    ba = paired_bootstrap(y_true, y_pred, baseline, n_resamples=300, seed=3)
    assert ba["diff"] == pytest.approx(-ab["diff"])
    assert ba["low"] == pytest.approx(-ab["high"])
    assert ba["high"] == pytest.approx(-ab["low"])


def test_paired_bootstrap_rejects_different_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap(["neutro", "neutro"], ["neutro", "neutro"], ["neutro"])
