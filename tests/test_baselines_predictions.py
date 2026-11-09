import pytest

from ptbr_review_lora.baselines import make_tfidf_lr
from ptbr_review_lora.predictions import load_predictions, save_predictions


def test_tfidf_lr_learns_obvious_words():
    texts = ["ótimo produto", "ótimo, recomendo", "péssimo produto", "péssimo, não recomendo"] * 5
    labels = ["positivo", "positivo", "negativo", "negativo"] * 5
    model = make_tfidf_lr(class_weight="balanced").fit(texts, labels)
    assert list(model.predict(["ótimo ótimo", "péssimo péssimo"])) == ["positivo", "negativo"]


def test_predictions_round_trip(tmp_path):
    path = tmp_path / "dev" / "model.csv"
    save_predictions(path, ["neutro", "positivo", "negativo"])
    assert load_predictions(path, n_rows=3) == ["neutro", "positivo", "negativo"]


def test_load_predictions_rejects_wrong_number_of_rows(tmp_path):
    path = tmp_path / "model.csv"
    save_predictions(path, ["neutro", "positivo"])
    with pytest.raises(ValueError):
        load_predictions(path, n_rows=3)
