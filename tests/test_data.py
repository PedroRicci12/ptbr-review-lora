import pandas as pd
import pytest

from ptbr_review_lora.data import (
    PERSONAL_COLUMNS,
    git_blob_sha,
    polarity_group,
    prepare_b2w,
    prepare_repro,
    remove_overlap,
    remove_text_duplicates,
    split_repro,
    stars_to_label,
)


def make_review(i: int, rating: int = 5, text: str | None = None, **extra) -> dict:
    return {
        "reviewer_id": f"r{i}",
        "product_id": f"p{i}",
        "submission_date": "2018-01-01 00:00:00",
        "review_text": text if text is not None else f"texto {i}",
        "overall_rating": rating,
        "reviewer_birth_year": 1990,
        "reviewer_gender": "F",
        "reviewer_state": "SP",
        **extra,
    }


def test_git_blob_sha_matches_git():
    # Value from `git hash-object` on a file containing "hello\n".
    assert git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


@pytest.mark.parametrize(
    ("rating", "label"),
    [(1, "negativo"), (2, "negativo"), (3, "neutro"), (4, "positivo"), (5, "positivo")],
)
def test_stars_to_label(rating, label):
    assert stars_to_label(rating) == label


@pytest.mark.parametrize("rating", [0, 6])
def test_stars_to_label_rejects_out_of_range(rating):
    with pytest.raises(ValueError):
        stars_to_label(rating)


@pytest.mark.parametrize(
    ("polarity", "group"),
    [
        ("['NEGATIVO']", "negativo"),
        ("['POSITIVO']", "positivo"),
        ("['NEUTRO']", "neutro"),
        ("['NEGATIVO', 'POSITIVO']", "misto"),
    ],
)
def test_polarity_group(polarity, group):
    assert polarity_group(polarity) == group


def test_polarity_group_rejects_unknown_combination():
    with pytest.raises(ValueError):
        polarity_group("['NEUTRO', 'POSITIVO']")


def test_prepare_b2w_drops_personal_data_empty_text_and_duplicates():
    raw = pd.DataFrame(
        [make_review(1, rating=1), make_review(1, rating=1), make_review(2, text="  "), make_review(3, rating=3)]
    )
    df = prepare_b2w(raw)
    assert not set(PERSONAL_COLUMNS) & set(df.columns)
    assert list(df["reviewer_id"]) == ["r1", "r3"]
    assert list(df["label"]) == ["negativo", "neutro"]


def test_prepare_repro_maps_mixed_to_neutral():
    raw = pd.DataFrame(
        [make_review(1, polarity="['NEGATIVO', 'POSITIVO']"), make_review(2, polarity="['POSITIVO']")]
    )
    df = prepare_repro(raw)
    assert not set(PERSONAL_COLUMNS) & set(df.columns)
    assert list(df["polarity_group"]) == ["misto", "positivo"]
    assert list(df["label"]) == ["neutro", "positivo"]


def test_remove_overlap_removes_only_matching_reviews():
    b2w = pd.DataFrame([make_review(i) for i in range(5)])
    # Same reviewer and product as review 4, but different text: a different review.
    repro = pd.DataFrame([make_review(1), make_review(3), make_review(4, text="outro texto")])
    clean, removed = remove_overlap(b2w, repro)
    assert removed == 2
    assert list(clean["reviewer_id"]) == ["r0", "r2", "r4"]


def test_remove_text_duplicates_ignores_case_and_edge_spaces():
    b2w = pd.DataFrame(
        [
            make_review(1, text="Chegou com a caixa amassada"),
            make_review(2, text="  CHEGOU COM A CAIXA AMASSADA "),
            make_review(3, text="Chegou com a caixa intacta"),
        ]
    )
    repro = pd.DataFrame([make_review(9, text="chegou com a caixa amassada")])
    clean, removed = remove_text_duplicates(b2w, repro)
    assert removed == 2
    assert list(clean["reviewer_id"]) == ["r3"]


def make_repro(n_per_group: int) -> pd.DataFrame:
    groups = ["negativo", "positivo", "neutro", "misto"]
    rows = [{"id": f"{g}{i}", "polarity_group": g} for g in groups for i in range(n_per_group)]
    return pd.DataFrame(rows)


def test_split_repro_sizes_disjoint_and_stratified():
    repro = make_repro(100)
    dev, test = split_repro(repro, dev_size=80, seed=0)
    assert len(dev) == 80
    assert len(test) == 320
    assert not set(dev["id"]) & set(test["id"])
    assert dev["polarity_group"].value_counts().to_dict() == {g: 20 for g in ["negativo", "positivo", "neutro", "misto"]}


def test_split_repro_is_deterministic():
    repro = make_repro(50)
    dev_a, _ = split_repro(repro, dev_size=40, seed=7)
    dev_b, _ = split_repro(repro, dev_size=40, seed=7)
    assert list(dev_a["id"]) == list(dev_b["id"])
