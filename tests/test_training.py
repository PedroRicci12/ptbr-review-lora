import pandas as pd
import pytest

from ptbr_review_lora.data import LABELS, split_dev
from ptbr_review_lora.training import IGNORE_INDEX, build_example, pad_batch, sample_balanced


def _pool(n_per_label=(50, 10, 80)):
    rows = [
        {"review_text": f"{label} {i}", "label": label}
        for label, n in zip(LABELS, n_per_label)
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def test_sample_balanced_counts_and_determinism():
    sample = sample_balanced(_pool(), per_class=8, seed=3)
    assert sample["label"].value_counts().to_dict() == {label: 8 for label in LABELS}
    assert sample.index.is_unique
    pd.testing.assert_frame_equal(sample, sample_balanced(_pool(), per_class=8, seed=3))


def test_sample_balanced_is_shuffled_not_grouped_by_label():
    labels = sample_balanced(_pool(), per_class=8, seed=3)["label"].tolist()
    assert labels != sorted(labels, key=LABELS.index)


def test_sample_balanced_fails_when_a_class_is_too_small():
    with pytest.raises(ValueError, match="neutro"):
        sample_balanced(_pool(), per_class=20)


def test_build_example_masks_the_prompt():
    example = build_example([1, 2, 3], [7, 8])
    assert example["input_ids"] == [1, 2, 3, 7, 8]
    assert example["labels"] == [IGNORE_INDEX] * 3 + [7, 8]


def test_build_example_requires_an_answer():
    with pytest.raises(ValueError):
        build_example([1, 2], [])


def test_pad_batch_pads_right_and_ignores_padding():
    batch = pad_batch([build_example([1, 2], [9]), build_example([1], [9])], pad_id=0)
    assert batch["input_ids"] == [[1, 2, 9], [1, 9, 0]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"] == [[IGNORE_INDEX, IGNORE_INDEX, 9], [IGNORE_INDEX, 9, IGNORE_INDEX]]


def _dev():
    groups = ["negativo"] * 40 + ["neutro"] * 8 + ["misto"] * 20 + ["positivo"] * 32
    return pd.DataFrame({"polarity_group": groups, "review_text": [f"r{i}" for i in range(100)]})


def test_split_dev_sizes_disjoint_and_index_kept():
    dev = _dev()
    train, select = split_dev(dev, train_size=75, seed=42)
    assert len(train) == 75 and len(select) == 25
    assert set(train.index).isdisjoint(select.index)
    assert sorted(train.index.tolist() + select.index.tolist()) == list(range(100))
    assert (select["review_text"] == dev.loc[select.index, "review_text"]).all()


def test_split_dev_is_stratified():
    train, select = split_dev(_dev(), train_size=75, seed=42)
    assert select["polarity_group"].value_counts().to_dict() == {
        "negativo": 10, "positivo": 8, "misto": 5, "neutro": 2,
    }
