"""Model / tokenizer construction shared by all three training modes."""

from __future__ import annotations

from transformers import BertForSequenceClassification, BertTokenizerFast

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2  # SST-2: negative / positive


def build_tokenizer() -> BertTokenizerFast:
    return BertTokenizerFast.from_pretrained(MODEL_NAME)


def build_model() -> BertForSequenceClassification:
    return BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
