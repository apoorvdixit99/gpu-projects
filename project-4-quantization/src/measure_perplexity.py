"""Perplexity measurement on the fixed synthetic corpus.

Perplexity = exp(mean negative log-likelihood per token).
Lower is better — a lower value means the model assigns higher probability
to the corpus text.  The same 20-sentence corpus is used for all precision
levels so results are directly comparable.
"""

from __future__ import annotations

import math

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from corpus import SENTENCES


def measure_perplexity(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
) -> float:
    device = next(model.parameters()).device
    model.eval()

    total_nll    = 0.0
    total_tokens = 0

    with torch.no_grad():
        for sentence in SENTENCES:
            ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
            # model(labels=input_ids) returns mean NLL per token as .loss
            loss     = model(ids, labels=ids).loss
            n_tokens = ids.size(1)
            total_nll    += loss.item() * n_tokens
            total_tokens += n_tokens

    return math.exp(total_nll / total_tokens)
