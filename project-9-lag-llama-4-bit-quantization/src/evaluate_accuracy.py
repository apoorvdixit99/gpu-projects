"""Zero-shot forecast accuracy: FP32 vs NF4, on GluonTS built-in datasets.

Lag-Llama is a probabilistic foundation model, so accuracy is measured the
way the paper's own zero-shot benchmarks are: sample `num_samples` forecast
trajectories per series and score the resulting empirical distribution with
GluonTS's `Evaluator` (CRPS via mean_wQuantileLoss, MASE, sMAPE, MSIS).
"""

from __future__ import annotations

from itertools import islice

import torch
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions

from load_model import build_predictor

METRIC_KEYS = {
    "MASE": "MASE",
    "sMAPE": "sMAPE",
    "mean_wQuantileLoss": "CRPS_approx",
    "MSIS": "MSIS",
}


def evaluate_dataset(
    dataset_name: str,
    precision: str,
    context_length: int = 32,
    num_samples: int = 100,
    max_series: int | None = None,
) -> dict:
    dataset = get_dataset(dataset_name)
    prediction_length = dataset.metadata.prediction_length

    test_data = dataset.test
    if max_series is not None:
        test_data = list(islice(test_data, max_series))

    predictor, _ = build_predictor(
        precision=precision,
        context_length=context_length,
        prediction_length=prediction_length,
        num_parallel_samples=num_samples,
    )

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data, predictor=predictor, num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    evaluator = Evaluator()
    agg_metrics, _ = evaluator(iter(tss), iter(forecasts))

    row = {
        "dataset": dataset_name,
        "backend": f"lagllama_{precision}",
        "num_series": len(forecasts),
    }
    for gluonts_key, label in METRIC_KEYS.items():
        row[label] = round(float(agg_metrics[gluonts_key]), 4)

    print(
        f"  [{precision}] {dataset_name:15s} | "
        f"MASE={row['MASE']:.3f}  sMAPE={row['sMAPE']:.3f}  "
        f"CRPS~{row['CRPS_approx']:.3f}"
    )

    del predictor
    torch.cuda.empty_cache()
    return row


def evaluate(
    dataset_names: list[str],
    precisions: list[str],
    context_length: int = 32,
    num_samples: int = 100,
    max_series: int | None = None,
) -> list[dict]:
    rows = []
    for dataset_name in dataset_names:
        for precision in precisions:
            rows.append(
                evaluate_dataset(
                    dataset_name, precision, context_length, num_samples, max_series
                )
            )
    return rows
