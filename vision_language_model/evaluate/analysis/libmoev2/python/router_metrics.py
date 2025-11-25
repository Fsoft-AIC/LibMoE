"""
Aggregation helpers for router entropy / margin metrics stored in analyst logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping

import numpy as np

from . import log_utils


def aggregate_metric(
    runs_root: Path,
    dataset: str,
    *,
    metric_filter: Callable[[str], bool],
    expected_layers: Iterable[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate a metric across all runs contained in ``runs_root`` for a given dataset.
    The returned mapping is ``model_name -> {layer_id: average_value}``.
    """
    results: Dict[str, Dict[str, float]] = {}

    for run_dir in sorted(runs_root.glob("*")):
        if not run_dir.is_dir():
            continue

        data_path = run_dir / f"{dataset}.json"
        if not data_path.exists():
            continue

        try:
            data = log_utils.load_json(data_path)
        except OSError:
            continue

        layer_values = log_utils.collect_layer_metric(
            data,
            metric_filter=metric_filter,
            aggregation=np.mean,
            expected_layers=expected_layers,
        )
        averaged = {layer: float(np.mean(values)) for layer, values in layer_values.items() if values}
        if not averaged:
            continue

        metadata_path = run_dir / "results.json"
        if not metadata_path.exists():
            continue
        metadata = log_utils.read_results_metadata(metadata_path)
        model_name, _ = log_utils.parse_model_and_checkpoint(metadata)
        results[model_name] = averaged

    return results


def aggregate_entropy(runs_root: Path, dataset: str, *, expected_layers: Iterable[str] | None = None) -> Dict[str, Dict[str, float]]:
    return aggregate_metric(
        runs_root,
        dataset,
        metric_filter=lambda name: "entropy_weight_topk" in name,
        expected_layers=expected_layers,
    )


def aggregate_margin(runs_root: Path, dataset: str, *, expected_layers: Iterable[str] | None = None) -> Dict[str, Dict[str, float]]:
    return aggregate_metric(
        runs_root,
        dataset,
        metric_filter=lambda name: "router_magine" in name,
        expected_layers=expected_layers,
    )
