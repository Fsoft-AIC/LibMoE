"""
Reusable helpers that drive the expert-selection analyses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from . import log_utils


SelectionStore = MutableMapping[str, MutableMapping[str, Dict[str, List[List[int]]]]]
SelectionScores = Dict[str, Dict[str, Dict[str, List[float]]]]


def collect_selected_experts(
    dataset: str,
    analyst_root: Path,
    *,
    selected_key: str,
    allowed_layers: Iterable[str] | None,
    expected_layers: Iterable[str] | None,
    file_suffix: str = "",
) -> SelectionStore:
    aggregated: SelectionStore = {}

    for run_dir in sorted(analyst_root.glob("*")):
        if not run_dir.is_dir():
            continue
        data_path = run_dir / f"{dataset}.json"
        if not data_path.exists():
            continue

        try:
            data = log_utils.load_json(data_path)
        except OSError:
            continue

        samples, selected = log_utils.extract_metrics_and_selected(
            data,
            selected_key=selected_key,
            allowed_layers=allowed_layers,
            expected_layers=expected_layers,
        )

        metadata_path = run_dir / "results.json"
        if not metadata_path.exists():
            continue
        metadata = log_utils.read_results_metadata(metadata_path)
        model_name, checkpoint = log_utils.parse_model_and_checkpoint(metadata)

        write_processed_samples(run_dir, dataset, samples, file_suffix=file_suffix)
        aggregated.setdefault(model_name, {})[checkpoint] = selected

    return aggregated


def compute_switch_scores(
    selected_data: Mapping[str, Mapping[str, Dict[str, List[List[int]]]]],
    *,
    checkpoints: Sequence[str],
    layers: Iterable[str],
    metric_fn: Callable[[Sequence, Sequence], float],
) -> SelectionScores:
    results: SelectionScores = {}
    for model_name, checkpoints_map in selected_data.items():
        model_scores: Dict[str, Dict[str, List[float]]] = {}
        for idx in range(1, len(checkpoints)):
            current_step = checkpoints[idx]
            prev_step = checkpoints[idx - 1]
            if current_step not in checkpoints_map or prev_step not in checkpoints_map:
                continue

            step_scores: Dict[str, List[float]] = {}
            for layer_id in layers:
                layer_current = checkpoints_map[current_step].get(str(layer_id))
                layer_prev = checkpoints_map[prev_step].get(str(layer_id))
                if not layer_current or not layer_prev:
                    continue
                score = metric_fn(layer_current, layer_prev)
                step_scores[str(layer_id)] = [score]
            if step_scores:
                model_scores[str(idx - 1)] = step_scores
        if model_scores:
            results[model_name] = model_scores
    return results


def analyse_datasets(
    datasets: Sequence[str],
    models: Sequence[str],
    *,
    analyst_root: Path,
    output_root: Path,
    checkpoints: Sequence[str],
    layers: Iterable[str],
    metric_fn: Callable[[Sequence, Sequence], float],
    selected_key: str,
    allowed_layers: Iterable[str] | None,
    expected_layers: Iterable[str] | None,
    file_suffix: str = "",
) -> None:
    for dataset in datasets:
        for model in models:
            model_root = analyst_root / model / "analysts"
            if not model_root.exists():
                continue

            aggregated = collect_selected_experts(
                dataset,
                model_root,
                selected_key=selected_key,
                allowed_layers=allowed_layers,
                expected_layers=expected_layers,
                file_suffix=file_suffix,
            )
            if not aggregated:
                continue

            output_dir = output_root / model / "analysts"
            output_dir.mkdir(parents=True, exist_ok=True)

            save_path = output_dir / f"{dataset}_data_selected_final{file_suffix}.json"
            write_json(save_path, aggregated)

            scores = compute_switch_scores(
                aggregated,
                checkpoints=checkpoints,
                layers=layers,
                metric_fn=metric_fn,
            )
            scores_path = output_dir / f"{dataset}_score_selected_final{file_suffix}.json"
            write_json(scores_path, scores)


def write_processed_samples(
    run_dir: Path,
    dataset: str,
    samples: List[Dict[str, Dict[str, object]]],
    *,
    file_suffix: str = "",
) -> None:
    if not samples:
        return
    output_path = run_dir / f"{dataset}_processed_data{file_suffix}.json"
    write_json(output_path, samples)


def write_json(path: Path, payload: object) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
