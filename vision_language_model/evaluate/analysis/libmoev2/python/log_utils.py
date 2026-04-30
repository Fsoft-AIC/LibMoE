"""
Utility helpers for parsing LibMoE analyst JSON logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence

import ujson as json


LayerLog = Mapping[str, Mapping[str, object]]
MetricFilter = Callable[[str], bool]


def load_json(path: Path | str) -> MutableMapping[str, object]:
    with open(Path(path), "r") as handle:
        return json.load(handle)


def iter_vision_layer_logs(data: Mapping[str, object]) -> Iterator[LayerLog]:
    logs: Sequence = data.get("logs", [])  # type: ignore[assignment]
    for sample in logs:
        sample_logs = sample.get("logs_metrics_vision", [[{}]])  # type: ignore[assignment]
        if not sample_logs or not sample_logs[0]:
            continue
        layer_logs = sample_logs[0][0]
        if isinstance(layer_logs, Mapping):
            yield layer_logs  # type: ignore[return-value]


def has_required_layers(layer_logs: LayerLog, expected: Iterable[str]) -> bool:
    return all(layer in layer_logs for layer in expected)


def collect_layer_metric(
    data: Mapping[str, object],
    metric_filter: MetricFilter,
    *,
    aggregation: Callable[[Iterable[float]], float] | None = None,
    expected_layers: Iterable[str] | None = None,
    skip_prefix: str = "time_inference",
) -> Dict[str, List[float]]:
    metrics: Dict[str, List[float]] = {}
    for layer_logs in iter_vision_layer_logs(data):
        if expected_layers and not has_required_layers(layer_logs, expected_layers):
            continue

        for layer_id, value in layer_logs.items():
            if skip_prefix and layer_id.startswith(skip_prefix):
                continue
            layer_bucket = metrics.setdefault(str(layer_id), [])
            if not isinstance(value, Mapping):
                continue
            for metric_name, metric_value in value.items():
                if not metric_filter(metric_name):
                    continue
                if isinstance(metric_value, (int, float)):
                    layer_bucket.append(float(metric_value))
                elif isinstance(metric_value, Sequence):
                    numbers = [float(item) for item in metric_value]  # type: ignore[arg-type]
                    if aggregation:
                        layer_bucket.append(float(aggregation(numbers)))
                    else:
                        layer_bucket.extend(numbers)
    return metrics


def collect_selected_experts(
    data: Mapping[str, object],
    *,
    key_substring: str = "selected_experts",
    expected_layers: Iterable[str] | None = None,
    skip_prefix: str = "time_inference",
    allowed_layers: Iterable[str] | None = None,
) -> Dict[str, List[List[int]]]:
    experts: Dict[str, List[List[int]]] = {}
    allowed = {str(layer) for layer in allowed_layers} if allowed_layers else None
    for layer_logs in iter_vision_layer_logs(data):
        if expected_layers and not has_required_layers(layer_logs, expected_layers):
            continue

        for layer_id, value in layer_logs.items():
            if skip_prefix and layer_id.startswith(skip_prefix):
                continue
            if allowed is not None and str(layer_id) not in allowed:
                continue
            if not isinstance(value, Mapping):
                continue
            for metric_name, metric_value in value.items():
                if key_substring not in metric_name:
                    continue
                experts.setdefault(str(layer_id), [])
                if isinstance(metric_value, Sequence):
                    experts[str(layer_id)].append(list(metric_value))  # type: ignore[arg-type]
    return experts


def extract_metrics_and_selected(
    data: Mapping[str, object],
    *,
    selected_key: str = "selected_experts",
    skip_prefix: str = "time_inference",
    allowed_layers: Iterable[str] | None = None,
    expected_layers: Iterable[str] | None = None,
) -> tuple[List[Dict[str, Dict[str, object]]], Dict[str, List[List[int]]]]:
    selected = collect_selected_experts(
        data,
        key_substring=selected_key,
        expected_layers=expected_layers,
        skip_prefix=skip_prefix,
        allowed_layers=allowed_layers,
    )
    samples: List[Dict[str, Dict[str, object]]] = []
    allowed = {str(layer) for layer in allowed_layers} if allowed_layers else None

    for layer_logs in iter_vision_layer_logs(data):
        if expected_layers and not has_required_layers(layer_logs, expected_layers):
            continue

        sample_payload: Dict[str, Dict[str, object]] = {}
        for layer_id, value in layer_logs.items():
            if skip_prefix and layer_id.startswith(skip_prefix):
                continue
            if allowed is not None and str(layer_id) not in allowed:
                continue
            if not isinstance(value, Mapping):
                continue
            layer_metrics: Dict[str, object] = {}
            for metric_name, metric_value in value.items():
                if selected_key in metric_name:
                    continue
                layer_metrics[metric_name] = metric_value
            if layer_metrics:
                sample_payload[str(layer_id)] = layer_metrics

        if sample_payload:
            samples.append(sample_payload)

    return samples, selected


def read_results_metadata(path: Path | str) -> Mapping[str, object]:
    return load_json(path)


def parse_model_and_checkpoint(metadata: Mapping[str, object]) -> tuple[str, str]:
    model_args = metadata.get("model_configs", {}).get("model_args", "")
    if not isinstance(model_args, str):
        return "unknown", "unknown"
    parts = model_args.split("/")
    if len(parts) < 2:
        return "unknown", "unknown"
    model_name = parts[-2]
    checkpoint = parts[-1].split(",")[0].split("-")[-1]
    return model_name, checkpoint
