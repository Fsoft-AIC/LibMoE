"""
Utilities for measuring expert selection saturation relative to the final checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

from . import constants, selection_metrics
from .selection_pipeline import collect_selected_experts, write_json


EXPECTED_VISION_LAYERS = tuple(str(idx) for idx in range(10))
ALL_LAYER_IDS = tuple(str(idx) for idx in range(27))
CHECKPOINT_SUFFIXES = tuple(cp.split("-")[-1] for cp in constants.CHECKPOINTS_665K)


def compute_saturation_scores(
    selected_data: Mapping[str, Mapping[str, Dict[str, list]]],
    *,
    checkpoints: Sequence[str],
    layers: Iterable[str],
) -> Dict[str, Dict[str, Dict[str, list]]]:
    results: Dict[str, Dict[str, Dict[str, list]]] = {}
    final_checkpoint = checkpoints[-1]

    for model_name, checkpoint_map in selected_data.items():
        if final_checkpoint not in checkpoint_map:
            continue
        final_layers = checkpoint_map[final_checkpoint]
        model_scores: Dict[str, Dict[str, list]] = {}

        for checkpoint in checkpoints[:-1]:
            if checkpoint not in checkpoint_map:
                continue
            step_scores: Dict[str, list] = {}
            for layer_id in layers:
                current = checkpoint_map[checkpoint].get(str(layer_id))
                reference = final_layers.get(str(layer_id))
                if not current or not reference:
                    continue
                score = selection_metrics.average_position_match(current, reference)
                step_scores[str(layer_id)] = [score]
            if step_scores:
                model_scores[str(int(checkpoint))] = step_scores

        if model_scores:
            results[model_name] = model_scores

    return results


def analyse_saturation(
    datasets: Sequence[str],
    models: Sequence[str],
    *,
    analyst_root: Path,
    output_root: Path,
    checkpoints: Sequence[str],
    layers: Iterable[str],
    selected_key: str,
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
                allowed_layers=layers,
                expected_layers=EXPECTED_VISION_LAYERS,
            )
            if not aggregated:
                continue

            output_dir = output_root / model / "analysts"
            output_dir.mkdir(parents=True, exist_ok=True)

            save_path = output_dir / f"{dataset}_data_selected_final.json"
            write_json(save_path, aggregated)

            saturation_scores = compute_saturation_scores(
                aggregated,
                checkpoints=checkpoints,
                layers=layers,
            )
            saturation_path = output_dir / f"{dataset}_saturation_order_final.json"
            write_json(saturation_path, saturation_scores)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse expert selection saturation.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mmmu_val", "mmstar"],
        help="Datasets to process.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "Full_smoe_sigmoidgating",
            "Full_smoe_share",
            "Full_smoe_plus_plus",
            "Full_smoe",
            "Full_smoe_tcmoe",
            "Full_xmoe",
            "Full_smoe_sharev3",
        ],
        help="Model directories to process.",
    )
    parser.add_argument(
        "--analyst-root",
        type=Path,
        default=constants.ANALYST_ROOT_665K,
        help="Root directory containing analyst runs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=constants.ANALYST_ROOT_665K,
        help="Directory to store aggregated summaries.",
    )
    parser.add_argument(
        "--selected-key",
        default="selected_experts",
        help="Substring that identifies selected expert entries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyse_saturation(
        args.datasets,
        args.models,
        analyst_root=args.analyst_root,
        output_root=args.output_root,
        checkpoints=CHECKPOINT_SUFFIXES,
        layers=ALL_LAYER_IDS,
        selected_key=args.selected_key,
    )


if __name__ == "__main__":
    main()
