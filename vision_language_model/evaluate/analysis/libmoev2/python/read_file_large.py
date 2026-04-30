"""
CLI utilities for analysing expert selection stability across checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import constants, selection_metrics
from .selection_pipeline import analyse_datasets as run_analysis


EXPECTED_VISION_LAYERS = tuple(str(idx) for idx in range(10))
ALL_LAYER_IDS = tuple(str(idx) for idx in range(27))
CHECKPOINT_SUFFIXES = tuple(cp.split("-")[-1] for cp in constants.CHECKPOINTS_665K)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse expert selection stability.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mme", "mmmu_val", "mmstar", "mathvista_testmini"],
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
    run_analysis(
        args.datasets,
        args.models,
        analyst_root=args.analyst_root,
        output_root=args.output_root,
        checkpoints=CHECKPOINT_SUFFIXES,
        layers=ALL_LAYER_IDS,
        metric_fn=selection_metrics.average_switch_fraction,
        selected_key=args.selected_key,
        allowed_layers=ALL_LAYER_IDS,
        expected_layers=EXPECTED_VISION_LAYERS,
    )


if __name__ == "__main__":
    main()
