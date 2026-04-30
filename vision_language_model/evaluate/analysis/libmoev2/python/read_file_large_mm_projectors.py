"""
Specialised CLI for analysing mm_projector expert selections.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import constants, selection_metrics
from .selection_pipeline import analyse_datasets as run_analysis


MM_PROJECTOR_LAYER = ("mm_projector",)
CHECKPOINT_SUFFIXES = tuple(cp.split("-")[-1] for cp in constants.CHECKPOINTS_665K)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse mm_projector selection stability.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mme"],
        help="Datasets to process.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Full_smoe_sigmoidgating"],
        help="Model directories to process.",
    )
    parser.add_argument(
        "--analyst-root",
        type=Path,
        default=constants.ANALYST_ROOT_1M2,
        help="Root directory containing analyst runs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=constants.ANALYST_ROOT_1M2,
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
        layers=MM_PROJECTOR_LAYER,
        metric_fn=selection_metrics.average_switch_fraction,
        selected_key=args.selected_key,
        allowed_layers=MM_PROJECTOR_LAYER,
        expected_layers=None,
        file_suffix="_mm_projector",
    )


if __name__ == "__main__":
    main()
