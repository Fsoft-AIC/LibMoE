"""
Shared constants and configuration helpers for the LibMoE v2 analysis toolkit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Tuple


# ---- Filesystem layout -------------------------------------------------------
ANALYST_ROOT_665K = Path("/cm/archive/anonymous_new/revise_checkpoints/Xphi35-siglip224/SMOE/665K36")
ANALYST_ROOT_1M2 = Path("/cm/archive/anonymous_new/revise_checkpoints/Xphi35-siglip224/SMOE/1M2")

RESULT_METRICS_PATH = Path(
    "/cm/shared/anonymous_h100/LibMoE/evaluate/analysis/results/result_metric_libmoev2_665k.json"
)

FIGURE_OUTPUT_DIR = Path("/cm/shared/anonymous_h100/LibMoE/evaluate/analysis/libmoev2/figures")


# ---- Benchmark checkpoints ---------------------------------------------------
CHECKPOINTS_665K: Tuple[str, ...] = (
    "checkpoint-4159",
    "checkpoint-8318",
    "checkpoint-12477",
    "checkpoint-16636",
    "checkpoint-20791",
)

CHECKPOINT_STEPS_665K: Tuple[int, ...] = tuple(int(item.split("-")[-1]) for item in CHECKPOINTS_665K)


# ---- Stage & benchmark metadata ----------------------------------------------
STAGE_LABELS: Mapping[str, str] = {
    "Full_smoe": "SMoE",
    "Full_xmoe": "XMoE",
    "Full_smoe_sigmoidgating": r"$\sigma$-MoE",
    "Full_smoe_share": "SharedE-V2",
    "Full_smoe_sharev3": "SharedE-V3",
    "Full_smoe_tcmoe": "TC-MoE",
    "Full_smoe_plus_plus": "MoE++",
}

STAGE_COLORS: Mapping[str, str] = {
    "Full_smoe_share": "#D55E00",
    "Full_smoe_perturbed": "#ff7f0e",
    "Full_smoe": "#CC79A7",
    "Full_competesmoev30": "#000000",
    "Full_xmoe": "#009E73",
    "Full_smoe_sigmoidgating": "#e377c2",
    "Full_smoe_sharev3": "#F80202",
    "Full_smoe_tcmoe": "#56B4E9",
    "Full_smoe_plus_plus": "#F0E442",
}

BENCHMARK_TITLES: Mapping[str, str] = {
    "pope": "POPE",
    "mmstar": "MMStar",
    "gqa": "GQA",
    "mmbench_en_dev": "MMBench EN",
    "ocrbench": "OCR Bench",
    "scienceqa_img": "SQA IMG",
    "ai2d": "AI2D",
    "mmerealworld_lite": "MME Real World",
    "mathvista_testmini": "MathVista Test Mini",
    "hallusion_bench_image": "Hallusion Bench Image",
    "seedbench_2_plus": "SeedBench2 Plus",
    "mmmu_val": "MMMU Val",
    "mmmu_pro_standard": "MMMU Pro Standard",
    "textvqa_val": "TextVQA Val",
    "infovqa_val": "InfoVQA",
    "mme": "MME",
    "avg": "Average",
}

BENCHMARKS_SELECTED: Tuple[str, ...] = (
    "ai2d",
    "gqa",
    "mmbench_en_dev",
    "textvqa_val",
    "ocrbench",
    "mmmu_val",
    "mmstar",
    "mme",
    "mmerealworld_lite",
    "pope",
    "hallusion_bench_image",
    "mathvista_testmini",
)


# ---- Default font configuration ----------------------------------------------
DEFAULT_FONT_FAMILY = "DejaVu Serif"
DEFAULT_FONT_SIZE = 16

