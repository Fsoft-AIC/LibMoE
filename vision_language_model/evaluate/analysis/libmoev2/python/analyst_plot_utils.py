from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import EngFormatter, MaxNLocator
import ujson as json

from .constants import (
    BENCHMARKS_SELECTED,
    BENCHMARK_TITLES,
    CHECKPOINTS_665K,
    DEFAULT_FONT_FAMILY,
    DEFAULT_FONT_SIZE,
    FIGURE_OUTPUT_DIR,
    RESULT_METRICS_PATH,
    STAGE_COLORS,
    STAGE_LABELS,
)


DATASETS_FOR_AVG: Tuple[str, ...] = BENCHMARKS_SELECTED
PERCENT_CHECKPOINTS: Tuple[str, ...] = CHECKPOINTS_665K

FIGURE_MARKER_STYLE = {
    "marker": "o",
    "linestyle": "-",
    "linewidth": 2.5,
    "markersize": 5,
}

ARROW_UP = ""
ARROW_DOWN = "\u2193"
N_COLS = 3


@dataclass
class StageCurves:
    name: str
    label: str
    color: str
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]]
    average_steps: np.ndarray
    average_scores: np.ndarray
    rank_frame: pd.DataFrame


def configure_matplotlib(font_family: str = DEFAULT_FONT_FAMILY) -> None:
    plt.rcParams["font.family"] = font_family
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"


def load_results(
    path: Path | str = RESULT_METRICS_PATH,
) -> MutableMapping[str, MutableMapping[str, MutableMapping[str, Dict[str, float]]]]:
    with open(Path(path), "r") as handle:
        return json.load(handle)


def find_missing_entries(
    results: Mapping[str, Mapping[str, Mapping[str, Dict[str, float]]]],
    benchmarks: Iterable[str] = BENCHMARKS_SELECTED,
    checkpoints: Sequence[str] = PERCENT_CHECKPOINTS,
) -> List[str]:
    missing: List[str] = []
    for stage, stage_logs in results.items():
        for bench in benchmarks:
            for checkpoint in checkpoints:
                if bench not in stage_logs or checkpoint not in stage_logs[bench]:
                    missing.append(f"{stage}/{bench}/{checkpoint}")
    return missing


def prepare_stage_curves(
    results: Mapping[str, Mapping[str, Mapping[str, Dict[str, float]]]],
    checkpoints: Sequence[str] = PERCENT_CHECKPOINTS,
    benchmarks: Sequence[str] = BENCHMARKS_SELECTED,
    datasets_for_avg: Sequence[str] = DATASETS_FOR_AVG,
) -> List[StageCurves]:
    stage_curves: List[StageCurves] = []
    stage_order = [stage for stage in STAGE_LABELS.keys() if stage in results]
    checkpoint_count = len(checkpoints)

    for stage_name in stage_order:
        stage_results = results.get(stage_name, {})
        curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        rank_source: Dict[str, List[float]] = {}
        avg_steps: Dict[str, List[float]] = {ckpt: [] for ckpt in checkpoints}
        avg_scores: Dict[str, List[float]] = {ckpt: [] for ckpt in checkpoints}

        for dataset_name, checkpoint_scores in stage_results.items():
            steps: List[float] = []
            scores: List[float] = []

            for checkpoint in checkpoints:
                payload = checkpoint_scores.get(checkpoint)
                if not payload:
                    continue

                step_value = _resolve_step_value(checkpoint, payload)
                score_value = float(payload["score"])

                steps.append(step_value)
                scores.append(score_value)

                if dataset_name in datasets_for_avg:
                    avg_steps[checkpoint].append(step_value)
                    avg_scores[checkpoint].append(score_value)

            if dataset_name in benchmarks and steps:
                curves[dataset_name] = (np.asarray(steps, dtype=float), np.asarray(scores, dtype=float))
                rank_source[dataset_name] = _pad_series(scores, checkpoint_count)

        if not curves:
            continue

        avg_steps_array = np.asarray(
            [_mean_or_nan(avg_steps[ckpt]) for ckpt in checkpoints], dtype=float
        )
        avg_scores_array = np.asarray(
            [_mean_or_nan(avg_scores[ckpt]) for ckpt in checkpoints], dtype=float
        )

        rank_frame = pd.DataFrame(rank_source, index=range(checkpoint_count))
        stage_curves.append(
            StageCurves(
                name=stage_name,
                label=STAGE_LABELS.get(stage_name, stage_name),
                color=STAGE_COLORS.get(stage_name, "#000000"),
                curves=curves,
                average_steps=avg_steps_array,
                average_scores=avg_scores_array,
                rank_frame=rank_frame,
            )
        )
    return stage_curves


def compute_average_rank(stage_curves: Sequence[StageCurves], checkpoint_count: int) -> pd.DataFrame:
    rank_per_checkpoint = pd.DataFrame(
        index=[curve.name for curve in stage_curves],
        columns=range(checkpoint_count),
        dtype=float,
    )

    for checkpoint_idx in range(checkpoint_count):
        performance: Dict[str, np.ndarray] = {}
        benchmark_names: Optional[Iterable[str]] = None

        for curve in stage_curves:
            if curve.rank_frame.empty or checkpoint_idx >= len(curve.rank_frame):
                continue

            row = curve.rank_frame.iloc[checkpoint_idx]
            benchmark_names = row.index
            performance[curve.name] = row.to_numpy(dtype=float)

        if not performance or benchmark_names is None:
            continue

        df_checkpoint = pd.DataFrame(performance, index=benchmark_names).T
        ranked = df_checkpoint.rank(axis=0, method="average", ascending=False)
        rank_per_checkpoint.iloc[:, checkpoint_idx] = ranked.mean(axis=1)

    return rank_per_checkpoint


def select_reference_steps(stage_curves: Sequence[StageCurves]) -> np.ndarray:
    for curve in stage_curves:
        steps = curve.average_steps
        if np.any(~np.isnan(steps)):
            return steps
    return np.full(len(PERCENT_CHECKPOINTS), np.nan, dtype=float)


def plot_metric_grid(
    stage_curves: Sequence[StageCurves],
    avg_rank: pd.DataFrame,
    avg_steps: np.ndarray,
    *,
    benchmarks: Sequence[str] = BENCHMARKS_SELECTED,
    benchmark_titles: Mapping[str, str] = BENCHMARK_TITLES,
    font_size: int = DEFAULT_FONT_SIZE,
    arrow_up: str = ARROW_UP,
    arrow_down: str = ARROW_DOWN,
    output_dir: Path | str = FIGURE_OUTPUT_DIR,
    show: bool = True,
) -> Path:
    fig, axes = _create_axes(len(benchmarks) + 2, font_size)
    axes_map = {bench: axes[idx] for idx, bench in enumerate(benchmarks)}
    avg_ax = axes[len(benchmarks)]
    rank_ax = axes[len(benchmarks) + 1]
    legend_handles: List[plt.Line2D] = []
    legend_labels: List[str] = []

    for curve in stage_curves:
        avg_mask = ~np.isnan(curve.average_steps) & ~np.isnan(curve.average_scores)
        if np.any(avg_mask):
            line, = avg_ax.plot(
                curve.average_steps[avg_mask],
                curve.average_scores[avg_mask],
                label=curve.label,
                color=curve.color,
                **FIGURE_MARKER_STYLE,
            )
            legend_handles.append(line)
            legend_labels.append(curve.label)

        for bench, ax in axes_map.items():
            if bench not in curve.curves:
                continue
            steps, values = curve.curves[bench]
            mask = ~np.isnan(steps) & ~np.isnan(values)
            if not np.any(mask):
                continue
            ax.plot(
                steps[mask],
                values[mask],
                color=curve.color,
                **FIGURE_MARKER_STYLE,
            )

        if curve.name in avg_rank.index:
            rank_values = avg_rank.loc[curve.name].to_numpy(dtype=float)
            rank_mask = ~np.isnan(avg_steps) & ~np.isnan(rank_values)
            if np.any(rank_mask):
                rank_ax.plot(
                    avg_steps[rank_mask],
                    rank_values[rank_mask],
                    color=curve.color,
                    **FIGURE_MARKER_STYLE,
                )

    for bench, ax in axes_map.items():
        title = benchmark_titles.get(bench, bench)
        ax.set_title(f"{title}{arrow_up}", fontsize=font_size + 2, pad=12)

    avg_ax.set_title(f"{benchmark_titles.get('avg', 'Average')}{arrow_up}", fontsize=font_size + 2, pad=12)
    avg_ax.set_ylabel(f"Average Accuracy{arrow_up}", fontsize=font_size, labelpad=5)

    rank_ax.set_title(f"Average Rank{arrow_down}", fontsize=font_size + 2, pad=12)
    rank_ax.set_ylabel(f"Average Rank{arrow_down}", fontsize=font_size, labelpad=5)

    _decorate_axes(axes, font_size)

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        fontsize=font_size - 2,
        ncol=min(7, len(legend_labels)),
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        handlelength=1.8,
        borderaxespad=0.6,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path / "timelearnmetric_steps_all_benchmarks.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return pdf_path


def plot_compact_summary(
    stage_curves: Sequence[StageCurves],
    avg_rank: pd.DataFrame,
    avg_steps: np.ndarray,
    *,
    font_size: int = DEFAULT_FONT_SIZE,
    arrow_up: str = ARROW_UP,
    arrow_down: str = ARROW_DOWN,
    output_dir: Path | str = FIGURE_OUTPUT_DIR,
    show: bool = True,
) -> Tuple[Path, Path]:
    fig, (ax_avg, ax_rank) = plt.subplots(1, 2, figsize=(14, 4))
    legend_handles: List[plt.Line2D] = []
    legend_labels: List[str] = []

    for curve in stage_curves:
        mask = ~np.isnan(curve.average_steps) & ~np.isnan(curve.average_scores)
        if np.any(mask):
            line, = ax_avg.plot(
                curve.average_steps[mask],
                curve.average_scores[mask],
                label=curve.label,
                color=curve.color,
                **FIGURE_MARKER_STYLE,
            )
            legend_handles.append(line)
            legend_labels.append(curve.label)

        if curve.name in avg_rank.index:
            rank_values = avg_rank.loc[curve.name].to_numpy(dtype=float)
            rank_mask = ~np.isnan(avg_steps) & ~np.isnan(rank_values)
            if np.any(rank_mask):
                ax_rank.plot(
                    avg_steps[rank_mask],
                    rank_values[rank_mask],
                    color=curve.color,
                    **FIGURE_MARKER_STYLE,
                )

    for axis in (ax_avg, ax_rank):
        axis.tick_params(axis="both", which="major", labelsize=font_size)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.yaxis.grid(True, linestyle="--", color="gray", alpha=0.6)
        axis.spines["right"].set_color("none")
        axis.spines["top"].set_color("none")
        axis.spines["left"].set_linewidth(1.0)
        axis.spines["bottom"].set_linewidth(1.0)
        axis.set_xlabel("Training Steps", fontsize=font_size)
        axis.xaxis.set_major_formatter(EngFormatter(unit=""))

    ax_avg.set_ylabel(f"Average Accuracy{arrow_up}", fontsize=font_size)
    ax_rank.set_ylabel(f"Average Rank{arrow_down}", fontsize=font_size)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        fontsize=font_size - 2,
        ncol=min(7, len(legend_labels)),
        bbox_to_anchor=(0.5, 1.15),
        frameon=False,
        handlelength=1.8,
        borderaxespad=0.6,
    )

    fig.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_path = output_path / "timelearnmetric_avg_2cols.pdf"
    png_path = output_path / "timelearnmetric_avg_2cols.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return pdf_path, png_path


def build_figures(
    results: Mapping[str, Mapping[str, Mapping[str, Dict[str, float]]]],
    *,
    checkpoints: Sequence[str] = PERCENT_CHECKPOINTS,
    benchmarks: Sequence[str] = BENCHMARKS_SELECTED,
    output_dir: Path | str = FIGURE_OUTPUT_DIR,
    font_size: int = DEFAULT_FONT_SIZE,
    show: bool = True,
) -> Dict[str, object]:
    configure_matplotlib()
    stage_curves = prepare_stage_curves(results, checkpoints=checkpoints, benchmarks=benchmarks)
    if not stage_curves:
        raise ValueError("No stage curves could be prepared from the provided results.")

    avg_steps = select_reference_steps(stage_curves)
    avg_rank = compute_average_rank(stage_curves, len(checkpoints))

    grid_pdf = plot_metric_grid(
        stage_curves,
        avg_rank,
        avg_steps,
        benchmarks=benchmarks,
        font_size=font_size,
        output_dir=output_dir,
        show=show,
    )
    summary_pdf, summary_png = plot_compact_summary(
        stage_curves,
        avg_rank,
        avg_steps,
        font_size=font_size,
        output_dir=output_dir,
        show=show,
    )

    missing = find_missing_entries(results, benchmarks=benchmarks, checkpoints=checkpoints)

    return {
        "grid_pdf": grid_pdf,
        "summary_pdf": summary_pdf,
        "summary_png": summary_png,
        "missing": missing,
        "stage_curves": stage_curves,
        "avg_rank": avg_rank,
    }


def _create_axes(count: int, font_size: int) -> Tuple[plt.Figure, List[plt.Axes]]:
    rows = int(np.ceil(count / N_COLS))
    fig, axs = plt.subplots(rows, N_COLS, figsize=(15, 4 * rows))
    axes = list(axs.flat) if isinstance(axs, np.ndarray) else [axs]

    # Hide any unused axes
    for ax in axes[count:]:
        ax.set_visible(False)

    used_axes = axes[:count] if count <= len(axes) else axes
    return fig, used_axes


def _decorate_axes(axes: Sequence[plt.Axes], font_size: int) -> None:
    for ax in axes:
        if not ax.get_visible():
            continue
        ax.set_xlabel("Training Steps", fontsize=font_size)
        ax.xaxis.set_major_formatter(EngFormatter(unit=""))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="both", which="major", labelsize=font_size)
        ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.6)
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)


def _resolve_step_value(checkpoint: str, payload: Mapping[str, float]) -> float:
    if "num_input_tokens_seen" in payload and payload["num_input_tokens_seen"] is not None:
        return float(payload["num_input_tokens_seen"])
    try:
        return float(checkpoint.split("-")[-1])
    except (IndexError, ValueError):
        return np.nan


def _pad_series(values: Sequence[float], target_length: int, fill_value: float = np.nan) -> List[float]:
    data = list(values)
    if len(data) >= target_length:
        return data[:target_length]
    return data + [fill_value] * (target_length - len(data))


def _mean_or_nan(values: Sequence[float]) -> float:
    if not values:
        return np.nan
    return float(np.mean(values))
