"""Auto-generated module extracted from entropy_analyst.ipynb."""
from __future__ import annotations

def run() -> None:
    from pathlib import Path
    from typing import Iterable, Mapping, Sequence

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import ujson as json
    from matplotlib.ticker import MaxNLocator

    plt.rcParams["font.family"] = "DejaVu Serif"
    DEFAULT_FONT_SIZE = 14

    BASE_DIR = Path("/cm/archive/anonymous_new/revise_checkpoints/Xphi35-siglip224/SMOE/665K36/analysts/entropy")
    LAYER_IDS = [str(i) for i in range(27)]
    MM_LAYER = "mm_projector"
    ALL_LAYER_IDS = LAYER_IDS + [MM_LAYER]
    MM_INDEX = len(LAYER_IDS)

    METHOD_LABELS = {
        "Full_smoe":               "SMoE-R",
        "Full_smoe_sigmoidgating": "SMoE-SG",
        "Full_smoe_share":         "SharedE-V2",
        "revise_Full_smoe_sharev3":"SharedE-V3",
        "Full_xmoe":               "X-MoE",
        "Full_smoe_tcmoe":         "TC-MoE",
        "Full_smoe_plus_plus":     "MoE++",
    }
    METHOD_ORDER = list(METHOD_LABELS.keys())

    DOMAIN_ABBREVIATIONS = {
        "celebrity": "celebs",
        "code_reasoning": "code_reason",
        "commonsense_reasoning": "common_sense",
        "numerical_calculation": "num_calc",
        "text_translation": "txt_trans",
    }

    TASK_GROUPS = {
        "Perception (Coarse-Grained Tasks)": ["existence", "count", "position", "color"],
        "Perception (Fine-Grained Tasks)": ["posters", "celebrity", "scene", "landmark", "artwork"],
        "Perception (OCR Task)": ["OCR"],
        "Cognition (Reasoning Tasks)": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"],
    }

    PALETTE = ["#CC79A7", "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#F0E442"]
    MARKERS = ["o", "s", "^", "D", "P", "X", "*"]

    def load_json(path: Path) -> dict:
        with path.open("r") as f:
            return json.load(f)


    def extract_method_name(results: Mapping) -> str:
        model_args = results.get("model_configs", {}).get("model_args", "")
        return model_args.split("/")[-1].split(",")[0]


    def compute_entropy_topk_np(weight_experts) -> float:
        weight_experts = np.asarray(weight_experts, dtype=np.float64)
        if weight_experts.size == 0:
            return float("nan")
        sums = weight_experts.sum(axis=-1, keepdims=True)
        sums[sums == 0] = 1.0
        probs = weight_experts / sums
        entropy = -np.sum(probs * np.log(probs + 1e-9), axis=-1)
        k = weight_experts.shape[-1]
        if k < 2:
            return 0.0
        max_entropy = np.log(k)
        return float(np.mean(entropy / max_entropy))


    def get_layer_metrics(entry: Mapping) -> Mapping[str, Mapping]:
        metrics = entry.get("logs_metrics_vision") or []
        if not metrics:
            return {}
        first = metrics[0] or []
        if not first:
            return {}
        return first[0] or {}


    def iter_run_dirs(root: Path) -> list[Path]:
        return [
            path for path in sorted(root.glob("*"))
            if (path / "mme.json").is_file() and (path / "results.json").is_file()
        ]


    def collect_entropy_records(run_dirs: Sequence[Path]) -> pd.DataFrame:
        records = []
        for run_dir in run_dirs:
            try:
                mme_logs = load_json(run_dir / "mme.json")
                result_logs = load_json(run_dir / "results.json")
            except FileNotFoundError:
                continue
            method = extract_method_name(result_logs)
            for entry in mme_logs.get("logs", []):
                category = entry.get("doc", {}).get("category", "unknown")
                layer_metrics = get_layer_metrics(entry)
                for idx, layer_id in enumerate(LAYER_IDS):
                    metrics = layer_metrics.get(layer_id)
                    if not metrics:
                        continue
                    dist_weights = metrics.get("dist_experts_top1")
                    if dist_weights is not None:
                        value = compute_entropy_topk_np(dist_weights)
                        if np.isfinite(value):
                            records.append({
                                "run": run_dir.name,
                                "method": method,
                                "category": category,
                                "layer": idx,
                                "layer_label": layer_id,
                                "metric": "dist_entropy",
                                "value": value,
                            })
                    weight_entropy = metrics.get("entropy_weight_topk")
                    if weight_entropy is not None and np.isfinite(weight_entropy):
                        records.append({
                            "run": run_dir.name,
                            "method": method,
                            "category": category,
                            "layer": idx,
                            "layer_label": layer_id,
                            "metric": "weight_entropy",
                            "value": float(weight_entropy),
                        })
                mm_metrics = layer_metrics.get(MM_LAYER)
                if mm_metrics:
                    weight_entropy = mm_metrics.get("entropy_weight_topk")
                    if weight_entropy is not None and np.isfinite(weight_entropy):
                        records.append({
                            "run": run_dir.name,
                            "method": method,
                            "category": category,
                            "layer": MM_INDEX,
                            "layer_label": MM_LAYER,
                            "metric": "weight_entropy",
                            "value": float(weight_entropy),
                        })
        return pd.DataFrame.from_records(
            records,
            columns=["run", "method", "category", "layer", "layer_label", "metric", "value"],
        )


    def to_layer_dict(df: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
        result: dict[str, dict[str, list[float]]] = {}
        if df.empty:
            return result
        for (method, category), group in df.groupby(["method", "category"]):
            ordered = group.sort_values("layer")
            result.setdefault(method, {})[category] = ordered["value"].tolist()
        return result


    def mean_of_last_layers(layer_dict: Mapping[str, Mapping[str, Sequence[float]]], n_last: int = 2) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for method, categories in layer_dict.items():
            method_summary = {}
            for category, values in categories.items():
                if len(values) >= n_last:
                    method_summary[category] = float(np.mean(values[-n_last:]))
            if method_summary:
                summary[method] = method_summary
        return summary


    def mean_per_category(layer_dict: Mapping[str, Mapping[str, Sequence[float]]]) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for method, categories in layer_dict.items():
            method_summary = {}
            for category, values in categories.items():
                if values:
                    method_summary[category] = float(np.mean(values))
            if method_summary:
                summary[method] = method_summary
        return summary


    def layer_means(df: pd.DataFrame, drop_layers: Sequence[int] = ()) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
        if df.empty:
            return result
        drop_set = set(drop_layers)
        for method, group in df.groupby("method"):
            filtered = group[~group["layer"].isin(drop_set)]
            ordered = filtered.sort_values("layer")
            if not ordered.empty:
                result[method] = ordered["value"].tolist()
        return result


    def summarize_task_groups(domain_summary: Mapping[str, Mapping[str, float]], groups: Mapping[str, Sequence[str]]) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for method, domain_values in domain_summary.items():
            method_summary = {}
            for group_name, subtasks in groups.items():
                vals = [domain_values[subtask] for subtask in subtasks if subtask in domain_values]
                if vals:
                    method_summary[group_name] = float(np.mean(vals))
            if method_summary:
                result[method] = method_summary
        return result


    def task_group_to_frame(summary: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
        rows = [
            {"method": method, "task_group": group, "value": value}
            for method, groups in summary.items()
            for group, value in groups.items()
        ]
        return pd.DataFrame(rows)


    def moving_average(arr: Sequence[float], window: int = 3) -> np.ndarray:
        arr_np = np.asarray(arr, dtype=np.float64)
        if window <= 1 or arr_np.size == 0:
            return arr_np
        pad = window // 2
        padded = np.concatenate(([arr_np[0]] * pad, arr_np, [arr_np[-1]] * pad))
        kernel = np.ones(window) / window
        return np.convolve(padded, kernel, mode="valid")


    def available_methods(methods: Iterable[str]) -> list[str]:
        method_set = set(methods)
        return [m for m in METHOD_ORDER if m in method_set]


    def compute_domain_range(domain_summary: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
        ranges: dict[str, float] = {}
        for method, values in domain_summary.items():
            if len(values) >= 2:
                maxima = max(values.values())
                minima = min(values.values())
                ranges[method] = float((maxima - minima) * 100)
        return ranges


    def plot_domain_entropy_grid(domain_summary: Mapping[str, Mapping[str, float]], *, method_order: Sequence[str] = METHOD_ORDER, abbreviations: Mapping[str, str] = DOMAIN_ABBREVIATIONS, font_size: int = DEFAULT_FONT_SIZE, n_cols: int = 4) -> None:
        methods = [m for m in method_order if m in domain_summary]
        if not methods:
            print("No data available for domain-level plot.")
            return
        n_rows = int(np.ceil(len(methods) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 4.5 * n_rows))
        axes = np.array(axes, ndmin=1).flatten()
        for ax in axes[len(methods):]:
            ax.set_visible(False)
        for ax, method in zip(axes, methods):
            domain_values = domain_summary[method]
            if not domain_values:
                ax.set_visible(False)
                continue
            items = sorted(domain_values.items(), key=lambda item: item[1], reverse=True)
            values = [val for _, val in items]
            labels = [abbreviations.get(name, name) for name, _ in items]
            ax.barh(labels, values, color="skyblue")
            ax.set_xlim(min(values) - 0.01, max(values) + 0.01)
            ax.set_title(METHOD_LABELS.get(method, method), fontsize=font_size + 2)
            ax.set_xlabel("Expert Allocation Entropy (norm.)", fontsize=font_size)
            ax.tick_params(axis="y", labelsize=font_size - 2)
            for idx, val in enumerate(values):
                ax.text(val + 0.001, idx, f"{val:.4f}", va="center", ha="left", fontsize=font_size - 2)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(False)
        plt.tight_layout()
        plt.show()


    def plot_task_group_line(task_group_summary: Mapping[str, Mapping[str, float]], *, method_order: Sequence[str] = METHOD_ORDER, font_size: int = DEFAULT_FONT_SIZE) -> None:
        df = task_group_to_frame(task_group_summary)
        if df.empty:
            print("No data available for task-group line plot.")
            return
        methods = [m for m in method_order if m in df["method"].unique()]
        if not methods:
            print("No matching methods for the requested order.")
            return
        fig, ax = plt.subplots(figsize=(4 * len(methods), 6))
        for idx, (group_name, group_df) in enumerate(df.groupby("task_group", sort=False)):
            values = group_df.set_index("method").reindex(methods)["value"]
            label_positions = [METHOD_LABELS.get(m, m) for m in methods]
            ax.plot(
                label_positions,
                values,
                color=PALETTE[idx % len(PALETTE)],
                marker=MARKERS[idx % len(MARKERS)],
                linewidth=2,
                label=group_name,
            )
        ax.set_xlabel("SMoE Variant", fontsize=font_size)
        ax.set_ylabel("Expert Weight Allocation Entropy", fontsize=font_size)
        ax.tick_params(axis="x", rotation=25, labelsize=font_size - 1)
        ax.tick_params(axis="y", labelsize=font_size - 1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(title="Task Group", fontsize=font_size - 2, title_fontsize=font_size)
        plt.tight_layout()
        plt.show()


    def plot_task_group_bar(task_group_summary: Mapping[str, Mapping[str, float]], *, method_order: Sequence[str] = METHOD_ORDER, font_size: int = DEFAULT_FONT_SIZE) -> None:
        df = task_group_to_frame(task_group_summary)
        if df.empty:
            print("No data available for task-group bar plot.")
            return
        methods = [m for m in method_order if m in df["method"].unique()]
        if not methods:
            print("No matching methods for the requested order.")
            return
        groups = list(dict.fromkeys(df["task_group"]))
        x = np.arange(len(methods))
        bar_width = 0.8 / max(len(groups), 1)
        fig, ax = plt.subplots(figsize=(4 * len(methods), 6))
        for idx, group_name in enumerate(groups):
            group_values = (
                df[df["task_group"] == group_name]
                .set_index("method")
                .reindex(methods)["value"]
            )
            ax.bar(
                x + idx * bar_width - (len(groups) - 1) * bar_width / 2,
                group_values,
                width=bar_width,
                color=PALETTE[idx % len(PALETTE)],
                label=group_name,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=25, fontsize=font_size - 1)
        ax.set_xlabel("SMoE Variant", fontsize=font_size)
        ax.set_ylabel("Expert Weight Allocation Entropy", fontsize=font_size)
        ax.tick_params(axis="y", labelsize=font_size - 1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(title="Task Group", fontsize=font_size - 2, title_fontsize=font_size)
        plt.tight_layout()
        plt.show()


    def plot_layer_entropy(layer_summary: Mapping[str, Sequence[float]], *, title: str, ylabel: str, smooth: bool = False, window: int = 3, font_size: int = DEFAULT_FONT_SIZE) -> None:
        methods = [m for m in METHOD_ORDER if m in layer_summary]
        if not methods:
            print(f"No data available for plot: {title}")
            return
        num_layers = len(next(iter(layer_summary.values())))
        x = np.arange(1, num_layers + 1)
        fig, ax = plt.subplots(figsize=(12, 5))
        for idx, method in enumerate(methods):
            series = np.asarray(layer_summary[method], dtype=np.float64)
            if smooth and window > 1:
                series = moving_average(series, window)
            color = PALETTE[idx % len(PALETTE)]
            ax.plot(
                x,
                series,
                lw=2,
                label=METHOD_LABELS.get(method, method),
                color=color,
            )
        ax.set_xlabel("Layer index", fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend_cols = min(4, len(methods))
        ax.legend(ncol=legend_cols, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=font_size - 1)
        ax.set_title(title, fontsize=font_size + 1)
        plt.tight_layout()
        plt.show()

    run_dirs = iter_run_dirs(BASE_DIR)
    print(f"Discovered {len(run_dirs)} evaluation folders under {BASE_DIR}.")

    entropy_records = collect_entropy_records(run_dirs)
    if entropy_records.empty:
        print("No entropy records were loaded. Check that the source directory contains evaluation outputs.")
    else:
        display_columns = ["run", "method", "category", "layer", "metric", "value"]
        entropy_records[display_columns].head()

    if entropy_records.empty:
        aggregated_layers = pd.DataFrame(columns=["metric", "method", "category", "layer", "value"])
        dist_layers = {}
        weight_layers = {}
        dist_last_layers = {}
        weight_domain_mean = {}
        task_group_summary = {}
        weight_layer_means = {}
        dist_layer_means = {}
    else:
        aggregated_layers = (
            entropy_records.groupby(["metric", "method", "category", "layer"], as_index=False)["value"].mean()
        )

        dist_layers = to_layer_dict(aggregated_layers[aggregated_layers["metric"] == "dist_entropy"])
        weight_layers = to_layer_dict(aggregated_layers[aggregated_layers["metric"] == "weight_entropy"])

        dist_last_layers = mean_of_last_layers(dist_layers, n_last=2)
        weight_domain_mean = mean_per_category(weight_layers)
        task_group_summary = summarize_task_groups(weight_domain_mean, TASK_GROUPS)

        weight_layer_means = layer_means(
            aggregated_layers[aggregated_layers["metric"] == "weight_entropy"].groupby(["method", "layer"], as_index=False)["value"].mean(),
            drop_layers=(MM_INDEX,),
        )

        dist_layer_means = layer_means(
            aggregated_layers[aggregated_layers["metric"] == "dist_entropy"].groupby(["method", "layer"], as_index=False)["value"].mean()
        )

    domain_ranges = compute_domain_range(dist_last_layers)
    if domain_ranges:
        print("Entropy span across subtasks (Δ × 100):")
        for method in available_methods(domain_ranges.keys()):
            print(f"  {METHOD_LABELS.get(method, method)}: {domain_ranges[method]:.2f}")

    if dist_last_layers:
        plot_domain_entropy_grid(dist_last_layers)
    else:
        print("Domain-level entropy summary not available.")

    if task_group_summary:
        plot_task_group_line(task_group_summary)
    else:
        print("Task-group summary not available.")

    if task_group_summary:
        plot_task_group_bar(task_group_summary)
    else:
        print("Task-group summary not available.")

    if weight_layer_means:
        plot_layer_entropy(
            weight_layer_means,
            title="Expert Weight Allocation Entropy by Layer",
            ylabel="Expert Weight Allocation Entropy",
            smooth=False,
        )
    else:
        print("Weight entropy per-layer summary not available.")

    if dist_layer_means:
        plot_layer_entropy(
            dist_layer_means,
            title="Expert Allocation Entropy (Top-1 Dist.) by Layer",
            ylabel="Expert Allocation Entropy (norm.)",
            smooth=True,
            window=3,
        )
    else:
        print("Distribution entropy per-layer summary not available.")


if __name__ == "__main__":
    run()
