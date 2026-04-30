"""Auto-generated module extracted from progress_changes_selected_pretrain.ipynb."""
from __future__ import annotations

def run() -> None:
    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import glob
    import ujson as json
    import torch
    from tqdm import tqdm
    from safetensors import safe_open
    import torch.nn as nn
    from matplotlib.ticker import MaxNLocator

    # Set font globally to Times New Roman
    plt.rcParams['font.family'] = 'DejaVu Serif'
    fontsize = 19

    with open("/cm/archive/anonymous_new/revise_checkpoints/Xphi35-siglip224/SMOE/665K36/analysts/entropy/0722_0217_llava_v1.5_mme_llava_model_args_3775f6/mme.json", "r") as f:
        data = json.load(f)

    data['logs'][0]

    # Merge logs results
    root_paths = "/cm/archive/anonymous_new/revise_checkpoints/Xphi35-siglip224/SMOE/665K36"
    model_list = [
        "Full_smoe_sigmoidgating",
        "Full_smoe_share",
        "Full_smoe_plus_plus",
        "Full_smoe",
        "Full_smoe_tcmoe",
        "Full_xmoe",
        "Full_smoe_sharev3"
    ]
    datas = [
        "mme",
        "mmmu_val",
        "mmstar",
        "mathvista_testmini"
    ]

    data_agg = {}
    for name_data in datas:
        data_agg[name_data] = {}
        for name_ml in model_list:
            path = f"{root_paths}/{name_ml}/analysts/{name_data}_score_selected_final.json"
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                data_agg[name_data][name_ml] = data.get(name_ml, data.get("revise_" + name_ml))
            except FileNotFoundError:
                print(f"File not found: {path}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON from: {path}")

    data_agg

    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    Plot expert-change rates of multiple MoE variants on four multimodal benchmarks.
    ‒ Colour-blind friendly palette (Okabe-Ito)
    ‒ Uniform X-axis labels: 20-40 · 40-60 · 60-80 · 80-100
    ‒ No black border around markers.

    Author: Nam (2025-07-21)
    """
    from itertools import cycle
    from pathlib import Path

    # ========== 1. CONFIGURATION ========== #
    FIG_W, FIG_H = 20, 4  # figure size (inch)
    FONTSIZE_TITLE = 14
    FONTSIZE_LABEL = 12
    FONTSIZE_LEGEND = 11

    root_paths = "/cm/archive/anonymous_new/revise_checkpoints/Xphi35-siglip224/SMOE/665K36"
    model_list = [
        "Full_smoe_sigmoidgating",
        "Full_smoe_share",
        "Full_smoe_sharev3",
        "Full_smoe_plus_plus",
        "Full_smoe",
        "Full_smoe_tcmoe",
        "Full_xmoe",
    ]
    datas = ["mme", "mathvista_testmini", "validate", "hellaswag"]

    pretty_name = {
        "Full_smoe": "SMoE",
        "Full_smoe_share": "SharedE-V2",
        "Full_smoe_sharev3": "SharedE-V3",
        "Full_smoe_sigmoidgating": r"$\sigma$-MoE",
        "Full_smoe_plus_plus": "SMoE++",
        "Full_smoe_tcmoe": "TC-MoE",
        "Full_xmoe": "X-MoE",
    }

    # ---------- Fixed X labels ---------- #
    x_labels = ["20-40", "40-60", "60-80", "80-100"]
    x_labels_pretrain = ["10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80","80-90", "90-100" ]

    x_pos = np.arange(len(x_labels))  # [0, 1, 2, 3]
    x_pos_pretrain = np.arange(len(x_labels_pretrain))

    def tilt_xticks(ax, rotation=30):
        ax.tick_params(axis='x', labelrotation=rotation)
        for t in ax.get_xticklabels():
            t.set_ha('right')
            t.set_rotation_mode('anchor')

    # ========== 2. LOAD & COMPUTE VALUES ========== #
    def load_all_scores() -> dict:
        """Return data_agg[data][model] = {percent: {layer: [score,…]}}"""
        data_agg = {d: {} for d in datas}
        for dname in datas:
            for m in model_list:
                _new = m.replace("revise_", "")
                jpath = (Path(root_paths) / _new / "analysts" /
                         f"{dname}_score_selected_final.json")
                if not jpath.is_file():
                    continue
                with open(jpath) as f:
                    raw = json.load(f)
                try:
                    data_agg[dname][_new] = raw[m]
                except KeyError:
                    data_agg[dname][_new] = raw["revise_" + m]
        return data_agg

    def avg_per_bucket(model_dict: dict):
        """{percent: {layer: [score,…]}} → (mean_list, sorted_perc)"""
        sorted_perc = sorted(model_dict.keys(), key=lambda x: float(x))
        means = [float(np.mean([v[0] for v in model_dict[p].values()]))
                 for p in sorted_perc]
        return means, sorted_perc

    data_agg = load_all_scores()

    plot_values = {d: {} for d in datas}
    for d in datas:
        for m in model_list:
            try:
                vals, _ = avg_per_bucket(data_agg[d][m])
                plot_values[d][m] = vals
            except KeyError:
                continue

    with open("/cm/shared/anonymous_h100/LibMoE/evaluate/analysis/results/router_change_rate_158m_pretrain_final.json", "r") as f:
        plot_values_pretrain = json.load(f)

    for k, v in plot_values_pretrain.items():
        plot_values[k] = v

    # ========== 3. COLOR & MARKER (Okabe-Ito) ========== #
    cvd_palette = [
        "#0072B2", "#D55E00", "#009E73", "#F0E442",
        "#CC79A7", "#56B4E9", "#E69F00", "#000000",
    ]
    markers = cycle(["o", "s", "^", "v", "D", "P", "X", "*"])

    method_order = model_list
    method2color = {m: cvd_palette[i % len(cvd_palette)] for i, m in enumerate(method_order)}
    method2marker = {m: next(markers) for m in method_order}

    # ========== 4. PLOT ========== #
    fig, axes = plt.subplots(1, 4, figsize=(FIG_W, FIG_H))

    title_map = {
        "mme": "MME",
        "mmmu_val": "MMMU-Val",
        "mmstar": "MMStar",
        "mathvista_testmini": "MathVista-Mini",
        "blimp": "BLiMP",
        "hellaswag": "HellaSwag",
        "validate": "Validation"
    }

    for i, dname in enumerate(datas):
        ax = axes[i]
        for m in method_order:
            x_pos = np.arange(len(plot_values[dname][m]))
            try:
                ax.plot(
                    x_pos,
                    plot_values[dname][m],
                    label=pretty_name.get(m, m),
                    color=method2color[m],
                    marker=method2marker[m],
                    linewidth=4,
                    markersize=10,
                    markeredgewidth=0,  # no border
                )
                ax.set_title(title_map.get(dname, dname), fontsize=FONTSIZE_TITLE)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels_pretrain, fontsize=FONTSIZE_LABEL)
                ax.tick_params(axis='x', labelrotation=30)
                ax.set_xlabel("Data Percentage", fontsize=FONTSIZE_LABEL)
                ax.set_ylabel("Expert Change Rate", fontsize=FONTSIZE_LABEL)
                ax.yaxis.grid(True, linestyle="--", alpha=.6)
                ax.tick_params(axis="y", labelsize=FONTSIZE_LABEL)
                ax.spines[["right", "top"]].set_visible(False)
            except KeyError:
                print(f"Missing data for {dname}, {m}")

    # ---------- 5. LEGEND ---------- #
    handles, labels = [], []
    for m in method_order:
        handles.append(plt.Line2D(
            [], [], linestyle="-", linewidth=4,
            color=method2color[m], marker=method2marker[m],
            markersize=10, markeredgewidth=0,
            label=pretty_name.get(m, m)))
        labels.append(pretty_name.get(m, m))

    fig.legend(handles, labels, loc="upper center",
               ncol=int(len(method_order)/2), fontsize=FONTSIZE_LEGEND,
               bbox_to_anchor=(0.5, 1.18))

    plt.tight_layout()
    out_path = Path("expert_change_rates_shareE_inve.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved to {out_path.resolve()}")

    method_order

    m

    model_list


if __name__ == "__main__":
    run()
