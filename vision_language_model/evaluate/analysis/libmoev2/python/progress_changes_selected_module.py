"""Auto-generated module extracted from progress_changes_selected.ipynb."""
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
    datas = ["blimp", "hellaswag"]

    pretty_name = {
        "Full_smoe": "SMoE-R",
        "Full_smoe_share": "SMoE-Share V2",
        "Full_smoe_sharev3": "SMoE-Share V3",
        "Full_smoe_sigmoidgating": "SMoE-Sigmoid",
        "Full_smoe_plus_plus": "SMoE++",
        "Full_smoe_tcmoe": "TC-MoE",
        "Full_xmoe": "X-MoE",
    }

    # ---------- Fixed X labels ---------- #
    x_labels = ["20-40", "40-60", "60-80", "80-100"]
    x_pos = np.arange(len(x_labels))  # [0, 1, 2, 3]

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
                    raise FileNotFoundError(jpath)
                with open(jpath) as f:
                    raw = json.load(f)
                try:
                    data_agg[dname][_new] = raw[m]
                except KeyError:
                    data_agg[dname][_new] = raw["revise_" + m]
        return data_agg

    with open("/cm/shared/anonymous_h100/LibMoE/evaluate/analysis/results/router_change_rate_158m_pretrain.json", "r") as f:
        plot_values = json.load(f)

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
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))

    title_map = {
        "mme": "MME",
        "mmmu_val": "MMMU-Val",
        "mmstar": "MMStar",
        "mathvista_testmini": "MathVista-Mini",
    }

    for i, dname in enumerate(datas):
        ax = axes[i]
        for m in method_order:
            try:
                ax.plot(
                    x_pos,
                    plot_values[dname][m].values(),
                    label=pretty_name.get(m, m),
                    color=method2color[m],
                    marker=method2marker[m],
                    linewidth=4,
                    markersize=10,
                    markeredgewidth=0,  # no border
                )
            except KeyError:
                print(f"Missing data for {dname}, {m}")

        ax.set_title(title_map.get(dname, dname), fontsize=FONTSIZE_TITLE)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=FONTSIZE_LABEL)
        ax.set_xlabel("Data Percentage", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Expert Change Rate", fontsize=FONTSIZE_LABEL)
        ax.yaxis.grid(True, linestyle="--", alpha=.6)
        ax.tick_params(axis="y", labelsize=FONTSIZE_LABEL)
        ax.spines[["right", "top"]].set_visible(False)

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
               ncol=len(method_order), fontsize=FONTSIZE_LEGEND,
               bbox_to_anchor=(0.5, 1.18))

    plt.tight_layout()
    out_path = Path("expert_change_rates.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved to {out_path.resolve()}")


if __name__ == "__main__":
    run()
