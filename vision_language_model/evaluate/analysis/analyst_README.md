# LibMoE v2 Analysis Toolkit (Refactored)

> ⚠️ **Heads up:** This folder has been **fully refactored** from the historical
> research drop. Paths, function names, and behaviours now follow a consistent,
> production-friendly layout. Treat this directory as the maintained toolkit—not
> as a mirror of the original notebooks.

## Directory layout

```
evaluate/analysis/libmoev2/
├─ python/                # All Python helpers (package: evaluate.analysis.libmoev2.python)
│  ├─ __init__.py
│  ├─ analyst_plot_utils.py
│  ├─ constants.py
│  ├─ log_utils.py
│  ├─ selection_metrics.py
│  ├─ selection_pipeline.py
│  ├─ router_metrics.py
│  ├─ read_file_large*.py
│  ├─ *_module.py         # Notebook runner modules (see below)
│  └─ …
├─ analyst_data.ipynb     # Thin front-ends that import from python/*
├─ convert_entropy.ipynb
├─ …                      # Other notebooks
├─ entropy_top3/          # Cached JSON metrics
└─ router_margin/         # Cached JSON metrics / plots
```

Importing `evaluate.analysis.libmoev2` re-exports everything from the
`python/` package for backwards compatibility, but new code should prefer
`evaluate.analysis.libmoev2.python`.

## Module reference

### Core utilities

| Module | Purpose | Key entry-points |
| --- | --- | --- |
| `python/constants.py` | Centralises filesystem paths, checkpoint ids, plotting defaults. Update this file whenever analyst directories move. | `ANALYST_ROOT_*`, `CHECKPOINTS_665K`, colour maps |
| `python/log_utils.py` | Streaming parsers for analyst JSON logs (vision-layer payloads, metadata, selected experts). | `extract_metrics_and_selected`, `collect_layer_metric` |
| `python/selection_metrics.py` | Similarity metrics between expert selections across checkpoints. | `average_switch_fraction`, `average_overlap_fraction`, `average_position_match` |
| `python/selection_pipeline.py` | High-level workflows that ingest multiple runs, persist cleaned artefacts, and compute per-checkpoint scores. | `analyse_datasets`, `collect_selected_experts` |
| `python/router_metrics.py` | Aggregates router entropy / margin stats across checkpoints. | `aggregate_entropy`, `aggregate_margin` |
| `python/analyst_plot_utils.py` | Shared plotting utilities used by `analyst_data.ipynb`. | `build_figures`, `find_missing_entries` |

### CLI / batch scripts

Run these modules with `python -m …` to batch-create JSON summaries:

| Module | Description | Example |
| --- | --- | --- |
| `python/read_file_large.py` | Computes expert-switch fractions across checkpoints for all vision layers. | `python -m evaluate.analysis.libmoev2.python.read_file_large --datasets mme mmstar --models Full_smoe Full_xmoe` |
| `python/read_file_large_mm_projectors.py` | Same as above but constrained to the `mm_projector` layer (defaults to the 1M2 analyst root). | `python -m evaluate.analysis.libmoev2.python.read_file_large_mm_projectors --datasets mme` |
| `python/read_file_large_saturation.py` | Compares each checkpoint against the final checkpoint to measure saturation (position matches). | `python -m evaluate.analysis.libmoev2.python.read_file_large_saturation --datasets mmstar` |

### Notebook runner modules

Each notebook in this folder is now a two-cell front-end that imports a runner
module and calls `run()`. This keeps notebooks declarative while allowing CLI
usage:

| Notebook | Runner module | Generates |
| --- | --- | --- |
| `diversity.ipynb` | `python/diversity_module.py` | Diversity losses across layers/datasets. |
| `drop_top1.ipynb` | `python/drop_top1_module.py` | Score deltas from removing top experts (PDF/PNG charts). |
| `entropy_analyst.ipynb` | `python/entropy_analyst_module.py` | Entropy trend plots and CSV summaries. |
| `init_gate.ipynb` | `python/init_gate_module.py` | Gate initialisation diagnostics. |
| `init_weight.ipynb` | `python/init_weight_module.py` | Balance-loss visualisations from W&B exports. |
| `progress_changes_selected.ipynb` | `python/progress_changes_selected_module.py` | Expert-switch progression over checkpoints. |
| `progress_changes_selected_pretrain.ipynb` | `python/progress_changes_selected_pretrain_module.py` | Same as above for the pre-train sweep. |
| `visual_data.ipynb` | `python/visual_data_module.py` | Dataset composition / sunburst figures. |
| `analyst_data.ipynb` | `python/analyst_plot_utils.py` | Multi-panel accuracy / rank trends (PDF + 1×2 summary). |
| `convert_entropy.ipynb` | `python/router_metrics.py` | Writes `router_entropy_*.json` into `entropy_top3/`. |
| `convert_margin.ipynb` | `python/router_metrics.py` | Writes `router_margin_*.json` into `router_margin/`. |

Programmatic usage example:

```python
from evaluate.analysis.libmoev2.python import diversity_module

diversity_module.run()
```

## Working with refactored code

1. **Keep `constants.py` authoritative.** Update paths (e.g., new checkpoints) in
   one place and every CLI/notebook will follow.
2. **Reuse `selection_pipeline.analyse_datasets`** for any workflow that consumes
   `selected_experts` logs. It already writes processed samples, aggregates by
   checkpoint, and computes switch metrics.
3. **Cache outputs under the provided folders.** CLI scripts will overwrite JSONs
   in `entropy_top3/` or `router_margin/`. Commit only curated artefacts.
4. **Backwards compatibility:** The top-level package re-exports modules so
   legacy imports like `from evaluate.analysis.libmoev2 import selection_pipeline`
   still work, but new code should import from the explicit `python` namespace.

By adhering to this structure we keep the analysis stack reproducible, auditable,
and easy to extend while making clear that the implementation has diverged from
the original research release.
