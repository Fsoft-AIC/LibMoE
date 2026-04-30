import lib
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


runs = [
    "c4_moeut_big_matched_rope_preln",
    "c4_moeut_big_matched_rope_preln_linear_norot_nosm1",
    "c4_moeut_big_matched_rope_postln",

    "c4_baseline_small_rope_long_nodrop",
    "c4_moeut_small_matched_rope_noln_long",
    "c4_moeut_small_matched_rope_postln"
]


runs = lib.get_runs(runs)

cgroups = [
    {"size": 44, "task": "c4_transformer", "name": "44M"},
    {"size": 244, "task": "c4_transformer", "name": "244M"},
]

model_order = [
    {"moe.nonorm": False, "transformer.variant": lambda x: "preln" not in x},
    {"moe.nonorm": False, "transformer.variant": lambda x: "preln" in x},
    {"moe.nonorm": True, "transformer.variant": lambda x: "preln" in x},
]

labels = [
    "PostLN",
    "PreLN",
    "PeriLN",
]

defaults = {
    "moe.nonorm": False
}

def get_config(run, name):
    if name in run.config:
        return run.config[name]
    elif name in defaults:
        return defaults[name]
    else:
        raise ValueError("Invalid config", name)

grouped_runs = []
run_types = []
ordered_runs = []

for cg in cgroups:
    thisg = []
    cgruns = [r for r in runs if r.config["task"] == cg["task"] and abs(r.summary["n_params"]/1e6 - cg["size"])<10]

    for im, mod in enumerate(model_order):
        thisr = [r for r in cgruns if all(get_config(r,k)==v if not callable(v) else v(get_config(r,k)) for k, v in mod.items())]

        if len(thisr) == 0:
            pass
        elif len(thisr) == 1:
            thisg.append(thisr[0])
            ordered_runs.append(thisr[0])
            run_types.append(im)
        else:
            raise ValueError("Multiple runs found for ", cg, mod)

    grouped_runs.append(thisg)

gsizes = [len(g) for g in grouped_runs]

indices = []
centroids = []
curri = 0
for gs in gsizes:
    centroids.append(curri + (gs-1) / 2)
    indices += list(range(curri, curri + gs))
    curri += gs + 1


fig, ax=plt.subplots(figsize=(4, 2))
for i in range(len(model_order)):
    x = [indices[j] for j in range(len(ordered_runs)) if run_types[j] == i]
    y = [ordered_runs[j].summary["validation/val/perplexity"] for j in range(len(ordered_runs)) if run_types[j] == i]

    plt.bar(x, y, label=labels[i])

plt.legend(ncol=1)
plt.xticks(centroids, [cg["name"] for cg in cgroups])

ax.set_yscale('log')

from matplotlib.ticker import ScalarFormatter
formatter = ScalarFormatter()
formatter.set_scientific(False)
plt.gca().yaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_minor_formatter(formatter)

# plt.minorticks_off()

# plt.ylim(8,45)
# plt.yticks([8,10,20, 30, 40],[8,10,20, 30, 40])
plt.ylabel("Perplexity")
plt.tight_layout()
plt.savefig("layernorm.pdf")
