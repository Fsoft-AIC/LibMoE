import lib
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# SUT LOWREG 2 problem

runs = lib.get_runs([
    "thestack_baseline_big_rope_nodrop",
    "thestack_moeut_big_matched_rope_noln",

    "thestack_moeut_gigantic_switchead_multilang",
    "thestack_baseline_gigantic_rope_nodrop_multilang",
])

cgroups = [
    {"size": 244, "name": "244M"},
    {"size": 728, "name": "728M"},
]

model_order = [
    "preln_rope",
    "preln_moe_universal",
]

labels = [
    "Baseline",
    "MoEUT",
]

grouped_runs = []
run_types = []
ordered_runs = []

for cg in cgroups:
    thisg = []
    cgruns = [r for r in runs if abs(r.summary["n_params"]/1e6 - cg["size"])<10]

    for im, mod in enumerate(model_order):
        thisr = [r for r in cgruns if r.config["transformer.variant"]==mod]

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

# plt.legend(ncol=3)
plt.legend()
plt.xticks(centroids, [cg["name"] for cg in cgroups])

ax.set_yscale('log')
# plt.minorticks_off()


from matplotlib.ticker import ScalarFormatter
formatter = ScalarFormatter()
formatter.set_scientific(False)
plt.gca().yaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_minor_formatter(formatter)

# plt.ylim(8,45)
# plt.yticks([8,10,20, 30, 40],[8,10,20, 30, 40])
plt.ylabel("Perplexity")
plt.xlabel("Number of parameters")
plt.tight_layout()
plt.savefig("code.pdf")
