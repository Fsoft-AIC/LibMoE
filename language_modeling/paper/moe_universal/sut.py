import lib
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# SUT LOWREG 2 problem

runs = lib.get_runs([
    # "c4_small_sut_lowreg2_fixed_act_lowloss", # Small with ACT
    # "c4_small_sut_lowreg2_fixed",
    "c4_small_sut_lowreg2_fixed_44m",

    "c4_sut_big_lessloss2_ffflopmatch_attmatch_actloss", # Big, ACT
    # "c4_sut_big_lessloss2_ffflopmatch_attmatch", #Big, NO ACT

    # All PES2O big
    # "pes2o_sut_big_lessloss2_ffflopmatch_attmatch", # Big, NO ACT
    "pes2o_sut_big_lessloss2_ffflopmatch_attmatch_actloss", # Big, ACT

    # All PES2O small
    "pes2o_small_sut_lowreg2_fixed_44m", # Small, both

    "c4_baseline_small_rope_long_nodrop", "c4_moeut_small_matched_rope_noln_long",
    "c4_baseline_big_rope_nodrop", "c4_moeut_big_matched_rope_preln_linear_norot_nosm1",
    "pes2o_baseline_small_rope_long_nodrop", "pes2o_moeut_small_matched_rope_noln_long",
    "pes2o_baseline_big_rope_nodrop", "pes2o_moeut_big_matched_rope_noln"
])

cgroups = [
    {"size": 44, "task": "c4_transformer", "name": "C4, 44M"},
    {"size": 244, "task": "c4_transformer", "name": "C4, 244M"},
    {"size": 44, "task": "pes2o_transformer", "name": "peS2o, 44M"},
    {"size": 244, "task": "pes2o_transformer", "name": "peS2o, 244M"},
]

model_order = [
    "preln_rope",
    "preln_moe_universal",
    # "sut_universal",
    "actsut_universal"
]

labels = [
    "Baseline",
    "MoEUT",
    # "SUT w.o. ACT",
    "SUT"
]

grouped_runs = []
run_types = []
ordered_runs = []

for cg in cgroups:
    thisg = []
    cgruns = [r for r in runs if r.config["task"] == cg["task"] and abs(r.summary["n_params"]/1e6 - cg["size"])<10]

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


fig, ax=plt.subplots(figsize=(5, 2))
for i in range(len(model_order)):
    x = [indices[j] for j in range(len(ordered_runs)) if run_types[j] == i]
    y = [ordered_runs[j].summary["validation/val/perplexity"] for j in range(len(ordered_runs)) if run_types[j] == i]

    plt.bar(x, y, label=labels[i])

plt.legend(ncol=3)
plt.xticks(centroids, [cg["name"] for cg in cgroups])

ax.set_yscale('log')
plt.minorticks_off()

plt.ylim(8,45)
plt.yticks([8,10,20, 30, 40],[8,10,20, 30, 40])
plt.ylabel("Perplexity")
plt.tight_layout()
plt.savefig("sut.pdf")
