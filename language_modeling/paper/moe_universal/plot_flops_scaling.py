import lib
import numpy as np
import matplotlib.pyplot as plt
from common import get_train_flops


runs = lib.get_runs([
    "c4_baseline_big_rope_nodrop", "c4_moeut_big_matched_rope_preln_linear_norot_nosm1",
    "c4_baseline_bigger_rope_nodrop", "c4_moeut_bigger_matched_rope_nonorm_g3",
    "c4_baseline_small_rope_long_nodrop", "c4_moeut_small_matched_rope_noln_long",
    "c4_moeut_gigantic_switchead", "c4_baseline_gigantic_rope_nodrop",
    "c4_moeut_mid_matched_rope_noln", "c4_baseline_mid_rope_nodrop",

    "c4_moeut_big_matched_rope_noln_g18", "c4_moeut_mid_matched_rope_noln_g18",
    "c4_moeut_bigger_matched_rope_nonorm_g24", "c4_moeut_gigantic_switchead_g36",
    "c4_moeut_small_matched_rope_noln_long_g16",

    "c4_baseline_1b_rope_nodrop", "c4_moeut_1b_switchead"
])


for r in runs:
    assert r.config["dropout"] == 0

runs = {
    "Baseline": [r for r in runs if r.config["transformer.variant"] == "preln_rope"],
    "MoEUT": [r for r in runs if r.config["transformer.universal.group_size"] < 10 and r.config["transformer.variant"] in {"preln_moe_universal", "moe_universal"}],
    "$\\sigma$-MoE": [r for r in runs if r.config["transformer.universal.group_size"] > 10 and r.config["transformer.variant"] in {"preln_moe_universal", "moe_universal"}],
}

flops = {
    k: [get_train_flops(r) for r in v] for k, v in runs.items()
}


plt.figure(figsize=(4, 2))

plt.ticklabel_format(axis='y', style='plain')


for name, thisr in runs.items():
    order = np.argsort(flops[name])
    x = [flops[name][i] for i in order]
    y = [thisr[i].summary["validation/val/perplexity"] for i in order]

    plt.plot(x, y, label=name, marker="o")


from matplotlib.ticker import ScalarFormatter

plt.legend()
plt.xlabel("Total training MACs")
plt.ylabel("Perplexity")
plt.yscale("log")
# plt.xscale("log")
formatter = ScalarFormatter()
formatter.set_scientific(False)
plt.gca().yaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_minor_formatter(formatter)
plt.tight_layout()
plt.savefig("scaling_flops.pdf")

