import lib
import numpy as np
import matplotlib.pyplot as plt

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


runs = {
    "Baseline": [r for r in runs if r.config["transformer.variant"] == "preln_rope"],
    "MoEUT": [r for r in runs if r.config["transformer.universal.group_size"] < 10 and r.config["transformer.variant"] in {"preln_moe_universal", "moe_universal"}],
    "$\\sigma$-MoE": [r for r in runs if r.config["transformer.universal.group_size"] > 10 and r.config["transformer.variant"] in {"preln_moe_universal", "moe_universal"}],
}

n_params = {
    k: [r.summary["n_model_weights"]/1e6 for r in v] for k, v in runs.items()
}


plt.figure(figsize=(4, 2))

plt.ticklabel_format(axis='y', style='plain')

# Calculate intersection point between the last baseline perplexity and curve of the MoEUT model
order = np.argsort(n_params["Baseline"])
target_ppl = runs["Baseline"][order[-1]].summary["validation/val/perplexity"]

order = np.argsort(n_params["MoEUT"])
pt_before = 0
for i in range(1, len(order)):
    if runs["MoEUT"][order[i]].summary["validation/val/perplexity"] > target_ppl:
        pt_before = i
        break

x1 = n_params["MoEUT"][order[pt_before]]
x2 = n_params["MoEUT"][order[pt_before+1]]

y1 = runs["MoEUT"][order[pt_before]].summary["validation/val/perplexity"]
y2 = runs["MoEUT"][order[pt_before+1]].summary["validation/val/perplexity"]

x = x1 + (x2 - x1) * (target_ppl - y1) / (y2 - y1)



# Plot lines

for name, thisr in runs.items():
    order = np.argsort(n_params[name])
    x = [n_params[name][i] for i in order]
    y = [thisr[i].summary["validation/val/perplexity"] for i in order]

    plt.plot(x, y, label=name, marker="o")


from matplotlib.ticker import ScalarFormatter

plt.legend()
plt.xlabel("Number of model parameters (M)")
plt.ylabel("Perplexity")
plt.yscale("log")
# plt.xscale("log")
formatter = ScalarFormatter()
formatter.set_scientific(False)
plt.gca().yaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_minor_formatter(formatter)
plt.tight_layout()
plt.savefig("scaling.pdf")

