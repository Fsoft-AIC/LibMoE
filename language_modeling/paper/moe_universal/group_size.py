import lib
import numpy as np
import matplotlib.pyplot as plt


runs = [
    "c4_moeut_big_matched_rope_noln_g1",
    "c4_moeut_big_matched_rope_preln_linear_norot_nosm1",
    "c4_moeut_big_matched_rope_noln_g3",
    "c4_moeut_big_matched_rope_noln_g6",
    "c4_moeut_big_matched_rope_noln_g9",
    "c4_moeut_big_matched_rope_noln_g18",
    "c4_moeut_big_matched_rope_noln_aabb",
]


runs = lib.get_runs(runs)
runs = list(sorted(runs, key=lambda r: (int(r.config["transformer.variant"] == "preln_rope"), int(r.config["transformer.universal.group_type"]!="aabb"), r.config["transformer.universal.group_size"])))

def get_label(r):
    if r.config["transformer.variant"] == "preln_rope":
        return "Baseline"

    res = f"G={r.config['transformer.universal.group_size']}"
    if  r.config['transformer.universal.group_type'] == 'aabb':
        res = "AABB"
    return res

from matplotlib.ticker import ScalarFormatter

fig, ax = plt.subplots(figsize=(4, 2))
plt.bar(range(len(runs)), [r.summary["validation/val/perplexity"] for r in runs])
plt.xlabel("Variant")
plt.ylabel("Perplexity")
ax.set_yscale('log')
formatter = ScalarFormatter()
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)
ax.yaxis.set_minor_formatter(formatter)
plt.xticks(range(len(runs)), [get_label(r) for r in runs])
plt.tight_layout()
plt.savefig("group_ppl.pdf")


