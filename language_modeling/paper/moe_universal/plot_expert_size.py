import lib
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

runs = [
    "c4_moeut_big_matched_rope_preln_linear_norot_nosm1",
    "c4_moeut_big_matched_rope_noln_es256",
    "c4_moeut_big_matched_rope_noln_es512",
    "c4_moeut_big_matched_rope_noln_es1024"
]


runs = lib.get_runs(runs)
runs = list(sorted(runs, key=lambda r: int(r.config["moe.expert_size"])))

fig, ax=plt.subplots(figsize=(4, 2))
plt.bar(range(len(runs)), [r.summary["validation/val/perplexity"] for r in runs])
plt.xticks(range(len(runs)), [r.config["moe.expert_size"] for r in runs])
plt.xlabel("$d_{expert}$")
plt.ylabel("Perplexity")
ax.set_yscale('log')

from matplotlib.ticker import ScalarFormatter
formatter = ScalarFormatter()
formatter.set_scientific(False)
plt.gca().yaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_minor_formatter(formatter)
plt.tight_layout()
plt.savefig("moe_expert_size.pdf", bbox_inches='tight')
