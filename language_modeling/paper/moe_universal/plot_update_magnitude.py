import lib
import numpy as np
import matplotlib.pyplot as plt


run = lib.get_runs(["c4_moeut_small_matched_rope_noln_long"])[0]
n_layers = run.config["transformer.encoder_n_layers"]

abs_updates = [run.summary[f"activation_norm/abs_update_layer_{i}"] for i in range(0, n_layers)]

plt.figure(figsize=(4, 2))
plt.bar(range(n_layers), abs_updates)
plt.xlabel("Layer")
plt.ylabel("Update magnitude")
plt.tight_layout()
plt.savefig("update_magnitude.pdf")


in_norms = [run.summary[f"activation_norm/in_layer_{i}"] for i in range(0, n_layers)]

plt.figure(figsize=(4, 2))
plt.bar(range(n_layers), in_norms)
plt.xlabel("Layer")
plt.ylabel("Residual norm")
plt.xlim(0, n_layers)
plt.tight_layout()
plt.savefig("in_norm.pdf")