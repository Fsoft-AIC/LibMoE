import os
import sys
import lib
import matplotlib as mpl

# Disable LaTeX rendering
mpl.rcParams['text.usetex'] = False

# Ensure no W&B logging will be performed
sys.argv = "main.py -log tb -name tst -reset 1 -lm.eval.enable 0 -log tb -batch_size 16 -restore /cm/shared/anonymous/moeut_training_code/tmp/smoe-sigmoid-60000.pth".split(" ")

# Pretend we are in the main directory
sys.path.append("../../")

from main import initialize
import torch
import torch.nn.functional as F
from layers import MoE
from matplotlib import pyplot as plt

os.chdir("../../")

helper, task = initialize()

count_logs = {}

def patch_module(module):
    myid = id(module)
    if myid in count_logs:
        return

    count_logs[myid] = {}

    old_forward = module.forward
    def new_forward(self, x):
        index_sel_counts = self.index_sel_counts
        res = old_forward(x)
        this_count = self.index_sel_counts - index_sel_counts

        if self.layer not in count_logs[myid]:
            count_logs[myid][self.layer] = 0

        count_logs[myid][self.layer] += this_count
        return res

    module.forward = new_forward.__get__(module)

for m in task.model.modules():
    if isinstance(m, MoE):
        patch_module(m)

task.validate()

order = list(sorted(count_logs.keys(), key=lambda x: min(count_logs[x].keys())))

def plot_group(gid):
    n_experts = count_logs[order[0]][1].shape[0]

    counts = torch.zeros(len(count_logs[order[gid]]), n_experts)
    order2 = list(sorted(count_logs[order[gid]].keys()))
    for j, o in enumerate(order2):
        counts[j] += count_logs[order[gid]][o].cpu()

    total_counts = counts.float() / counts.sum(0, keepdim=True)
    tresh = torch.quantile(total_counts, 0.9, dim=0)
    total_counts = total_counts / total_counts.sum(0, keepdim=True)
    total_counts = total_counts * (total_counts > tresh)
    total_counts = total_counts * torch.arange(counts.shape[0], dtype=torch.float)[..., None]
    total_counts = total_counts.sum(0)
    order3 = total_counts.argsort(descending=False)

    counts2 = counts[:, order3]

    from matplotlib.colors import LogNorm
    counts2 = counts2.float() / counts2.sum(0, keepdim=True)
    fig, ax=plt.subplots(figsize=(4, 2))
    plt.imshow(counts2.cpu().numpy(), aspect='auto', cmap='viridis', interpolation="nearest")
    plt.ylabel("Layer")
    plt.xlabel("Expert ID")
    plt.yticks(range(len(counts2)), order2)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"paper/moe_universal/expert_layer_g{gid}.pdf", bbox_inches='tight', dpi=300)
    return fig

for g in range(len(order)):
    fig = plot_group(g)
    plt.close(fig)
