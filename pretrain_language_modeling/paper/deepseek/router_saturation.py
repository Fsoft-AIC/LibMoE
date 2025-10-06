import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'
# os.chdir("../../")  # Pretend we are in the main directory
os.environ["MOE_TYPE"] = "moe_layer_deepseek"
os.environ.get("MOE_TYPE", "moe_layer")

from main import initialize
import torch
import glob
import torch.nn.functional as F
from layers import MoE
import matplotlib.pyplot as plt
import pickle
import numpy as np

print(MoE)

checkpoint_dir = "/cm/archive/anonymous/moeut_training_code/save/slimpajama_moe_no_attmoe_660M_standardlb_deepseek_shared_only/checkpoint"
checkpoint_list_file = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
# checkpoint_list_file.sort()

for checkpoint_file_path in checkpoint_list_file:
    # Ensure no W&B logging will be performed
    print("Process:",checkpoint_file_path)
    sys.argv = f"main.py -log tb -name tst -reset 1 -lm.eval.enable 0 -log tb -batch_size 20 -restore {checkpoint_file_path}".split(" ")

    output_path = f"/cm/archive/anonymous/moeut_training_code/paper/deepseek/router_saturation/smoe_shared/{sys.argv[-1].split('/')[-1].split('.')[0]}.npy"
    if os.path.exists(output_path):
        continue

    helper, task = initialize()
    task.create_data_fetcher()

    orig_run_model_valid = task.run_model_validation

    nexp = task.helper.args.moe.n_experts
    ntok = task.helper.args.sentencepiece.n_pieces
    nlayers = task.helper.args.transformer.encoder_n_layers
    ngrp = 18

    token_counts = 0

    counts = torch.zeros(ngrp, nlayers // ngrp, nexp, ntok)

    print(counts.size())  # torch.Size([16, 1, 66, 8000])

    global this_data

    def run_model_validation(self, data):
        global token_counts
        global this_data

        token_counts = token_counts + F.one_hot(data["data"].flatten().long(), ntok).sum(0)

        this_data = data
        return orig_run_model_valid(data)

    task.run_model_validation = run_model_validation.__get__(task)
    print(task)

    id_map = {}

    def patch_module(module):
        myid = id(module)
        if myid in id_map:
            return

        gid = len(id_map)
        id_map[myid] = gid

        # sel_val, sel_index = self.topk(
        def new_topk(self, *args, **kwargs):
            nonlocal gid
            global this_data
            data = this_data["data"][:-1].T

            sel_val, sel_index = MoE.topk(self, *args, **kwargs)

            assert data.shape == sel_index.shape[:-1]

            data = data.reshape(-1)

            # Shape of counts[gid]: nexp, ntok
            # Linear index: expert * ntok + tok

            seli = sel_index.flatten(end_dim=-2) * ntok
            addi = seli + data[..., None]
            addi = addi.flatten().cpu()
            # breakpoint()
            counts[gid][self.layer // ngrp].flatten().index_add_(0, addi, torch.ones_like(addi, dtype=torch.float32))

            return sel_val, sel_index

        module.topk = new_topk.__get__(module)

    for m in task.model.modules():
        if isinstance(m, MoE):
            patch_module(m)

    validation_results = task.validate()
    print(validation_results)

    order = torch.argsort(token_counts, descending=True).cpu()
    token_counts_o = token_counts.cpu()[order]
    counts_o = counts[:, :, :, order]

    save_experts_assign = np.zeros((counts_o.size(0), counts_o.size(3), counts_o.size(2)))

    for layer in range(counts_o.size(0)):
        counts_o_layer = counts_o[layer, 0, :, :]
        counts_o_layer = counts_o_layer / counts_o_layer.sum(0, keepdim=True)
        counts_o_layer = counts_o_layer.T

        # get top_k
        score, index = counts_o_layer.topk(6, dim=-1, sorted=False)
        score = score / (score.sum(dim=-1, keepdim=True) + 1e-20)
        counts_o_layer_topk = torch.zeros_like(counts_o_layer)
        counts_o_layer_topk = counts_o_layer_topk.scatter(-1, index, score)

        save_experts_assign[layer] = counts_o_layer_topk.numpy()

    # Save the results
    np.save(output_path, save_experts_assign)

    # end the process
    print(f"Finished processing {checkpoint_file_path}")
    print(f"Results saved to {output_path}")

    # Clean up to prevent memory leaks
    del task
    del helper

    # # Load and verify the saved data
    # loaded_data = np.load("/cm/shared/anonymous/moeut_training_code/paper/deepseek/router_saturation/smoe/model-10000.npy")
    # print(loaded_data.shape)