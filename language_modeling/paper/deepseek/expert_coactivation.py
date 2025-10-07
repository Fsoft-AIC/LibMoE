import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from matplotlib.colors import LinearSegmentedColormap

from typing import Any, List


# helper function
def top_k_numpy(logits: np.ndarray, top_k: int, dim: int) -> np.ndarray:
    """Implementation of top_k in pytorch using numpy's partition function.
    Args:
        logits:  shape (L, T, N) — L layers, T tokens, N experts
        top_k:   K, number of experts selected per token
        dim:     dimension to perform top_k on
    Returns:
        top_k_index:  shape (L, T, N) - values at top_k positions different from 0, other is 0
        top_k_logits: shape (L, T, N) - values at top_k positions are the original values, other is 0
    """
    top_k_index = np.zeros_like(logits)
    top_k_index[np.arange(logits.shape[0])[:, None, None], np.arange(logits.shape[1])[None, :, None], np.argpartition(logits, -top_k, axis=dim)[:, :, -top_k:]] = 1
    
    top_k_value = logits * top_k_index

    return top_k_index, top_k_value


def extract_coactivation(logits: np.ndarray, top_k: int = 8) -> np.ndarray:
    n_layers, n_tokens, n_experts = logits.shape
    result = np.zeros((n_layers, n_experts, n_experts))
    
    top_k_index, top_k_value = top_k_numpy(logits, top_k, dim=-1)
    result = np.matmul(top_k_index.transpose(0, 2, 1), top_k_index)
    return result


def extract_coactivation_from_folder(logits_folder: str, top_k: int = 8) -> np.ndarray:
    logits_list_path = glob.glob(os.path.join(logits_folder, "*.npy"))
    logits_list_path.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
    
    coactivation_list = []
    for path in logits_list_path:
        logits = np.load(path)
        coactivation = extract_coactivation(logits, top_k)
        coactivation_list.append(coactivation)
    return np.array(coactivation_list)

def plot_coactivation(coactivation: np.ndarray, output_dir: str, top_k_plot: int = 15) -> None:
    cmap = LinearSegmentedColormap.from_list("white_to_tartan", ["white", "#0072B2"])
    
    for checkpoint_step in range(coactivation.shape[0]):
        for layer in range(coactivation.shape[1]): # num_layer
            expert_coactivation_layer = coactivation[checkpoint_step][layer]
            expert_coactivation_without_diag = expert_coactivation_layer - np.diag(np.diag(expert_coactivation_layer))
            
            # top_k_index and top_k_value of expert_coactivation_without_diag.sum(0)
            top_k_index = np.argsort(expert_coactivation_without_diag.sum(0))[- top_k_plot:]
            top_k_value = expert_coactivation_without_diag[top_k_index][:, top_k_index]
            
            plt.figure(figsize=(10, 10))
            plt.imshow(top_k_value, cmap=cmap)
            plt.xticks(range(top_k_plot), top_k_index)
            plt.yticks(range(top_k_plot), top_k_index)
            # increase font size of xticks and yticks
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            # save fig
            
            os.makedirs(os.path.join(output_dir, f"step_{checkpoint_step}"), exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"step_{checkpoint_step}", f"layer_{layer}.pdf"), format='pdf', bbox_inches='tight')
            plt.close()
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run tests with specified parameters.")
    parser.add_argument('--logits_folder', type=str, required=True, help='The folder containing the logits.')
    parser.add_argument('--top_k', type=int, default=8, help='The number of experts to keep.')
    parser.add_argument('--save_folder', type=str, required=True, help='The folder to save the coactivation matrix.')
    args = parser.parse_args()
    
    expert_coactivation = extract_coactivation_from_folder(args.logits_folder)
    last_2save_dir = "/".join(args.logits_folder.split("/")[-2:])
    print(last_2save_dir)
    output_dir_logit = os.path.join(args.save_folder, last_2save_dir)
    output_dir_plot = os.path.join(args.save_folder, "figs", last_2save_dir)
    
    if not os.path.exists(output_dir_logit):
        os.makedirs(output_dir_logit)
    if not os.path.exists(output_dir_plot):
        os.makedirs(output_dir_plot)
    np.save(os.path.join(output_dir_logit, "coactivation.npy"), expert_coactivation)
    plot_coactivation(expert_coactivation, output_dir_plot, 15)
