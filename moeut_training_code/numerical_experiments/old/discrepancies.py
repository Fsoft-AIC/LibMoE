import numpy as np
import sys
import itertools


rbar = [0,1,4,6] # cf. Lemma 1 in main text.

# Exact-fitted Settings
def loss_SGaME_exact(params, true_params, K):
    beta1, beta0, a, b, sigma = params
    n_components, n_features = beta1.shape 
    beta1_true, beta0_true, a_true, b_true, sigma_true = true_params
    n_components_true, n_features = beta1_true.shape 
    
    D = get_dist(params, true_params) # (n_component, n_component_true)
    
    vor = get_voronoi(D)

    subsets = list(itertools.combinations(range(n_components), K))
    loss_list = []
    for subset in subsets:
        loss_list.append(compute_loss(vor, D, subset, beta0, beta0_true))

    return max(loss_list)


def compute_loss(vor, D, indices, beta0, beta0_true):
    n_components = beta0.shape[0]
    unique, counts = np.unique(vor, return_counts=True)
    loss = 0
    for i in indices:
            j = vor[i]
            if counts[vor[i]] == 1:
                loss += np.exp(beta0[i]) * D[i, vor[i]]
    
    for j in indices:
        exp_beta0 = 0

        for i in range(n_components):
            if vor[i] == j:
                exp_beta0 += np.exp(beta0[i])

        loss += np.abs(exp_beta0 - np.exp(beta0_true[j]))

    return loss

def get_voronoi(D):
    n_components, n_components_true = D.shape
    vor=[]
    for i in range(n_components):
        for j in range(n_components_true):
            if D[i,j] == np.min(D[i,:]):
                vor.append(j)
    return vor

def get_dist(params, true_params):
    beta1, beta0, a, b, sigma = params
    n_components, n_features = beta1.shape 
    beta1_true, beta0_true, a_true, b_true, sigma_true = true_params
    n_components_true, n_features = beta1_true.shape 

    D = np.empty((n_components, n_components_true))
    for i in range(n_components):
        for j in range(n_components_true):
            D[i, j] = gauss_dist_SGaME(beta1[i], a[i], b[i], sigma[i],\
                beta1_true[j], a_true[j], b_true[j], sigma_true[j])
    return D

def gauss_dist_SGaME(beta1, a, b, sigma, beta1_true, a_true, b_true, sigma_true):
    return np.linalg.norm(beta1 - beta1_true) + np.linalg.norm(a - a_true) \
            + np.linalg.norm(b - b_true) + np.linalg.norm(sigma - sigma_true)


