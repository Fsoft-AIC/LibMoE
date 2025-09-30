import numpy as np
# from functions import *
import sys
import multiprocessing as mp
from multiprocessing import Pool, get_context # Work on MacBook Air (M1, 2020, 8 cores).
import time 
import datetime
import logging
# from gllim import GLLiM
from topk_gmoe import *

import models

start_time = time.time() # Calculate the runtime of a programme in Python.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default=1, type=int, help='Type number.')
parser.add_argument('-nc', '--n_components', default=2, type=int, help='Number of mixture components.')
parser.add_argument('-K', '--topK', default=2, type=int, help='Top K.')
parser.add_argument('-r' ,'--reps', default=10, type=int, help='Number of replications per sample size.')
parser.add_argument('-np','--nproc', default=8, type=int, help='Number of processes to run in parallel.') # Work on MacBook Air (M1, 2020, 8 cores)
parser.add_argument('-mi','--maxit', default=2000, type=int, help='Maximum EM iterations.')
parser.add_argument('-e', '--eps', default=1e-6, type=float, help='EM stopping criterion.')
parser.add_argument('-nnum', '--n_num', default=200, type=int, help='Number of different choices of sample size.')
parser.add_argument('-nsmax', '--ns_max', default=1000, type=int, help='Number of sample size maximum.')
parser.add_argument('-nsmin', '--ns_min', default=100, type=int, help='Number of sample size maximum.')
parser.add_argument('-errorbar', '--errorbar', default=0, type=int, help='Number of trials for emNMoE of meteoritsSim package.')
parser.add_argument('-verbose', '--verbose', default=0,\
                    type=int, help='Log-likelihood should be printed or not during EM iterations.')

args = parser.parse_args()

print(args)




model = args.model                    # Type number
n_components = args.n_components                            # Number of mixture components
n_proc = args.nproc                   # Number of cores to use
reps  = args.reps                      # Number of replications to run per sample size
max_iter = args.maxit                 # Maximum EM iterations
eps = args.eps                        # EM Stopping criterion.
n_num = args.n_num                    # Number of different choices of sample size.
ns_max = args.ns_max                    # Number of sample size maximum.
ns_min  = args.ns_min                    # Number of sample size minimum.
errorbar = args.errorbar  
verbose = args.verbose                    


from models import *
import os

logging.basicConfig(
    filename=
    'std_mod' + str(model) + '_k' + str(n_components) +\
                    '.log', filemode='w', format='%(asctime)s %(message)s'
)
    
n_components_true = 2

if (n_components_true == n_components):
    ns = np.concatenate([np.linspace(ns_min, 9*ns_min, 5), np.linspace(10*ns_min, ns_max, n_num-5)])
else:
    ns = np.linspace(ns_min, ns_max, n_num)

## Test.
## Test for 100 different choices of n between 10^2 and 10^5 on MacBook Air (M1, 2020, 8 cores). 
# ns = np.concatenate([np.linspace(100, 1000, 20), np.linspace(1000, ns_max, n_num-20)])
# ns = np.concatenate([np.linspace(ns_min, 9*ns_min, 5), np.linspace(10*ns_min, ns_max, n_num-5)])
# ns = np.concatenate([np.linspace(ns_min, 9*ns_min, 10), np.linspace(10*ns_min, ns_max, n_num-10)])
# ns = np.linspace(ns_min, ns_max, n_num)
# ns = np.concatenate([np.linspace(100, 900, 5), np.linspace(1000, 100000, n_num-5)]) 
# ns = np.concatenate([np.linspace(100, 900, 5), np.linspace(1000, 100000, n_num-5)]) 
# print(ns)
# print("Chose Model " + str(model))
# print(model)


# Main EM algorithm.

def process_chunk_SGaME(bound):
    """ Run EM on a range of sample sizes. """
    ind_low = bound[0]
    ind_high= bound[1]

    m = ind_high - ind_low

    seed_ctr = 2023 * ind_low   # Random seed

    betas = np.array([[-8,0], [25,0]]).T # [(d+1)xK]
    As = np.array([[15, -5], [-20,20]]).T # [(d+1)xK]
    Sigmas  = np.array([0.3, 0.4]) # [1xK]
    
    n_features = 1

    true_params = (
    betas[:, 1].reshape(-1, n_features),
    betas[:, 0].reshape(-1, n_features), 
    As[:, 1].reshape(-1, n_features), 
    As[:, 0].reshape(-1, n_features), 
    Sigmas
    )
    
    chunk_weights_coef = np.empty((m, reps, n_components, n_features))
    chunk_weights_intercept = np.empty((m, reps, n_components))
    chunk_means_coef = np.empty((m, reps, n_components, n_features))
    chunk_means_intercept = np.empty((m, reps, n_components))
    chunk_variances = np.empty((m, reps, n_components))    

    
    for i in range(ind_low, ind_high):
        n_samples = int(ns[i])
        print("n_samples =", n_samples)
        for rep in range(reps):
            print("\t rep =", rep)
                
            # Sample from the mixture. 
            X, y = sample_topKSGaME(
                n_samples=n_samples, 
                true_params=true_params, 
                true_topK=1, seed = 0
            )

            np.random.seed(seed_ctr + 1)
            params = init_params(true_params=true_params, n_samples=n_samples)
            params_init = {
                'weights_coef_': params[0],
                'weights_intercept_': params[1],
                'means_coef_': params[2],
                'means_intercept_': params[3],
                'variances_': params[4]
            }


            # Using em_SGaME via a partition of starting values near the true components.
            result = TopKSGaME(
                n_components=2, 
                topK=1, 
                params_init_dict=params_init
            ).fit(X, y)._get_parameters()
            
            logging.warning(
                'Model ' + str(model) + ', rep:' + str(rep) +\
                            ', n:' + str(n_samples) + ", nind:" + str(i)
                )

            chunk_weights_coef[i-ind_low, rep, :, :] = result[0]
            chunk_weights_intercept[i-ind_low, rep, :] = result[1]   
            chunk_means_coef[i-ind_low, rep, :, :] = result[2]
            chunk_means_intercept[i-ind_low, rep, :] = result[3]
            chunk_variances[i-ind_low, rep, :] = result[4]    
            # chunk_iters[i-ind_low, rep]             = out[3]   

            seed_ctr += 1

    return (
        chunk_weights_coef, 
        chunk_weights_intercept, 
        chunk_means_coef, 
        chunk_means_intercept,
        chunk_variances
    )


# Multiprocessing.

proc_chunks_SGaME = []

## Uniform distribution.
Del = n_num // n_proc 

for i in range(n_proc):
    if i == n_proc-1:
        proc_chunks_SGaME.append(( (n_proc-1) * Del, n_num) )

    else:
        proc_chunks_SGaME.append(( (i*Del, (i+1)*Del ) ))

if n_proc == 1: # For quick test.
    proc_chunks_SGaME = [(50, 51)]

# elif ((n_proc == 8) & (n_num == 10)): # 8 Cores n_num = 100
#     proc_chunks_SGaME = [(91, 96), (96, 100)] # 8 Cores for 100 different choices of n.

elif ((n_proc == 8) & (n_num == 100)): # 8 Cores n_num = 100
    proc_chunks_SGaME = [(0, 25), (25, 40), (40, 55), (55, 70), (70, 82), (82, 90),\
                    (90, 96), (96, 100)] # 8 Cores for 100 different choices of n.
        
elif ((n_proc == 8) & (n_num == 150)): # 8 Cores n_num = 150
    proc_chunks_SGaME = [(0, 50), (50, 90), (90, 110), (110, 125), (125, 137), (137, 143),\
                    (143, 148), (148, 150)] # 8 Cores for 100 different choices of n.        
        
elif ((n_proc == 8) & (n_num == 200)): # 8 Cores n_num = 200
    proc_chunks_SGaME = [(0, 80), (80, 120), (120, 150), (150, 170), (170, 188), (188, 194),\
                    (194, 198), (198, 200)] # 8 Cores for 100 different choices of n.

elif n_proc == 12: # 12 Cores
    proc_chunks_SGaME = [(0, 25), (25, 40), (40, 50), (50, 60), (60, 67), (67, 75),\
                    (75, 80), (80, 85), (85, 90), (90, 94), (94, 97), (97, 100)]        
else:
    print("Please modify proc_chunks according to the core of your computer.!")
    
with get_context("fork").Pool(processes=n_proc) as pool:  # Work on MacBook Air (M1, 2020).
    proc_results_SGaME = [pool.apply_async(process_chunk_SGaME,
                                      args=(chunk,))
                    for chunk in proc_chunks_SGaME]

    result_chunks_SGaME = [r.get() for r in proc_results_SGaME]

# Save the result for SGaME models.
done_beta1 = np.concatenate([result_chunks_SGaME[j][0] for j in range(n_proc)], axis=0)
done_beta0 = np.concatenate([result_chunks_SGaME[j][1] for j in range(n_proc)], axis=0)
done_a = np.concatenate([result_chunks_SGaME[j][2] for j in range(n_proc)], axis=0)
done_b = np.concatenate([result_chunks_SGaME[j][3] for j in range(n_proc)], axis=0)
done_sigma = np.concatenate([result_chunks_SGaME[j][4] for j in range(n_proc)], axis=0)


if not os.path.exists("results"):
    os.makedirs("results")

np.save("results/result_model" + str(model) +"_n_c" + str(n_components) +"_ns_min" +\
        str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
            "_rep"  + str(reps) + "_beta1.npy", done_beta1)

np.save("results/result_model" + str(model) +"_n_c" + str(n_components) +"_ns_min" +\
        str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
            "_rep"  + str(reps) + "_beta0.npy", done_beta0)

np.save("results/result_model" + str(model) +"_n_c" + str(n_components) +"_ns_min" +\
        str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
            "_rep"  + str(reps) + "_a.npy", done_a)    

np.save("results/result_model" + str(model) +"_n_c" + str(n_components) +"_ns_min" +\
        str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
            "_rep"  + str(reps) + "_b.npy", done_b) 

np.save("results/result_model" + str(model) + "_n_c" + str(n_components) +"_ns_min" +\
        str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
            "_rep"  + str(reps) + "_sigma.npy", done_sigma)      
    

print("--- %s seconds ---" % (time.time() - start_time))


