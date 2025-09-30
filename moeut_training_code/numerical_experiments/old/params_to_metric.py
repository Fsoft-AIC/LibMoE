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
from discrepancies import *

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

dists = np.empty((n_num, reps))

done_beta1 = np.load("results_topKSGaME/result_model" + str(model) +"_n_c" + str(n_components) +"_ns_min" +\
        str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
            "_rep"  + str(reps) + "_beta1.npy")

done_beta0 = np.load("results_topKSGaME/result_model" + str(model) +"_n_c" + str(n_components) +"_ns_min" +\
        str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
            "_rep"  + str(reps) + "_beta0.npy")

done_a = np.load("results_topKSGaME/result_model" + str(model) +"_n_c" + str(n_components) +"_ns_min" +\
        str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
            "_rep"  + str(reps) + "_a.npy")    

done_b = np.load("results_topKSGaME/result_model" + str(model) +"_n_c" + str(n_components) +"_ns_min" +\
        str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
            "_rep"  + str(reps) + "_b.npy") 

done_sigma = np.load("results_topKSGaME/result_model" + str(model) + "_n_c" + str(n_components) +"_ns_min" +\
        str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
            "_rep"  + str(reps) + "_sigma.npy")

done_beta0 = betas[:, 0].reshape(-1, n_features)
done_beta0 = np.tile(done_beta0, (n_num, reps, 1, 1))
print(done_beta0.shape)

done_params = (
    done_beta1,
    done_beta0,
    done_a,
    done_b,
    done_sigma
)

# print(done_beta0)

# for param in done_params:
#     # print(param.shape)

for i in range(n_num):
    for j in range(reps):
        if model == 1:
            params = (
                done_beta1[i, j, :, :],
                done_beta0[i, j, :],
                done_a[i, j, :, :],
                done_b[i, j, :],
                done_sigma[i, j, :]
            )
            dists[i,j] = loss_SGaME_exact(params, true_params, K=1)
             
        else:
            sys.exit("Model unrecognized.")  

np.save("results_topKSGaME/result_model" + str(model) +"_n_c" + str(n_components) +"_ns_min" +\
        str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
            "_rep"  + str(reps)+ "_loss.npy", dists)