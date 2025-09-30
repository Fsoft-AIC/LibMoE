import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logging
import models

from scipy.optimize import curve_fit


import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-K', '--n_components', default=2, type=int, help='Number of mixture components.')
parser.add_argument('-r' ,'--reps', default=10, type=int, help='Number of replications per sample size.')
parser.add_argument('-np','--nproc', default=8, type=int, help='Number of processes to run in parallel.') # Work on MacBook Air (M1, 2020, 8 cores)
parser.add_argument('-mi','--maxit', default=2000, type=int, help='Maximum EM iterations.')
parser.add_argument('-e', '--eps', default=1e-6, type=float, help='EM stopping criterion.')
parser.add_argument('-m', '--m_func', default=None, type=str, help='Gating kernel function.')
parser.add_argument('-nnum', '--n_num', default=200, type=int, help='Number of different choices of sample size.')
parser.add_argument('-nsmax', '--ns_max', default=1000, type=int, help='Number of sample size maximum.')
parser.add_argument('-nsmin', '--ns_min', default=100, type=int, help='Number of sample size maximum.')
parser.add_argument('-errorbar', '--errorbar', default=0, type=int, help='Number of trials for emNMoE of meteoritsSim package.')
parser.add_argument('-verbose', '--verbose', default=1,\
                    type=int, help='Log-likelihood should be printed or not during EM iterations.')

args = parser.parse_args()

print(args)

# model = args.model                    # Type number
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
m_func = args.m_func         


logging.basicConfig(
    filename=
    '_k' + str(n_components) +\
                    '.log', filemode='w', format='%(asctime)s %(message)s'
)

params_true = models.get_params(1)
n_components_true, n_features = params_true[2].shape

params_true = {
        'weights_coef_': params_true[0],
        'weights_intercept_': params_true[1],
        'sigmas_': params_true[2],
    }

    

if (n_components_true == n_components):
    ns = np.concatenate([np.linspace(ns_min, 9*ns_min, 5), np.linspace(10*ns_min, ns_max, n_num-5)])
else:
    ns = np.linspace(ns_min, ns_max, n_num)


def O_poly(x, a, b):
    return a * x**b


def plot_model(n0=0):
    
    D = np.load("results/result" + "_K" + str(n_components) + "_M_" + m_func + "_ns_min" +\
        str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
            "_rep"  + str(reps) + "_loss.npy")  
    
    
    #####
    
    # # Original.
    fig = plt.figure()
    
    loss        = np.mean(D, axis=1)
    yerr        = 2*np.std(D, axis=1)
    
    yerr_std  = 2*np.std(yerr)
    
    lab="temp"
    
    # Y = np.array(np.log(loss)).reshape([-1,1])
    if (n_components_true == n_components):
        label = "$\mathcal{D}_1(\widehat G_n, G_{*})$"
    else:
        label = "$\mathcal{D}_2(\widehat G_n, G_{*})$"
    

    params, _ = curve_fit(O_poly, ns[n0:], loss[n0:])


    a_fit, b_fit = params   

    y_fit = O_poly(ns[n0:], a_fit, b_fit)
    
    text_size = 17

    lw = 2
    elw = 0.7
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",  # Choose a font that supports LaTeX symbols
        "font.serif": ["Computer Modern Roman"],
    })

    if (errorbar == 1):
        plt.errorbar(ns[n0:], loss[n0:], yerr=yerr[n0:], color='blue', capsize=2, linestyle = '-', lw=lw, elinewidth=elw, label=label)
    else:
        plt.plot(ns[n0:], loss[n0:], color='blue', capsize=2, linestyle = '-', lw=lw, elinewidth=elw, label=label)

    plt.plot(
        ns[n0:], y_fit, lw=lw, color='orange', linestyle = '-.', 
        label=f'${a_fit:.1f}~n^{{ {b_fit:.2f} }}$'
    )
    

    # plt.grid(True, which="both", ls="-")
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("log(sample size)", fontsize=text_size)
    plt.ylabel("log(loss)", fontsize=text_size)#"$\log$ " + lab)
    plt.legend(loc="upper right", title="", prop={'size': 10})

    # Create the plots folder if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.savefig("plots_SGaME/plot_model" + str(model) +"_K" + str(K) + "_n0_" +\
                str(n0) +"_ns_min" + str(int(ns[0])) +"_ns_max" +\
                    str(int(ns[-1])) + "_n_tries" + str(n_tries) + "_n_num" + str(n_num) +"_errorbar"+str(errorbar) +"_rep" + str(D.shape[1])+\
                    ".pdf", bbox_inches = 'tight',pad_inches = 0)
    

if (n_components > n_components_true):
    for i in range(0, 50):
        plot_model(n0 = i)
else:
    plot_model(n0 = 0)