import warnings
import copy

import numpy as np
from numpy.linalg import inv
import sys
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
import sklearn.cluster as cluster

from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from itertools import compress
import time


def sample_topKSGaME(n_samples, true_params, true_topK, seed = 0):
    """
    Draw n_samples samples (Xi, Yi), i = 1,...,n_samples, from a supervised Gaussian locally-linear mapping (GLLiM).

    Parameters
    ----------
    betas : ([nb_mixture, dim_X +1], np.array)
        Intercept and coefficience vectors for weights.
    As : ([nb_mixture, dim_X +1] np.array)
        Intercept vector and Regressor matrix of location parameters of normal distribution.            
    Sigmas : ([nb_mixture, dim_data_Y, dim_data_Y] np.array)
        Scale parameters (Gaussian experts' covariance matrices) of normal distribution.            
    nb_data : int
        Sample size.    
    seed : int
        Starting number for the random number generator.    
                    
    Returns
    -------
    data_X : ([nb_data, dim_data_X+1] np.array) dim_data_X = dim_X+1 with the first column := 1.
        Input random sample.
    data_Y : ([nb_data] np.array)
        Output random sample.

    """
    ######################################################################################## 
    # Sample the input data: data_X.
    ########################################################################################   
    
    # ## Quick test
    # seed = 0
    # nb_data = 4
    
    (beta1, beta0, a, b, sigmas) = true_params
    n_components = beta1.shape[0]
    rng = np.random.default_rng(seed)
    n_features = 1
    # Draw nb_data data_X samples from uniform distribution on [0,1].
    
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    
    
    ########################################################################################
    # Sample the output data: data_Y via latent_Z variable.
    ########################################################################################
    # dim_data_Y = As.shape[1]
    # dim_data_Y = l
    y = np.zeros(n_samples)

    # Calculate the softmax gating network probabilites

    gating_prob = topK_softmax(X, beta1, beta0, true_topK)
    
    # ## Test whether the sum of the rows in gating_prob is equal to 1 in each column.
    # ## print(gating_prob)
    # print('Sum of the rows in gating_prob is equal to 1 = ', \
    #       np.all(np.sum(gating_prob, axis = 1).reshape((nb_data,1))-np.ones((nb_data,1))<1e-16))
    
    latent_Z =  np.zeros((n_samples, n_components))
    kclass_Y =  np.zeros(n_samples)
    

    for n in range(n_samples):
        Znk = rng.multinomial(1, gating_prob[n], size = 1)[0]
        latent_Z[n] = Znk
        zn = np.where(Znk == 1)[0]
        kclass_Y[n] = zn[0]

        y[n] = rng.multivariate_normal(
            mean = (X[n].dot(a[zn[0], :]) + b[zn[0]]),
            cov = sigmas[zn[0]].reshape((1, 1))
        )
        
    return (X, y)

def init_params(true_params, n_samples):
    """ Starting values for EM algorithm. """
    # n_samples = 1000
    n_components = 2
    n_features = 1
    

    (beta1_true, beta0_true, a_true, b_true, sigma_true) = true_params

    beta1_init = np.zeros((n_components, n_features))
    beta0_init = np.zeros(n_components)
    a_init = np.zeros((n_components, n_features))
    b_init = np.zeros(n_components)
    sigma_init = np.zeros(n_components)
    
    inds = range(n_components)

    # Make a partition of starting values near the true components.
    while True:
        s_inds = np.random.choice(inds, size=n_components)
        unique,counts = np.unique(s_inds, return_counts=True)

        if unique.size == n_components:
            break

    for k in range(n_components):
        beta1_init[k] = (beta1_true[s_inds[k]] + np.random.normal(0, 0.005*n_samples**(-0.083), size=(n_features)))
        beta0_init[k] = (beta0_true[s_inds[k]] + np.random.normal(0, 0.005*n_samples**(-0.083)))
        a_init[k] = a_true[s_inds[k]] + np.random.normal(0, 0.005*n_samples**(-0.083), size=(n_features))
        b_init[k] = (b_true[s_inds[k]] + np.random.normal(0, 0.005*n_samples**(-0.083)))
        sigma_init[k] = sigma_true[s_inds[k]] + np.abs(np.random.normal(0, 0.0005*n_samples**(-0.25)))
        
    return (beta1_init, beta0_init, a_init, b_init, sigma_init)

def topK_softmax(X, weights_coef, weights_intercept, K = 2):
    """
    Calculate the softmax gating network probabilites.

    Parameters
    ----------
    X: ([n_samples, n_features + 1] np.array)
        Input sample. dim_data_X = dim_X+1 with the first column := 1.    
    beta0 : ([nb_mixture], np.array)
        Mixing intercept parameter.
    beta1 : ([nb_mixture, dim_data_X], np.array)
        Mixing coefficient parameter.
        
    Returns
    -------
    gating_prob: ([nb_data, nb_mixture], np.array)
        Softmax gating network probabilites.
    """
    n_samples, _ = X.shape
    n_components, _ = weights_coef.shape
    gating_prob_sample = np.zeros((n_samples, n_components))
    
    mat_Xbeta = np.exp(X.dot(weights_coef.T) + weights_intercept.T)
    topK_Xbeta = keep_topK(mat_Xbeta, K)
    rowSum_topK_Xbeta = np.sum(topK_Xbeta, axis=1).reshape(n_samples,1)
    sum_topK_Xbeta = np.tile(rowSum_topK_Xbeta, reps = (1,n_components))

    gating_prob_sample = topK_Xbeta/sum_topK_Xbeta

    return gating_prob_sample

def keep_topK(input, K, axis=1):
    idx = np.argsort(-input)
    idx = np.take(idx, np.arange(K), axis=axis)
    output = np.zeros(input.shape)
    val = np.take_along_axis(input, idx, axis=axis)
    for row in np.arange(input.shape[0]):
        output[row, idx[row]] = input[row, idx[row]]
    return output

def Obj(X, y, tau, beta1, beta0, Gammak):
    #Compute the value of the objective function
    
    # Val_Obj = (tau.reshape(1,X.shape[0]))@((X@u.reshape(X.shape[1],1) \
    #                                         - Y.reshape(X.shape[0],1))**2) #[1x1]

    Val_Obj = tau.dot(((X.dot(beta1) + beta0 - y)**2)) #[1x1]
    
    if Gammak==0:
        Val_Obj = Val_Obj/2    
    
    return Val_Obj

def SoTh(numerator, Gammak):
    if Gammak==0:
        return numerator
    elif numerator>Gammak:
        return numerator-Gammak
    elif numerator<-Gammak:
        return numerator+Gammak
    else:
        return 0

def Fs(X, tau, beta1, beta0, K):
    
    # n = X.shape[0]
    # K = beta.shape[1] + 1
    
    pik = topK_softmax(X, beta1, beta0, K)
    
    Qw = np.sum(tau*np.log(pik))
    
    return Qw

def CoorLQk(X, y, tau, coef, intercept, Gammak=0):
    
    esp_CoorLQk = 1e-6 ## Stopping condition
    n_samples, n_features = X.shape

    beta1 = copy.deepcopy(coef)
    beta0 = copy.deepcopy(intercept)

    Val = Obj(X, y, tau, beta1, beta0, Gammak) # (27): Y:=c_k, tau:=d_k, u:=beta_new[k,:]
    Xmat = X #[nxd1]
    # Sum = (tau.reshape(1,n))@(Xmat**2) #[1xn nxd1 = 1xd1]
    # print('Shape(Sum) = ', Sum.shape)
    TauU = np.sum(tau) #[1x1]
    iiter_CoorLQk = 0
    
    while True:
        Val1 = Val
        # print(beta0)
        # print(beta1.shape)
        
        for j in range(n_features):

            rij = y - (X.dot(beta1.T) + beta0) + beta1[j] * Xmat[:, j] 
            #(n,) - ((nxd) dot (d,) + (1,)) + (n,) = (n,)
            # [ nx1* nx1 * nx1 = nx1--> np.sum() = 1x1]    
            numerator = np.sum(rij*(tau)*(Xmat[:,j]), axis = 0)
            # print('rij =', rij)
            # print('shape(rij) =', rij.shape)
            # [nx1 * nx1]
            denominator = np.sum((tau)*(Xmat[:,j])**2,axis = 0)
            # beta1[j] = SoTh(numerator, Gammak)/denominator # (28) or (21)
            beta1[j] = numerator/denominator # (28) or (21) No penalization!
            
            # [1xn @(nx1 - nxd1 @ d1x1) = 1x1]
            # beta0 = (tau.reshape(1,n))@(y.reshape(n,1) - Xmat@(beta1.reshape(d1,1)))
            beta0 = tau.dot(y - Xmat.dot(beta1))
            beta0 = beta0/TauU # (29) or (22) [1x1]
            
        
        Val = Obj(X, y, tau, beta1, beta0, Gammak)
        
        # Stopping criterion.
        if ((Val1 - Val) < esp_CoorLQk):
            # print('Val1 - Val = ', Val1 - Val)
            # print('Val1 - Val < esp_CoorLQk? ', (Val1 - Val) < esp_CoorLQk)
            # print(np.all((Val1 - Val) < esp_CoorLQk))
            break
        iiter_CoorLQk += 1
        # print('iiter_CoorLQk = ', iiter_CoorLQk)
    # return Val
    return (beta1, beta0, iiter_CoorLQk)


class TopKSGaME:
    ''' Top K Sparse Softmax Gating Mixture of Experts  '''
    
    def __init__(
        self,
        n_components=1,
        topK=1,
        tol=1e-6,
        max_iter=2000,
        n_init=1,
        params_init_dict=None,
        random_state=None,
        warm_start=False, 
        verbose=0, 
        verbose_interval=1
    ):
        self.n_components = n_components
        self.topK = topK
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.params_init_dict = params_init_dict
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = 0
        self.n_features_in_ = 0

    def _initialize_parameters(self):
        """ Initialization of the Gaussian mixture parameters.
        """
        self.weights_coef_ = self.params_init_dict['weights_coef_']
        self.weights_intercept_ = self.params_init_dict['weights_intercept_']
        self.means_coef_ = self.params_init_dict['means_coef_']
        self.means_intercept_ = self.params_init_dict['means_intercept_']
        self.variances_ = self.params_init_dict['variances_']
    
    def fit(self, X, y):
        """Estimate model parameters with the EM algorithm.

        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for ``max_iter``
        times until the change of likelihood or lower bound is less than
        ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
        If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
        initialization is performed upon the first call. Upon consecutive
        calls, training starts where it left off.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : array-like of shape (n_samples,)
            List of data points.

        Returns
        -------
        self : object
            The fitted mixture.
        """
        # parameters are validated in fit_predict
        self.fit_predict(X, y)
        return self

    def fit_predict(self, X, y):
        """Estimate model parameters using X and predict the labels for y.

        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        .. versionadded:: 0.20

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        # X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        # if X.shape[0] < self.n_components:
        #     raise ValueError(
        #         "Expected n_samples >= n_components "
        #         f"but got n_components = {self.n_components}, "
        #         f"n_samples = {X.shape[0]}"
        #     )
        # self._check_parameters(X)

        # if we enable warm_start, we will have a unique initialization
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = self.random_state

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters()

            lower_bound = -np.inf if do_init else self.lower_bound_

            if self.max_iter == 0:
                best_params = self._get_parameters()
                best_n_iter = 0
            else:
                for n_iter in range(1, self.max_iter + 1):
                    prev_lower_bound = lower_bound

                    log_prob_norm, log_resp = self._e_step(X, y)
                    # print("Running m-step for means...")
                    self._m_step_means(X, y, np.exp(log_resp))
                    # print("Running m-step for weights...")
                    self._m_step_weights(X, np.exp(log_resp))
                    _, log_resp = self._e_step(X, y)
                    # print("Running m-step for variances...")
                    self._m_step_variances(X, y, np.exp(log_resp))
                    log_prob_norm, _ = self._e_step(X, y)

                    lower_bound = log_prob_norm

                    change = lower_bound - prev_lower_bound
                    self._print_verbose_msg_iter_end(n_iter, change)

                    if abs(change) < self.tol:
                        self.converged_ = True
                        break

                self._print_verbose_msg_init_end(lower_bound)

                if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                    max_lower_bound = lower_bound
                    best_params = self._get_parameters()
                    best_n_iter = n_iter

        # Should only warn about convergence if max_iter > 0, otherwise
        # the user is assumed to have used 0-iters initialization
        # to get the initial means.
        if not self.converged_ and self.max_iter > 0:
            warnings.warn(
                "Initialization %d did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data." % (init + 1)
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X, y)

        return log_resp.argmax(axis=1)

    def _e_step(self, X, y):
        """E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in y given X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X, y)
        return np.mean(log_prob_norm), log_resp

    def _m_step_means(self, X, y, resp):
        _, n_features = X.shape
        coeff = np.zeros((self.n_components, n_features))
        intercept = np.zeros(self.n_components)
        for k in range(self.n_components):
            (coeff[k], intercept[k], iiter_mAb_SGaME) = CoorLQk(X, y, resp[:,k], self.means_coef_[k,:], self.means_intercept_[k])
            # print('M-step for means converged after iterations = ', iiter_mAb_SGaME)
        
        self.means_coef_, self.means_intercept_ = (coeff, intercept)

    def _m_step_variances(self, X, y, resp):
        variances = np.zeros(self.n_components)
        for k in range(self.n_components):
            temp = resp[:,k].dot(
                (y - (X.dot(self.means_coef_[k]) + self.means_intercept_[k]))**2)
            variances[k] = temp/np.sum(resp[:,k]) 
        
        self.variances_ = variances

    def _m_step_weights(self, X, resp, max_iter_mbeta=1000):
        n_samples = X.shape[0]
        P_k = np.zeros(n_samples)
        d_k = 1/4*np.ones(n_samples)
        c_k = np.zeros(n_samples)
        
        Stepsize = 0.5
        esp_Q = 1e-5 #threshold for Q value
        beta1_new  = copy.deepcopy(self.weights_coef_)
        beta0_new  = copy.deepcopy(self.weights_intercept_)

        for iiter_mbeta in range(max_iter_mbeta):
            beta1_old = beta1_new
            beta0_old = beta0_new
            Q_old = Fs(X, resp, beta1_old, beta0_old, K=self.topK)
        
            for k in range(self.n_components):
                #First: compute the quadratic approximation w.r.t (w_k): L_Qk
                P = topK_softmax(X, beta1_old, beta0_old, self.topK)
                P_k = P[:, k]
                c_k = X.dot(beta1_new[k,:]) + beta0_new[k] + 4 * (resp[:, k] - P_k)
                #Second: coordinate descent for maximizing L_Qk
                beta1_new[k,:], beta0_new[k], _  = CoorLQk(X, c_k, d_k, beta1_new[k, :], beta0_new[k])
            
            Q_new = Fs(X, resp, beta1_new, beta0_new, K=self.topK)
            # Backtracking line search.
            t = 1 
            while (
                Q_new < Q_old and 
                beta1_new != beta1_old and 
                beta0_new != beta0_old
                ):
                t = t*Stepsize
                beta1_new = beta1_new * t + beta1_old * (1 - t)
                beta0_new = beta0_new * t + beta0_old * (1 - t)
                Q_new = Fs(X, resp, beta1_new, beta0_new, K=self.topK)
            
            if ((Q_new - Q_old) < esp_Q):
                print('mbeta_SGaME converged after iterations = ', iiter_mbeta)
                break

        self.weights_coef_, self.weights_intercept_ = (beta1_new, beta0_new)

    def _get_parameters(self):
        return (
            self.weights_coef_,
            self.weights_intercept_,
            self.means_coef_,
            self.means_intercept_,
            self.variances_,
        )

    def _set_parameters(self, params):
        (
            self.weights_coef_,
            self.weights_intercept_,
            self.means_coef_,
            self.means_intercept_,
            self.variances_,
        ) = params

    def _estimate_weighted_log_prob(self, X, y):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        return self._estimate_log_prob(X, y) + self._estimate_log_weights(X)

    def _estimate_log_weights(self, X):
        """Estimate log-weights in EM algorithm, E[ log pi ] in VB algorithm.

        Returns
        -------
        log_weight : array, shape (n_components, )
        """
        weights = topK_softmax(
            X,
            weights_coef=self.weights_coef_,
            weights_intercept=self.weights_intercept_,
            K=self.topK
        )
        return np.log(weights)

    def _estimate_log_prob(self, X, y):
        """Estimate the log-probabilities log P(X | Z).

        Compute the log-probabilities per each component for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob : array, shape (n_samples, n_component)
        """
        n_samples, _ = X.shape
        means = X.dot(self.means_coef_.T) + self.means_intercept_.T
        return np.array(
            [[multivariate_normal.logpdf(y[i], mean=means[i, k], cov=self.variances_[k])\
                for k in range(self.n_components)] for i in range(n_samples)]
        )

    def _estimate_log_prob_resp(self, X, y):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(Y | X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X, y)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp
    
    def _print_verbose_msg_init_beg(self, n_init):
        """Print verbose message on initialization."""
        if self.verbose == 1:
            print("Initialization %d" % n_init)
        elif self.verbose >= 2:
            print("Initialization %d" % n_init)
            self._init_prev_time = time.time()
            self._iter_prev_time = self._init_prev_time

    def _print_verbose_msg_iter_end(self, n_iter, diff_ll):
        """Print verbose message on initialization."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose == 1:
                print("  Iteration %d" % n_iter)
            elif self.verbose >= 2:
                cur_time = time.time()
                print(
                    "  Iteration %d\t time lapse %.5fs\t ll change %.5f"
                    % (n_iter, cur_time - self._iter_prev_time, diff_ll)
                )
                self._iter_prev_time = cur_time

    def _print_verbose_msg_init_end(self, ll):
        """Print verbose message on the end of iteration."""
        if self.verbose == 1:
            print("Initialization converged: %s" % self.converged_)
        elif self.verbose >= 2:
            print(
                "Initialization converged: %s\t time lapse %.5fs\t ll %.5f"
                % (self.converged_, time.time() - self._init_prev_time, ll)
            )