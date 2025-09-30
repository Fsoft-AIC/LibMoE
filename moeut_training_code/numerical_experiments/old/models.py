import numpy as np

def get_params(n):
    # # SGaME model.
    # # Multiple output. l > 1
    # betas   = np.array([[-7,15],[0,0]]) # [K*(d+1)]
    # As      = np.array([np.array([[8,-15]]), np.array([[.6,.4]])]) # [K*l*(d+1)]
    # Sigmas  = np.array([np.array([[.3]]), np.array([[.4]])]) # [K*l*l]
    
    # ## Single output. l = 1
    # betas   = np.array([[-7,15],[0,0]]) # [K*(d+1)]
    # As      = np.array([[8,-15], [.6,.4]]) # [K*(d+1)]
    # Sigmas  = np.array([.3,.4]) # [K*1]
    # # # Test.
    # # print(betas.shape)
    # # print(As.shape)
    # # print(Sigmas.shape)
    
    # ## Model 1: Bad result sam meteorits package.
    # ## Single output. l = 1
    # betas   = np.array([[0,0],[8,0]]) # [(d+1)xK]
    # As      = np.array([[0,0], [-2.5,2.5]]) # [(d+1)xK]
    # Sigmas  = np.array([[1,1]]) # [1xK]
    # # Test.
    # # print(betas)
    # # print(As)
    # # print(Sigmas)
    
    # # For rpy2.
    # betasr = r.matrix(r.c(0,8), ncol=1)
    # Asr = r.matrix(r.c(0, -2.5, 0, 2.5), ncol= 2)
    # Sigmasr = r.c(1,1)
    
    ## Model 2 Chamroukhi 2010.
    # ## Model 2. Hope with near mean clusters.
    # betas   = np.array([[-7,0], [15,0]]) # [(d+1)xK]
    # As      = np.array([[8,-15], [.6,.4]]) # [(d+1)xK]
    # Sigmas  = np.array([[0.3, 0.4]]) # [1xK]
    
    # # For rpy2. Error: rpy2.rinterface_lib.embedded.RRuntimeError: Erreur dans solve.default((t(Xbeta) %*% Xbeta)) : 
    # #le système est ent singulier : conditionnement de la réciproque = 7.07047e-18

    # betasr = r.matrix(r.c(-7,15), ncol=1)
    # Asr = r.matrix(r.c(8, 0.6, -15, 0.4), ncol= 2)
    # Sigmasr = r.c(0.3, 0.4)
    
    # ## Model 3. Hope with these seperation clusters.
    # betas   = np.array([[-7,0], [15,0]]) # [(d+1)xK]
    # As      = np.array([[2, 2], [-15,1]]) # [(d+1)xK]
    # Sigmas  = np.array([[0.3, 0.4]]) # [1xK]
    
    # # For rpy2. Error: rpy2.rinterface_lib.embedded.RRuntimeError: Erreur dans solve.default((t(Xbeta) %*% Xbeta)) : 
    # #le système est ent singulier : conditionnement de la réciproque = 7.07047e-18

    # betasr = r.matrix(r.c(-7,15), ncol=1)
    # Asr = r.matrix(r.c(2, -15, 2, 1), ncol= 2)
    # Sigmasr = r.matrix(r.c(0.3, 0.4), ncol = 2)
    
    ## Example of a lambda shape with little overlap.
    betas   = np.array([[-8,0], [25,0]]).T # [(d+1)xK]
    As      = np.array([[15, -5], [-20,20]]).T # [(d+1)xK]
    Sigmas  = np.array([[0.3, 0.4]]).T # [1xK]
    
    # For rpy2. Error: rpy2.rinterface_lib.embedded.RRuntimeError: Erreur dans solve.default((t(Xbeta) %*% Xbeta)) : 
    #le système est ent singulier : conditionnement de la réciproque = 7.07047e-18

    # betasr = r.matrix(r.c(-8, 25), ncol = 1)
    # Asr = r.matrix(r.c(15, -20, -5, 20), ncol = 2)
    # Sigmasr = r.matrix(r.c(0.3, 0.4), ncol = 2)
        
        
    return (betas, As, Sigmas)