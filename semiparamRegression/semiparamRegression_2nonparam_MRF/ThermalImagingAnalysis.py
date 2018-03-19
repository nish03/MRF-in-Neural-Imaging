import ActivityPatterns as ap
import numpy as np
import h5py
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import pixel_mrf_model as pm

def semiparamRegression(S2, X, B, B2, P, P2, num_knots,num_clusters, noPixels):
    """Apply semiparametric regression framework to imaging data.
    S: m x n data cube with m time series of length n
    X: length m vector of discretized parametric function
    B: non parametric basis
    P: penalty matrix of non-parametric basis
    """
    m = np.mean(S2,axis=0)
    S2 = S2 - m
    G = np.concatenate([X, B2, B]).transpose();
    [noFixedComponents, noTimepoints] = X.shape
    [noParamComponents, noTimepoints] = B2.shape
    assert (noFixedComponents == 1), "The hypothesis test only works for a single parametric component."
    # compute Penalty term
    E1 = 0 * np.eye(noFixedComponents)
    S_P = linalg.block_diag(E1, P2, P)
    Pterm = S_P.transpose().dot(S_P)
    # allocate intermediate storage 
    lambdas= np.linspace(0.1,10,10)
    GtG = G.transpose().dot(G)
    AIC = np.zeros([len(lambdas),noPixels])
    Z = np.zeros([len(lambdas),noPixels])
    for i in range(0,len(lambdas)):
        # fit model using penalised normal equations
        lambda_i = lambdas[i]
        GtGpD = GtG + lambda_i * Pterm;
        GTGpDsG = linalg.solve(GtGpD,G.transpose())
        beta = GTGpDsG.dot(S2)
        # MRF regularization
        beta_mrf = pm.pixel_mrf_model(num_knots,num_clusters,beta,S2,G,noPixels) 
        Y_hat = G.dot(beta_mrf)
        beta_refit = GTGpDsG.dot(S2 - Y_hat)
        # compute model statistics
        seqF = G.dot(beta_refit)
        eGlobal = S2 - seqF
        RSS = np.sum(eGlobal ** 2, axis=0)
        df = np.trace(GTGpDsG.dot(G));
        covA_1 = linalg.solve(GtGpD,GtG)
        covA = linalg.solve(GtGpD.transpose(),covA_1.transpose()).transpose()
        # covariance matrix of our components
        s_square = RSS / (noTimepoints-df-1)
        # Z-value of our parametric component
        Z[i,] = beta_refit[0,:] / np.sqrt(s_square * covA[0,0])
        # compute AICc
        AIC_i = np.log(RSS) + (2 * (df+1)) / (noTimepoints-df-2)
        AIC[i,] = AIC_i

    minAICcIdx = np.argmin(AIC,axis=0)
    Z = Z.transpose()
    Z_minAIC = Z[np.arange(Z.shape[0]), minAICcIdx]

    return Z_minAIC
