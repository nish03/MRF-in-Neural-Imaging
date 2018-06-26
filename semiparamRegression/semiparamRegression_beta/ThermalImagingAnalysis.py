import matplotlib.pyplot as plt
import numpy as np
import time 
import h5py
import scipy.linalg as linalg
import ActivityPatterns as ap
import ThermalImagingAnalysis as tai
import scipy.io
import sklearn.metrics
from sklearn import mixture
import opengm
from scipy.linalg import norm
from numpy.linalg import lstsq
from numpy.linalg import inv

def semiparamRegression_noMRF(S2, X, B, P):
    """Apply semiparametric regression framework to imaging data.
    S: m x n data cube with m time series of length n
    X: length m vector of discretized parametric function
    B: non parametric basis
    P: penalty matrix of non-parametric basis
    """
    lenTS,noPixels = S2.shape
    m = np.mean(S2,axis=0)
    S2 = S2 - m
    G = np.concatenate([X, B]).transpose();
    [noFixedComponents, noTimepoints] = X.shape
    [nonParamComponents,noTimepoints] = B.shape
    assert (noFixedComponents == 1), "The hypothesis test only works for a single parametric component."
    # compute Penalty term
    E1 = 0 * np.eye(noFixedComponents)
    S_P = linalg.block_diag(E1,P)
    Pterm = S_P.transpose().dot(S_P)
    # allocate intermediate storage 
    lambdas= [1] #np.linspace(0.1,10,10)
    GtG = G.transpose().dot(G)
    AIC = np.zeros([len(lambdas),noPixels])
    Z = np.zeros([len(lambdas),noPixels])
    for i in range(0,len(lambdas)):
        # fit model using penalised normal equations
        lambda_i = lambdas[i]
        GtGpD = GtG + lambda_i * Pterm;
        GTGpDsG = linalg.solve(GtGpD,G.transpose())
        beta = GTGpDsG.dot(S2)
        # compute model statistics
        seqF = G.dot(beta)
        eGlobal = S2 - seqF
        RSS = np.sum(eGlobal ** 2, axis=0)
        df = np.trace(GTGpDsG.dot(G));
        covA_1 = linalg.solve(GtGpD,GtG)
        covA = linalg.solve(GtGpD.transpose(),covA_1.transpose()).transpose()
        # covariance matrix of our components
        s_square = RSS / (noTimepoints-df-1)
        # Z-value of our parametric component
        Z[i,] = beta[0,:] / np.sqrt(s_square * covA[0,0])
        # compute AICc
        AIC_i = np.log(RSS) + (2 * (df+1)) / (noTimepoints-df-2)
        AIC[i,] = AIC_i

    minAICcIdx = np.argmin(AIC,axis=0)
    Z = Z.transpose()
    Z_minAIC = Z[np.arange(Z.shape[0]), minAICcIdx]
    return Z_minAIC



def semiparamRegression(S2, X, B, P, num_clusters, noPixels, lambda_pairwise):
    """Apply semiparametric regression framework to imaging data.
    S: m x n data cube with m time series of length n
    X: length m vector of discretized parametric function
    B: non parametric basis
    P: penalty matrix of non-parametric basis
    """
    m = np.mean(S2,axis=0)
    S2 = S2 - m
    G = np.concatenate([X, B]).transpose();
    [noFixedComponents, noTimepoints] = X.shape
    assert (noFixedComponents == 1), "The hypothesis test only works for a single parametric component."
    # compute Penalty term
    E1 = 0 * np.eye(noFixedComponents)
    S_P = linalg.block_diag(E1,P)
    Pterm = S_P.transpose().dot(S_P)
    # allocate intermediate storage 
    lambdas= [1] #np.linspace(0.1,10,10)
    GtG = G.transpose().dot(G)
    AIC = np.zeros([len(lambdas),noPixels])
    '''
    prepare potential values for our MRF
    '''
    Z = np.zeros([num_clusters,len(lambdas),noPixels])
    for i in range(0,len(lambdas)):
        # fit model using penalised normal equations
        lambda_i = lambdas[i]
        GtGpD = GtG + lambda_i * Pterm;
        GTGpDsG = linalg.solve(GtGpD,G.transpose())
        beta = GTGpDsG.dot(S2)
        gmm = mixture.GaussianMixture(n_components=num_clusters,covariance_type = 'diag')
        gmm.fit(beta[1:,].T)
        means = gmm.means_
        for l in range(0,num_clusters):
            mt = means.T[:,l]
            mt = mt[...,np.newaxis]
            mt2 = np.repeat(mt,noPixels,axis=1)
            Y_gmm = B.transpose().dot(mt2)
            beta_refit = GTGpDsG.dot(S2 - Y_gmm)
            # compute model statistics
            seqF = G.dot(beta_refit)
            eGlobal = S2 - Y_gmm - seqF
            RSS = np.sum(eGlobal ** 2, axis=0)
            df = np.trace(GTGpDsG.dot(G));
            covA_1 = linalg.solve(GtGpD,GtG)
            covA = linalg.solve(GtGpD.transpose(),covA_1.transpose()).transpose()
            # covariance matrix of our components
            s_square = RSS / (noTimepoints-df-1)
            # refit Z-value of our parametric component
            Z[l,i,] = beta_refit[0,:] / np.sqrt(s_square * covA[0,0])

    '''
    MRF inference
    '''
    # for each cluster, pixel: compute Z_min (marginalize lambda)
    Zcp = Z.min(axis=1)
    #unary potential
    gm = opengm.graphicalModel([num_clusters]*noPixels)
    unary_potential = np.zeros([noPixels, num_clusters],dtype=np.float32)
    for l in range(0,num_clusters):
    	unary_potential[:,l] = -1 * abs(Zcp[l,])
    fids = gm.addFunctions(unary_potential)
    gm.addFactors(fids,np.arange(noPixels))
    #pairwise potential 
    pairwise_potential = np.zeros(num_clusters*num_clusters,dtype=np.float32).reshape(num_clusters,num_clusters)       
    for l in range(num_clusters):         
        for k in range(num_clusters):            
            pairwise_potential[l,k] = lambda_pairwise*norm(means[l,:] - means[k,:],ord=2)      
    fid = gm.addFunction(pairwise_potential)
    vis = opengm.secondOrderGridVis(640,480)
    gm.addFactors(fid,vis)
    inf_trws = opengm.inference.TrwsExternal(gm)   
    visitor=inf_trws.timingVisitor()
    inf_trws.infer(visitor)
    argmin=inf_trws.arg()
    # we dont need to recompute Z as we already have the Z value for each pixel and label
    Z_mrf = np.zeros([noPixels])
    for j in range(noPixels):
        Z_mrf[j] = Zcp[argmin[j],j]
    return Z_mrf
