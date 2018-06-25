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


def semiparamRegression(S2, X, B, P, noPixels, lambda_pairwise):
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
    lambdas= np.linspace(0.1,10,10) #need to decide on its values 
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

  
    #spatial regularization on spline coefficients
    #unary potential
    gm = opengm.graphicalModel([len(lambdas)]*noPixels)
    pixel_unaries = np.zeros([noPixels, len(lambdas)],dtype=np.float32)
    pixel_unaries[:,i] = 1 / Z[i,]
    fids = gm.addFunctions(pixel_unaries)
    gm.addFactors(fids,np.arange(noPixels))
    #pairwise potential
    gmm = mixture.GaussianMixture(n_components=len(lambdas),covariance_type = 'diag')
    gmm.fit(P.dot(beta[1:,]).T)
    means = gmm.means_   
    pairwise_energy = np.zeros(len(lambdas)*len(lambdas),dtype=np.float32).reshape(len(lambdas),len(lambdas))           
    for l in range(len(lambdas)):             
        for k in range(len(lambdas)):                    
            pairwise_energy[l,k] = lambda_pairwise*norm(means[l,:] - means[k,:]) 
           
    fid = gm.addFunction(pairwise_energy)
    vis = opengm.secondOrderGridVis(640,480)
    gm.addFactors(fid,vis)
    inf_trws = opengm.inference.TrwsExternal(gm)   
    visitor=inf_trws.timingVisitor()
    inf_trws.infer(visitor)
    argmin=inf_trws.arg()
    pbeta = np.zeros((noPixels,len(lambdas)))
    pbeta = [means[i,:] for i in argmin]
    pbeta = np.asarray(pbeta)
    pbeta = pbeta.T
    beta  = lstsq(P,pbeta)
    beta = np.asarray(beta[0])
    Y_mrf = B.transpose().dot(beta)
    beta_refit = GTGpDsG.dot(S2 - Y_mrf)
    # compute model statistics
    seqF = G.dot(beta_refit)
    eGlobal = S2 - Y_mrf - seqF
    RSS = np.sum(eGlobal ** 2, axis=0)
    # covariance matrix of our components
    s_square = RSS / (noTimepoints-df-1)
    # Z-value of our parametric component
    Z = beta_refit[0,:] / np.sqrt(s_square * covA[0,0])
    return Z
