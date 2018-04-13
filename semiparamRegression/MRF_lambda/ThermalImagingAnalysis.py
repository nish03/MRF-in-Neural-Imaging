import time
import numpy as np
import h5py
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import opengm

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
    pixel_unaries = np.zeros([noPixels, len(lambdas)],dtype=np.float32)
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
        #AIC_i = np.log(RSS) + (2 * (df+1)) / (noTimepoints-df-2)
        #AIC[i,] = AIC_i
        pixel_unaries[:,i] = 1 / Z[i,]
    n_labels_pixels = len(lambdas)
    Z = Z.transpose()
    pixel_regularizer = opengm.differenceFunction(shape=[n_labels_pixels,n_labels_pixels],norm=1,weight=lambda_pairwise,truncate=None)
    gm = opengm.graphicalModel([n_labels_pixels]*noPixels)
    fids = gm.addFunctions(pixel_unaries)
    gm.addFactors(fids,np.arange(noPixels))
    fid = gm.addFunction(pixel_regularizer)
    vis = opengm.secondOrderGridVis(640,480)
    gm.addFactors(fid,vis)
    #inf_trws=opengm.inference.TrwsExternal(gm)
    #inf_trws = opengm.inference.BeliefPropagation(gm,parameter=opengm.InfParam(damping=0.9))
    inf_trws = opengm.inference.TreeReweightedBp(gm)
    
    visitor=inf_trws.timingVisitor()
    inf_trws.infer(visitor)
    argmin=inf_trws.arg()
    h5f = h5py.File('res_ogm.h5', 'w')
    h5f.create_dataset('amin', data=argmin)
    h5f.close()
    #opengm.hdf5.save(gm,'ogm_lambdaMRF.h5','lambdaMRF') ## doesnt work
    Z_new = np.zeros([noPixels])
    for i in range(0,noPixels):   
		Z_new[i] = Z[i,argmin[i]]
    return Z_new
