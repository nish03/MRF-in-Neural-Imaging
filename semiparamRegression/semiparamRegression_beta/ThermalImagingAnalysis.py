import matplotlib.pyplot as plt
import numpy as np
import time 
import h5py
import scipy.linalg as linalg
import ActivityPatterns as ap
import ThermalImagingAnalysis as tai
import scipy.io
import sklearn.metrics
import time
from sklearn.cluster import KMeans
from sklearn import mixture
import opengm
from scipy import ndimage
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
    #pairwise potential 
    pairwise_potential = np.zeros(num_clusters*num_clusters,dtype=np.float32).reshape(num_clusters,num_clusters)       

    GtGpD = GtG + np.eye(513) 
    GTGpDsG = linalg.solve(GtGpD,G.transpose())
    beta_noP = GTGpDsG.dot(S2)

    for i in range(0,len(lambdas)):
        # fit model using penalised normal equations
        lambda_i = lambdas[i]
        GtGpD = GtG + lambda_i * Pterm;
        GTGpDsG = linalg.solve(GtGpD,G.transpose())
        beta = GTGpDsG.dot(S2)
        noKnots,noPixels = beta.shape
        Kb = np.zeros([num_clusters,noKnots-1,noPixels])
        print('precomputing data for pairwise potential')
        start_time = time.time()
        for l in range(0,num_clusters):
              start_time = time.time()
              Kb[l,:,:] = smoothBetaWithGaussian(beta_noP[1:,],l+1)
              runtime = time.time() - start_time
              print(str(l) + " (" + str(runtime) + "s)")
 
        runtime = time.time() - start_time
        print("Elapsed time " + str(runtime) + "s")
 
        start_time = time.time()
        for l in range(0,num_clusters):
            start_time = time.time()
            for k in range(l,num_clusters):
		#val = norm(smoothBetaWithGaussian(beta[1:,],l+1) - smoothBetaWithGaussian(beta[1:,],k+1), ord=2)
                val = norm(Kb[l] - Kb[k])
                pairwise_potential[l,k] = val
                pairwise_potential[k,l] = val
            runtime = time.time() - start_time
            print(str(l) + " (" + str(runtime) + "s)")

	pairwise_potential = pairwise_potential / np.max(pairwise_potential)
        runtime = time.time() - start_time
        print("Elapsed time " + str(runtime) + "s")
        seqF = G.dot(beta)
        eGlobal = S2 - seqF
        RSS = np.sum(eGlobal ** 2, axis=0)
        df = np.trace(GTGpDsG.dot(G));
        covA_1 = linalg.solve(GtGpD,GtG)
        covA = linalg.solve(GtGpD.transpose(),covA_1.transpose()).transpose()
        # covariance matrix of our components
        s_square = RSS / (noTimepoints-df-1)
        # refit Z-value of our parametric component
        Zast = beta[0,:] / np.sqrt(s_square * covA[0,0])
        print('Z*_0 = ' + str(Zast[0]))
        for l in range(0,num_clusters):
            mt2 = Kb[l,:,:]
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
            print('Z_0 = ' + str(Z[l,i,0]))
            
    '''
    MRF inference
    '''
    # for each cluster, pixel: compute Z_min (marginalize lambda)
    Zcp = Z.min(axis=1)
    print('storing Z')
    np.save('Z.npy',Z)
    print('done')
    #unary potential
    gm = opengm.graphicalModel([num_clusters]*noPixels)
    unary_potential = np.zeros([noPixels, num_clusters],dtype=np.float32)
    for l in range(0,num_clusters):
#    	unary_potential[:,l] = -1 * abs(Zcp[l,]) 
        unary_potential[:,l] = Zcp[l,] - Zast 
    unary_potential = unary_potential / np.max(abs(unary_potential))
    fids = gm.addFunctions(unary_potential)
    gm.addFactors(fids,np.arange(noPixels))
    np.save('pw_potential.npy',pairwise_potential)
    np.save('unary_potential.npy',unary_potential)
    fid = gm.addFunction(pairwise_potential)
    vis = opengm.secondOrderGridVis(640,480)
    gm.addFactors(fid, vis)
    inf_trws = opengm.inference.TrwsExternal(gm)   
    visitor=inf_trws.timingVisitor()
    inf_trws.infer(visitor)
    argmin=inf_trws.arg()
    np.save('amin.npy',argmin)
    # we dont need to recompute Z as we already have the Z value for each pixel and label
    Z_mrf = np.zeros([noPixels])
    for j in range(noPixels):
        Z_mrf[j] = Zcp[argmin[j],j]
    return Z_mrf

def smoothBetaWithGaussian(b,h):
    h = h
    b = b.transpose()
    noPixels,noKnots = b.shape
    b = np.reshape(b,[640, 480, noKnots])
    y_g3d = ndimage.gaussian_filter(b[:,:,:],sigma=(h,h,1))
    y_g3d = np.reshape(y_g3d,[640*480, noKnots]).transpose()
    return y_g3d
