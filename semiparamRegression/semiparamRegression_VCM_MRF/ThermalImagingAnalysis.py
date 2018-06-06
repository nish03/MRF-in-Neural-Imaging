import ActivityPatterns as ap
import numpy as np
import h5py
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import bspline

def semiparamRegressio_VCM(S, T, B, P):
    """Apply semiparametric regression framework with varying coefficient model to imaging data.
    S: m x n data cube with m time series of length n
    T: length n vector of timestamps
    B: non parametric basis
    P: penalty matrix of non-parametric basis
    """
    [noTimepoints, noPixels] = S.shape
    m = np.mean(S,axis=0)
    S2 = S - m
    """
    Build VCM basis
    """
    val = ap.computeBoxcarActivityPattern(T,sigma=30)
    val_neg,vp = val.nonzero()
    csep = np.cos((2*np.pi*1/60000) * T);
    ssep = np.sin((2*np.pi*1/60000) * T);
    csep[val_neg] = 0;
    ssep[val_neg] = 0;
    ycos = np.diag(csep);
    ysin = np.diag(ssep);
    """
    modulate B-Spline basis
    magic script to create spline basis [Bsep,~,Dsep] = computePenBSplineBasis(noTimepoints,2,1,10);
    """
    Bsep = bspline.createBasis(orderSpline = 2, noKnots = 10);
    print("-> " + str(Bsep.shape))
    BcosSEP = ycos.dot(Bsep).transpose();
    BsinSEP = ysin.dot(Bsep).transpose();
    nothing, noFixedEffects = Bsep.shape
    noFixedEffects *= 2
    """
    create design matrix
    """
    G = np.concatenate([BcosSEP, BsinSEP, B]).transpose();
    GwithoutFixedEffects = B.transpose()
    """
    fit model
    """
    # compute Penalty term
    E1 = 0 * np.eye(noFixedEffects)
    S_P = linalg.block_diag(E1,P)
    Pterm = S_P.transpose().dot(S_P)
    # allocate intermediate storage
    lambdas= np.linspace(0.1,10,10)
    GtG = G.transpose().dot(G)
    GtGwoFE = GwithoutFixedEffects.transpose().dot(GwithoutFixedEffects)
    AIC = np.zeros([len(lambdas),noPixels])
    F = np.zeros([len(lambdas),noPixels])
    for i in range(0,len(lambdas)):
        """
        fixed effects
        """
        lambda_i = lambdas[i]
        GtGpD = GtG + lambda_i * Pterm
        GTGpDsG = linalg.solve(GtGpD,G.transpose())
        beta = GTGpDsG.dot(S2)
        seqF = G.dot(beta)
        # compute model statistics
        eGlobal = S2 - seqF
        RSS = np.sum(eGlobal ** 2, axis=0)
        df = np.trace(GTGpDsG.dot(G))
        """
        Extra Sum of Squares F-test
        """
        F_i = ((RSS_woFE - RSS) / df_woFE)  /    (RSS / (noTimepoints - df))
        # covariance matrix of our components
        # Z-value of our parametric component
        F[i,] = F_i
        # compute AICc
        AIC_i = np.log(RSS) + (2 * (df+1)) / (noTimepoints-df-2)
        AIC[i,] = AIC_i

    minAICcIdx = np.argmin(AIC,axis=0)
    F = F.transpose()
    F_minAIC = F[np.arange(F.shape[0]), minAICcIdx]
    
    return F_minAIC
