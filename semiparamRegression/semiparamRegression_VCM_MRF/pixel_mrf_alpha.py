import numpy
from math import sqrt
import opengm
from sklearn import mixture
import time

def pixel_mrf_coeff(num_clusters, coeff, X, noPixels, lambda_pairwise): 
    datapoints, DIMENSIONS = coeff.T.shape
    gmm = mixture.GaussianMixture(n_components=num_clusters,covariance_type = 'diag')
    gmm.fit(coeff.T)
    means  = gmm.means_
    n_labels_pixels = num_clusters
    n_pixels=noPixels 
    pixel_unaries = numpy.zeros((n_pixels,n_labels_pixels),dtype=numpy.float32)
    for l in range(n_labels_pixels):
        mt = means.T[:,l]
        mt = mt[...,numpy.newaxis]
        mt2 = numpy.repeat(mt,datapoints,axis=1)
        est = X.transpose().dot(coeff) - X.T.dot(mt2)
        pixel_unaries[:,l] = numpy.sqrt(numpy.sum(est**2, axis=0))
    pixel_regularizer = opengm.differenceFunction(shape=[n_labels_pixels,n_labels_pixels],norm=1,weight=lambda_pairwise,truncate=None)
    gm = opengm.graphicalModel([n_labels_pixels]*n_pixels)
    fids = gm.addFunctions(pixel_unaries)
    gm.addFactors(fids,numpy.arange(n_pixels))
    fid = gm.addFunction(pixel_regularizer)
    vis = opengm.secondOrderGridVis(640,480)
    gm.addFactors(fid,vis)
    inf_trws=opengm.inference.TrwsExternal(gm)
    visitor=inf_trws.timingVisitor()
    inf_trws.infer(visitor)
    argmin=inf_trws.arg()
    centroid_labels = numpy.zeros((n_pixels,1))
    centroid_labels = [means[i,:] for i in argmin]
    centroid_labels = numpy.asarray(centroid_labels)
    coeff = centroid_labels.T
    return coeff
