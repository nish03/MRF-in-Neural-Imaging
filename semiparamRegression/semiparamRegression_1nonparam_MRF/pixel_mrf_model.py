import numpy
from math import sqrt
import tensorflow as tf
import opengm
import gmm_tensorflow

def pixel_mrf_model(num_knots,num_clusters,beta,S2,G,noPixels): 
    alpha_param = beta[0,:].reshape(1,-1)
    beta_nonparam = beta[1:].transpose()
    datapoints, DIMENSIONS = beta_nonparam.shape
    means = gmm_tensorflow(COMPONENTS = num_clusters, DIMENSIONS = DIMENSIONS , TRAINING_STEPS =1000, TOLERANCE=10e-6, beta_nonparam)
    n_labels_pixels = num_clusters
    n_pixels=noPixels 
    pixel_unaries = numpy.zeros((n_pixels,n_labels_pixels),dtype=numpy.float32)
    for i in range(n_pixels):
        for l in range(n_labels_pixels):
            pixel_unaries[i,l] = numpy.linalg.norm(S2[:,i] - G.dot(numpy.concatenate([alpha_param[:,i], means.T[:,l]])))
    pixel_regularizer = opengm.differenceFunction(shape=[n_labels_pixels,n_labels_pixels],norm=1,weight=1.0/n_labels_pixels,truncate=None)
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
    centroid_labels = numpy.zeros((n_pixels,num_knots))
    centroid_labels = [means[i,:] for i in argmin]
    centroid_labels = numpy.asarray(centroid_labels)
    beta = centroid_labels.T
    return beta
