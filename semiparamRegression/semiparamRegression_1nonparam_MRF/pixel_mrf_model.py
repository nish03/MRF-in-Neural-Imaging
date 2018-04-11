import numpy
from math import sqrt
import opengm
from sklearn import mixture
import time

def pixel_mrf_model(num_knots,num_clusters,beta,S2,G,noPixels): 
    alpha_param = beta[0,:].reshape(1,-1)
    beta_nonparam = beta[1:].transpose()
    datapoints, DIMENSIONS = beta_nonparam.shape
    start_time = time.time()
    gmm = mixture.GaussianMixture(n_components=num_clusters,covariance_type = 'diag')
    datapoints, DIMENSIONS = beta_nonparam.shape
    gmm.fit(beta_nonparam)
    means  = gmm.means_
    print('GMM elapsed: ' + str(time.time() - start_time) + ' s')
    n_labels_pixels = num_clusters
    n_pixels=noPixels 
    pixel_unaries = numpy.zeros((n_pixels,n_labels_pixels),dtype=numpy.float32)
    start_time = time.time()
    for l in range(n_labels_pixels):
         mt = means.T[:,l]
         mt = mt[...,numpy.newaxis]
         mt2 = numpy.repeat(mt,datapoints,axis=1)
         cc = numpy.concatenate([alpha_param, mt2])
         est = S2 - G.dot(numpy.concatenate([alpha_param, mt2]))
         pixel_unaries[:,l] = numpy.sqrt(numpy.sum(est**2, axis=0))
    print('UNARIES elapsed: ' + str(time.time() - start_time) + ' s')
    pixel_regularizer = opengm.differenceFunction(shape=[n_labels_pixels,n_labels_pixels],norm=1,weight=0.1,truncate=None)
    gm = opengm.graphicalModel([n_labels_pixels]*n_pixels)
    fids = gm.addFunctions(pixel_unaries)
    gm.addFactors(fids,numpy.arange(n_pixels))
    fid = gm.addFunction(pixel_regularizer)
    vis = opengm.secondOrderGridVis(640,480)
    gm.addFactors(fid,vis)
    inf_trws=opengm.inference.TrwsExternal(gm)
    visitor=inf_trws.timingVisitor()
    start_time = time.time()
    inf_trws.infer(visitor)
    argmin=inf_trws.arg()
    print('MRF elapsed: ' + str(time.time() - start_time) + ' s')
    centroid_labels = numpy.zeros((n_pixels,num_knots))
    centroid_labels = [means[i,:] for i in argmin]
    centroid_labels = numpy.asarray(centroid_labels)
    beta = centroid_labels.T
    return beta
