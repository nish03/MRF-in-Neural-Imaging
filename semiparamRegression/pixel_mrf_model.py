import numpy
from sklearn.decomposition import PCA
from math import sqrt
from sklearn import mixture
import opengm
from sklearn.preprocessing import StandardScaler

def fast_norm(x):
    return sqrt(x.dot(x.conj()))

def pixel_mrf_model(num_knots,beta,S2,B,noPixels):
    #taking principal components   
    pca = PCA(n_components=num_knots)
    pca.fit(beta.T)
    var1= numpy.cumsum(numpy.round(pca.explained_variance_ratio_, decimals=3)*100)
    components = numpy.argmax(numpy.unique(var1)) + 1
    pca = PCA(n_components=components)
    pca.fit(beta.T)
    eigenvector_matrix = pca.components_
    beta = beta.T.dot(eigenvector_matrix.T)
    #discretization
    num_clusters = 5
    gmm = mixture.GaussianMixture(n_components=num_clusters)
    gmm.fit(beta)
    means  = gmm.means_
    #means for mrf
    means_inv_PCA = eigenvector_matrix.T.dot(means.T)
    #start mrf potentials assginment
    n_labels_pixels = num_clusters
    n_pixels=noPixels 
    pixel_unaries = numpy.zeros((n_pixels,n_labels_pixels),dtype=numpy.float32)
    for i in range(n_pixels):
        for l in range(n_labels_pixels):
            pixel_unaries[i,l] = fast_norm(S2[:,i] - B.T.dot(means_inv_PCA[:,l])) #L2 norm
    #define pixel regularizer
    pixel_regularizer = opengm.differenceFunction(shape=[n_labels_pixels,n_labels_pixels],norm=1,weight=1.0/n_labels_pixels,truncate=None)
    #initialise graphical model
    gm = opengm.graphicalModel([n_labels_pixels]*n_pixels)
    #pixel wise unary factors
    fids = gm.addFunctions(pixel_unaries)
    gm.addFactors(fids,numpy.arange(n_pixels))
    #pixel wise pairwise factors
    fid = gm.addFunction(pixel_regularizer)
    vis = opengm.secondOrderGridVis(640,480)
    gm.addFactors(fid,vis)
    #Inference
    inf_trws=opengm.inference.TrwsExternal(gm, parameter=opengm.InfParam(steps=50))
    visitor=inf_trws.timingVisitor()
    inf_trws.infer(visitor)
    argmin=inf_trws.arg()
    #inverse discretization
    centroid_labels = numpy.zeros((n_pixels,num_knots))
    centroid_labels = [means[i,:] for i in argmin]
    centroid_labels = numpy.asarray(centroid_labels)
    #inverse PCA
    beta = eigenvector_matrix.T.dot(centroid_labels.T)
    return beta
