import numpy
from sklearn.decomposition import PCA
from sklearn import mixture
import opengm
import scipy.linalg as linalg

def evaluation_fft_mrf(num_knots,num_clusters, S,B,P, noPixels, lambdas): 
    m = numpy.mean(S,axis=0)
    S = S - m
    Pterm = P.transpose().dot(P)
    BtB = B.dot(B.T)
    BtBpD = BtB + lambdas * Pterm
    BTBpDsB = linalg.solve(BtBpD,B)
    beta = BTBpDsB.dot(S)
    pca = PCA(n_components=num_knots)
    pca.fit(beta.T)
    var1= numpy.cumsum(numpy.round(pca.explained_variance_ratio_, decimals=3)*100)
    components = numpy.argmax(numpy.unique(var1)) + 1
    pca = PCA(n_components=components)
    pca.fit(beta.T)
    eigenvector_matrix = pca.components_
    beta = beta.T.dot(eigenvector_matrix.T)
    gmm = mixture.GaussianMixture(n_components=num_clusters,covariance_type = 'diag')
    gmm.fit(beta)
    means  = gmm.means_
    means_inv_PCA = eigenvector_matrix.T.dot(means.T)
    BtM = B.T.dot(means_inv_PCA)
    n_labels_pixels = num_clusters
    n_pixels=noPixels 
    pixel_unaries = numpy.zeros((n_pixels,n_labels_pixels),dtype=numpy.float32)
    for i in range(n_pixels):
        temp = S[:,i,numpy.newaxis] - BtM[:]
        pixel_unaries[i,:] = numpy.linalg.norm(temp,axis=0)
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
    beta = eigenvector_matrix.T.dot(centroid_labels.T)
    Y_mrf = B.transpose().dot(beta)
    return Y_mrf