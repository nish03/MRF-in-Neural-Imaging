#import packages
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy 
import h5py
import opengm
import time 
from math import sqrt
import scipy.io
import numpy.linalg
from  pyclustering.cluster import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.cluster import KMeans
from scipy.spatial import distance
import pywt


#define pspline function
def basis(dx,dy,dt, knots):  
    X = numpy.linspace(0, dt - 1, dt)
    no_of_splines = knots
    order_of_spline = 3
    knots = numpy.linspace(0, 1, 1 + no_of_splines - order_of_spline)
    difference = numpy.diff(knots[:2])[0]
    x = (numpy.ravel(deepcopy(X)) - X[0]) / float(X[-1] - X[0])
    x = numpy.r_[x, 0., 1.]
    x = numpy.r_[x]
    x = numpy.atleast_2d(x).T
    n = len(x)
    corner = numpy.arange(1, order_of_spline + 1) * difference
    new_knots = numpy.r_[-corner[::-1], knots, 1 + corner]
    new_knots[-1] += 1e-9
    basis = (x >= new_knots[:-1]).astype(numpy.int) * (x < new_knots[1:]).astype(numpy.int)
    basis[-1] = basis[-2][::-1]
    maxi = len(new_knots) - 1
    for m in range(2, order_of_spline + 2):
        maxi -= 1
        mask_l = new_knots[m - 1 : maxi + m - 1] != new_knots[:maxi]
        mask_r = new_knots[m : maxi + m] != new_knots[1 : maxi + 1]
        left_numerator = (x - new_knots[:maxi][mask_l]) * basis[:, :maxi][:, mask_l]
        left_denominator = new_knots[m-1 : maxi+m-1][mask_l] - new_knots[:maxi][mask_l]
        left = numpy.zeros((n, maxi))
        left[:, mask_l] = left_numerator/left_denominator
        right_numerator = (new_knots[m : maxi+m][mask_r]-x) * basis[:, 1:maxi+1][:, mask_r]
        right_denominator = new_knots[m:maxi+m][mask_r] - new_knots[1 : maxi+1][mask_r]
        right = numpy.zeros((n, maxi))
        right[:, mask_r] = right_numerator/right_denominator
        prev_bases = basis[-2:]
        basis = left + right
        
    basis = basis[:-2]  
    return basis
	
#define penalty function	
def penalty(data, basis, lambda_param):	
    dict_penalty = dict()
    D = numpy.identity(basis.shape[1])
    D_k = numpy.diff(D,n=2,axis=-1)  
    spline_coeff = numpy.linalg.solve(numpy.dot(basis.T,basis)+lambda_param*numpy.dot(D_k,D_k.T),numpy.dot(basis.T,data.T))
    dict_penalty['spline_coeff'] = spline_coeff
    data_hat_spline = basis.dot(spline_coeff)
    s = numpy.sum((data.T-data_hat_spline)**2)
    Q = numpy.linalg.inv(numpy.dot(basis.T,basis) + lambda_param * numpy.dot(D_k,D_k.T))
    t = numpy.sum(numpy.diag(Q.dot(numpy.dot(basis.T,basis))))
    dict_penalty['GCV'] = s / (basis.shape[0] - t)**2
    dict_penalty['AIC'] = -2*numpy.log(numpy.exp(-(data.T - data_hat_spline)**2/(2)) / (2 * numpy.pi)**0.5 ).sum() + 2*t
    dict_penalty['MSE'] =  ((data.T - data_hat_spline) ** 2).mean()
    return dict_penalty


#define discretization function
def discretization(data,dx,dy):
    dict_disc = dict()
    data_list = data.tolist()   
    initial_centers = kmeans_plusplus_initializer(data_list, 2).initialize()  
    xmeans_instance = xmeans.xmeans(data_list, initial_centers, ccore = True)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    centers  = xmeans_instance.get_centers()
    kmeans = KMeans(n_clusters=len(clusters))
    kmeans.fit(data)
    labels = kmeans.predict(data)
    means  = kmeans.cluster_centers_
    imgplot = plt.imshow(labels.reshape(dx,dy))
    plt.show()
    dict_disc['num_clusters'] = len(clusters)
    dict_disc['means'] = means
    return dict_disc
	
#define MRF model
def mrf(data, basis, num_knots, means, num_clusters, dx, dy):
    numLabels = num_clusters
    numVar=dx*dy
    numberOfStates=numpy.ones(numVar,dtype=opengm.index_type)*numLabels
    gm=opengm.graphicalModel(numberOfStates)
    def fast_norm(x):
        return sqrt(x.dot(x.conj()))
    unary_energy = numpy.zeros((numVar,numLabels),dtype=numpy.float32)
    for i in range(numVar):
        for l in range(numLabels):
            unary_energy[i,l] = fast_norm(data.T[:,i] - basis.dot(means.T[:,l])) #L2 norm
    
    unary_energy = unary_energy.reshape(dx,dy,numLabels)
    for x in range(dx):
        for y in range(dy):
            fid=gm.addFunction(unary_energy[x,y,:])
            gm.addFactor(fid,x*dy+y)
		    
    for x in range(dx):
        for y in range(dy):
            pairwise_energy = numpy.ones(numLabels*numLabels,dtype=numpy.float32).reshape(numLabels,numLabels)
            if (x + 1 < dx):
                for l in range(numLabels):
                    for k in range(numLabels):
                        pairwise_energy[l,k] = numpy.abs(l - k)
                pair_id = gm.addFunction(pairwise_energy)
                variableIndex0 = y + x * dy
                variableIndex1 = y + (x + 1) * dy
                gm.addFactor(pair_id, [variableIndex0, variableIndex1])
            if (y + 1 < dy):
                for l in range(numLabels):
                    for k in range(numLabels):
                        pairwise_energy[l,k] = numpy.abs(l - k)
                pair_id = gm.addFunction(pairwise_energy)	
                variableIndex0 = y + x * dy
                variableIndex1 = (y + 1) + x * dy
                gm.addFactor(pair_id, [variableIndex0, variableIndex1])     
	 
    inf_trws=opengm.inference.TrwsExternal(gm)
    visitor=inf_trws.timingVisitor()
    t0=time.time()
    inf_trws.infer(visitor)
    t1=time.time()
    print t1-t0
    argmin=inf_trws.arg()
    print "energy ",gm.evaluate(argmin)
    print "bound", inf_trws.bound()
    result=argmin.reshape(dx,dy)
    imgplot = plt.imshow(result)
    plt.title('TRWS')
    plt.show()
    centroid_labels = numpy.zeros((numVar,num_knots))
    centroid_labels = [means[i,:] for i in argmin]
    centroid_labels = numpy.asarray(centroid_labels)
    Y_hat_mrf = basis.dot(centroid_labels.T)
    return Y_hat_mrf
    

#Evaluation
pTrainDataT = '626510_sinus_50pxDiameter.mat'
pTrainDataT = scipy.io.loadmat(pTrainDataT)
groundtruth = pTrainDataT['groundtruth']
posIdx = numpy.flatnonzero(groundtruth.T)
dx,dy=    groundtruth.shape   
imgplot = plt.imshow(groundtruth.reshape(dx,dy))
plt.show()
S1024_raw = pTrainDataT['S1024_raw']
dx,dy,dt  = S1024_raw.shape
S1024_raw = S1024_raw.reshape(dx*dy, dt)
imgplot = plt.imshow(S1024_raw[:,0].reshape(dx,dy))
plt.show()
S1024_sinusPtnr = pTrainDataT['S1024_sinusPtnr']
S1024_signalPtnr = S1024_sinusPtnr.reshape(dx*dy, dt)
imgplot = plt.imshow(S1024_signalPtnr[:,0].reshape(dx,dy))
plt.show()
targetPattern = pTrainDataT['targetPattern']
y = numpy.fft.fft(targetPattern)
noElementsToAnalyse = 128
contributionWeCareAbout = 90.0 / 100.0      # 90 %
energyContribution = numpy.zeros(noElementsToAnalyse)
for i in range(2,noElementsToAnalyse+2):
      energyContribution[i-2] = sum(sum(abs(y[:,1:i]))) / sum(sum(abs(y[:,1:512])))
  
contributionWeCareAbout = numpy.repeat(90.0/100.0,noElementsToAnalyse)
val,relevantSpectralComponents = (abs(energyContribution - contributionWeCareAbout)).min(0), (abs(energyContribution - contributionWeCareAbout)).argmin(0)

basis = basis(dx,dy,dt,80)
num_knots = basis.shape[1]
dict_penalty = penalty(S1024_raw, basis, 0.02)
print "GCV", dict_penalty['GCV']
print "AIC", dict_penalty['AIC']
print "MSE", dict_penalty['MSE']
spline_coeff = dict_penalty['spline_coeff']
dict_disc = discretization(spline_coeff.T,dx,dy)
num_clusters = dict_disc['num_clusters']
means = dict_disc['means']
Y_hat_mrf = mrf(S1024_raw, basis, num_knots, means, num_clusters, dx, dy)
Sraw_mrf = Y_hat_mrf.T

Yraw = abs(numpy.fft.fft(S1024_raw))
Yraw_mrf = abs(numpy.fft.fft(Sraw_mrf))

energyAtTargetSite_raw = sum(sum(Yraw[posIdx,1:])) / len(posIdx)
energyAtTargetSite_raw_mrf = sum(sum(Yraw_mrf[posIdx,1:])) / len(posIdx)
energyAtTargetSite_raw_rel = sum(sum(Yraw[posIdx,1:relevantSpectralComponents])) / len(posIdx)
energyAtTargetSite_raw_mrf_rel = sum(sum(Yraw_mrf[posIdx,1:relevantSpectralComponents])) / len(posIdx)


# smooth the data being superimposed by our target and compute statistics
dict_penalty = penalty(S1024_signalPtnr, basis, 0.02)
print "GCV", dict_penalty['GCV']
print "AIC", dict_penalty['AIC']
print "MSE", dict_penalty['MSE']
spline_coeff = dict_penalty['spline_coeff']
dict_disc = discretization(spline_coeff.T,dx,dy)
num_clusters = dict_disc['num_clusters']
means = dict_disc['means']
s_mrf = mrf(S1024_signalPtnr, basis, num_knots, means, num_clusters, dx, dy)
Ssignal_mrf = s_mrf.T


Ysignal = abs(numpy.fft.fft(S1024_signalPtnr)) 
Ysignal_mrf = abs(numpy.fft.fft(Ssignal_mrf)) 

energyAtTargetSite = sum(sum(Ysignal.T[posIdx,1:])) / len(posIdx)
energyAtTargetSite_3DS = sum(sum(Ysignal_mrf[posIdx,1:])) / len(posIdx)
energyAtTargetSite_rel = sum(sum(Ysignal.T[posIdx,1:relevantSpectralComponents])) / len(posIdx)
energyAtTargetSite_3DS_rel = sum(sum(Ysignal_mrf[posIdx,1:relevantSpectralComponents])) / len(posIdx)

S1024_signalPtnr = S1024_signalPtnr.T
S1024_raw        = S1024_raw.T
ss_raw = sum(sum( (S1024_raw[posIdx,:] - S1024_signalPtnr[posIdx,:])**2 )) / len(posIdx)
ss_mrf = sum(sum( (Sraw_mrf[posIdx,:] - Ssignal_mrf[posIdx,:])**2 )) / len(posIdx)
