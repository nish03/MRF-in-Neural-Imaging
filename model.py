#########################################################
#import packages
#########################################################
print "importing libraries"

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy 
import opengm
import time 
from math import sqrt
import scipy.io
import numpy.linalg
from sklearn.cluster import KMeans
from sklearn import mixture
import vigra
from vigra import graphs
import h5py


print "Loading data"
t0=time.time()
#############################################################
# Load raw Data
#############################################################
#load training data
# pTrainDataT = '626510_sinus_50pxDiameter.mat'
# pTrainDataT = scipy.io.loadmat(pTrainDataT)
# #ground truth
# groundtruth = pTrainDataT['groundtruth']
# posIdx = numpy.flatnonzero(groundtruth.T)
# dx,dy=    groundtruth.shape   
# #raw data
# S1024_raw = pTrainDataT['S1024_raw']
# dx,dy,dt  = S1024_raw.shape
# S1024_raw = S1024_raw.reshape(dx*dy, dt)

f = h5py.File("sep_1072240.mat", "r")
S1024_raw = numpy.array(f["img"].value)
dy,dx,dt  = S1024_raw.T.shape
S1024_raw =  S1024_raw.reshape((S1024_raw.shape[0], -1))
f.close()
S1024_raw = S1024_raw.T

t1=time.time()
print "Loading data took", t1-t0, "secs"
#############################################################
#neuronal activity 
#############################################################
t_start = time.time()
avg = numpy.zeros(dt)
for i in range(dt):
       avg[i] = (S1024_raw[:,i].mean())

neuro_activity_max = numpy.argmax(avg)

# x = numpy.linspace(0, dt-1, dt)
# matplotlib.pyplot.figure(figsize=(10,8))
# matplotlib.pyplot.plot(x, avg, label = 'Averaged time course', linewidth = 1.0, color='r')
# matplotlib.pyplot.xlabel(r'Time points (t)',fontweight='bold',fontsize=10)
# matplotlib.pyplot.ylabel(r'Intensity',fontweight='bold', fontsize=10)
# matplotlib.pyplot.legend(prop={'size': 12})
# matplotlib.pyplot.title(r'Time vs Intensity', fontsize=15)
# matplotlib.pyplot.show()
	   
image = S1024_raw[:,neuro_activity_max].reshape(dx,dy)
image = numpy.float32(image)
# imgplot = plt.imshow(S1024_raw[:,neuro_activity_max].reshape(dx,dy))
# plt.show()
	   
print "Peak neuronal activity happens at: ",  neuro_activity_max, "time point"

##########################################################
#define basis function
########################################################## 
print "Running P-Spline" 
t0 = time.time()

num_knots = 80
X = numpy.linspace(0, dt - 1, dt)
no_of_splines = num_knots
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

###########################################################
#define penalty function
###########################################################	
lambda_param = 0.0
D = numpy.identity(basis.shape[1])
D_k = numpy.diff(D,n=1,axis=-1)  
spline_coeff_raw = numpy.linalg.solve(numpy.dot(basis.T,basis)+lambda_param*numpy.dot(D_k,D_k.T),numpy.dot(basis.T,S1024_raw.T))

t1 = time.time()
print "Total time for running P-Spline", t1-t0, "secs"

###########################################################
#define discretization function
###########################################################
print "Discretization starts"

num_clusters = 4
gmm = mixture.GaussianMixture(n_components=num_clusters)
t0=time.time()
gmm.fit(spline_coeff_raw.T)
t1=time.time()
print "Time taken for discretization of raw data is: ", t1-t0
labels = gmm.predict(spline_coeff_raw.T)
# imgplot = plt.imshow(labels.reshape(dx,dy))
# plt.show()
means  = gmm.means_



##############################################################
#define pixel wise unary and pairwise potentials
##############################################################
print "Define unary and pairwise potentials"
t0 = time.time()
n_labels_pixels = num_clusters
n_pixels=dx*dy
def fast_norm(x):
    return sqrt(x.dot(x.conj()))

#define pixel unaries 
pixel_unaries = numpy.zeros((n_pixels,n_labels_pixels),dtype=numpy.float32)
for i in range(n_pixels):
    for l in range(n_labels_pixels):
        pixel_unaries[i,l] = fast_norm(S1024_raw.T[:,i] - basis.dot(means.T[:,l])) #L2 norm

#define pixel regularizer
pixel_regularizer = opengm.differenceFunction(shape=[n_labels_pixels,n_labels_pixels],norm=1,weight=1.0/n_labels_pixels,truncate=None)			
#define region adjacency graph and region wise potentials
superpixelDiameter = 1                   # super-pixel size
slicWeight = 1                           # SLIC color - spatial weight
labels, n_segments = vigra.analysis.slicSuperpixels(image, slicWeight, superpixelDiameter) 
labels = vigra.analysis.labelImage(labels) -1
gridGraph = graphs.gridGraph(image.shape)
rag = graphs.regionAdjacencyGraph(gridGraph, labels)
nodeFeatures = rag.accumulateNodeFeatures(image)
nodeFeatures = nodeFeatures.reshape(-1,1)
nCluster   = 2
g = mixture.GaussianMixture(n_components=nCluster)
g.fit(nodeFeatures)
clusterProb = g.predict_proba(nodeFeatures)
probs = numpy.clip(clusterProb, 0.00001, 0.99999)
#define superpixel_unaries
superpixel_unaries = -1.0*numpy.log(probs)
#define superpixel regularizer
superpixel_regularizer = opengm.differenceFunction(shape=[nCluster,nCluster],norm=1,weight=1.0/nCluster,truncate=None)
#define interlayer regularizer
interlayer_regularizer = opengm.differenceFunction(shape=[n_labels_pixels,n_labels_pixels],norm=1,weight=1.0/n_labels_pixels,truncate=None)

###############################################################
#initialise and define graphical model
###############################################################
rag_edges = rag.uvIds()
rag_edges = numpy.sort(rag_edges,axis=1)
rag_edges += n_pixels

n_variables = n_pixels + n_segments
n_inter_edges = n_pixels
n_pixel_edges = (dx-1)*dy + (dy-1)*dx
n_segment_edges = rag_edges.shape[0]            #check this is right
n_edges = n_pixel_edges + n_segment_edges + n_inter_edges

print "Assigning unary and pairwise potentials to factor graph"
t0 = time.time()
#initialise graphical model
gm = opengm.graphicalModel([n_labels_pixels]*n_pixels + [nCluster]*n_segments)
gm.reserveFunctions(n_variables + 3,'explicit') # the unary functions plus the 3 types of regularizer
gm.reserveFactors(n_variables + n_edges)

#pixel wise unary factors
fids = gm.addFunctions(pixel_unaries)
gm.addFactors(fids,numpy.arange(n_pixels), finalize=False)

#pixel wise pairwise factors
fid = gm.addFunction(pixel_regularizer)
vis = opengm.secondOrderGridVis(dx,dy)
gm.addFactors(fid,vis, finalize=False)

#superpixel wise unary factors
fids = gm.addFunctions(superpixel_unaries)
gm.addFactors(fids, n_pixels + numpy.arange(n_segments), finalize=False)

#superpixel wise pairwise factors
fid = gm.addFunction(superpixel_regularizer)
gm.addFactors(fid, numpy.sort(rag_edges, axis=1), finalize=False)

#inter layer pairwise factors
fid = gm.addFunction(interlayer_regularizer)
vis = numpy.dstack([numpy.arange(n_pixels).reshape(dx,dy), labels]).reshape((-1,2))
vis[:,1] += n_pixels
gm.addFactors(fid, vis, finalize=False)

gm.finalize()
t1 = time.time()
print "Total time to assign potentials to graph", t1-t0, "secs"
################################################################
#Perform inference
################################################################
print "Inference started"
inf_trws=opengm.inference.TrwsExternal(gm, parameter=opengm.InfParam(steps=50))
visitor=inf_trws.timingVisitor()
t0=time.time()
inf_trws.infer(visitor)
t1=time.time()
print "Inference took:", t1-t0
argmin=inf_trws.arg()

################################################################
#extract superpixel node features and project to adjacency graph
################################################################
arg_pixels = argmin[0:n_pixels]
arg_superpixels = argmin[n_pixels:]
argImg = rag.projectNodeFeaturesToGridGraph(arg_superpixels.astype(numpy.uint32))

t_end = time.time()
print "Executing time of script", t_end - t_start, "secs"
imgplot = plt.imshow(argImg)
plt.show()

# argfinal = argImg.reshape(n_pixels)
# centroid_labels = numpy.zeros((n_pixels,num_knots))
# centroid_labels = [means[i,:] for i in argfinal]
# centroid_labels = numpy.asarray(centroid_labels)
# Y_hat_mrf = basis.dot(centroid_labels.T)
# Sraw_mrf = Y_hat_mrf.T


