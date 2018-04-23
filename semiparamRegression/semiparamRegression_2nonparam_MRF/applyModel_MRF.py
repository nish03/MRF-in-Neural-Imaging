#import packages
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy 
import numpy as np
import time 
import h5py
import scipy.linalg as linalg
import ActivityPatterns as ap
import pixel_mrf_model as pm
import ThermalImagingAnalysis as tai
import scipy.io
import sklearn.metrics

# Data and parametric component
f = h5py.File("/scratch/p_optim/nish/Master-Thesis/semiparamRegression_2nonparam_MRF/626510_sep.mat", "r")
S = numpy.array(f["S1024"].value)
T = numpy.array(f["T1024"].value)
S2 = S[0:1024,]
T2 = T[0:1024,]
del S;
del T;
noTimepoints, noPixels = S2.shape


#First non parametric component
g = '/scratch/p_optim/nish/Master-Thesis/Penalties/LearnedPenalties_Gaussian_BSpline_knots_428.mat'
g = scipy.io.loadmat(g)
B = g['B'].transpose()
P = g['BPdir2']


#mrf regularization
h = '/scratch/p_optim/nish/Master-Thesis/Penalties/LearnedPenalties_Gaussian_BSpline_knots_428.mat'
h = scipy.io.loadmat(h)
B2 = h['B'].transpose()
P2 = h['BPdir2']


#f_P = h5py.File("/scratch/p_optim/nish/Master-Thesis/semiparamRegression_2nonparam_MRF/Penalty_Gaussian_1024fr_2.5Hz_TruncatedWaveletBasis.mat", "r")
#P = f_P["BPdir2"].value        # learned penalty matrix
#P = P.transpose()              # P appears to be stored as transposed version of itself
#B = f_P["B"].value             # basis matrix 



#compute gaussian activity pattern
X = ap.computeGaussianActivityPattern(numpy.squeeze(T2)).transpose();
num_knots =  P2.shape[0] 
num_clusters = 12
lambda_pairwise = 2.0

#semiparametric regression
Z = tai.semiparamRegression(S2, X, B, B2, P, P2, num_knots, num_clusters, noPixels, lambda_pairwise)
plt.imshow(Z.reshape(640,480).transpose())
plt.show()



#accuracy after pixel_mrf model
groundtruthImg = numpy.array(f["groundtruthImg"].value)
groundtruth_foreground = numpy.where(groundtruthImg > 0)[0]
groundtruth_background = numpy.where(groundtruthImg == 0)[0]
true_positive =  len(numpy.where(abs(Z[groundtruth_foreground,]) >= 5.2)[0])                                  
false_positive = len(numpy.where(abs(Z[groundtruth_foreground,]) < 5.2)[0])
true_negative = len(numpy.where(abs(Z[groundtruth_background,]) < 5.2)[0])
false_negative = len(numpy.where(abs(Z[groundtruth_background,]) >= 5.2)[0])
true_positive_rate = true_positive / numpy.float32(len(groundtruth_foreground))
false_positive_rate = false_positive / numpy.float32(len(groundtruth_background))
accuracy  = (true_positive + true_negative) / numpy.float32(len(groundtruth_background) + len(groundtruth_foreground))
  

#F1 score metrics for better evaluation
Z_true = groundtruthImg.flatten()
for i in range(len(Z_true)):
    if Z_true[i] != 0:
       Z_true[i] = 1
    else:
       Z_true[i] = 0

Z_pred = numpy.zeros(len(Z_true))
for i in range(len(Z_pred)):
    if Z[i] >= 5.2:
       Z_pred[i] = 1
    else:
       Z_pred[i] = 0
    
F1 = sklearn.metrics.f1_score(Z_true, Z_pred, average='binary')

#MRF on Z
n_labels = 2
n_pixels=noPixels 
threshold = 5.2
pixel_unaries = numpy.zeros((n_pixels,n_labels),dtype=numpy.float32)
for l in range(n_pixels):
    pixel_unaries[l,0] = Z[l,] - threshold
    pixel_unaries[l,1] = threshold - Z[l,]

pixel_regularizer = opengm.differenceFunction(shape=[n_labels,n_labels],norm=1,weight=2,truncate=None)
gm = opengm.graphicalModel([n_labels]*n_pixels)
fids = gm.addFunctions(pixel_unaries)
gm.addFactors(fids,numpy.arange(n_pixels))
fid = gm.addFunction(pixel_regularizer)
vis = opengm.secondOrderGridVis(640,480)
gm.addFactors(fid,vis)
inf_trws=opengm.inference.TrwsExternal(gm)
visitor=inf_trws.timingVisitor()
inf_trws.infer(visitor)
Z_new =inf_trws.arg()
true_positive =  len(numpy.where(abs(Z_new[groundtruth_foreground,]) >= 1)[0])                                  
false_positive = len(numpy.where(abs(Z_new[groundtruth_foreground,]) < 1)[0])
true_negative = len(numpy.where(abs(Z_new[groundtruth_background,]) < 1)[0])
false_negative = len(numpy.where(abs(Z_new[groundtruth_background,]) >= 1)[0])
true_positive_rate = true_positive / numpy.float32(len(groundtruth_foreground))
false_positive_rate = false_positive / numpy.float32(len(groundtruth_background))
accuracy  = (true_positive + true_negative) / numpy.float32(len(groundtruth_background) + len(groundtruth_foreground))
Z_true = groundtruthImg.flatten()
for i in range(len(Z_true)):
    if Z_true[i] != 0:
       Z_true[i] = 1
    else:
       Z_true[i] = 0


Z_pred = numpy.zeros(len(Z_true))
for i in range(len(Z_pred)):
    if Z_new[i] >= 1:
       Z_pred[i] = 1.0
    else:
       Z_pred[i] = 0.0
  

F1 = sklearn.metrics.f1_score(Z_true, Z_pred, average='binary')
