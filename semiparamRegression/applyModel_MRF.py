#import packages
#import packages
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy 
import time 
import h5py
import ActivityPatterns as ap
import pixel_mrf_model as pm
import region_mrf_model as rm
import ThermalImagingAnalysis as tai
import scipy.io
from sklearn.metrics import f1_score


# Data, parametric and non-parametric components 
f = h5py.File("/scratch/p_optim/nish/Master-Thesis/semiparamRegression_2nonparam_MRF/626510_sep.mat", "r")
g = '/scratch/p_optim/nish/Master-Thesis/Penalties/LearnedPenalties_Gaussian_BSpline_knots_638.mat'
g = scipy.io.loadmat(g)


S = numpy.array(f["S1024"].value)
T = numpy.array(f["T1024"].value)
#f_P = h5py.File("/scratch/p_optim/nish/Master-Thesis/semiparamRegression_2nonparam_MRF/Penalty_Gaussian_1024fr_2.5Hz_TruncatedWaveletBasis.mat", "r")
#P = f_P["BPdir2"].value        # learned penalty matrix
#P = P.transpose()              # P appears to be stored as transposed version of itself
#B = f_P["B"].value             # basis matrix 
B = g['B'].transpose()
P = g['BPdir2']

S2 = S[0:1024,]
T2 = T[0:1024,]
del S;
del T;
noTimepoints, noPixels = S2.shape

#compute gaussian activity pattern
X = ap.computeGaussianActivityPattern(numpy.squeeze(T2)).transpose();
num_knots = P.shape[0]
num_clusters = 10

#semiparametric regression
Z = tai.semiparamRegression(S2, X, B, P, num_knots, num_clusters, noPixels)
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
    
F1 = f1_score(Z_true, Z_pred, average='binary')
