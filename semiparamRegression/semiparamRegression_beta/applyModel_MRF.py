import matplotlib.pyplot as plt
import numpy as np
import time 
import h5py
import scipy.linalg as linalg
import ActivityPatterns as ap
import scipy.io
import sklearn.metrics
from sklearn import mixture
import opengm
from scipy.linalg import norm
from numpy.linalg import lstsq

# Data and parametric component
f = h5py.File("/scratch/p_optim/nish/Master-Thesis/semiparamRegression_2nonparam_MRF/626510_sep.mat", "r")
S = np.array(f["S1024"].value)
T = np.array(f["T1024"].value)
S2 = S[0:1024,]
T2 = T[0:1024,]
noTimepoints, noPixels = S2.shape

#non parametric component
g = '/scratch/p_optim/nish/Master-Thesis/Penalties/LearnedPenalties_Gaussian_BSpline_knots_428.mat'
g = scipy.io.loadmat(g)
B = g['B'].transpose()
P = g['BPdir2']

#compute gaussian activity pattern
X = ap.computeGaussianActivityPattern(np.squeeze(T2)).transpose();
lambda_pairwise = 1
#semiparametric regression
Z = tai.semiparamRegression(S2, X, B, P, num_knots, noPixels,lambda_pairwise)
plt.imshow(Z.reshape(640,480).transpose())
plt.show()

#accuracy after pixel_mrf model
groundtruthImg = np.array(f["groundtruthImg"].value)
groundtruth_foreground = np.where(groundtruthImg > 0)[0]
groundtruth_background = np.where(groundtruthImg == 0)[0]
true_positive =  len(np.where(abs(Z[groundtruth_foreground,]) >= 5.2)[0])                                  
false_positive = len(np.where(abs(Z[groundtruth_foreground,]) < 5.2)[0])
true_negative = len(np.where(abs(Z[groundtruth_background,]) < 5.2)[0])
false_negative = len(np.where(abs(Z[groundtruth_background,]) >= 5.2)[0])
true_positive_rate = true_positive / np.float32(len(groundtruth_foreground))
false_positive_rate = false_positive / np.float32(len(groundtruth_background))
accuracy  = (true_positive + true_negative) / np.float32(len(groundtruth_background) + len(groundtruth_foreground))
  

#F1 score metrics for better evaluation
Z_true = groundtruthImg.flatten()
for i in range(len(Z_true)):
    if Z_true[i] != 0:
       Z_true[i] = 1
    else:
       Z_true[i] = 0

Z_pred = np.zeros(len(Z_true))
for i in range(len(Z_pred)):
    if Z[i] >= 5.2:
       Z_pred[i] = 1
    else:
       Z_pred[i] = 0
    
F1 = sklearn.metrics.f1_score(Z_true, Z_pred, average='binary')
