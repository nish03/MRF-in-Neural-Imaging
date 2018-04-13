#import packages
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy 
import time 
import h5py
import ActivityPatterns as ap
import ThermalImagingAnalysis as tai
import scipy.io
import sklearn.metrics

# Data, parametric and non-parametric components 
f = h5py.File("/home/nico/Code/626510_sep.mat", "r")
g = '/home/nico/Code/LearnedPenalties_Gaussian_BSpline_knots_415.mat'
g = scipy.io.loadmat(g)


S = numpy.array(f["S1024"].value)
T = numpy.array(f["T1024"].value) 
B = g['B'].transpose()
P = g['BPdir2']

S2 = S[0:1024,]
T2 = T[0:1024,]
del S;
del T;
noTimepoints, noPixels = S2.shape

#compute gaussian activity pattern
X = ap.computeGaussianActivityPattern(numpy.squeeze(T2)).transpose();

#semiparametric regression
Z = tai.semiparamRegression(S2, X, B, P, noPixels, 1)
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

print("RES " + str(accuracy) + ": " + str(true_positive_rate) + ": " + str(false_positive_rate))

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
print("RES " + str(F1))
