#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import matplotlib.pyplot as plt
import ThermalImagingAnalysis as tai
import ActivityPatterns as ap
import numpy as np
import scipy.io
import numpy
from sklearn.metrics import f1_score

pPenalty = "/scratch/p_optim/nish/Master-Thesis/semiparamRegression_2nonparam_MRF/Penalty_Gaussian_1024fr_2.5Hz_TruncatedWaveletBasis.mat"
#pData = "sep_1072240.mat"
pData = "/scratch/p_optim/nish/Master-Thesis/semiparamRegression_2nonparam_MRF/626510_sep.mat"

f = h5py.File(pData, "r")
S = f["S1024"].value
#S = f["img"].value
#T = f["T"].value
T = f["T1024"].value

#f_P = h5py.File(pPenalty, "r")
#P = f_P["BPdir2"].value   # learned penalty matrix
#print('[INFO] P is being transposed\n')
#P = P.transpose() # P appears to be stored as transposed version of itself
#B = f_P["B"].value        # basis matrix

g = '/scratch/p_optim/nish/Master-Thesis/Penalties/LearnedPenalties_Gaussian_BSpline_knots_415.mat'
g = scipy.io.loadmat(g)
B =  g['B'].transpose()
P =  g['BPdir2']

S2 = S[0:1024,]
T2 = T[0:1024,]
del S;
del T;

X = ap.computeGaussianActivityPattern(np.squeeze(T2)).transpose();
Z = tai.semiparamRegressionRaw(S2,X,B,P);


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
    
F1 = f1_score(Z_true, Z_pred, average='binary')
