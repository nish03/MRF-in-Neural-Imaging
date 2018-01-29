#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import matplotlib.pyplot as plt
import ThermalImagingAnalysis as tai
import ActivityPatterns as ap
import numpy as np

pPenalty = "Penalty_Gaussian_1024fr_2.5Hz_TruncatedWaveletBasis.mat"
pData = "FILENAME_WITH_INTRAOP_THERMAL_IMAGING_DATA"
f = h5py.File(pData, "r")
S = f["seq"].value
T = f["T"].value

f_P = h5py.File(pPenalty, "r")
P = f_P["BPdir2"].value
B = f_P["B"].value

noTimepoints, noPixels = S.shape

S2 = S[1:noTimepoints:4,:];
T2 = T[1:noTimepoints:4];
S2 = S2[0:1024,]
T2 = T2[0:1024,]
del S;
del T;

X = ap.computeGaussianActivityPattern(np.squeeze(T2)).transpose();
Z = tai.semiparamRegression(S2,X,B,P);
plt.imshow(np.reshape(Z,[640, 480]).transpose())