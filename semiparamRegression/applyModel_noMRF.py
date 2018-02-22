#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import matplotlib.pyplot as plt
import ThermalImagingAnalysis as tai
import ActivityPatterns as ap
import numpy as np

pPenalty = "Penalty_Gaussian_1024fr_2.5Hz_TruncatedWaveletBasis.mat"
#pData = "sep_1072240.mat"
pData = "../../SEP/626510_sep.mat"

f = h5py.File(pData, "r")
S = f["S1024"].value
#S = f["img"].value
#T = f["T"].value
T = f["T1024"].value

f_P = h5py.File(pPenalty, "r")
P = f_P["BPdir2"].value   # learned penalty matrix
print('[INFO] P is being transposed\n')
P = P.transpose() # P appears to be stored as transposed version of itself
B = f_P["B"].value        # basis matrix

S2 = S[0:1024,]
T2 = T[0:1024,]
del S;
del T;

X = ap.computeGaussianActivityPattern(np.squeeze(T2)).transpose();
Z = tai.semiparamRegressionRaw(S2,X,B,P);

with h5py.File("result.h5","w") as f:
  d1 = f.create_dataset('Z',data=Z)
