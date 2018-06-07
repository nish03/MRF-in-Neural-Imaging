import ThermalImagingAnalysis as tai
import ActivityPatterns as ap
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time




# load data
f_P = h5py.File("/scratch/p_optim/nish/Master-Thesis/semiparamRegression_2nonparam_MRF/Penalty_Gaussian_1024fr_2.5Hz_TruncatedWaveletBasis.mat","r")
f = h5py.File("/scratch/p_optim/nish/Master-Thesis/semiparamRegression_2nonparam_MRF/626510_sep.mat", "r")
S = np.array(f["S1024"].value)
T = np.squeeze(np.array(f["T1024"].value))
f.close()
S2 = S[0:1024,]
T2 = T[0:1024,]
P = f_P["BPdir2"].value   # learned penalty matrix
print('[INFO] P is being transposed\n')
P = P.transpose() # P appears to be stored as transposed version of itself
B = f_P["B"].value # basis matrix
val = ap.computeBoxcarActivityPattern(T,sigma=30)
val_neg,vp = val.nonzero()
# REGRESSION ANALYSIS
X = ap.computeGaussianActivityPattern(np.squeeze(T2)).transpose();
start_time = time.time()
F = tai.semiparamRegressio_VCM(S2,T2,B,P);
elapsed_time = time.time() - start_time
print('elapsed (CPU): ' + str(elapsed_time) + ' s')
