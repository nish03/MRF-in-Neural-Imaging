#import packages
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy 
import time 
import h5py
import MRF_model as em
import scipy.io
from sklearn import mixture
import scipy.linalg as linalg

#Load raw and signal data
f = '/projects/p_optim/Nico/SEP_SemiReg/626510_gaussian_50pxDiameter.mat'
f = scipy.io.loadmat(f)
groundtruth = f['groundtruth']
posIdx = numpy.flatnonzero(groundtruth.T)
S_raw = f['S1024_raw']
S_raw = S_raw.reshape(-1,S_raw.shape[-1]).T
noTimepoints, noPixels = S_raw.shape
plt.imshow(S_raw[0,:].reshape(480,640))
plt.show()
S_signal = f['S1024_gaussianPtnr']
S_signal = S_signal.reshape(-1,S_signal.shape[-1]).T
T = f['T1024']

#load basis and penalty
print "Raw Data"
f_P = scipy.io.loadmat("/scratch/p_optim/nish/Master-Thesis/Penalties/LearnedPenalties_Gaussian_BSpline_knots_428.mat")
P = f_P["BPdir2"]        # learned penalty matrix
B = f_P["B"].transpose()             # basis matrix 
num_knots = P.shape[0]
num_clusters = 12


#fft analysis of target pattern
T = T.flatten()
y = numpy.fft.fft(T)
y = y.flatten()
noElementsToAnalyse = 128
contributionWeCareAbout = 90.0 / 100.0      # 90 %
energyContribution = numpy.zeros(noElementsToAnalyse)
for i in range(1,noElementsToAnalyse):
      energyContribution[i] = sum(abs(y[0:i])) / sum(abs(y[0:512]))

x = numpy.linspace(0, 512, 512)
plt.plot(x, energyContribution , label = 'Energy contribution', linewidth = 1.0, color='black')
plt.xlabel(r'Positive spectral coponents',fontweight='bold',fontsize=10)
plt.ylabel(r'Energy Contribution ',fontweight='bold', fontsize=10)
plt.legend(prop={'size': 12})
plt.show()
  
contributionWeCareAbout = numpy.repeat(90.0/100.0,noElementsToAnalyse)
val = min(abs(energyContribution - contributionWeCareAbout))
relevantSpectralComponents = (abs(energyContribution - contributionWeCareAbout)).argmin(0)
print "%.2f%% of the energy is contained in the first %d spectral components\n", 100*energyContribution[relevantSpectralComponents],relevantSpectralComponents 

##########################################################
#smooth our raw data and compute statistics
##########################################################
print "Data with raw Signal"

BtB = B.dot(B.T)
BTBpDsB = linalg.solve(BtB ,B)
beta = BTBpDsB.dot(S_raw)
S_raw_noMRF = B.transpose().dot(beta)
lambda_pairwise = 0

S_raw_mrf = em.evaluation_fft_mrf(num_knots, num_clusters, beta, S_raw, B, noPixels, lambda_pairwise)
#fft analysis of raw data
Y_raw = abs(numpy.fft.fft(S_raw))
Y_raw_mrf = abs(numpy.fft.fft(S_raw_mrf))

#energies
energyAtTargetSite_raw = sum(sum(Y_raw[:,posIdx])) / len(posIdx)
energyAtTargetSite_raw_mrf = sum(sum(Y_raw_mrf[:,posIdx])) / len(posIdx)
energyAtTargetSite_raw_rel = sum(sum(Y_raw[0:relevantSpectralComponents,posIdx])) / len(posIdx)
energyAtTargetSite_raw_mrf_rel = sum(sum(Y_raw_mrf[0:relevantSpectralComponents,posIdx])) / len(posIdx)

print "-> RAW: FFT Analysis of raw data:  Y_raw Y_raw_mrf", energyAtTargetSite_raw, energyAtTargetSite_raw_mrf
print "-> RAW: FFT Analysis of raw data at relevant spectral components: Y_raw Y_raw_mrf", energyAtTargetSite_raw_rel, energyAtTargetSite_raw_mrf_rel

########################################################################
#smooth the data being superimposed by our target and compute statistics
########################################################################
print "Data with Target Signal" 

beta = BTBpDsB.dot(S_signal)
S_signal_noMRF = B.transpose().dot(beta)
lambda_pairwise = 0

S_signal_MRF = em.evaluation_fft_mrf(num_knots, num_clusters, beta, S_signal, B, noPixels, lambda_pairwise)
#fft analysis of signal data
Y_signal = abs(numpy.fft.fft(S_signal))  #fft and take absolute values
Y_signal_mrf = abs(numpy.fft.fft(S_signal_MRF))   #fft and take absolute values

#energies
energyAtTargetSite = sum(sum(Y_signal[:,posIdx])) / len(posIdx)
energyAtTargetSite_mrf = sum(sum(Y_signal_mrf[:,posIdx])) / len(posIdx)
energyAtTargetSite_rel = sum(sum(Y_signal[0:relevantSpectralComponents,posIdx])) / len(posIdx)
energyAtTargetSite_mrf_rel = sum(sum(Y_signal_mrf[0:relevantSpectralComponents,posIdx])) / len(posIdx)

print "-> TARGET: FFT Analysis of signal data: Y_signal,  Y_signal_mrf", energyAtTargetSite, energyAtTargetSite_mrf
print "-> TARGET: FFT Analysis of signal data at selected spectral components: Y_signal, Y_signal_mrf" , energyAtTargetSite_rel,  energyAtTargetSite_mrf_rel



######################################################################
#sum of squares
######################################################################
print "Sum of Squares:"
ss_raw = sum(sum( (S_raw[:,posIdx] - S_signal[:,posIdx])**2 )) / len(posIdx)
ss_mrf = sum(sum( (S_raw_mrf[:,posIdx] - S_signal_MRF[:,posIdx])**2 )) / len(posIdx)
ss_noMRF = sum(sum( (S_raw_noMRF[:,posIdx] - S_signal_noMRF[:,posIdx])**2 )) / len(posIdx) 
print "S_raw - S_signal:", ss_raw
print "S_raw_mrf - S_signal_mrf:", ss_mrf  
print "S_raw_noMRF - S_signal_noMRF:", ss_noMRF
print "The value should approach zero if the estimate match, otherwise the mrf model absorbed some energy of our target signal"







