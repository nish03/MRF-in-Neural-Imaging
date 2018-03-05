#import packages
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy 
import time 
import h5py
import basis_mrf as bm
import pixel_mrf_model as pm
import scipy.io

#Evaluation with fast fourier transform
f = '/projects/p_optim/Nico/SEP_SemiReg/626510_gaussian_50pxDiameter.mat'
f = scipy.io.loadmat(f)
groundtruth = f['groundtruth']
posIdx = numpy.flatnonzero(groundtruth.T)
S_raw = f['S1024_raw']
S_signal = f['S1024_gaussianPtnr']
T = f['T1024']

f_P = h5py.File("/scratch/p_optim/nish/Master-Thesis/semiparamRegression_2nonparam_MRF/Penalty_Gaussian_1024fr_2.5Hz_TruncatedWaveletBasis.mat", "r")
P = f_P["BPdir2"].value        # learned penalty matrix
P = P.transpose()              # P appears to be stored as transposed version of itself
B = f_P["B"].value             # basis matrix 
noTimepoints, noPixels = S2.shape


#fft analysis of target pattern
y = numpy.fft.fft(T)
noElementsToAnalyse = 128
contributionWeCareAbout = 90.0 / 100.0      # 90 %
energyContribution = numpy.zeros(noElementsToAnalyse)
for i in range(0,noElementsToAnalyse):
      energyContribution[i] = sum(abs(y[0:i,])) / sum(abs(y[2:512,]))
  
contributionWeCareAbout = numpy.repeat(90.0/100.0,noElementsToAnalyse)
val,relevantSpectralComponents = min(abs(energyContribution - contributionWeCareAbout)), (abs(energyContribution - contributionWeCareAbout)).argmin(0)
print "%.2f%% of the energy is contained in the first %d spectral components\n", 100*energyContribution[relevantSpectralComponents],relevantSpectralComponents 

##########################################################
#smooth our raw data and compute statistics
##########################################################
print "Raw Data"

S_raw_mrf = pm.pixel_mrf_model(num_knots,num_clusters,beta,S_raw,B, P, noPixels)
#fft analysis of signal data
Y_raw = abs(numpy.fft.fft(S_raw))
Y_raw_mrf = abs(numpy.fft.fft(S_raw_mrf))

#energies
energyAtTargetSite_raw = sum(sum(Yraw[posIdx,1:])) / len(posIdx)
energyAtTargetSite_raw_mrf = sum(sum(Yraw_mrf[posIdx,1:])) / len(posIdx)
energyAtTargetSite_raw_rel = sum(sum(Yraw[posIdx,1:relevantSpectralComponents])) / len(posIdx)
energyAtTargetSite_raw_mrf_rel = sum(sum(Yraw_mrf[posIdx,1:relevantSpectralComponents])) / len(posIdx)

print "-> RAW: FFT Analysis at target site:  Y_raw Y_raw_spline Y_raw_mrf", energyAtTargetSite_raw, energyAtTargetSite_raw_mrf);
print "-> RAW: FFT Analysis at target site of relevant spectral components: Y_raw Y_raw_spline Y_raw_mrf", energyAtTargetSite_raw_rel, energyAtTargetSite_raw_3DS_rel

########################################################################
#smooth the data being superimposed by our target and compute statistics
########################################################################
print "Data with Target Signal" 
S_signal_MRF = pm.pixel_mrf_model(S1024_signalPtnr,T1024,Spline_kx,Spline_ky,Spline_kt,1,1)
#fft analysis of signal data
Y_signal = abs(numpy.fft.fft(S2))  #fft and take absolute values
Y_signal_mrf = abs(numpy.fft.fft(S_hat_mrf))   #fft and take absolute values

#energies
energyAtTargetSite = sum(sum(Y_signal[posIdx,1:])) / len(posIdx)
energyAtTargetSite_rel = sum(sum(Y_signal[posIdx,1:relevantSpectralComponents])) / len(posIdx)
energyAtTargetSite_mrf = sum(sum(Y_signal_mrf[posIdx,1:])) / len(posIdx)
energyAtTargetSite_mrf_rel = sum(sum(Y_signal_mrf[posIdx,1:relevantSpectralComponents])) / len(posIdx)

print "-> TARGET: FFT Analysis at target site: Y_signal, Y_signal_spline, Y_signal_mrf", energyAtTargetSite, energyAtTargetSite_spline, Y_signal_mrf
print "-> TARGET: FFT Analysis at target site of selected spectral components: Y_signal, Y_signal_spline, Y_signal_mrf" , energyAtTargetSite_rel, energyAtTargetSite_spline_rel, energyAtTargetSite_mrf_rel);



######################################################################
#sum of squares
######################################################################
print "Sum of Squares:"
ss_raw = sum(sum( (S_raw[posIdx,:] - S_signal[posIdx,:])**2 )) / len(posIdx)
ss_mrf = sum(sum( (S_raw_mrf[posIdx,:] - S_signal_mrf[posIdx,:])**2 )) / len(posIdx)
print "S_raw - S_signal:", ss_raw
print "S_raw_spline - S_signal_spline:", ss_mrf  

print "The last SS value should approach zero if the estimate match, otherwise the 3DS absorbed some energy of our target signal"
