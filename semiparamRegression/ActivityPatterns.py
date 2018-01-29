import h5py
import numpy as np
import matplotlib.pyplot as plt

#pDataset = 'Brudzinski.mat'
#f = h5py.File(pDataset, "r");  
#T = np.array(f["T"].value);
#f.close();

def computeBoxcarActivityPattern(T, lag=0, phaseDuration=30, stim_width=30,sigma=10,mu=0):
    timing = T;   
    noTimepoints = len(T);    
    act = np.zeros( (noTimepoints,1) );
    i_beginStimulation = np.linspace(1,19,10);
    t_phaseShift = i_beginStimulation*phaseDuration;
    
    timing = (T-T[1])/1000;
    
    for i in range(0,len(t_phaseShift)):
        t_il = t_phaseShift[i]
        t_ir = t_il + phaseDuration
        idx_il = np.argmin(np.abs(timing - t_il))
        idx_ir = np.argmin(np.abs(timing - t_ir))
        act[idx_il : idx_ir] = 1
        
    return act

def computeGaussianActivityPattern(T, lag=0, phaseDuration=30, stim_width=30,sigma=10,mu=15):
    timing = T;   
    noTimepoints = len(T);    
    act = np.zeros( (noTimepoints,1) );
    i_beginStimulation = np.linspace(1,19,10);
    t_phaseShift = i_beginStimulation*phaseDuration;
    
    timing = (T-T[1])/1000;
    actIdx = 0;
    
    for i in range(0,len(t_phaseShift)):
        print("i " + str(i))
        t_il = t_phaseShift[i]
        t_ir = t_il + phaseDuration
        idx_il = np.argmin(np.abs(timing - t_il));
        idx_ir = np.argmin(np.abs(timing - t_ir));
        if(idx_il==idx_ir):
            break;
        
        idx = (timing[idx_il:idx_ir] - timing[idx_il]);
        gaussian = gauss_function(idx,mu,sigma);
        
        gauss_il = np.argmin(abs(idx - (mu-1.5*sigma))); # war vorher 2 (24.8.) 
        gauss_ir = np.argmin(abs(idx - (mu+1.5*sigma))); # war vorher 2 (24.8.)
        gaussian[1:gauss_il] = 0;
        gaussian[gauss_ir+1:] = 0;
        
        gaussian = gaussian - np.min(gaussian[gauss_il:gauss_ir]);
        gaussian = gaussian / np.max(gaussian[gauss_il:gauss_ir]);
        
        act[idx_il : idx_ir, actIdx] = gaussian
        
    return act

def gauss_function(idx, mu, sigma):
    p1 = -.5 * np.power((idx - mu)/sigma,2);
    p2 = (sigma * np.sqrt(2*np.pi));
    f = np.exp(p1) / p2; 
    return f