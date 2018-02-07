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

def computeGaussianActivityPattern(T, lag=0, phaseDuration=30, stim_width=30,sigma=10,mu=30):
    timing = T;   
    noTimepoints = len(T);    
    act = np.zeros( (noTimepoints,1) );
    i_beginStimulation = np.linspace(0,19,10);
    
    timing = (T-T[1])/1000;
    actIdx = 0;
    startWithRestPeriod = 1    

    for i in range(0,19):
        mu_i = startWithRestPeriod*phaseDuration +  i * 2*phaseDuration
        idx_il = np.argmin(abs(timing - (mu_i-1.5*sigma))); # war vorher 2 (24.8.) 
        idx_ir = np.argmin(abs(timing - (mu_i+1.5*sigma))); # war vorher 2 (24.8.)
        if(idx_il==idx_ir):
            break;
        gaussian = gauss_function(timing,mu_i,sigma);
        
        gaussian[1:idx_il] = 0;
        gaussian[idx_ir+1:] = 0;
        
        gaussian = gaussian - np.min(gaussian[idx_il:idx_ir]);
        gaussian = gaussian / np.max(gaussian[idx_il:idx_ir]);
        act[idx_il:idx_ir, actIdx] = gaussian[idx_il:idx_ir]
        
    return act

def gauss_function(idx, mu, sigma):
    p1 = -.5 * np.power((idx - mu)/sigma,2);
    p2 = (sigma * np.sqrt(2*np.pi));
    f = np.exp(p1) / p2; 
    return f
