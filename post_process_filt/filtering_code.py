'Modfiied from Chang. et al. 2020, OT for change point detection: https://github.com/kevin-c-cheng/OtChangePointDetection'



import numpy as np
import scipy.io as sio
import scipy.signal as scisig



def filter_and_get_peaks(cp_stats, filter_wind=10,peak_thresh=0.8):
    '''takes in cp statistic array and returns filtered array with peak locations.
    filter_wind: filtering window (length of one half of the filter)
    thresh: for scipy's threshold finder algorithm
    '''
    w2ConvFilter = sio.loadmat("./post_process_filt/TwoSampConvFilter.mat")["filter2"].flatten()
    w2ConvFilter = w2ConvFilter[0::int(np.ceil(len(w2ConvFilter)/(2*filter_wind)))]-0.166
    w2ConvFilter = w2ConvFilter / np.sum(w2ConvFilter)

    w2SampC = np.convolve(cp_stats, w2ConvFilter, mode='same')
    pkIdx = scisig.find_peaks(w2SampC, height=peak_thresh, prominence=np.max(w2SampC) / 1000, width=2)
    return w2SampC,pkIdx[0]
