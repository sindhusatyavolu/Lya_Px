import numpy as np
from Lya_Px.params import *
from collections import defaultdict


def avg_over_healpixels(results):
    '''
    Gather results from all healpixels and average over them
    results (list): list of tuples, each containing the results from a single healpix
    Returns:
    k_arr (np.ndarray): 1D array of shape (N_FFT,), k-space grid in 1/A
    px_avg (dict): dictionary with keys as tuples (z_bin, theta_bin) and values as dimensionless Px arrays of shape (N_FFT)
    px_var (dict): dictionary with keys as tuples (z_bin, theta_bin) and values as variance of Px arrays of shape (N_FFT)
    px_weights (dict): dictionary with keys as tuples (z_bin, theta_bin) and values as Px of weights of shape (N_FFT)
    p1d_avg (dict): dictionary with keys as z_bin and values as P1D array of shape (N_FFT)
    covariance (dict): dictionary with keys as tuples (z_bin, theta_bin) and values as covariance matrix of Px arrays of shape (N_FFT, N_FFT)

    '''
    px_all = defaultdict(list)  # key = (z, theta_bin), value = list of Px arrays
    px_weights_all = defaultdict(list)  
    p1d_all = defaultdict(list)

    # accumulate results in only redshift and theta bins that exist for each healpixel
    for k_arr, px_dict,p1d_dict, px_weights in results:
        for key in px_dict:
            px_all[key].append(px_dict[key])  
            px_weights_all[key].append(px_weights[key])
            p1d_all[key[0]].append(p1d_dict[key[0]])

    px_avg = {}
    px_var = {}
    p1d_avg = {}
    covariance = {}
    px_avg_weights = {}
    for key in px_all:
        # stack by healpix
        stacked = np.stack(px_all[key]) 
        stacked_weights = np.stack(px_weights_all[key])
        # average over healpixels
        px_avg[key] = np.mean(stacked, axis=0)
        px_var[key] = np.var(stacked, axis=0)
        px_weights[key] = np.mean(stacked_weights, axis=0)  
        px_avg_weights[key] = np.mean(stacked_weights, axis=0)  # count non-zero elements
        p1d_avg[key[0]] = np.mean(np.stack(p1d_all[key[0]]), axis=0)
        covariance[key] = np.cov(stacked, rowvar=False) 
    
    return k_arr, px_avg, px_var, px_avg_weights, p1d_avg, covariance



