import numpy as np
from Lya_Px.params import *
from collections import defaultdict
from Lya_Px.covariance_matrix import compute_cov

def calculate_estnorm(W, R, L):
    '''
    W (np.ndarray): vector length N, average FFT of the weights per healpix 
    R (np.ndarray): vector length N, resolution in Fourier space
    L (float): length of the spectra (in physical units, e.g. Angstroms or Mpc)
    Returns:
    estnorm (np.ndarray): vector length N, to be multiplied by every P1D mode of the measurement
    '''
    R2 = R.real**2 + R.imag**2
    denom = np.absolute(np.fft.ifft(np.fft.fft(W)* np.fft.fft(R2)))    
    estnorm = np.absolute(L/denom)
    return 1/estnorm            


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
    no_of_pairs = defaultdict(list)  
    weights_average = defaultdict(list)  

    # accumulate results in only redshift and theta bins that exist for each healpixel
    for k_arr, px_dict,p1d_dict, px_weights, npairs, w_avg in results:
        for key in px_dict:
            px_all[key].append(px_dict[key])  
            px_weights_all[key].append(px_weights[key])
            p1d_all[key[0]].append(p1d_dict[key[0]])
            no_of_pairs[key].append(npairs[key])
            weights_average[key].append(w_avg[key])

    px_avg = {}
    px_var = {}
    p1d_avg = {}
    covariance = {}
    px_avg_weights = {}
    for key in px_all:
        #print(key)
        #print('no_of_pairs = ', no_of_pairs[key])
        #print('px_all = ', np.average(np.stack(px_all[key]), axis=0, weights=no_of_pairs[key])) 
        #print('px_all = ', np.average(np.stack(px_all[key]), axis=0, weights=[1,1,1]))
        #print('px_all = ', np.mean(np.stack(px_all[key]), axis=0)) 
 
        # stack by healpix
        stacked_px = np.stack(px_all[key])
        
        stacked_weights = np.stack(px_weights_all[key])
       
        print('shape of stacked_px:', np.shape(stacked_px))
        print('shape of stacked_weights:', np.shape(stacked_weights))
        fft_avg_res = np.ones(N_fft)
        L = N_fft * pw_A  # length of the spectra in Angstroms
        stacked_V_m = np.stack([calculate_estnorm(w,fft_avg_res , L) for w in stacked_weights])        
        print('shape of stacked_V_m:', np.shape(stacked_V_m))
        stacked_px_hat = np.zeros_like(stacked_px)
        ind = stacked_V_m > 0.0
        stacked_px_hat[ind] = stacked_px[ind] / stacked_V_m[ind]


        # average over healpixels
        #px_avg[key] = np.average(stacked_px, axis=0, weights=no_of_pairs[key]) # weighted average        
        px_var[key] = np.var(stacked_px, axis=0) # not weighted
        
        stacked_weights_avg  = np.stack(weights_average[key])
        px_avg_weights[key] =  np.average(stacked_weights_avg, axis=0,weights=no_of_pairs[key])  # weighted average
       
        p1d_avg[key[0]] = np.mean(np.stack(p1d_all[key[0]]), axis=0)

        #covariance[key] = np.cov(stacked, rowvar=False,aweights=no_of_pairs[key])  # covariance matrix of Px arrays


        mean_px , covariance[key] = compute_cov(stacked_px_hat, stacked_V_m)  # covariance matrix of Px arrays
        px_avg[key] = mean_px
        
        print('shape of covariance matrix:',np.shape(covariance[key]))
        #print('mean_px',mean_px)
        #print(covariance[key])

    return k_arr, px_avg, px_var, px_avg_weights, p1d_avg, covariance



