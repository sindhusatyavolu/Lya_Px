import numpy as np
from Lya_Px.params import *
from astropy.io import fits
# import argparse
from collections import defaultdict
import fitsio
if gauss_test:
    from Lya_Px.gaussian_skewer_maker import SkewerMaker
    from Lya_Px.input_power import InputPower

def get_p1d(all_skewers,N_fft):
    '''
    Function to compute the 1D power spectrum from the skewers in the given redshift bin
    all_skewers (list): list of Skewers objects, each containing the data for a single sightline and which redshift bin they belong to
    Returns:
    p1d (np.ndarray): 1D array of shape (N_FFT,), dimensionless P1D in the FFT grid
    p1d_norm (np.ndarray): 1D array of shape (N_FFT,), normalized P1D in the FFT grid

    '''
    
    p1d = np.zeros(N_fft)
    w = np.zeros(N_fft)

    if gauss_test:
        # Generate fake input power spectrum
        #input_p=InputPower(P0=0.5,k0=0.01,kF=0.1,f_px=0.7)        
        if add_spike:
            input_p = InputPower(P0=input_P0,k0=input_k0,kF=input_kF,f_px=input_f_px,add_spike=True,k_spike=k_spike,spike_width_npix=spike_width_npix,spike_height=spike_height)
        else:
            input_p = InputPower(P0=input_P0,k0=input_k0,kF=input_kF,f_px=input_f_px)
        
        # read the skewer details from the first one, assuming all are equal-sized
        Npix = all_skewers[0].delta_fft_grid.size
        L_wave = all_skewers[0].pw_A * Npix
        # initiate the skewer maker
        if input_res:
           maker=SkewerMaker(N=Npix,L=L_wave,input_power=input_p, seed=None,resolution=gauss_res)#,noise=1.0)   
        else:
           maker=SkewerMaker(N=Npix,L=L_wave,input_power=input_p, seed=None)#,noise=1.0)   

    for skewer in all_skewers:

        weight = skewer.weight_fft_grid

        if set_weights_one:
            weight = np.ones(skewer.weight_fft_grid.shape) # uncomment if you want equal weighting for all pixels
           
        if gauss_test:
            g_skewer_1,g_skewer_2=maker.make_skewer_pair()
            delta = g_skewer_1 
        else:
            delta = skewer.delta_fft_grid
       

 
        fft_weighted_delta = np.fft.fft(delta * weight)
        p1d += np.abs(fft_weighted_delta)**2
        
        fft_weight = np.fft.fft(weight)
        w += np.abs(fft_weight)**2

    # Normalize
    #p1d_norm = (pw_A / N_fft) * (1 / len(all_skewers)) * p1d
    p1d_norm = p1d#*(1/len(all_skewers))
    W_m = w#/len(all_skewers)

    return p1d, p1d_norm, W_m
