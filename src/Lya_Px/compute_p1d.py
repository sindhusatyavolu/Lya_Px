import numpy as np
from Lya_Px.params import *
from astropy.io import fits
# import argparse
from collections import defaultdict
import fitsio


def get_p1d(all_skewers):
    '''
    Function to compute the 1D power spectrum from the skewers in the given redshift bin
    all_skewers (list): list of Skewers objects, each containing the data for a single sightline and which redshift bin they belong to
    Returns:
    p1d (np.ndarray): 1D array of shape (N_FFT,), dimensionless P1D in the FFT grid
    p1d_norm (np.ndarray): 1D array of shape (N_FFT,), normalized P1D in the FFT grid

    '''
    
    p1d = np.zeros(N_fft)
    
    for skewer in all_skewers:

        delta = skewer.delta_fft_grid
        weight = skewer.weight_fft_grid
        
        fft_weighted_delta = np.fft.fft(delta * weight)
        p1d += np.abs(fft_weighted_delta)**2

    # Normalize
    p1d_norm = (pw_A / N_fft) * (1 / len(all_skewers)) * p1d
    return p1d, p1d_norm
