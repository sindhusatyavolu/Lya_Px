import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import sys
import h5py
from Lya_Px.params import *
from Lya_Px.functions import get_skewers, create_skewer_class, get_p1d
from Lya_Px.px_from_pix import get_px
from Lya_Px.auxiliary import angular_separation,save_to_hdf5,wave_to_velocity


def compute_px(healpix, z_alpha, dz, theta_min_array, theta_max_array, wave_desi):
    print('healpix = ', healpix)
    wave_desi_min = wave_desi[0]
    skewers = get_skewers(healpix, deltas_path)
    
    result_dict = {}  # key = (z, θmin, θmax), value = Px array
    px_weights = {}
    p1d_dict = {}

    for z in range(len(z_alpha)):
        lam_cen = LAM_LYA * (1 + z_alpha[z])
        lam_min = LAM_LYA * (1 + z_alpha[z] - 0.5 * dz[z])
        lam_max = LAM_LYA * (1 + z_alpha[z] + 0.5 * dz[z])
        i_cen = round((lam_cen - wave_desi_min) / pw_A)
        wave_fft_grid = wave_desi[i_cen - N_fft//2 : i_cen + N_fft//2]
        k_arr = np.fft.fftfreq(N_fft)*2*np.pi/pw_A

        mask_fft_grid = np.ones(N_fft)
        j_min = round((lam_min - wave_fft_grid[0]) / pw_A)
        j_max = round((lam_max - wave_fft_grid[0]) / pw_A)
        mask_fft_grid[:j_min] = 0
        mask_fft_grid[j_max:] = 0

        all_skewers = [s for s in skewers if z_alpha[z] in s.z_bins]
        
        if not all_skewers:
            continue

        # (optional) compute P1D here if needed
        p1d,p1d_norm = get_p1d(all_skewers, wave_fft_grid, mask_fft_grid)

        for theta in range(len(theta_min_array)):
            result = get_px(all_skewers, theta_min_array[theta], theta_max_array[theta])
            no_of_pairs = result[4]
            if no_of_pairs == 0:
                continue

            px = result[0] * (pw_A / N_fft)  # normalize
            z_bin = z_alpha[z]
            theta_bin = (theta_min_array[theta], theta_max_array[theta])
            result_dict[(z_bin, theta_bin)] = px
            px_weights[(z_bin, theta_bin)] = result[1]
            p1d_dict[z_bin] = p1d_norm

    
    return k_arr, result_dict, p1d_dict ,px_weights


