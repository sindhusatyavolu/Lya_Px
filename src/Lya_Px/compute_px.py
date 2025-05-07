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
    
    # get sightlines in each healpix pixel, and the redshift bins which they belong to
    skewers = get_skewers(healpix, deltas_path)
    
    result_dict = {}  # key = (z, θmin, θmax), value = Px array
    px_weights = {}
    p1d_dict = {}

    for z in range(len(z_alpha)):
        # wavelength range covered by this redshift bin
        lam_cen = LAM_LYA * (1 + z_alpha[z])
        lam_min = LAM_LYA * (1 + z_alpha[z] - 0.5 * dz[z])
        lam_max = LAM_LYA * (1 + z_alpha[z] + 0.5 * dz[z])
        
        # define FFT grid as N_FFT pixels in the DESI wavelength grid, centered on the center of the redshift bin. Note that the FFT grid can be larger the DESI grid in the redshift bin
        i_cen = round((lam_cen - wave_desi_min) / pw_A)
        wave_fft_grid = wave_desi[i_cen - N_fft//2 : i_cen + N_fft//2]
        # FFT grid in k-space, has negative values as well
        k_arr = np.fft.fftfreq(N_fft)*2*np.pi/pw_A

        mask_fft_grid = np.ones(N_fft)
        # zero out the mask values that fall outside the redshift bin wavelength range
        j_min = round((lam_min - wave_fft_grid[0]) / pw_A)
        j_max = round((lam_max - wave_fft_grid[0]) / pw_A)
        mask_fft_grid[:j_min] = 0
        mask_fft_grid[j_max:] = 0

        # gather the sightlines that fall either partially or completely within the redshift bin wavelength range.
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


