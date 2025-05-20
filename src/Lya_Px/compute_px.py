import numpy as np
from Lya_Px.params import *
from Lya_Px.make_skewers import get_skewers
from Lya_Px.compute_p1d import get_p1d
from Lya_Px.px_from_skewers import get_px


def compute_px(healpix, z_alpha, dz, theta_min_array, theta_max_array, wave_desi):    
    '''
    healpix (int): integer, healpix pixel number 
    z_alpha (np.ndarray): 1D array of shape (N,), redshift bin centers (unitless)
    dz (np.ndarray): 1D array of shape (N,), redshift bin widths (unitless)
    theta_min_array (np.ndarray): 1D array of shape (M,), minimum angular separations in radians
    theta_max_array (np.ndarray): 1D array of shape (M,), maximum angular separations in radians
    wave_desi (np.ndarray): 1D array of shape (L,), DESI observed wavelength grid in Angstrom
    Returns:
    k_arr (np.ndarray): 1D array of shape (N_FFT), k-space grid in 1/A
    result_dict (dict): dictionary with keys as tuples (z_bin, theta_bin) and values as dimensionless Px arrays of shape (N_FFT)
    p1d_dict (dict): dictionary with keys as z_bin and values as P1D array of shape (N_FFT)
    px_weights (dict): dictionary with keys as tuples (z_bin, theta_bin) and values as Px of weights of shape (N_FFT)
    no_of_pairs (int): number of pairs of sightlines that were used to compute the Px

    '''
    print('healpix = ', healpix)
    wave_desi_min = wave_desi[0]
    
    # get list of sightline objects in each healpix pixel, which includes the redshift bins which they belong to
    skewers = get_skewers(healpix, deltas_path)
    
    result_dict = {}  # key = (z, θmin, θmax), value = Px array
    px_weights = {}
    p1d_dict = {}
    npairs = {}

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

        # create mask for the FFT grid
        for skewer in skewers:
            skewer.mask_function(wave_fft_grid, lam_min, lam_max)

        # gather the sightlines that fall either partially or completely within the redshift bin wavelength range.
        all_skewers = [s for s in skewers if z_alpha[z] in s.z_bins]
        
        if not all_skewers:
            continue
    
        for skewer in all_skewers:
            # map the sightline onto the FFT grid
            skewer.map_to_fftgrid(wave_fft_grid)

        # compute P1D 
        p1d,p1d_norm = get_p1d(all_skewers)

        for theta in range(len(theta_min_array)):
            # measure Px
            result = get_px(all_skewers, theta_min_array[theta], theta_max_array[theta])
            # check if there are any pairs of sightlines in this theta bin
            no_of_pairs = result[3]
            if no_of_pairs == 0:
                continue
            
            # create keys for Px dict    
            z_bin = z_alpha[z]
            theta_bin = (theta_min_array[theta], theta_max_array[theta])

            # store Px and weights in dicts for each z_bin and theta bin
            result_dict[(z_bin, theta_bin)] = result[0] # dimensionles Px; has to be normalized 
            px_weights[(z_bin, theta_bin)] = result[1]  # Px of weights for normalization
            p1d_dict[z_bin] = p1d_norm # normalized P1D in the redshift bin
            npairs[(z_bin, theta_bin)] = no_of_pairs # number of pairs of sightlines in this theta bin
    
    return k_arr, result_dict, p1d_dict ,px_weights, npairs


