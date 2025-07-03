import numpy as np
import h5py
from Lya_Px.params import *
from collections import defaultdict

# function to measure the angular separation between two points on the sky
def angular_separation(ra1, dec1, ra2, dec2): 
    # Calculate the difference in right ascension
    delta_ra = ra2 - ra1  # in radians
    
    # Apply the formula for angular separation
    angular_distance = np.arccos(np.sin(dec1) * np.sin(dec2) +
                                 np.cos(dec1) * np.cos(dec2) * np.cos(delta_ra))
    
    return angular_distance # in radians

# observed wavelength to velocity units conversion
def wave_to_velocity(wave):
    return (wave - LAM_LYA) / LAM_LYA * c_SI

# angular separation to transverse distance conversion

# save outputs
def save_to_hdf5(filename,z,dz,px,k_arr,theta_min_array,theta_max_array,px_var,px_cov,px_weights,p1d,pw_A):
    '''
    Function to write to hdf5 file
    Parameters:
    ----------
    filename (str): name of the output file
    z (float): redshift bin center
    dz (float): redshift bin width
    px (np.ndarray): 2D array of shape (N_FFT, M), where N_FFT is the number of FFT pixels and M is the number of theta bins
    k_arr (np.ndarray): 1D array of shape (N_FFT), k-space grid in 1/A
    theta_min_array (np.ndarray): 1D array of shape (M,), minimum angular separations in radians
    theta_max_array (np.ndarray): 1D array of shape (M,), maximum angular separations in radians
    px_var (np.ndarray): 2D array of shape (N_FFT, M), variance of Px
    px_cov (np.ndarray): 2D array of shape (N_FFT, M), covariance matrix of Px
    px_weights (np.ndarray): 2D array of shape (N_FFT, M), Px of weights
    p1d (np.ndarray): 1D array of shape (N_FFT,), P1D array
    pw_A (float): pixel width in Angstroms

    '''
    with h5py.File(filename, 'w') as f:
        # shared data
        f.create_dataset('k_arr', data=k_arr)

        # shared metadata 
        f.attrs['z'] = z
        f.attrs['dz'] = dz
        f.create_dataset('p1d',data=p1d)
        f.attrs['N_fft'] = len(k_arr)
        f.attrs['pixel_width_A'] = pw_A
        
        # group for each theta bin
        for i in range(len(px)):
            g = f.create_group('theta_%d_%d'%(theta_min_array[i]*RAD_TO_ARCMIN,theta_max_array[i]*RAD_TO_ARCMIN))
            g.create_dataset('px', data=px[i])
            g.create_dataset('px_var', data=px_var[i])
            g.create_dataset('px_weights', data=px_weights[i])
            g.create_dataset('covariance', data=px_cov[i])
            g.attrs['theta_min'] = theta_min_array[i]
            g.attrs['theta_max'] = theta_max_array[i]

    return None

def save_results(px_avg, px_weights, p1d_avg, covariance, k_arr, z_alpha, dz, output_path, healpixlist, pw_A):
    '''
    Save the results to hdf5 files for each z_bin 
    Parameters:
    ----------
    px_avg (dict): dictionary with keys as tuples (z_bin, theta_bin) and values as dimensionless Px arrays of shape (N_FFT)
    px_var (dict): dictionary with keys as tuples (z_bin, theta_bin) and values as variance of Px arrays of shape (N_FFT)
    px_weights (dict): dictionary with keys as tuples (z_bin, theta_bin) and values as Px of weights of shape (N_FFT)
    p1d_avg (dict): dictionary with keys as z_bin and values as P1D array of shape (N_FFT)
    covariance (dict): dictionary with keys as tuples (z_bin, theta_bin) and values as covariance matrix of Px arrays of shape (N_FFT, N_FFT)
    k_arr (np.ndarray): 1D array of shape (N_FFT), k-space grid in 1/A
    z_alpha (np.ndarray): 1D array of shape (N,), redshift bin centers 
    dz (np.ndarray): 1D array of shape (N,), redshift bin widths 
    output_path (str): path to the output directory
    healpixlist (list): list of shape M with healpix numbers
    pw_A (float): pixel width in Angstroms

    '''
    filename = output_path + f'px-nhp_{len(healpixlist)}-zbins_{len(z_alpha)}-thetabins_{len(theta_array)}.hdf5'

    with h5py.File(filename, 'w') as f:
        # shared data
        f.create_dataset('k_arr', data=k_arr)
        
        f.attrs['N_fft'] = len(k_arr)
        f.attrs['pixel_width_A'] = pw_A
        
        #print(len(px_avg), 'z-theta bins found')
        #print(list(px_avg.keys()))
        # group for each z and theta bin
        for i in range(len(px_avg)):
            z_bin, theta_bin = list(px_avg.keys())[i]
            theta_min, theta_max = theta_bin
            g = f.create_group(f'z_{z_bin:.1f}_theta_{theta_min*RAD_TO_ARCMIN:.1f}_{theta_max*RAD_TO_ARCMIN:.1f}')
            g.create_dataset('p1d',data=p1d_avg[z_bin])
            g.create_dataset('px', data=px_avg[(z_bin, theta_bin)])
            #g.create_dataset('px_var', data=px_var[(z_bin, theta_bin)])
            g.create_dataset('px_weights', data=px_weights[(z_bin, theta_bin)])
            g.create_dataset('covariance', data=covariance[(z_bin, theta_bin)])
            g.attrs['theta_min'] = theta_min
            g.attrs['theta_max'] = theta_max

    print('Saved to', filename)
    return None


def save_hp(results,output_path,healpixlist,z_alpha, dz, pw_A):

    px_all = defaultdict(list)  # key = (z, theta_bin), value = list of Px arrays
    px_weights_all = defaultdict(list)  
    p1d_all = defaultdict(list)
    no_of_pairs = defaultdict(list)  
    #weights_average = defaultdict(list)  

    # accumulate results in only redshift and theta bins that exist for each healpixel --- no all healpixels are needed

    for k_arr, px_dict,p1d_dict, px_weights, npairs in results:
        for key in px_dict:
            px_all[key].append(px_dict[key])  
            px_weights_all[key].append(px_weights[key])
            p1d_all[key[0]].append(p1d_dict[key[0]])
            no_of_pairs[key].append(npairs[key])
            #weights_average[key].append(w_avg[key])

    filename = output_path + f'px-nhp_{len(healpixlist)}-zbins_{len(z_alpha)}-thetabins_{len(theta_array)}.hdf5'

    # save results to hdf5 file
    with h5py.File(filename,'w') as f:
        
        f.create_dataset('k_arr', data=k_arr)
        
        f.attrs['N_fft'] = len(k_arr)
        f.attrs['pixel_width_A'] = pw_A
        f.attrs['z'] = z_alpha
        f.attrs['dz'] = dz

        # group for each z and theta bin, save all contributing healpix results
        for key in px_all:
            z_bin, theta_bin = key
            theta_min, theta_max = theta_bin
            g = f.create_group(f'z_{z_bin:.2f}_theta_{theta_min*RAD_TO_ARCMIN:.2f}_{theta_max*RAD_TO_ARCMIN:.2f}')
            g.create_dataset('p1d', data=p1d_all[key[0]])
            #print('px_all[key] shape:', np.shape(px_all[key]))
            g.create_dataset('px', data=px_all[key])
            g.create_dataset('px_weights', data=px_weights_all[key])
            g.create_dataset('no_of_pairs', data=no_of_pairs[key])
            #g.create_dataset('avg_weights_per_healpix', data=weights_average[key])
            g.attrs['theta_min'] = theta_min
            g.attrs['theta_max'] = theta_max
    
    print('Saved to', filename)        

    return None
