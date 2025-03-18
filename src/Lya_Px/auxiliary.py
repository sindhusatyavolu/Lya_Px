import numpy as np
from config import *

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

# anglular separation to transverse distance conversion

# save outputs
def save_to_hdf5(filename,px,k_arr,theta_min_array,theta_max_array,N_fft,dvel,N_skewers):
    with h5py.File(filename, 'w') as f:
        # shared data
        f.create_dataset('k_arr', data=k_arr)

        # shared metadata 
        f.attrs['N_fft'] = N_fft
        f.attrs['dvel'] = dvel
        f.attrs['N_skewers'] = N_skewers
        
        # group for each theta bin
        for i in range(len(px)):
            g = f.create_group('theta_%d_%d'%(theta_min_array[i]*RAD_TO_ARCMIN,theta_max_array[i]*RAD_TO_ARCMIN))
            g.create_dataset('px', data=px[i])
            g.attrs['theta_min'] = theta_min_array[i]
            g.attrs['theta_max'] = theta_max_array[i]

    return None

