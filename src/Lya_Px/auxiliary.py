import numpy as np
import h5py
from Lya_Px.params import *

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
def save_to_hdf5(filename,z,dz,px,k_arr,theta_min_array,theta_max_array,px_var,px_weights,p1d):
    with h5py.File(filename, 'w') as f:
        # shared data
        f.create_dataset('k_arr', data=k_arr)

        # shared metadata 
        f.attrs['z'] = z
        f.attrs['dz'] = dz
        f.create_dataset('p1d',data=p1d)
        
        # group for each theta bin
        for i in range(len(px)):
            g = f.create_group('theta_%d_%d'%(theta_min_array[i]*RAD_TO_ARCMIN,theta_max_array[i]*RAD_TO_ARCMIN))
            g.create_dataset('px', data=px[i])
            g.create_dataset('px_var', data=px_var[i])
            g.create_dataset('px_weights', data=px_weights[i])
            g.attrs['theta_min'] = theta_min_array[i]
            g.attrs['theta_max'] = theta_max_array[i]

    return None

def save_results(px_avg, px_var, px_weights, p1d_avg, k_arr, z_alpha, dz, output_path, healpixlist):
    for z_bin in range(len(z_alpha)):
        # Pull out all keys that match this z_bin
        matching_keys = [key for key in px_avg if np.isclose(key[0], float(z_alpha[z_bin]))]

        theta_mins = [key[1][0] for key in matching_keys]
        theta_maxs = [key[1][1] for key in matching_keys]
        px_data    = [px_avg[key] for key in matching_keys]
        px_vars    = [px_var[key] for key in matching_keys]
        px_weights_data = [px_weights[key] for key in matching_keys]
        p1d = [p1d_avg[z_alpha[z_bin]]]

        filename = output_path + f'px-nhp_{len(healpixlist)}_zbin_{z_alpha[z_bin]:.1f}.hdf5'

        save_to_hdf5(
            filename,
            z_alpha[z_bin],
            dz[z_bin],
            px_data,
            k_arr,
            theta_mins,
            theta_maxs,
            px_vars,
            px_weights_data,
            p1d
        )
        print('Saved to', filename)
    return None