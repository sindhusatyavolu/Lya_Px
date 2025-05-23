import numpy as np
from Lya_Px.params import *
from Lya_Px.avg_healpix import avg_over_healpixels
from Lya_Px.auxiliary import save_results
import cProfile
import pstats
from Lya_Px.compute_px import compute_px
from multiprocessing import Pool
import os

def main():
    
    #First define the official DESI observed wavelength grid (all wavelengths that we could possibly care about)
    wave_desi_N = 5000
    # I know for sure that there is a pixel at 3600A, so let's make sure we cover that one
    wave_desi_min = 3600-500*pw_A 
    wave_desi_max = wave_desi_min+wave_desi_N*pw_A
    print('{:.2f} < lambda < {:.2f} [A]'.format(wave_desi_min, wave_desi_max))
    print('{:.3f} < z < {:.3f}'.format(wave_desi_min/LAM_LYA-1, wave_desi_max/LAM_LYA-1))
    wave_desi=np.linspace(wave_desi_min,wave_desi_max,wave_desi_N+1)
    
    # Get theta bins in which Px will be measured
    theta_min_array = theta_array[:,0]*ARCMIN_TO_RAD # theta in Radians
    theta_max_array = theta_array[:,1]*ARCMIN_TO_RAD # theta in Radians 

    args = [(hp, z_alpha, dz, theta_min_array, theta_max_array, wave_desi) for hp in healpixlist] # z_alpha and dz are the redshift bin and bin widths, hp is the healpix number

    # Use multiprocessing to compute px for each healpix
    with Pool(ncpus) as pool:
        results = pool.starmap(compute_px, args)
    
    # average over healpixels and compute covariance
    k_arr, px_avg, px_var, px_weights, p1d_avg, covariance = avg_over_healpixels(results)  

    # save results to hdf5 file
    save_results(px_avg, px_var, px_weights, p1d_avg, covariance, k_arr, z_alpha, dz, output_path, healpixlist,pw_A)
    

if __name__=="__main__":
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats("time").print_stats(10)







