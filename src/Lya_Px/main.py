import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import sys
import h5py
from params import *
from functions import read_deltas, create_skewer_class, get_p1d
from px_from_pix import get_px
from auxiliary import angular_separation,save_to_hdf5,wave_to_velocity
import cProfile
import pstats

def main():
    #First define the official DESI wavelength grid (all wavelengths that we could possibly care about)
    wave_desi_N = 5000
    # I know for sure that there is a pixel at 3600A, so let's make sure we cover that one
    wave_desi_min = 3600-500*pw_A 
    wave_desi_max = wave_desi_min+wave_desi_N*pw_A
    print('{:.2f} < lambda < {:.2f} [A]'.format(wave_desi_min, wave_desi_max))
    print('{:.3f} < z < {:.3f}'.format(wave_desi_min/LAM_LYA-1, wave_desi_max/LAM_LYA-1))
    wave_desi=np.linspace(wave_desi_min,wave_desi_max,wave_desi_N+1)
    
    # Take as inputs the redshift and theta bins 
    theta_min_array = theta_array[:,0]*ARCMIN_TO_RAD
    theta_max_array = theta_array[:,1]*ARCMIN_TO_RAD

    skewers_by_healpix = {}

    # For each healpix file, sort sightlines into respective redshift bins
    for hp in range(len(healpixlist)):
        
        file  =  read_deltas(healpixlist[hp],deltas_path)
        Skewers = create_skewer_class()
        skewers = []
        for hdu in file[1:]:
                wave_data=10.0**(hdu.data['LOGLAM'])
                delta_data=hdu.data['DELTA']
                weight_data=hdu.data['WEIGHT']            
                RA = hdu.header['RA']
                Dec = hdu.header['DEC']
                z_qso = hdu.header['Z']
                skewer = Skewers(wave_data, delta_data, weight_data,RA, Dec, z_qso,z_alpha,dz)
                skewers.append(skewer)

        skewers_by_healpix[hp] = skewers 

    print(skewers_by_healpix[0][0].z_bins)
    print(type(skewers_by_healpix))

    # compute P1D for each redshift bin
    for z in range(len(z_alpha)):
        print('z = {:.3f}'.format(z_alpha[z]),'dz = {:.3f}'.format(dz[z]))
        # create fft grid for a given redshift bin
        lam_cen = LAM_LYA * (1 + z_alpha[z])
        lam_min = LAM_LYA * (1 + z_alpha[z] - 0.5 * dz[z])
        lam_max = LAM_LYA * (1 + z_alpha[z] + 0.5 * dz[z])
        i_cen = round((lam_cen - wave_desi_min) / pw_A)
        wave_fft_grid = wave_desi[i_cen - N_fft//2 : i_cen + N_fft//2]
        k_arr = np.fft.fftfreq(N_fft)*2*np.pi/pw_A
       
        mask_fft_grid = np.ones(N_fft)
        j_min=round((lam_min-wave_fft_grid[0])/pw_A)
        j_max=round((lam_max-wave_fft_grid[0])/pw_A)
        print(j_min,j_max)
        mask_fft_grid[:j_min]=0
        mask_fft_grid[j_max:]=0

        # get all skewers that have data in this redshift bin
        all_skewers = []
        for healpix in range(len(healpixlist)):
            for skewer in skewers_by_healpix[healpix]:
                if z_alpha[z] in skewer.z_bins:
                    all_skewers.append(skewer)

        # compute P1D
        p1d, p1d_norm = get_p1d(all_skewers,wave_fft_grid,mask_fft_grid)

        if P1D:
            plt.plot(k_arr[:N_fft//2],p1d_norm[:N_fft//2])
            plt.show()

        # for each theta bin, compute Px 
        px = np.empty((len(theta_min_array),N_fft))
        px_var = np.empty((len(theta_min_array),N_fft))
        px_weights = np.empty((len(theta_min_array),N_fft))

        for theta in range(len(theta_min_array)):
            result = get_px(all_skewers,theta_min_array[theta],theta_max_array[theta])
            px[theta,:] = result[0]
            px_weights[theta,:] = result[1]
            px_var[theta,:] = result[2]
            px_sum  = result[3]
            #print(px_sum-px[theta,:])
            
            assert np.allclose(px_sum,px[theta,:])
            
            px_norm =  px[theta,:] * (pw_A / N_fft)           # in A
            
            if plot_px:
                plt.plot(k_arr[:N_fft//2],px_norm[:N_fft//2],label='%f-%f arcmin'%(theta_min_array[theta]*RAD_TO_ARCMIN,theta_max_array[theta]*RAD_TO_ARCMIN))
        
        plt.xlabel('$k$ [A]')
        plt.ylabel(r'$P(k)$[A]')
        plt.legend()
        plt.show()

        # save results to file
        filename = output_path + 'px_z_{:.3f}.h5'.format(z_alpha[z])
        save_to_hdf5(filename,z_alpha[z],dz[z],px,k_arr,theta_min_array,theta_max_array,px_var,px_weights,p1d_norm)    
        

if __name__=="__main__":
    main()







"""
        path = "/Users/ssatyavolu/px-500-2.20-0.20.hdf5"
        with h5py.File(path, 'r') as f:
            # Load shared datasets
            k_arr = f['k_arr'][:]
            p1d = f['p1d'][:]

            # Load attributes
            n_fft = f.attrs['N_fft']
            dvel = f.attrs['dvel']
            N_skewers = f.attrs['N_skewers']

            # Loop over all theta groups
            px = []
            px_var = []
            px_weights = []
            theta_bins = []
            # sort only the theta_* groups
            theta_keys = sorted([key for key in f.keys() if key.startswith('theta_')],
                            key=lambda k: float(k.split('_')[1]))  # sort by theta_min in arcmin

            for key in theta_keys:
                g = f[key]
                px.append(g['px'][:])
                px_var.append(g['px_var'][:])
                px_weights.append(g['px_weights'][:])
                theta_bins.append((g.attrs['theta_min'], g.attrs['theta_max']))
        
        
        k_arr = np.array(k_arr)
        px = np.array(px)*pw_A/N_fft
        px_var = np.array(px_var)
        px_weights = np.array(px_weights)

        print(px[-1]/px_norm)
        plt.plot(k_arr[:N_fft//2],px[-1][:N_fft//2]/px_norm[:N_fft//2],label='500-2.20-0.20')
        plt.xlabel('k [A]')
        plt.ylabel(r'$P(k)/P^{updated}(k)$')
        plt.legend()
        plt.savefig('ratio_px_old_v_new.png',dpi=350,bbox_inches='tight')
        plt.show()
        sys.exit()
"""