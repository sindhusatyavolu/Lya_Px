import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import sys
import h5py
from config import *
from functions import *
from px_from_pix import *
from auxiliary import *
from make_plots import *

import argparse

import cProfile
import pstats

def main():
    # First define the official DESI wavelength grid (all wavelengths that we could possibly care about)
    wave_desi_N = 5000
    # I know for sure that there is a pixel at 3600A, so let's make sure we cover that one
    wave_desi_min = 3600-500*pw_A 
    wave_desi_max = wave_desi_min+wave_desi_N*pw_A
    print('{:.2f} < lambda < {:.2f} [A]'.format(wave_desi_min, wave_desi_max))
    print('{:.3f} < z < {:.3f}'.format(wave_desi_min/LAM_LYA-1, wave_desi_max/LAM_LYA-1))
    wave_desi=np.linspace(wave_desi_min,wave_desi_max,wave_desi_N+1)

    z_alpha, dz, healpix, out_path, theta_array, deltas_path = read_inputs()  

    for i in range(len(z_alpha)):
        print('z=%.2f, dz=%.2f, healpix=%d'%(z_alpha[i],dz[i],healpix))
        fft_grid = create_fft_grid(wave_desi_min,z_alpha[i],dz[i],wave_desi_N,wave_desi)
        
        wave_fft_grid = fft_grid['wave_fft_grid']
        k_arr = fft_grid['k']
        k_vel = fft_grid['k_vel']

        mask_fft_grid = map_to_fft_grid(wave_fft_grid,fft_grid['lam_cen'],fft_grid['lam_min'],fft_grid['lam_max']) 
        
        # Load deltas
        file = read_deltas(healpix,deltas_path)

        Skewers = create_skewer_class()

        # get sightlines from the delta file and map them to the FFT grid, for a given redshift bin
        skewers = []
        count = 0
        
        for hdu in file[1:]:
            wave_data=10.0**(hdu.data['LOGLAM'])
            delta_data=hdu.data['DELTA']
            weight_data=hdu.data['WEIGHT']            
            RA = hdu.header['RA']
            Dec = hdu.header['DEC']
            z_qso = hdu.header['Z']
            #if count%100 == 0:
            #    print(wave_data)

            # ignore skewers with no data at all in the redshift bin
            #if wave_data[-1]<lam_min or wave_data[0]>lam_max:
            #    continue

            skewer = Skewers(wave_data, delta_data, weight_data, None, None, None, None, RA, Dec, z_qso)
            skewer.map_to_fftgrid(wave_fft_grid,mask_fft_grid)
            skewers.append(skewer)
            count += 1

        # check that the first skewer is mapped correctly
        print(skewers[0].RA,skewers[0].Dec,skewers[0].z_qso)
        
        N_skewers = len(skewers)
        print('Number of skewers:',N_skewers)
        norm_factor = pw_A/N_fft #*1/N_skewers # ignoring the resolution function for now

        vel = wave_to_velocity(wave_fft_grid) # in km/s
        dv = np.mean(np.diff(vel)) # in km/s
        norm_factor_vel = dv/N_fft*1/N_skewers

        # compute P1D
        p1d = get_p1d(skewers)
        p1d_norm = norm_factor*p1d
        if P1D:
            plt.plot(k_arr[:N_fft//2],p1d_norm[:N_fft//2])
            plt.title('z=%.2f, dz=%.2f, healpix=%d'%(z_alpha[i],dz[i],healpix))
            plt.xlabel('k [1/A]')
            plt.ylabel('P1D [A]')
            #plt.savefig('p1d-%d.png'%(healpix))
            plt.show()
            #clear image
            #plt.clf()
        
        theta_min_array = theta_array[:,0]*ARCMIN_TO_RAD
        theta_max_array = theta_array[:,1]*ARCMIN_TO_RAD
        print('theta_min_array:',theta_min_array)
        print('theta_max_array:',theta_max_array)

        px = np.empty((len(theta_min_array),N_fft))
        px_var = np.empty((len(theta_min_array),N_fft))
        px_weights = np.empty((len(theta_min_array),N_fft))

        for l in range(len(theta_min_array)):
            print('theta_min:',theta_min_array[l],'theta_max:',theta_max_array[l])
            result = get_px(skewers,theta_min_array[l],theta_max_array[l])
            px[l,:] = result[0]
            px_weights[l,:] = result[1]
            px_var[l,:] = result[2]
            px_sum  = result[3]
            print(px_sum-px[l,:])
            assert np.allclose(px_sum,px[l,:])
            px[l,:] *= norm_factor

        if plot_px:
            for k in range(len(theta_min_array)):
                plt.plot(k_arr[:N_fft//2],px[k,:N_fft//2]*norm_factor,label='%f-%f arcmin'%(theta_min_array[k]*RAD_TO_ARCMIN,theta_max_array[k]*RAD_TO_ARCMIN))
            plt.title('z=%.2f, dz=%.2f, healpix=%d'%(z_alpha[i],dz[i],healpix))
            plt.xlabel('$k$ [1/A]')
            plt.ylabel(r'$P_{\times}$ [A]')
            plt.legend()
            plt.savefig(out_path+'px-%d-%d-%d-%d-%d.png'%(healpix,theta_min_array[0]*RAD_TO_ARCMIN,theta_max_array[0]*RAD_TO_ARCMIN,theta_min_array[1]*RAD_TO_ARCMIN,theta_max_array[1]*RAD_TO_ARCMIN),bbox_inches='tight',dpi=350)
            plt.show()

        if plot_px_vel:
            for i in range(len(theta_min_array)):
                plt.plot(k_vel[:N_fft//2],(k_vel[:N_fft//2]*px[i,:N_fft//2]*norm_factor_vel)/np.pi,label='%f-%f arcmin'%(theta_min_array[i]*RAD_TO_ARCMIN,theta_max_array[i]*RAD_TO_ARCMIN))
                plt.title('z=%.2f, dz=%.2f, healpix=%d'%(z_alpha,dz,healpix))
                plt.xscale('log')
                #plt.yscale('log')
                plt.xlabel('$k$ [s/km]')
                plt.ylabel(r'$kP_{\times}/\pi$')
                plt.legend()
                plt.show()
                #plt.savefig(out_path+'px-%d-%d-%d-%d-%d-vel-selected.png'%(healpix,theta_min_array[0]*RAD_TO_ARCMIN,theta_max_array[0]*RAD_TO_ARCMIN,theta_min_array[1]*RAD_TO_ARCMIN,theta_max_array[1]*RAD_TO_ARCMIN))
        # save the results
        #outfilename = out_path+'/'+'px-%d-%.2f-%.2f.hdf5'%(healpix,z_alpha[i],dz[i])
        #save_to_hdf5(outfilename,px,k_arr,theta_min_array,theta_max_array,N_fft,dv,N_skewers,px_var,px_weights,p1d) # nfft, variance, average over skewers of square of weighted mask fft grid  


    # save the results
    #outfilename = out_path+'/'+'px-%d-%.2f-%.2f.hdf5'%(healpix,z_alpha,dz)
    #save_to_hdf5(outfilename,px,k,theta_min_array,theta_max_array,N_fft,dv,N_skewers,px_var,px_weights,p1d) # nfft, variance, average over skewers of square of weighted mask fft grid  


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()

    stats = pstats.Stats(pr)
    stats.sort_stats("cumulative").print_stats(30)  # top 30 slowest entries

