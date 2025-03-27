import numpy as np
from Lya_Px.params import *
from Lya_Px.auxiliary import angular_separation
from astropy.io import fits
import argparse

def read_deltas(healpix,deltas_path):
    delta_file=deltas_path+'delta-%d.fits.gz'%(healpix)
    file = fits.open(delta_file)
    return file

def create_skewer_class():
    class Skewers:
        def __init__(self, wave_data, delta_data, weight_data, RA, Dec, z_qso,redshifts,redshift_bins):
            self.wave_data = wave_data
            self.delta_data = delta_data
            self.weight_data = weight_data
            self.weight_data *= (self.wave_data/4500)**3.8 

            self.RA = RA
            self.Dec = Dec
            self.z_qso = z_qso
            
            wave_min = wave_data[0]
            wave_max = wave_data[-1]
            
            lam_bin = LAM_LYA*(1+redshifts)
            lam_min = lam_bin - 0.5*redshift_bins*LAM_LYA
            lam_max = lam_bin + 0.5*redshift_bins*LAM_LYA 
            
            self.z_bins = []
            self.z_bins_width = []

            for i in range(len(redshifts)):
                if wave_min < lam_max[i] and wave_max > lam_min[i]:
                    self.z_bins.append(float(redshifts[i]))
                    self.z_bins_width.append(float(redshift_bins[i]))

        def map_to_fftgrid(self,wave_fft_grid,mask_fft_grid):

            
            delta_fft_grid = np.zeros(N_fft)
            weight_fft_grid = np.zeros(N_fft)

            # Map the observed spectrum to the FFT grid for this particular redshift bin    
            j_min_data=round((self.wave_data[0]-wave_fft_grid[0])/pw_A)
            j_max_data=round((self.wave_data[-1]-wave_fft_grid[0])/pw_A)
            

            # figure out whether the spectrum is cut at low-z or at high-z
            loz_cut=False
            hiz_cut=False
            if j_min_data < 0:
                loz_cut=True
                if j_max_data >=0:
                    delta_fft_grid[:j_max_data]=self.delta_data[-j_min_data+1:]
                    weight_fft_grid[:j_max_data]=self.weight_data[-j_min_data+1:]
            if j_max_data >= N_fft:
                hiz_cut=True
                if j_min_data < N_fft:
                    delta_fft_grid[j_min_data:]=self.delta_data[:N_fft-j_max_data-1]
                    weight_fft_grid[j_min_data:]=self.weight_data[:N_fft-j_max_data-1]
            if loz_cut==False and hiz_cut==False:
                delta_fft_grid[j_min_data:j_max_data+1]=self.delta_data
                weight_fft_grid[j_min_data:j_max_data+1]=self.weight_data


            weight_fft_grid *= mask_fft_grid
             
            self.delta_fft_grid = delta_fft_grid
            self.weight_fft_grid = weight_fft_grid
            self.mask_fft_grid = mask_fft_grid
            
            return None 
                               
    return Skewers


def get_p1d(all_skewers,wave_fft_grid,mask_fft_grid):
    p1d = np.zeros(N_fft)
    p1d = np.zeros(N_fft)
    for skewer in all_skewers:
        skewer.map_to_fftgrid(wave_fft_grid,mask_fft_grid)
        delta = skewer.delta_fft_grid
        weight = skewer.weight_fft_grid
        fft_weighted_delta = np.fft.fft(delta * weight)
        p1d += np.abs(fft_weighted_delta)**2

    # Normalize
    p1d_norm = (pw_A / N_fft) * (1 / len(all_skewers)) * p1d
    return p1d, p1d_norm


def create_fft_grid(wave_desi_min,z_alpha,dz,wave_desi_N,wave_desi):
    fft_grid = {}

    # figure out the center of the bin and its edges, in observed wavelength
    lam_cen = LAM_LYA*(1+z_alpha)
    lam_min = LAM_LYA*(1+z_alpha-0.5*dz)
    lam_max = LAM_LYA*(1+z_alpha+0.5*dz)
    print(lam_min,lam_cen,lam_max)
    fft_grid['lam_cen'] = lam_cen
    fft_grid['lam_min'] = lam_min
    fft_grid['lam_max'] = lam_max

    # Create FFT grid in observed wavelength 
    # the FFT grid will have a fixed length of pixels (N_fft)  
    k = np.fft.fftfreq(N_fft)*2*np.pi/pw_A
    fft_grid['k'] = k

    # figure out the index of the global (desi) grid that is closer to the center of the redshift bin
    i_cen = round((lam_cen-wave_desi_min)/pw_A) 
    wave_fft_grid = wave_desi[i_cen-N_fft//2:i_cen+N_fft//2] 

    fft_grid['wave_fft_grid'] = wave_fft_grid

    if i_cen-N_fft//2 < 0 or i_cen+N_fft//2 > wave_desi_N:
        print('FFT grid is out of bounds, try different N_fft')
        exit(1) 

    print(wave_fft_grid[0],'< lambda <',wave_fft_grid[-1])

    # velocity grid
    vel = wave_to_velocity(wave_fft_grid) # in km/s
    dv = np.mean(np.diff(vel)) # in km/s
    print(dv,np.diff(vel))
    k_vel = np.fft.fftfreq(N_fft,d=dv)*2*np.pi # s/km
    fft_grid['k_vel'] = k_vel

    return fft_grid



