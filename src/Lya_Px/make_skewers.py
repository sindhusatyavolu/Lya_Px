import numpy as np
from Lya_Px.params import *
# import argparse
from collections import defaultdict
import fitsio

def get_skewers(healpix,deltas_path):
    '''
    healpix (int): integer, healpix pixel number 
    deltas_path (str): path to the directory containing the delta files
    Returns:
    skewers (list): list of Skewers objects, each containing the data for a single sightline

    '''

    # read skewers from the healpix pixel
    delta_file=deltas_path+'delta-%d.fits.gz'%(healpix)
    
    Skewers = create_skewer_class()
    skewers = []

    f =  fitsio.FITS(delta_file)

    for hdu in range(1,len(f)):
        RA = f[hdu].read_header()['RA']
        Dec = f[hdu].read_header()['DEC']
        z_qso = f[hdu].read_header()['Z']
        loglam = f[hdu].read()['LOGLAM']
        wave_data=10.0**(loglam)
        delta_data=f[hdu].read('DELTA')
        weight_data=f[hdu].read('WEIGHT')
        # create skewer object for each sightline in the healpix pixel
        skewer = Skewers(wave_data, delta_data, weight_data,RA, Dec, z_qso,z_alpha,dz)
        skewers.append(skewer)
   
    return skewers
    
def create_skewer_class():
    class Skewers:
        def __init__(self, wave_data, delta_data, weight_data, RA, Dec, z_qso,redshifts,redshift_bins):
            '''
            wave_data (np.ndarray): 1D array of shape (N,), observed wavelength in Angstrom
            delta_data (np.ndarray): 1D array of shape (N,), delta in real space
            weight_data (np.ndarray): 1D array of shape (N,), weight in real space
            RA (float): right ascension in radians
            Dec (float): declination in radians
            z_qso (float): redshift of the quasar
            redshifts (np.ndarray): 1D array of shape (M,), redshift bin centers
            redshift_bins (np.ndarray): 1D array of shape (M,), redshift bin widths

            '''
            self.wave_data = wave_data # observed wavelength in Angstrom
            self.delta_data = delta_data # delta in real space
            self.weight_data = weight_data # weight in real space
            # Equation 
            self.weight_data *= (self.wave_data/4500)**3.8  

            self.RA = RA  # radians
            self.Dec = Dec # radians
            self.z_qso = z_qso 
            
            wave_min = wave_data[0] 
            wave_max = wave_data[-1]
            
            lam_bin = LAM_LYA*(1+redshifts)  
            lam_min = lam_bin - 0.5*redshift_bins*LAM_LYA  
            lam_max = lam_bin + 0.5*redshift_bins*LAM_LYA 
            
            self.z_bins = []
            self.z_bins_width = []

            # find the redshift bins that overlap with the observed wavelength range of this skewer
            for i in range(len(redshifts)):
                if wave_min < lam_max[i] and wave_max > lam_min[i]:
                    self.z_bins.append(float(redshifts[i]))
                    self.z_bins_width.append(float(redshift_bins[i]))

        
        def map_to_fftgrid(self,wave_fft_grid):
            '''
            Function to map sightline to the FFT grid
            wave_fft_grid (np.ndarray): 1D array of shape (N_FFT,), FFT grid in observed wavelength
            mask_fft_grid (np.ndarray): 1D array of shape (N_FFT,), mask in the FFT grid
            delta_fft_grid (np.ndarray): 1D array of shape (N_FFT,), deltas in the FFT grid, array elements are zeroed out if they fall outside the redshift bin wavelength range
            weight_fft_grid (np.ndarray): 1D array of shape (N_FFT,), weight in the FFT grid corresponding to deltas
            
            '''
            
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


            weight_fft_grid *= self.mask_fft_grid
             
            self.delta_fft_grid = delta_fft_grid # real space delta in FFT grid
            self.weight_fft_grid = weight_fft_grid # real space weight in FFT grid
            #self.mask_fft_grid = mask_fft_grid # real space mask in FFT grid 
            self.pw_A = pw_A
            return None 
        
        def mask_function(self,wave_fft_grid,lam_min,lam_max):
            '''
            Function to create mask for the FFT grid
            wave_fft_grid (np.ndarray): 1D array of shape (N_FFT,), FFT grid in observed wavelength
            lam_min (float): minimum wavelength of the redshift bin in Angstrom
            lam_max (float): maximum wavelength of the redshift bin in Angstrom
            mask_fft_grid (np.ndarray): 1D array of shape (N_FFT,), mask in the FFT grid
    
            ''' 
            self.mask_fft_grid = np.ones(N_fft)
            # zero out the mask values that fall outside the redshift bin wavelength range
            j_min = round((lam_min - wave_fft_grid[0]) / pw_A)
            j_max = round((lam_max - wave_fft_grid[0]) / pw_A)
            self.mask_fft_grid[:j_min] = 0
            self.mask_fft_grid[j_max:] = 0
            return None

                               
    return Skewers



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


  