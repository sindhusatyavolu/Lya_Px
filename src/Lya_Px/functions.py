import numpy as np
from config import *
from auxiliary import *
from astropy.io import fits
import argparse
from config import *

def read_inputs():
    parser = argparse.ArgumentParser(description="Run Lyman alpha cross power spectrum analysis with given parameters.")

    # All required arguments
    parser.add_argument("--redshifts", type=str, help="Column1 :Redshift value (e.g., 2.2) Column2: Redshift bin width (e.g., 0.2)")
    parser.add_argument("output_path", type=str, help="Directory to save output files")
    parser.add_argument("--theta_file", type=str, required=True,
                        help="Path to theta_values.txt file (e.g., /path/to/theta_values.txt)")
    parser.add_argument("--deltas_path", type=str, required=True,help="Path to delta files")

    args = parser.parse_args()

    # Extract the arguments
    redshifts = np.loadtxt(args.redshifts,skiprows=1) # redshifts and redshift bin widths
    z_alpha = redshifts[:,0] # redshift bin center
    dz = redshifts[:,1] # redshift bin width
    out_path = args.output_path # output path
    theta_file = args.theta_file
    # Print to verify values
    print(f"Redshift: {z_alpha}")
    print(f"Redshift bin: {dz}")
    print(f"Output directory: {out_path}")
    print(f"Theta file: {theta_file}")

    # read theta values from file with first column as theta_min and second column as theta_max
    theta_array = np.loadtxt(theta_file, skiprows=1)   # first row is header  (S:keep or change?)
    # Load theta_values.txt from the provided path
    #assert theta_min_array.size == theta_max_array.size

    deltas_path = args.deltas_path 
    
    #'/global/cfs/cdirs/desi/science/lya/mock_analysis/develop/ifae-ql/qq_desi_y3/v1.0.5/analysis-0/jura-124/raw_bao_unblinding/deltas_lya/Delta/'
    return z_alpha, dz, out_path, theta_array, deltas_path

def read_deltas(healpix,deltas_path):
    delta_file=deltas_path+'delta-%d.fits.gz'%(healpix)
    file = fits.open(delta_file)
    return file


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

