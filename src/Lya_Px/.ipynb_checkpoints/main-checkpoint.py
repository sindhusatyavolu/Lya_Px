import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from config import *
from functions import *
from px_from_pix import *
import sys

# First define the official DESI wavelength grid (all wavelengths that we could possibly care about)
wave_desi_N = 5000
# I know for sure that there is a pixel at 3600A, so let's make sure we cover that one
wave_desi_min = 3600-500*pw_A 
wave_desi_max = wave_desi_min+wave_desi_N*pw_A
print('{:.2f} < lambda < {:.2f} [A]'.format(wave_desi_min, wave_desi_max))
print('{:.3f} < z < {:.3f}'.format(wave_desi_min/LAM_LYA-1, wave_desi_max/LAM_LYA-1))
wave_desi=np.linspace(wave_desi_min,wave_desi_max,wave_desi_N+1)


if len(sys.argv) != 4:
    print("Usage: python main.py <redshift_bin> <redshift_bin_width> <healpix pixel>")
    sys.exit(1)

z_alpha = float(sys.argv[1]) # redshift bin center
dz = float(sys.argv[2]) # redshift bin width

# figure out the center of the bin and its edges, in observed wavelength
lam_cen = LAM_LYA*(1+z_alpha)
lam_min = LAM_LYA*(1+z_alpha-0.5*dz)
lam_max = LAM_LYA*(1+z_alpha+0.5*dz)
print(lam_min,lam_cen,lam_max)

# Create FFT grid in observed wavelength 
# the FFT grid will have a fixed length of pixels (N_fft)  
k = np.fft.fftfreq(N_fft)*2*np.pi/pw_A

# figure out the index of the global (desi) grid that is closer to the center of the redshift bin
i_cen = round((lam_cen-wave_desi_min)/pw_A) 
wave_fft_grid = wave_desi[i_cen-N_fft//2:i_cen+N_fft//2] 
print(wave_fft_grid[0],'< lambda <',wave_fft_grid[-1])

mask_fft_grid = np.ones(N_fft) # placeholder for the mask in the FFT grid
# while we use i to refer to indices in the global (desi) grid, we use j to refer to the FFT grid of this redshift
j_cen = round((lam_cen-wave_fft_grid[0])/pw_A) 
# this should alway be N_fft/2
print(j_cen, N_fft//2)
# figure out the indices (in the FFT grid) that fall within the redshift bin (top hat binning)
j_min = round((lam_min-wave_fft_grid[0])/pw_A)
j_max = round((lam_max-wave_fft_grid[0])/pw_A)
print(j_min,j_max)
mask_fft_grid[:j_min]=0
mask_fft_grid[j_max:]=0

# Read inputs

healpix = int(sys.argv[3])
deltas_path='/global/cfs/cdirs/desi/science/lya/mock_analysis/develop/ifae-ql/qq_desi_y3/v1.0.5/analysis-0/jura-124/raw_bao_unblinding/deltas_lya/Delta/'
delta_file=deltas_path+'delta-%d.fits.gz'%(healpix)
file = fits.open(delta_file)

# get sightlines from the delta file that fall within the fft grid
skewers = get_skewers(wave_fft_grid,mask_fft_grid,file)

# compute separations
separation_angles = get_separations(skewers)

# compute the power spectrum for each separation
get_px(skewers,separation_angles)

# save the results


