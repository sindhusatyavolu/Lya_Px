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

if len(sys.argv) != 5:
    print("Usage: python main.py <redshift_bin> <redshift_bin_width> <healpix pixel> <output_path>")
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

if i_cen-N_fft//2 < 0 or i_cen+N_fft//2 > wave_desi_N:
    print('FFT grid is out of bounds, try different N_fft')
    exit(1) 

print(wave_fft_grid[0],'< lambda <',wave_fft_grid[-1])

# velocity grid
vel = wave_to_velocity(wave_fft_grid) # in km/s
dv = np.mean(np.diff(vel)) # in km/s
print(dv,np.diff(vel))
k_vel = np.fft.fftfreq(N_fft,d=dv)*2*np.pi # s/km

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

deltas_path = '/global/cfs/cdirs/desi/science/lya/mock_analysis/develop/ifae-ql/qq_desi_y3/v1.0.5/analysis-0/jura-124/raw_bao_unblinding/deltas_lya/Delta/'
file = read_deltas(healpix,deltas_path)

class Skewers:
    def __init__(self, wave_data, delta_data, weight_data, delta_fft_grid, weight_fft_grid, j_min_data,j_max_data, fft_delta,fft_weight,fft_weighted_detla, RA, Dec, z_qso):
        self.wave_data = wave_data
        self.delta_data = delta_data
        self.weight_data = weight_data
        self.RA = RA
        self.Dec = Dec
        self.z_qso = z_qso

        self.delta_fft_grid = delta_fft_grid
        self.weight_fft_grid = weight_fft_grid
        self.j_min_data = None
        self.j_max_data = None
        self.fft_delta = None
        self.fft_weight = None
        self.fft_weighted_delta = None

    def map_to_fftgrid(self,wave_fft_grid,mask_fft_grid):
        
        self.weight_data *= (self.wave_data/4500)**3.8
        
        j_min_data=round((self.wave_data[0]-wave_fft_grid[0])/pw_A)
        j_max_data=round((self.wave_data[-1]-wave_fft_grid[0])/pw_A)
        self.j_min_data = j_min_data
        self.j_max_data = j_max_data
        
        # map the data deltas and weights into the FFT grid
        delta_fft_grid=np.zeros(N_fft)
        weight_fft_grid=np.zeros(N_fft)
        
        # have to make sure FFT grid is larger than the data grid that falls within the redshift bin
        if (j_max_data-j_min_data) >= N_fft:
            print('Data grid is larger than FFT grid, increase N_fft')
            exit(1)

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
        
        weight_fft_grid = weight_fft_grid*mask_fft_grid

        self.delta_fft_grid = delta_fft_grid  # real space 
        self.weight_fft_grid = weight_fft_grid # real space 
        
        self.fft_delta = np.fft.fft(self.delta_fft_grid) # fourier space
        self.fft_weight = np.fft.fft(self.weight_fft_grid) # fourier space
        self.fft_weighted_delta = np.fft.fft(self.delta_fft_grid*self.weight_fft_grid) # fourier space

    
# get sightlines from the delta file that fall within the fft grid
skewers = []
for hdu in file[1:]:
    wave_data=10.0**(hdu.data['LOGLAM'])
    delta_data=hdu.data['DELTA']
    weight_data=hdu.data['WEIGHT']
    
    RA = hdu.header['RA']
    Dec = hdu.header['DEC']
    z_qso = hdu.header['Z']

    skewer = Skewers(wave_data, delta_data, weight_data, None, None, None, None, None, None, None, RA, Dec, z_qso)
    skewer.map_to_fftgrid(wave_fft_grid,mask_fft_grid)
    skewers.append(skewer)

print(skewers[0].RA,skewers[0].Dec,skewers[0].z_qso)
N_skewers = len(skewers)
print('Number of skewers:',N_skewers)
norm_factor = pw_A/N_fft*1/N_skewers # ignoring the resolution function for now

if P1D:
    # compute P1D
    p1d = get_p1d(skewers)
    p1d_norm = norm_factor*p1d
    plt.plot(k[:N_fft//2],p1d_norm[:N_fft//2])
    plt.title('z=%.2f, dz=%.2f, healpix=%d'%(z_alpha,dz,healpix))
    plt.xlabel('k [1/A]')
    plt.ylabel('P1D [A]')
    plt.show()
    plt.savefig('p1d-%d.png'%(healpix))
    #clear image
    plt.clf()


# create a 2d array of theta_min and theta_max values corresponding to theta bin in which Px will be measured
theta_min_array = np.array([10,15])*ARCMIN_TO_RAD
theta_max_array = np.array([15,20])*ARCMIN_TO_RAD

assert theta_min_array.size == theta_max_array.size

px = np.empty((len(theta_min_array),N_fft))

for i in range(len(theta_min_array)):
    px[i,:] = get_px(skewers,theta_min_array[i],theta_max_array[i])
    px[i,:] *= norm_factor

if plot_px:
    for i in range(len(theta_min_array)):
        plt.plot(k[:N_fft//2],px[i,:N_fft//2],label='%f-%f arcmin'%(theta_min_array[i]*RAD_TO_ARCMIN,theta_max_array[i]*RAD_TO_ARCMIN))
    plt.title('z=%.2f, dz=%.2f, healpix=%d'%(z_alpha,dz,healpix))
    plt.xlabel('k [1/A]')
    plt.ylabel('Px [A]')
    plt.legend()
    plt.show()
    plt.savefig('px-%d-%d-%d-%d-%d-old.png'%(healpix,theta_min_array[0]*RAD_TO_ARCMIN,theta_max_array[0]*RAD_TO_ARCMIN,theta_min_array[1]*RAD_TO_ARCMIN,theta_max_array[1]*RAD_TO_ARCMIN))


# compute variance




# save the results
out_path = str(sys.argv[4])
#np.savez(out_path+'/'+'px-%d-%.2f-%.2f.npz'%(healpix,z_alpha,dz),k=k[:N_fft//2],px=px[:,:N_fft//2],k_vel=k_vel[:N_fft//2],theta_min=theta_min,theta_max=theta_max)

# nfft, variance, average over skewers of square of weighted mask fft grid  

# compare with forestflow 