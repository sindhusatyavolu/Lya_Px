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

# get sightlines from the delta file that fall within the fft grid
skewers = get_skewers(wave_fft_grid,mask_fft_grid,file)
N_skewers = len(skewers)
norm_factor = pw_A/N_fft*1/N_skewers # ignoring the resolution function for now

if P1D:
    # compute P1D
    p1d = get_p1d(skewers)
    print('Number of skewers:',N_skewers)
    p1d_norm = pw_A/N_fft*1/N_skewers*p1d
    plt.plot(k[:N_fft//2],p1d_norm[:N_fft//2])
    plt.title('z=%.2f, dz=%.2f, healpix=%d'%(z_alpha,dz,healpix))
    plt.xlabel('k [1/A]')
    plt.ylabel('P1D [A]')
    plt.show()
    plt.savefig('p1d-%d.png'%(healpix))
    #clear image
    plt.clf()

# compute separations
separation_angles, skewer_pairs_indices = get_separations(skewers)
separation_angles  = np.array(separation_angles)
skewer_pairs_indices = np.array(skewer_pairs_indices)
print('minimum and maximum separation in degrees:',separation_angles.min()*RAD_TO_DEG,separation_angles.max()*RAD_TO_DEG)
print(separation_angles[(separation_angles*RAD_TO_ARCMIN>6) & (separation_angles*RAD_TO_ARCMIN<9)])


#print(skewer_pairs[(separation_angles*RAD_TO_ARCMIN>6) & (separation_angles*RAD_TO_ARCMIN<9)])
#print(angular_separation(skewers[0]['RA'],skewers[0]['Dec'],skewers[1]['RA'],skewers[1]['Dec']))


theta_min = np.array([6])*ARCMIN_TO_RAD
theta_max = np.array([9])*ARCMIN_TO_RAD
print(theta_min,theta_max)

assert len(theta_min) == len(theta_max)

px = np.zeros((len(theta_min),N_fft))

for i in range(len(theta_min)):
        # select pairs of skewers that fall within the angular separation bin
        skewer_pairs_thetabin = skewer_pairs_indices[(separation_angles>theta_min[i]) & (separation_angles<theta_max[i])]
        print(skewer_pairs_thetabin)
        print(skewer_pairs_thetabin.shape)
        print(angular_separation(skewers[skewer_pairs_thetabin[0][0]]['RA'],skewers[skewer_pairs_thetabin[0][1]]['DEC'],skewers[skewer_pairs_thetabin[1][0]]['RA'],skewers[skewer_pairs_thetabin[1][1]]['DEC']))
        print(len(skewer_pairs_thetabin))
        px[i,:] = get_px(skewer_pairs_thetabin,skewers)
        px *= norm_factor

if plot_px:
    for i in range(len(theta_min)):
        plt.plot(k[:N_fft//2],px[i,:N_fft//2],label='%f-%f arcmin'%(theta_min[i]*RAD_TO_ARCMIN,theta_max[i]*RAD_TO_ARCMIN))
    plt.title('z=%.2f, dz=%.2f, healpix=%d'%(z_alpha,dz,healpix))
    plt.xlabel('k [1/A]')
    plt.ylabel('Px [A]')
    plt.legend()
    plt.show()
    plt.savefig('px-%d.png'%(healpix))


# compute variance


# save the results
out_path = str(sys.argv[4])
np.savez(out_path+'/'+'px-%d-%.2f-%.2f.npz'%(healpix,z_alpha,dz),k=k[:N_fft//2],px=px[:,:N_fft//2],k_vel=k_vel[:N_fft//2],theta_min=theta_min,theta_max=theta_max)

# nfft, variance, average over skewers of square of weighted mask fft grid  

# compare with forestflow 