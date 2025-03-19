import numpy as np
import sys
from config import *
import matplotlib.pyplot as plt
import h5py

path = str(sys.argv[1]) 
output_path = str(sys.argv[2])

# read hdf5 file

with h5py.File(path, 'r') as f:
    # Load shared datasets
    k_arr = f['k_arr'][:N_fft//2]
    p1d = f['p1d'][:N_fft//2]

    # Load attributes
    N_fft = f.attrs['N_fft']
    dvel = f.attrs['dvel']
    N_skewers = f.attrs['N_skewers']

    # Loop over all theta groups
    px = []
    px_var = []
    px_weights = []
    theta_bins = []

    for key in f.keys():
        if key.startswith('theta_'):
            g = f[key]
            px.append(g['px'][:])
            px_var.append(g['px_var'][:])
            px_weights.append(g['px_weights'][:])
            theta_bins.append((g.attrs['theta_min'], g.attrs['theta_max']))

px = np.array(px)[:,:N_fft//2]
px_var = np.array(px_var)[:,:N_fft//2]
px_weights = np.array(px_weights)[:,:N_fft//2]

print(np.shape(px))

# simple binning of the power spectrum
N_bins = 4
dk = (k_arr[-1]-k_arr[0])/N_bins
k_bins  = np.arange(k_arr[0],k_arr[-1],dk)

# find indices where the k values are in the bins
k_indices = np.digitize(k_arr,k_bins) -1 

# bin the power spectrum
px_binned = np.zeros(N_bins)
for i in range(N_bins):
    px_binned[i] = np.mean(px[0][k_indices==i])

# Plot Px
plt.plot(k_arr,px[0],label='Px')
plt.plot(k_bins,px_binned,label='Px binned')
plt.xlabel('k [1/A]')
plt.ylabel('Px')
plt.legend()
plt.savefig(output_path+'binned_px_comp.png')
plt.show()
# compare with forestflow 


