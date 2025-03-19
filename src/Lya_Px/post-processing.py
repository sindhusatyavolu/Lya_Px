import numpy as np
import sys
from config import *
import matplotlib.pyplot as plt

path = str(sys.argv[1]) 
output_path = str(sys.argv[2])

# read hdf5 file
f = h5py.File(path, 'r')
px = f['px']
k_arr = f['k_arr']
theta_min_array = f['theta_min']
theta_max_array = f['theta_max']
N_fft = f.attrs['N_fft']
dvel = f.attrs['dvel']
N_skewers = f.attrs['N_skewers']
px_var = f['px_var']
px_weights = f['px_weights']
p1d = f['p1d']


# simple binning of the power spectrum
N_bins = 4
dk = (k_arr[-1]-k_arr[0])/N_bins
k_bins  = np.arange(k_arr[0],k_arr[-1]+dk,dk)

# find indices where the k values are in the bins
k_indices = np.digitize(k_arr,k_bins)

# bin the power spectrum
px_binned = np.zeros(N_bins)
for i in range(N_bins):
    px_binned[i] = np.mean(px[k_indices==i])

# Plot Px
plt.plot(k_arr,px,label='Px')
plt.plot(k_bins,px_binned,'o',label='Px binned')
plt.savefig(output_path+'binned_px_comp.png')
plt.show()
# compare with forestflow 


