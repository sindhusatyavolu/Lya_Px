import numpy as np
import sys
from config import *
import matplotlib.pyplot as plt

path = str(sys.argv[1]) 
output_path = str(sys.argv[2])

k_arr = np.load(path)['k'] # in 1/A
px = np.load(path)['px'] # in A
karr_v = np.load(path)['k_vel'] # in s/km

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


