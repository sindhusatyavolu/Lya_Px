import numpy as np
import sys
from config import *
import matplotlib.pyplot as plt

path = str(sys.argv[1]) 

k_arr = np.load(path)['k'] # in 1/A
px = np.load(path)['px'] # in A
karr_v = np.load(path)['k_vel'] # in s/km

# bin power spectrum
pos_k = k_arr[:N_fft//2]
k_bins, bin_edges = np.histogram(pos_k,bins=20) #np.linspace(pos_k.min(),pos_k.max(),20)

k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])  # Compute bin centers

# Digitize k values to find bin indices
bin_indices = np.digitize(k_arr, k_bins) - 1 # Subtract 1 to have indices starting from 0

# Take moving average of Pk in each k bin
Pbins = np.zeros(len(k_bins))
for n in range(len(k_bins)):
    Pbins[n] = np.mean(px[bin_indices==n])

# Plot the binned power spectrum
plt.plot(k_arr, px, label='Unbinned')
plt.plot(k_bins, Pbins, label='Binned')
plt.xlabel('k [1/A]')
plt.ylabel('P(k) [A]')
plt.legend()
plt.show()
plt.savefig('binned_px.png')







