import numpy as np
import sys
from config import *
import matplotlib.pyplot as plt

path = str(sys.argv[1]) 
output_path = str(sys.argv[2])

k_arr = np.load(path)['k'] # in 1/A
px = np.load(path)['px'] # in A
karr_v = np.load(path)['k_vel'] # in s/km

# define full length of FFT grid (in Angstroms)
L_fft=N_fft*pw_A
# we will rebin the wavenumbers to make them more independent, and better measured
k0_fft=2*np.pi/L_fft
dk_bin=k0_fft*4
print('dk =',dk_bin)
# stop roughly at 1/4 of the Nyquist frequency for now (equivalent to rebinning 4 pixels)
k_Ny_fft=np.pi/pw_A
k_max=k_Ny_fft/4
print('k < ',k_max)
k_edges=np.arange(0.01*dk_bin,k_max+dk_bin,dk_bin)
Nk=k_edges.size-1
print('Nk =',Nk)

# Define a binning function for each k value
B_m=np.zeros([Nk,N_fft])
for i in range(Nk):
    inbin=(abs(k)>k_edges[i]) & (abs(k)<k_edges[i+1])
    B_m[i,inbin]=1 

# mean wavenumber in each bin (without weights)
k_A=np.zeros(Nk)
for i in range(Nk):
    k_A[i]=np.sum(B_m[i]*abs(k))/np.sum(B_m[i])

plt.plot(k[:N_fft//2],B_m[0][:N_fft//2],color='blue')
plt.axvline(x=k_A[0],color='blue',ls=':')
plt.plot(k[:N_fft//2],B_m[5][:N_fft//2],color='red')
plt.axvline(x=k_A[5],color='red',ls=':')
plt.axhline(y=0,color='black')

#plt.legend()
plt.xlim([0,1.2*k_A[10]])
plt.ylim([0,1])
plt.xlabel('k')
plt.ylabel('B_m')

# plot bin function 
#plt.plot(k_arr, Binning(k_arr))
plt.savefig('%s/bin_func_new.png'%(output_path))
plt.show()

# Plot Px


# compare with forestflow 


