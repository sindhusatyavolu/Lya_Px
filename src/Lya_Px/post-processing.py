import numpy as np
import sys
from config import *
import matplotlib.pyplot as plt
import h5py
show_plots = 0
plot_p1d = False

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
    # sort only the theta_* groups
    theta_keys = sorted([key for key in f.keys() if key.startswith('theta_')],
                    key=lambda k: float(k.split('_')[1]))  # sort by theta_min in arcmin

    for key in theta_keys:
        g = f[key]
        px.append(g['px'][:])
        px_var.append(g['px_var'][:])
        px_weights.append(g['px_weights'][:])
        theta_bins.append((g.attrs['theta_min'], g.attrs['theta_max']))

px = np.array(px)[:,:N_fft//2]
px_var = np.array(px_var)[:,:N_fft//2]
px_weights = np.array(px_weights)[:,:N_fft//2]

print(np.shape(px))


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

# define the rebinning vector B_alpha_m in the notes (including negative frequencies!)
B_m=np.zeros([Nk,N_fft//2])
for i in range(Nk):
    #print(i,k_edges[i],k_edges[i+1])
    inbin=(abs(k_arr)>k_edges[i]) & (abs(k_arr)<k_edges[i+1])
    B_m[i,inbin]=1    

# mean wavenumber in each bin (without weights)
k_A=np.zeros(Nk)
for i in range(Nk):
    k_A[i]=np.sum(B_m[i]*abs(k_arr))/np.sum(B_m[i])

if show_plots==1:
    plt.plot(k_arr,B_m[0],color='blue')
    plt.axvline(x=k_A[0],color='blue',ls=':')
    plt.plot(k_arr,B_m[5],color='red')
    plt.axvline(x=k_A[5],color='red',ls=':')
    plt.axhline(y=0,color='black')

    #plt.legend()
    plt.xlim([0,1.2*k_A[10]])
    plt.ylim([0,1])
    plt.xlabel('k')
    plt.ylabel('B_m')

# iFFT B_m to get B_a (for the convolution theorem)
B_a=np.empty([Nk,N_fft//2])
for i in range(Nk):
    B_a[i]=np.fft.ifft(B_m[i]).real

if show_plots==2:
    x = pw_A*np.arange(N_fft//2)
    plt.plot(x,B_a[0],color='blue',label='k_A={:.3f}'.format(k_A[0]))
    plt.plot(x,B_a[2],color='red',label='k_A={:.3f}'.format(k_A[2]))
    plt.plot(x,B_a[4],color='green',label='k_A={:.3f}'.format(k_A[4]))
    plt.xlabel('x')
    plt.ylabel('B_a')
    plt.legend()

# rebin the 1d power spectrum by convolving with B_m i.e; take product in real space and fft.
p1d_Q_a = np.fft.ifft(p1d)
p1d_BQ_m=np.empty([Nk,N_fft//2])
for i in range(Nk):
    p1d_BQ_m[i]=np.fft.fft(B_a[i]*p1d_Q_a).real


if show_plots==3:
    plt.plot(k_arr,p1d_BQ_m[0],color='blue')
    plt.axvline(x=k_A[0],color='blue',ls=':')
    plt.plot(k_arr,p1d_BQ_m[5],color='red')
    plt.axvline(x=k_A[5],color='red',ls=':')
    plt.axhline(y=0,color='black')

# normalise p1d 
p1d_A_A=np.empty(Nk)
for i in range(Nk):
    p1d_A_A[i]=np.sum(p1d_BQ_m[i])/pw_A
# actual summary statistics
p1d_Theta_A=np.zeros_like(p1d_A_A)
for i in range(Nk):
    p1d_Theta_A[i]=np.sum(B_m[i]*p1d)/p1d_A_A[i]

if plot_p1d:
    plt.plot(k_A,p1d_Theta_A,label='masked measurement')
    #plt.plot(k_A,true_p1d_A,label='true P1D')
    #plt.plot(k_A,model_A,label='masked model')
    plt.xlabel('k')
    plt.ylabel('P1D(k)')
    plt.legend()

# rebin Px

px_BQ_m=np.empty([len(theta_bins),Nk,N_fft//2])
for j in range(len(theta_bins)):
    px_Q_a = np.fft.ifft(px[j])    
    for i in range(Nk):
        px_BQ_m[j][i]=np.fft.fft(B_a[i]*px_Q_a).real

if show_plots==4:
    plt.plot(k_arr,px_BQ_m[0][0][:N_fft//2],color='blue')
    plt.axvline(x=k_A[0],color='blue',ls=':')
    plt.plot(k_arr,px_BQ_m[0][5][:N_fft//2],color='red')
    plt.axvline(x=k_A[5],color='red',ls=':')
    plt.axhline(y=0,color='black')
    plt.xlim([0,k_max/2])
    plt.xlabel('k')
    plt.ylabel('p1d_BQ_m')
    plt.xlabel('k [1/A]')
    plt.ylabel('Px [A]')    

# normalise px
px_A_A=np.empty((len(theta_bins),Nk))
for j in range(len(theta_bins)):
    for i in range(Nk):
        px_A_A[j][i]=np.sum(px_BQ_m[j][i])/pw_A
for j in range(len(theta_bins)):
    print(np.array(theta_bins[j])*RAD_TO_ARCMIN, np.mean(px[j]))

# actual summary statistics
px_Theta_A=np.zeros_like(px_A_A)
for j in range(len(theta_bins)):
    for i in range(Nk):
        px_Theta_A[j][i]=np.sum(B_m[i]*px[j])#/px_A_A[j][i]        

for l in range(len(theta_bins)):
    plt.plot(k_A,px_Theta_A[l],label='%0.1f-%0.1f arcmin'%(theta_bins[l][0]*RAD_TO_ARCMIN,theta_bins[l][1]*RAD_TO_ARCMIN))

print('theta bins:',np.array(theta_bins)*RAD_TO_ARCMIN)
#plt.plot(k_A,true_p1d_A,label='true P1D')
#plt.plot(k_A,model_A,label='masked model')
plt.xlabel('k [1/A]')
plt.ylabel('PX(k)[A]')
plt.legend()
plt.savefig(output_path+'binned_px_comp.png')
plt.show()


# compare with forestflow 




"""
# simple binning of the power spectrum
N_bins = 4
dk = (k_arr[-1]-k_arr[0])/N_bins
k_bins  = np.linspace(k_arr[0], k_arr[-1], N_bins + 1)

# find indices where the k values are in the bins
k_indices = np.digitize(k_arr,k_bins) -1 

# bin the power spectrum
px_binned = np.zeros((len(theta_bins),N_bins+1))
for i in range(N_bins+1):
    for j in range(len(theta_bins)):
        px_binned[j,i] = np.mean(px[j][k_indices==i])

# Plot Px
for j in range(len(theta_bins)):
    plt.plot(k_arr,px[j],label='Px %f-%f'%(theta_bins[j][0]*RAD_TO_ARCMIN,theta_bins[j][1]*RAD_TO_ARCMIN))
    plt.plot(k_bins,px_binned[j],label='Px binned %f-%f'%(theta_bins[j][0]*RAD_TO_ARCMIN,theta_bins[j][1]*RAD_TO_ARCMIN))
#plt.plot(k_arr,px[0],label='Px')
#plt.plot(k_bins,px_binned,label='Px binned')
"""