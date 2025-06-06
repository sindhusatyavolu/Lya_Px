import numpy as np
import sys
import matplotlib.pyplot as plt
import h5py
#from Lya_Px.params import *
import argparse

def main():
    show_plots = 0
    plot_p1d = False
    
    parser = argparse.ArgumentParser(description="Postprocessing Px files")

    parser.add_argument("--path_to_px_file", type=str, required=True,
                        help="Path to Px hdf5 file (e.g., /path/to/px.hdf5)")
    parser.add_argument("--output_path", type=str, required=True,help="Path to output directory where plots etc will be saved")
    #parser.add_argument("--save_plots", type=int, required=True, help="0: no, 1: Bin function 2: IFFT of Bin function 3: Binned P1D 4: Binned Px")

    args = parser.parse_args()

    path = args.path_to_px_file 
    output_path = args.output_path
    

    # read hdf5 file
    with h5py.File(path, 'r') as f:
        # Load shared datasets
        k_arr = f['k_arr'][:]
        p1d = f['p1d'][:]

        # Load attributes
        z = f.attrs['z']
        dz = f.attrs['dz']

        # Loop over all theta groups
        px = []
        px_var = []
        px_weights = []
        px_cov = []
        theta_bins = []
        # sort only the theta_* groups
        theta_keys = sorted([key for key in f.keys() if key.startswith('theta_')],
                        key=lambda k: float(k.split('_')[1]))  # sort by theta_min in arcmin

        for key in theta_keys:
            g = f[key]
            px.append(g['px'][:])
            px_var.append(g['px_var'][:])
            px_weights.append(g['px_weights'][:])
            px_cov.append(g['covariance'][:])
            theta_bins.append((g.attrs['theta_min'], g.attrs['theta_max']))
    
    
    k_arr = np.array(k_arr)
    px_norm = np.array(px)
    px_var = np.array(px_var)
    px_weights = np.array(px_weights)
    px_cov = np.array(px_cov)
    N_fft = len(k_arr)
    print(np.shape(px_norm))
    print(np.shape(px_cov)) # should be N_fft x N_fft

    #plt.plot(k_arr[:N_fft//2],px_norm[0][:N_fft//2])
    #plt.plot(k_arr[:N_fft//2],px_norm[1][:N_fft//2])
    #plt.plot(k_arr[:N_fft//2],px_norm[2][:N_fft//2])
    
    k_arr = np.array(k_arr)
    pw_A = 0.8
    N_fft = len(k_arr)
    px = np.array(px)*pw_A/N_fft
    px_var = np.array(px_var)
    px_weights = np.array(px_weights)
    
    #print(px[-1]/px_norm)
    plt.plot(k_arr[:N_fft//2],px[-1][:N_fft//2])
    plt.xlabel('k [A]')
    plt.ylabel(r'P(k)')
    plt.legend()
    
    plt.show()

    sys.exit() 

    # S: I find the binning here a bit complicated. Not sure I understand how it's useful. See the commented code at the end for the binning I implemented.

    """ Binning of the power spectrum """
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

    # define the rebinning vector B_m for each k bin (1 if k_arr is in the given k bin, 0 otherwise) 
    B_m=np.zeros([Nk,N_fft]) # includes negative k values
    for i in range(Nk):
        inbin=(abs(k_arr)>k_edges[i]) & (abs(k_arr)<k_edges[i+1])
        B_m[i,inbin]=1    

    # mean wavenumber in each bin (without weights)
    k_A=np.zeros(Nk)
    for i in range(Nk):
        k_A[i]=np.sum(B_m[i]*abs(k_arr))/np.sum(B_m[i])
        assert np.allclose(k_A[i],np.mean(abs(k_arr)[B_m[i]==1])) 


    if show_plots==1:
        plt.plot(k_arr[:N_fft//2],B_m[0][:N_fft//2],color='blue')
        plt.axvline(x=k_A[0],color='blue',ls=':')
        plt.plot(k_arr[:N_fft//2],B_m[5][:N_fft//2],color='red')
        plt.axvline(x=k_A[5],color='red',ls=':')
        plt.axhline(y=0,color='black')

        #plt.legend()
        plt.xlim([0,1.2*k_A[10]])
        plt.ylim([0,1])
        plt.xlabel('k')
        plt.ylabel('B_m')
        plt.show()

    # iFFT B_m to get B_a (for the convolution theorem)
    B_a=np.empty([Nk,N_fft])  # real space bins 
    for i in range(Nk):
        B_a[i]=np.fft.ifft(B_m[i]).real  

    if show_plots==2:
        x = pw_A*np.arange(N_fft)
        plt.plot(x,B_a[0],color='blue',label='k_A={:.3f}'.format(k_A[0]))
        plt.plot(x,B_a[2],color='red',label='k_A={:.3f}'.format(k_A[2]))
        plt.plot(x,B_a[4],color='green',label='k_A={:.3f}'.format(k_A[4]))
        plt.xlabel('x')
        plt.ylabel('B_a')
        plt.legend()
        plt.show()


    # rebin the 1d power spectrum by convolving with B_m i.e; take product in real space and fft.
    p1d_ifft = np.fft.ifft(p1d)    # length N_fft
    p1d_bins =np.empty([Nk,N_fft]) # convolved p1d (think of it as weights) in each k bin 
    for i in range(Nk): 
        p1d_bins[i]=np.fft.fft(B_a[i]*p1d_ifft).real

    if show_plots==3:
        plt.plot(k_arr[:N_fft//2],p1d_bins[0][:N_fft//2],color='blue')
        plt.axvline(x=k_A[0],color='blue',ls=':')
        plt.plot(k_arr[:N_fft//2],p1d_bins[5][:N_fft//2],color='red')
        plt.plot(k_arr[:N_fft//2],p1d_bins[20][:N_fft//2],color='green')
        plt.axvline(x=k_A[5],color='red',ls=':')
        plt.axhline(y=0,color='black')
        plt.show()  

    # normalise p1d 
    p1d_A_A=np.empty(Nk)  # total convolved p1d (sum of weights) in each k bin normalised by the pixel width (convention?)
    for i in range(Nk):
        p1d_A_A[i]=np.sum(p1d_bins[i])/pw_A

    # actual summary statistics
    p1d_Theta_A=np.zeros_like(p1d_A_A)  
    for i in range(Nk):
        p1d_Theta_A[i]=np.sum(B_m[i]*p1d)/p1d_A_A[i]

    if plot_p1d:
        plt.plot(k_A,p1d_Theta_A,label='masked measurement')
        #plt.plot(k_A,p1d_A_A,label='true P1D')
        #plt.plot(k_A,true_p1d_A,label='true P1D')
        #plt.plot(k_A,model_A,label='masked model')
        plt.xlabel('k')
        plt.ylabel('P1D(k)')
        plt.legend()
        plt.show()

    # rebin Px
    px_BQ_m=np.empty([len(theta_bins),Nk,N_fft])
    for j in range(len(theta_bins)):
        px_Q_a = np.fft.ifft(px[j])    
        for i in range(Nk):
            px_BQ_m[j][i]=np.fft.fft(B_a[i]*px_Q_a).real

    if show_plots==4:
        plt.plot(k_arr[:N_fft//2],px_BQ_m[0][0][:N_fft//2],color='blue')
        plt.axvline(x=k_A[0],color='blue',ls=':')
        plt.plot(k_arr[:N_fft//2],px_BQ_m[0][5][:N_fft//2],color='red')
        plt.axvline(x=k_A[5],color='red',ls=':')
        plt.axhline(y=0,color='black')
        plt.xlim([0,k_max/2])
        plt.xlabel('k')
        plt.ylabel('p1d_BQ_m')
        plt.xlabel('k [1/A]')
        plt.ylabel('Px [A]')    
        plt.show()
    print(np.sum(px_BQ_m[0][0]),np.sum(px_BQ_m[0][5]))    
    print(np.sum(px_BQ_m[1][0]),np.sum(px_BQ_m[1][5]))    


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
            px_Theta_A[j][i]=np.sum(B_m[i]*px[j])/px_A_A[j][i]        

    norm_factor = pw_A/N_fft*1/N_skewers # to give dimensions of A

    for l in range(len(theta_bins)):
        #plt.plot(k_arr[:N_fft//2],px[l][:N_fft//2],label='%0.1f-%0.1f arcmin'%(theta_bins[l][0]*RAD_TO_ARCMIN,theta_bins[l][1]*RAD_TO_ARCMIN))
        plt.plot(k_A,px_Theta_A[l]*norm_factor,label='%0.1f-%0.1f arcmin'%(theta_bins[l][0]*RAD_TO_ARCMIN,theta_bins[l][1]*RAD_TO_ARCMIN))
    
    print(theta_bins)
    print('theta bins:',np.array(theta_bins)*RAD_TO_ARCMIN)
    plt.xlabel('k [1/A]')
    plt.ylabel('PX(k)[A]')
    plt.legend()
    plt.savefig(output_path+'binned_px_comp.png',bbox_inches='tight',dpi=350)
    plt.show()


    # compare with forestflow 

    # compute covariance matrix of the binned power spectrum


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

if __name__ == '__main__':
    main()
