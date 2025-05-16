import numpy as np
from Lya_Px.params import *
from Lya_Px.auxiliary import angular_separation
if gauss_test:
    from Lya_Px.gaussian_skewer_maker import SkewerMaker
    from Lya_Px.input_power import InputPower

def get_px(all_skewers,theta_min,theta_max):
    '''
    Function to compute Px for given redshift and theta bin
    all_skewers (list): list of Skewers objects, each containing the data for a single sightline and which redshift bin they belong to
    theta_min (float): minimum angular separation in radians
    theta_max (float): maximum angular separation in radians
    Returns:
    px_ft (np.ndarray): 1D array of shape (N_FFT,), average of the product of the FFT of the delta over all sightlines in the FFT grid which is the dimensionless Px
    w_v_m (np.ndarray): 1D array of shape (N_FFT,), average of the product of the FFT of the weights over all sightlines in the FFT grid
    px_var (np.ndarray): 1D array of shape (N_FFT,), variance of the product of the FFT of the delta over all sightlines in the FFT grid
    len(products) (int): number of pairs of sightlines that were used to compute the Px

    '''
    px_ft = np.zeros(N_fft)
    # mean of w_m v_m^* (product of FFT of masks)
    # w_v_m = (w_m v_m^*).real (product of FFT mask, real part only)
    w_v_m=np.zeros(N_fft)
    if gauss_test:
        # Generate fake input power spectrum
        input_p=InputPower(P0=0.5,k0=0.01,kF=0.1,f_px=0.7)        
        # read the skewer details from the first one, assuming all are equal-sized
        Npix = all_skewers[0].delta_fft_grid.size
        L_wave = all_skewers[0].pw_A * Npix
        # initiate the skewer maker
        maker=SkewerMaker(N=Npix,L=L_wave,input_power=input_p, seed=5432)
    products = []
    products_weight = []
    for i in range(1,len(all_skewers)):
        for j in range(i):
            separation_angle = angular_separation(all_skewers[j].RA,all_skewers[j].Dec,all_skewers[i].RA,all_skewers[i].Dec)

            if separation_angle>theta_min and separation_angle<theta_max:   # strictly less than theta_max and strictly greater than theta_min?
                delta1 = all_skewers[i].delta_fft_grid
                delta2 = all_skewers[j].delta_fft_grid
                weight1 = all_skewers[i].weight_fft_grid
                weight2 = all_skewers[j].weight_fft_grid


                if gauss_test:
                    # generate two correlated Gaussian skewers as a known test case, use these instead of real delta1 and delta2
                    g_skewer_1,g_skewer_2=maker.make_skewer_pair()
                    # weight1 = np.ones(all_skewers[i].weight_fft_grid.shape) # uncomment if you want equal weighting for all pixels
                    # weight2 = weight1 # uncomment if you want equal weighting for all pixels
                    weighted_delta1 = g_skewer_1 * weight1
                    weighted_delta2 = g_skewer_2 * weight2
                    #print('weighted delta1',np.nonzero(weighted_delta1))
                else:
                    weighted_delta1 = delta1*weight1
                    weighted_delta2 = delta2*weight2

                # FT of weighted delta
                weighted_delta1_ft = np.fft.fft(weighted_delta1)
                weighted_delta2_ft = np.fft.fft(weighted_delta2)

                # compute the product of the FT of the weighted delta 1 with the conjugate of the FT of the weighted delta 2 and take the real part
                px_sum = (weighted_delta1_ft*np.conj(weighted_delta2_ft)).real    
                products.append(px_sum)
                px_ft += px_sum
                # compute the product of the FT of the weights 1 with the conjugate of the FT of the weights 2
                fft_weight1 = np.fft.fft(weight1)
                fft_weight2 = np.fft.fft(weight2)
                products_weight.append((fft_weight1*np.conjugate(fft_weight2)).real)   
                
    # compute the variance of the products
    px_var = np.var(products,axis=0)
    #px_ave = np.mean(products,axis=0)
    #print('variance and mean computed',px_var)
    
    w_v_m = np.mean(products_weight,axis=0)
    #print('mean of product of fft of weights computed',w_v_m)   
    # print(len(products),'Number of pairs')
    return px_ft/len(products), w_v_m, px_var,len(products)


