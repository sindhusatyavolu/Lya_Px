import numpy as np
from Lya_Px.params import *
from Lya_Px.auxiliary import angular_separation

def get_px(all_skewers,theta_min,theta_max):
    
    #for skewer in all_skewers:
    #    if skewer.delta_fft_data is None:
    #        skewer.map_to_fftgrid(wave_fft_grid)

    px_ft = np.zeros(N_fft)
    # mean of w_m v_m^* (product of FFT of masks)
    # w_v_m = (w_m v_m^*).real (product of FFT mask, real part only)
    w_v_m=np.zeros(N_fft) 
    
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
                    weighted_delta1 = delta1*weight1
                    weighted_delta2 = delta2*weight2

                    # FT of weighted delta
                    weighted_delta1_ft = np.fft.fft(weighted_delta1)
                    weighted_delta2_ft = np.fft.fft(weighted_delta2)

                    # compute the products
                    products.append((weighted_delta1_ft*np.conj(weighted_delta2_ft)).real)
                    px_ft += (weighted_delta1_ft*np.conj(weighted_delta2_ft)).real
                    fft_weight1 = np.fft.fft(weight1)
                    fft_weight2 = np.fft.fft(weight2)
                    products_weight.append((fft_weight1*np.conjugate(fft_weight2)).real)   
                    
                    
    # compute the variance of the products
    px_var = np.var(products,axis=0)
    px_ave = np.mean(products,axis=0)
    print('variance and mean computed',px_ave)
    w_v_m = np.mean(products_weight,axis=0)
    print('mean of product of fft of weights computed',w_v_m)   
    print(len(products),'Number of pairs')
    return px_ft/len(products), w_v_m, px_var,px_ave, px_ave #*len(products)



