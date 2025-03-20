import numpy as np
from config import *
from auxiliary import *

def get_px(skewers,theta_min,theta_max):    
    px_ft = np.zeros(N_fft)
    # mean of w_m v_m^* (product of FFT of masks)
    # w_v_m = (w_m v_m^*).real (product of FFT mask, real part only)
    w_v_m=np.zeros(N_fft)    
    products = []
    products_weight = []
    for i in range(len(skewers)):
        for j in range(i+1,len(skewers)):
            separation_angle = angular_separation(skewers[j].RA,skewers[j].Dec,skewers[i].RA,skewers[i].Dec)
            if separation_angle>theta_min and separation_angle<theta_max:   # strictly less than theta_max and strictly greater than theta_min?
                delta1 = skewers[i].delta_fft_grid
                delta2 = skewers[j].delta_fft_grid
                weight1 = skewers[i].weight_fft_grid
                weight2 = skewers[j].weight_fft_grid
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
    w_v_m = np.mean(products_weight,axis=0)
    
    return px_ft, w_v_m, px_var,px_ave*len(products)




