import numpy as np
from config import *
from auxiliary import *

def get_px(skewers,theta_min,theta_max):
    # loop over the pairs of skewers
    for i1 in range(1,len(skewers)):
        sk1=skewers[i1]
        w1=sk1['weight_fft_grid']
        d1=sk1['delta_fft_grid']
        Np =  0
        for i2 in range(i1):
            sk2=skewers[i2]
            # compute angular separation, in radians
            ang=angular_separation(sk1['RA'],sk1['Dec'],sk2['RA'],sk2['Dec'])
            if ang<theta_min or ang>theta_max:
                continue
                
            # correlate Fourier modes
            w2=sk2['weight_fft_grid']
            d2=sk2['delta_fft_grid']
                            
            Np+=1
            
            # FFT of masks
            w_1_m = np.fft.fft(w1)
            w_2_m = np.fft.fft(w2)
            w_v_m += (w_1_m*np.conjugate(w_2_m)).real
     
            # and FFT again to obtain masked modes
            f_1_m=np.fft.fft(d1*w1)
            f_2_m=np.fft.fft(d2*w2)
            F_G_m += (f_1_m*np.conjugate(f_2_m)).real
    
    return F_G_m
                


"""
    px_ft = np.zeros(N_fft)
    for i in range(len(skewers)):
        for j in range(i+1,len(skewers)):
            if angular_separation(skewers[i]['RA'],skewers[i]['Dec'],skewers[j]['RA'],skewers[j]['Dec'])>theta_min and angular_separation(skewers[i]['RA'],skewers[i]['Dec'],skewers[j]['RA'],skewers[j]['Dec'])<theta_max:
                delta1 = skewers[i]['delta_fft_grid']
                delta2 = skewers[j]['delta_fft_grid']
                weight1 = skewers[i]['weight_fft_grid']
                weight2 = skewers[j]['weight_fft_grid']
                weighted_delta1 = delta1*weight1
                weighted_delta2 = delta2*weight2

                # FT of weighted delta
                weighted_delta1_ft = np.fft.fft(weighted_delta1)
                weighted_delta2_ft = np.fft.fft(weighted_delta2)

                # compute the products
                px_ft += (weighted_delta1_ft*np.conj(weighted_delta2_ft)).real
    return px_ft
"""

"""
    for skewer1_index, skewer2_index in skewer_pair_indices:
        skewer1 = skewers[skewer1_index]
        skewer2 = skewers[skewer2_index]
        # compute the fourier transform of delta1 and delta2
        delta1 = skewer1['delta_fft_grid']
        delta2 = skewer2['delta_fft_grid']
        weight1 = skewer1['weight_fft_grid']
        weight2 = skewer2['weight_fft_grid']
        weighted_delta1 = delta1*weight1
        weighted_delta2 = delta2*weight2

        # FT of weighted delta
        weighted_delta1_ft = np.fft.fft(weighted_delta1)
        weighted_delta2_ft = np.fft.fft(weighted_delta2)
        
        # compute the products
        px_ft += (weighted_delta1_ft*np.conj(weighted_delta2_ft)).real       
 
"""