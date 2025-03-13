import numpy as np
from config import *
from auxiliary import *

def get_px(skewers,theta_min,theta_max):
    # loop over the pairs of skewers
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