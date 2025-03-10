import numpy as np

def get_px(skewer_pair_indices,skewers):
    # loop over the pairs of skewers
    px_ft = np.zeros(N_fft)
    for i,j in skewer_pair_indices:
        skewer1=skewers[i]
        skewer2=skewers[j]
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
 
    return px_ft

