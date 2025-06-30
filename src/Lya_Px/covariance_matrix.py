import numpy as np
from Lya_Px.params import *
from collections import defaultdict

def compute_cov(px, weights):
    """Computes the covariance matrix using the subsampling technique

    Args:
        px: array of floats
            Px measurement in each healpix
        weights: array of floats
            Weights on the Px measurement in each healpix

    Returns:
        The covariance matrix
    """
    print("Computing mean Px...")
    mean_px = (px * weights).sum(axis=0)
    sum_weights = weights.sum(axis=0)
    w = sum_weights > 0.
    mean_px[w] /= sum_weights[w]
    
    
    meanless_px_times_weight = weights * (px - mean_px)

    print("Computing subsampling cov...")

    covariance = meanless_px_times_weight.T.dot(meanless_px_times_weight)

    sum_weights_squared = sum_weights * sum_weights[:, None]
    
    w = sum_weights_squared > 0.
    
    # covariance estimator C^_ij
    covariance[w] /= sum_weights_squared[w]

    num = np.matmul((weights**2).T,weights) + np.matmul(weights.T, weights**2) 
    correction_factor = 1+ np.matmul(weights.T, weights) - num/np.matmul(weights.T, weights)
    
    #print(np.matmul(weights.T, weights),num)
    #print("Correction factor:", correction_factor)
    
    # True covariance C_ij
    #covariance /= correction_factor
    #print("true Covariance:", covariance)
    # covariance of the mean 
    #covariance = covariance*np.matmul(weights.T, weights)
    #print("mean Covariance:", covariance)

    return mean_px, covariance

