import numpy as np
from Lya_Px.params import *


def calculate_estnorm(W, R, L):
    '''
    W (np.ndarray): vector length N, average FFT of the weights
    R (np.ndarray): vector length N, resolution in Fourier space
    L (float): length of the spectra (in physical units, e.g. Angstroms or Mpc)
    Returns:
    estnorm (np.ndarray): vector length N, to be multiplied by every P1D mode of the measurement
    '''
    R2 = R.real**2 + R.imag**2
    denom = np.absolute(np.fft.ifft(np.fft.fft(W)* np.fft.fft(R2)))
    estnorm = np.absolute(L/denom)
    return estnorm

def calculate_window_matrix(W, R, L):

    '''
    W (np.ndarray): average of (w1) conj(w2) where w1 and w2 are FFT of original weights per skewer
    R (np.ndarray): vector length N, resolution in Fourier space
    L (float): physical length of skewers (e.g., in Angstroms)
    Returns:
    window_matrix (np.ndarray): window matrix to be convolved with pure theory
    estnorm (np.ndarray): vector length N, to be multiplied by every P1D mode of the measurement
    '''
    R2 = R.real**2 + R.imag**2
    denom = np.absolute(np.fft.ifft(np.fft.fft(W)* np.fft.fft(R2)))
    estnorm = np.absolute(L/denom)
    N = estnorm.size
    window_matrix = np.zeros((N,N))
    for m in range(N):
        for n in range(N):
            window_matrix[m,n] = W[m-n]*R2[n] / denom[m]
    return window_matrix, estnorm


def masked_theory(window_matrix, model):
    return np.matmul(window_matrix, model)
  