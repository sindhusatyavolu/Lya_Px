import numpy as np
from Lya_Px.config import *
from Lya_Px.auxiliary import *
from astropy.io import fits

def read_deltas(healpix,deltas_path):
    delta_file=deltas_path+'delta-%d.fits.gz'%(healpix)
    file = fits.open(delta_file)
    return file

def get_p1d(skewers):
    p1d = np.zeros(N_fft)
    for skewer in skewers:
        wighted_delta_ft = np.fft.fft(skewer.delta_fft_grid * skewer.weight_fft_grid)
        p1d += np.abs(wighted_delta_ft)**2
    return p1d

