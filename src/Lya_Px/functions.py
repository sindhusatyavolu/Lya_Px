import numpy as np
from config import *
from auxiliary import *
from astropy.io import fits

def read_deltas(healpix,deltas_path):
    delta_file=deltas_path+'delta-%d.fits.gz'%(healpix)
    file = fits.open(delta_file)
    return file

def get_p1d(skewers):
    p1d = np.zeros(N_fft)
    for skewer in skewers:
        wighted_delta_ft = skewer.fft_weighted_delta
        p1d += np.abs(wighted_delta_ft)**2
    return p1d

