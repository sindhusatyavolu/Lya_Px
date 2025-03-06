import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


from config import *

# Read inputs
healpix = int(input("Enter the Healpix pixel number: "))
deltas_path='/global/cfs/cdirs/desi/science/lya/mock_analysis/develop/ifae-ql/qq_desi_y3/v1.0.5/analysis-0/jura-124/raw_bao_unblinding/deltas_lya/Delta/'
delta_file=deltas_path+'delta-%d.fits.gz'%(healpix)
file = fits.open(delta_file)




