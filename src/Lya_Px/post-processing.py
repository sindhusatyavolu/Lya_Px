import numpy as np
import sys
from config import *

path = str(sys.argv[1]) 

k_arr = np.load(path)['k'] # in 1/A
px = np.load(path)['px'] # in A

# convert k to s/km
radial_distance = 2*np.pi/k_arr # in A
print(radial_distance)
dv = c_SI*(1-wave_grid/LAM_LYA)

