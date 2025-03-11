import numpy as np
import sys
from config import *

path = str(sys.argv[1]) 

k_arr = np.load(path)['k'] # in 1/A
px = np.load(path)['px'] # in A




