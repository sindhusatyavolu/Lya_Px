import numpy as np
import sys
from config import *
import matplotlib.pyplot as plt

path = str(sys.argv[1]) 
output_path = str(sys.argv[2])

k_arr = np.load(path)['k'] # in 1/A
px = np.load(path)['px'] # in A
karr_v = np.load(path)['k_vel'] # in s/km

# Define a binning function 

def Binning(k):
    width = (k[-1]-k[0])/4 # in 1/A
    print('Width:',width)
    period = 2*width
    # periodic window function with given width and period = twice the width
    bin_func = np.zeros(len(k))
    for i in range(len(k)):
        if k[i] % period < width:
            bin_func[i] = 1
    return bin_func

# plot bin function 
plt.plot(k_arr, Binning(k_arr))
plt.savefig('%s/bin_func.png'%(output_path),dpi=400,bbox_inches='tight')
plt.show()

# Plot Px


# compare with forestflow 


