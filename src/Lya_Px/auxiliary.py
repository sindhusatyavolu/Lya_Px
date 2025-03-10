import numpy as np
from config import *

# function to measure the angular separation between two points on the sky
def angular_separation(ra1, dec1, ra2, dec2): 
    # Calculate the difference in right ascension
    delta_ra = ra2 - ra1  # in radians
    
    # Apply the formula for angular separation
    angular_distance = np.arccos(np.sin(dec1) * np.sin(dec2) +
                                 np.cos(dec1) * np.cos(dec2) * np.cos(delta_ra))
    
    return angular_distance # in radians

# observed wavelength to velocity units conversion


# anglular separation to transverse distance conversion


