import configparser
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.ini",help="Path to the configuration file")
args = parser.parse_args()

# Load from config.ini
config = configparser.ConfigParser()
config.read(args.config)

# Read the parameters
redshifts_file = config.get("parameters", "redshifts")
output_path = config.get("parameters", "output_path")
theta_file = config.get("parameters", "theta_file")
deltas_path = config.get("parameters", "deltas_path")

redshifts = np.loadtxt(redshifts_file,skiprows=1) # redshifts and redshift bin widths
z_alpha = redshifts[:,0] # redshift bin center
dz = redshifts[:,1] # redshift bin width

theta_array = np.loadtxt(theta_file, skiprows=1)
healpix_file = config.get("parameters", "healpix_file")
healpixlist = np.atleast_1d(np.loadtxt(healpix_file, dtype=int))

n_healpix = config.getint("parameters", "n_healpix")
#healpixlist = healpixlist[-1:]

# Load constants
LAM_LYA = config.getfloat("constants", "LAM_LYA")
c_SI = config.getfloat("constants", "c_SI")
pw_A = config.getfloat("constants", "pw_A")
N_fft = config.getint("constants", "N_fft")
PI = config.getfloat("constants", "PI")

# Load conversion factors
RAD_TO_ARCMIN = config.getfloat("constants", "RAD_TO_ARCMIN")
ARCMIN_TO_RAD = config.getfloat("constants", "ARCMIN_TO_RAD")
RAD_TO_DEG = config.getfloat("constants", "RAD_TO_DEG")
DEG_TO_RAD = config.getfloat("constants", "DEG_TO_RAD")


# Load postprocessing flags 
P1D = config.getboolean("postprocessing", "P1D")
plot_px = config.getboolean("postprocessing", "plot_px")
plot_px_vel = config.getboolean("postprocessing", "plot_px_vel")
bin_px = config.getboolean("postprocessing", "bin_px")




# 1 job that averages over all healpix pixels (one for each mock) 
# Take multiple redshifts as input, similar to theta file
# more modular -- call class and plot functions from other files
# deltas_path is hardcoded right now. take it also as input
# add redshifts and redshift bin to the output file


