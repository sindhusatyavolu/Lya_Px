import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Run Lyman alpha cross power spectrum analysis with given parameters.")

# All required arguments
parser.add_argument("redshift", type=float, help="Redshift value (e.g., 2.2)")
parser.add_argument("redshift_bin", type=float, help="Redshift bin width (e.g., 0.2)")
parser.add_argument("healpix", type=int, help="Healpix pixel number (e.g., 500)")
parser.add_argument("output_path", type=str, help="Directory to save output files")
parser.add_argument("--theta_file", type=str, required=True,
                    help="Path to theta_values.txt file (e.g., /path/to/theta_values.txt)")
parser.add_argument("--deltas_path", type=str, required=True,help="Path to delta files")

args = parser.parse_args()

# Extract the arguments
z_alpha = args.redshift # redshift bin center
dz = args.redshift_bin # redshift bin width
healpix = args.healpix # healpix pixel
out_path = args.output_path # output path
theta_file = args.theta_file
# Print to verify values
print(f"Redshift: {z_alpha}")
print(f"Redshift bin: {dz}")
print(f"Healpix pixel: {healpix}")
print(f"Output directory: {out_path}")
print(f"Theta file: {theta_file}")

# read theta values from file with first column as theta_min and second column as theta_max
theta_array = np.loadtxt(theta_file, skiprows=1)   # first row is header  (S:keep or change?)
# Load theta_values.txt from the provided path
print("Theta array loaded, shape:", theta_array.shape)
theta_min_array = theta_array[:,0]*ARCMIN_TO_RAD
theta_max_array = theta_array[:,1]*ARCMIN_TO_RAD

assert theta_min_array.size == theta_max_array.size


deltas_path = args.deltas_path 
#'/global/cfs/cdirs/desi/science/lya/mock_analysis/develop/ifae-ql/qq_desi_y3/v1.0.5/analysis-0/jura-124/raw_bao_unblinding/deltas_lya/Delta/'
