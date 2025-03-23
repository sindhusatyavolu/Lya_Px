# Lya_PxEC
Lyman alpha forest cross power spectrum estimator code 

## Installation

First, clone the repository and navigate to the directory

<pre>git clone --branch class-based-code https://github.com/sindhusatyavolu/Lya_PxEC.git
cd Lya_PxEC </pre>

## Create and activate a virtual environment
<pre>python3 -m venv venv
source venv/bin/activate</pre>

## Install package in editable mode
<pre>pip install -e .</pre>

The list of dependencies required are in pyproject.toml. For versions used, check requirements.txt.

## Usage

### For measuring the cross power spectrum

Run the command:

```python3 -m Lya_Px.main <redshift> <redshift_bin> <healpix> <output_path> --theta_file <path_to_theta_values.txt> ```

Takes as inputs the redshift of the bin center, redshift bin size, healpix pixel number, desired location of the outputs, and location of the file containing theta bins where you want to measure Px.

Outputs are stored in HDF5 format. Use h5ls or h5dump to check the file. Alternatively, check post-processing.py for the python script to read it.

### Post-processing the Px measurement

Primarily includes binning and plotting. Usage:

```python3 -m Lya_Px.post_processing <output_path> <plot_directory>```

Takes as input the path to the Px file and the directory where you want to save any outputs.






