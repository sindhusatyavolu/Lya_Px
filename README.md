# Lya_PxEC
Lyman alpha forest cross power spectrum estimator code 

## Installation

First, clone the repository and navigate to the directory

<pre>git clone https://github.com/sindhusatyavolu/Lya_PxEC.git 
cd Lya_PxEC </pre>

## Create and activate a virtual environment
<pre>python3 -m venv venv
source venv/bin/activate</pre>

## Install package in editable mode
<pre>pip install -e .</pre>

The list of dependencies required are listed in pyproject.toml. For versions used, check requirements.txt.

## Usage

# For measuring the cross power spectrum

Run the command:
<pre> python -m Lya_Px.main <redshift> <redshift_bin> <healpix> <output_path> --theta_file <path_to_theta_values.txt> </pre>


