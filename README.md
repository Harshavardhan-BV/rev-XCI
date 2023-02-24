# rev-XCI
A model for simulating different behaviours observed on X-Chromosome reactivation

## Structure of the repo
- codes
	- contains the base code for model
        - TS_fun.py: defined the differential equation
        - TS_fit.py: implements wrapper for calling solve_ivp, differential_evolution functions
        - vary_self.py: Iterates over the possible combinations of self-regulation
        - vary_cross.py : Iterates over the possible combinations of cross-regulation
        - vary_all.py: Iterates over all the possible combinations over self and cross-regulation
        - noise_sensitivity.py: Implements wrapper for calling solve_ivp function on model with noise 
        - ProcessRaw.py: Used to process the raw data files provided 
- input
	- contains the files taken as input
        - iPSC: Full X-reactivation data in iPSC reprogramming
        - Partial: Partial X-reactivation data
        - _timeshifted: The start is considered as Day 7
        - RawData not provided 
- output
	- contains output parameters from the model
- analysis
	- analysis scripts for processing data in output
        - common_fn.py: Contains the plotting and processing functions
        - vary_self.py: Analyses combinations of self-regulation
        - vary_cross.py : Analyses combinations of cross-regulation
        - vary_all.py: Analyses all the combinations
        - noise.py: Analyses the model output with added noise
- figures
	- figures produced by the analysis scripts are stored here
- writing
	- contains the writeup/slides drafts

## How to run
- Clone the repository
```
git clone https://github.com/Harshavardhan-BV/rev-XCI.git
cd rev-XCI
```
- You'll need all the packages listed in [requirements.txt](./requirements.txt). A virtual environment is recommended to avoid dependency conflicts.
``` 
# create a virtual environment
python -m venv ./venv/revXCI
# activate the virtual environment
source ./venv/revXCI/bin/activate
# install required packages
pip install -r requirements.txt
```
- Run the vary_*.py python files in codes to generate fits. vary_all is redundant with the other 2. 
``` 
cd codes

python vary_self.py
python vary_cross.py
# [xor] 
python vary_all.py
cd ..
```
- Generate plots with the vary_*.py files in analysis
``` 
cd analysis

python vary_self.py
python vary_cross.py
# [or]
python vary_all.py
cd ..
```
- Solve the equations with the fit parameters with added noise
```
cd code 
python Noise_sensitivity.py
cd ..
```
- Generate plots with noise with the noise.py files in analysis
```
cd analysis
python noise.py
cd ..
```
