# pvinsight-validation-hub
Scripts related to the PVInsight Validation Hub.

This particular setup shows the framework for feeding algorithms into a
pipeline, and running a test validation. Scripts are as follows:


process-az-tilt.ipynb: This jupyter notebook is the master script that
assesses algorithm submission performance, and outputs a performance score
for each test set for each submission. Function submissions are imported as
modules via the importlib package.

/data/ folder: Contains any neccessary metadata and time series data needed
to run the function. In this case we have a metadata file (metadata.csv)
and multiple time series data files (4_ac_power__315.csv,
1403_inv2_ac_power__4213.csv)

az-tilt-estimator.py: This is an algorithm submission, which is tested for
performance in the process-az-tilt.ipynb notebook. It takes parameters from the
metadata file as inputs, and outputs azimuth and tilt estimates that are
then compared to the ground-truth values.