# Project-1

HI, I am Sweekrity from CRI, Paris. I am working on low cost wearables. I am trying to improve the HR estimation and sp02 level so that it is comparable to HR and sp02 estimation by ECG.

Folders:
--------

Data:
*****
Data folder contains all the datasets being processed and analyzed. It also stores dataframe pickles that are being created during analysis.

Scripts:
********

a) extract_data.py:  
  This script imports data and converts it into a dataframe
  
b) preprocessing_data.py:
   This script has three functions:
   a) plotting time series data
   b) normalizing the data - scaling it 
   c) baselineshit - to remove the slope from the signal

c) analysis.py
   This script uses baselineshifted data and calculates
   a) sampling frequency
   b) signal to noise ratio (is being optimized)
   c) M-M peaks (is being optimized)
   d) Heart rate estimation (is being optimized)
