
Notebooks Felix Semler

Main:

ReplicateJelle.ipynb : 
https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:aalbers:lce_fit

Applying the methods developed by Jelle from 1T to nT

ShenyangSR2_RadialLCE.ipynb : 
https://xe1t-wiki.lngs.infn.it/doku.php?id=shenyang:s2_pattern_likelihood

Utilizing Shenyangs methods on SR2 nT - main focus is 1PMT model from ReplicateJelle, however, some work may be done on reimplementing for SR2 allthough this does not seem required

OfficialLCEs.ipynb : Loading and working with a variety of LCE's I could find as reference

process_raw_local.ipynb, retrieving_data.ipynb, : Notebooks used to process event_area_per_channel -> At the time i didnt know how i could do this on the chicago servers

filtering_data.ipynb : Generates hdf5 files of filtered data 


TODO:
Train_MLP.ipynb : Training MLP on varying LCE's to determine LCE 'correctness'

Anode_To_Gate.ipynb : The RLCE model predicts based on pre anode location : Need a model to shift down to gate 



Misc:

functions.py : Holds model definitions, plotting definitions, utility functions, etc.
