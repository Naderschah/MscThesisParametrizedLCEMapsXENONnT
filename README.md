# Data Driven Development of Light Collection Efficiency Maps for XENONnT

Contains the code required to replicate the main results of my thesis, alongside miscellaneous investigations. 

TODO Link to paper 

#### Abstract:

The XENONnT experiment utilizes computed S2 light collection efficiencies (LCEs) maps to discriminate between signal and background events. Current Monte Carlo maps fail to fully account for detector physics, causing systematic biases in event reconstruction. This work develops a physics-constrained neural network parametrization of the expected detector response using an extended PMT signal likelihood function, with the full model trained on $^{83m}\text{Kr}$ calibration data. To avoid position-dependent biases, both event positions and model parameters are co-evolved under this likelihood framework. The resulting parametrization achieves a reduced $\chi^2 = 1.223$ compared to Monte Carlo's reduced $\chi^2 = 2.301$, while producing significantly more homogeneous event density distributions throughout the detector volume. This improved, physics-constrained LCE model provides a foundation for enhanced position reconstruction training and systematic studies in XENONnT. The constrained co-evolution methodology may demonstrate applicability to other dual-phase time projection chamber experiments.

#### Git repo organization

The main results live in:
- ActualThesisWork/CurveFit_FirstBiased mod.ipynb
Here instructions can be found as to how to run all code found within the thesis, most people will only be interested in this file.
- ActualThesisWork/MC_AnodeMesh.ipynb
This contains the small monte carlo simulation I ran alongside some plotting that I did there

Miscellaneous files of relevance in that directory are 

functions.py : contains plotting, preprocessing, etc functions

layer_definitions.py : Contains tensorflow layer definitions alongside the means by which i save and load models (as my layers are not serializable) 

The reimplementations directory contains JAX, Numba, and Tensorflow (minimal form) implementations of the Likelihood functions, I intended full reimplementation in each framework but this was never realized so some leftovers are present from that. With the test_implementations.ipynb comparing these implementations and containing the exact likelihood Cython implementation. 

Data is available on the midway server under path TODO. This must be placed in the root path of the repo to get running, adjust docker-compose or paths as explained in ActualThesisWork/CurveFit_FirstBiased mod.ipynb. 

The remaining files are kept for completeness sake, allthough most content is likely useless. 
