# Data Driven Development of Light Collection Efficiency Maps for XENONnT

Contains the code required to replicate the main results of my thesis, alongside miscellaneous investigations. 

TODO Link to paper 

#### Abstract:

The XENONnT experiment utilizes computed S2 light collection efficiencies (LCEs) maps to discriminate between signal and background events. Current Monte Carlo maps fail to fully account for detector physics, causing systematic biases in event reconstruction. This work develops a physics-constrained neural network parametrization of the expected detector response using an extended PMT signal likelihood function, with the full model trained on $^{83m}\text{Kr}$ calibration data. To avoid position-dependent biases, both event positions and model parameters are co-evolved under this likelihood framework. The resulting parametrization achieves a reduced $\chi^2 = 1.223$ compared to Monte Carlo's reduced $\chi^2 = 2.301$, while producing significantly more homogeneous event density distributions throughout the detector volume. This improved, physics-constrained LCE model provides a foundation for enhanced position reconstruction training and systematic studies in XENONnT. The constrained co-evolution methodology may demonstrate applicability to other dual-phase time projection chamber experiments.

#### Git repo organization

Code is organized as follows, in TODO one can find a neatly formated version of hte working code (not tested but should work to just run the notebooks in sequence - this will take very long, 3+ days on my device). To do so adjust the docker entry points, I ran on my device rather than on server, minor adjustment is needed to get it running on server (documented in the first cells of the notebook). The docker file requires that paths be adjusted to run, specifically the link to the working directory, it must point to TODO when using the non-codedump folder and TODO when using the cleaned up folder. 

The codedump folder is in repo path TODO, this contains the code I actually worked with, it is very messy with lots of things no longer relevant, notebooks were often not run in sequence but parts or explicitly in weird orders to achieve some desired effect. Consult this only if more information is required, I also recommend contacting me directly if this becomes required. 

Data is available on the midway server under path TODO. This must be placed in TODO to get running. 

Likelihood Function reimplementations are available in folder TODO and are equivalent to the codedump folder. 

TODO Replication Notebook overview

TODO codedump notebook overview 

