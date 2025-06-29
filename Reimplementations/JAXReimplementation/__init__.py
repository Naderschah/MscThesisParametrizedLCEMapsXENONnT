

"""
Implementation Jax version:
- 0.4.30

All code is typed using pythons typing module
This enforces nothing and is only hints 

Used types are imported in the type_names.py file 

Design strategy:
- In tf i often used tf.where for masking rather than multilying by valid masks, i replaced all of the ones with simple functions but left them for complex functions
    - Conditionals are harder to evaluate for XLA, but are required for say Gaussian vs exact evaluation




NEeds to be done:
- 

"""