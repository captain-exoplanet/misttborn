# misttborn
MISTTBORN (MCMC Interface for Synthesis of Transits, Tomography, Binaries, and Others of a Relevant Nature) is a Python code for fitting various exoplanet and stellar datasets. This code has been developed mostly by Marshall Johnson. Note that the code is in many ways still a work in progress; not all planned features have been implemented, and while the code works for every case I have thrown at it yet, I have not rigorously tested it in edge cases. Bug reports appreciated!

NOTE: This repository is still being populated! Some code may be missing and the instructions may be incomplete.

Contents of this repository:
- [misttborn.py](./misttborn.py) The actual fitting code, instructions and documentation are [here](./misttborn.md)
- [misttbornplotter.py](./misttbornplotter.py) is a code to produce publication-quality plots from the MCMC output of MISTTBORN. Instructions and documentation are [here](./misttbornplotter.md)
- [examples](./examples) contains various example input files and datasets to demonstrate how to set up MISTTBORN.
- [kernels](./kernels) contains kernels for Gaussian process regression with the [celerite package](https://celerite.readthedocs.io/en/stable/#).
