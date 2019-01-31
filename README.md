# misttborn
MISTTBORN is a Python code for fitting various exoplanet and stellar datasets. This code has been developed mostly by Marshall Johnson. Note that the code is in many ways still a work in progress; not all planned features have been implemented, and while the code works for every case I have thrown at it yet, I have not rigorously tested it in edge cases. Bug reports appreciated!

NOTE: This repository is still being populated! Some code may be missing and the instructions may be incomplete.

FEATURES:

- Allows simultaneous fitting of multiple types of data within an MCMC framework.
- Can handle photometric transit/eclipse, radial velocity, Doppler tomographic, or individual line profile data, for an arbitrary number of datasets in an arbitrary number of photometric bands for an arbitrary number of planets.
- Allows the use of Gaussian process regression to handle correlated noise in photometric or Doppler tomographic data; implementation for radial velocity data is a planned feature.
- Can include dilution due to a nearby unresolved star in the transit fits, and an additional line component due to another star or scattered sun/moonlight in Doppler tomographic or line profile fits.
- Can be used for eclipsing binary fits, including a secondary eclipse and radial velocities for both stars.
- Produces diagnostic plots showing the data and best-fit models. The associated code MISTTBORNPLOTTER should be used to produce publication-quality plots and tables.
- User choice of several different sets of MCMC parameters for handling eccentricity (fixed to circular orbit, e & omega, esin(omega) & ecos(omega), or sqrt(e)sin(omega) & sqrt(e)cos(omega)) and transit parameters (a/R* & b, b & rho*, or a/R* & cos(i)).
- Can include radial velocity jitter as a fit parameter.
- Can include Gaussian priors on arbitrary fit parameters.
- Can fit limb darkening using the uniform sampling method of Kipping (2013).
- Can run an arbitrary number of multiprocessing threads.

CURRENT LIMITATIONS AND KNOWN BUGS:

- For multi-planet fits, there is currently NO enforcement of self-consistency among transit parameters if either a/R* & b or a/R* & cos(i) are used as MCMC fit parameters. I therefore strongly recommend using b & rho* as the fit parameters for a multi-planet system, in which case all planets are forced to use the same rho*. All planets do use the same limb darkening parameters, however.
- The code sometimes freezes for reasons that I have been unable to track down. This usually happens when either the eccentricity is not fixed to zero, or radial velocity jitter is included. The work-around is to split the MCMC run into multiple short runs, which usually successfully complete.
- In order to work around a similar problem the eccentricity is limited to be less than 0.99.
- Fits for stellar binaries currently only support eclipsing double-lined spectroscopic binaries; eclipsing single-lined binaries are not currently supported. However, if there is no significant secondary eclipse and the mass ratio is large fitting the system as an "exoplanet" should work adequately.
- I have not tested the code in the case where there are multiple planets in transits and RVs but only some of the planets transit.
- Only quadratic limb darkening is currently supported.
- Only certain kernels available in george and celerite are currently supported.
- The calculation of the Bayesian information criterion may not be correct.

DEPENDENCIES & REQUIRED PACKAGES:

- Standard Python packages: numpy, scipy, matplotlib, math, argparse, sys, os, multiprocessing, time, readcol
- The MCMC uses emcee
- The photometric models are produced by either batman or jktebop
- The radial velocity models for eccentric orbits are produced by RadVel
- The Doppler tomographic and line profile models are produced by horus (to be included in this repository)
- The Gaussian process regression is performed by either george or celerite
- Importing Doppler tomographic data requires either idlsave or pickle, depending upon the input format

SHORT INSTRUCTIONS:

The code is typicaly run from the command line as:
"python misttborn.py input.filename -p -v"
to, for example, run a fit only to photometric data. input.filename is the name of a file containing various input parameters for the code, which is described in detail below.

FULL LIST OF COMMAND-LINE FLAGS:

- -p, --photometry: perform a photometric fit.
- -r, --rvs: perform an RV fit.
- -t, --tomography: perform a Doppler tomographic fit
- -l, --line: fit a single spectral line (mostly useful for getting vsini)
- note that at least one of these first four need to be specified for the code to do anything
- -g, --gp: use Gaussian process regression (currently only implemented for photometry and Doppler tomography, only limted kernel selection currently available)
- -b, --binary: fit a stellar binary (i.e., two sets of spectral lines, both primary and secondary eclipses).
- -v, --verbose: print the log likelihood every MCMC step. Default is to not print anything.
- --startnew: force the code to start a new chain from the input parameters. Default is to check whether the specified output chain file already exists and, if it does, start the emcee run at the final position in that file.
- --plotstep: plot the data and model at every MCMC step. This will force the code to run a single multiprocessing thread.
- --plotbest: make plot(s) showing the best-fit photometric model instead of running the MCMC. See the description of the "plotfile" parameter below for a more detailed description.
- --ploterrors: plot error bars if --plotbest is specified; otherwise, no effect
- --plotresids: plot residuals if --plotbest is specified; otherwise, no effect
- --time: print the elapsed time to compute the best-fit model, log likelihood, and log prior for each MCMC step.
- --getprob: print the contributions to the ln(probability) of the model from each dataset and prior. If specified with --startnew will use the parameters given in the input file, otherwise the best-fit parameters calculated as for the --plotbest option. Will also plot the Bayesian information criterion (BIC) corresponding to the best-fit parameters.
- --bestprob: use with --plotbest to plot the most probable fit instead of the median of the values (good for highly correlated results)
- --skyline: include a line profile due to the sky (or another contaminating star) in the Doppler tomographic or line profile fit.
- --dilution: include dilution due to a nearby star in the photometric fit.
- -h, --help: print all flags and their short descriptions to the command line.
- Several other command-line options currently specified in the code (--ttvs, --fullcurve, --pt) are not currently fully implemented and should NOT be used.

STRUCTURE OF INPUT FILE:

The input files consists of two tab-separated columns, the first column being a parameter name and the second the parameter value, which can be either a number or a string. The order of parameters in the file is arbitrary, except with a few restrictions that I'll list below.
- nplanets: the number of planets to fit
- nwalkers: the number of MCMC walkers to use. Must be at least twice the number of fit parameters or emcee will refuse to run.
- nsteps: the number of MCMC steps to run.
- nthreads: the number of multiprocessing threads to run. For best performance should be no more than the number of cores on your machine.
- sysname: name of the system. For human readability only, is not actually used by the code.
- g1p#, g2p#: starting values of the quadratic limb darkening parameters, where # denotes an integer for each individual filter/photometric band.
- rhostar: starting value of the stellar density.
- Per#, epoch#, rprs#, bpar#: starting values for the period (in days), epoch (in days), Rp/R*, and impact parameter. Replace # with the planet number for each planet (e.g., Per1, Per2, etc.).
- ecc#, omega# (optional): same as above, but for eccentricity and omega. To force a circular orbit, simply omit these from the input file.
- aors# (optional): to use a/R* for each planet as a fit parameter instead of rhostar, simply include this parameter for each planet and omit rhostar, and also set the parameter rhobaflag (see below). Note that there is currently no mechanism to enforce the same stellar density for each planet when fitting with a/R* instead of rhostar, so I do not recommend using this option for multiplanet systems.
- rhobaflag (optional): specify whether to use a/R* (set rhobaflag = aorsb) or rhostar (set rhobaflag = rhostarb) as a fit parameter. Defaults to rhostarb.
- photfile#: path to the input lightcurve file(s), one per dataset (or filter). Each input file is assumed to consist of four columns: time (in days), normalized flux, error, and either 0 or 1 to denote the cadence, or the exposure time. The exposure time is assumed to be in days, except:
- expunit# (optional): include this parameter to specify the units of the exposure time. Acceptable values are: s or seconds, d or days, m or minutes, h or hours. # is the same integer as specifies the lightcurve files in photfile#.
- cadenceflag (optional): set to kepler, tess, or corot to interpret the fourth column of the input file as specifying the cadence for the specified transit mission: 0=short cadence, 1=long cadence. It will then convert these to the appropriate exposure time.
- photlcflag (optional): set to q to use the Kipping triangular sampling for the photometric limb darkening parameters, or g to just use the regular quadratic limb darkening parameters. Omitting this parameter will default to triangular sampling.
- perturbfile: path to the file specifying the size of the perturbations to each parameter for the starting positions of the MCMC chains. See below for details on the format of this file
- priorfile (optional): path to the file containing the priors. Again, I'll describe the format of this file below. If omitted, all parameters will be left free (except that I've hard-coded in physical limits to relevant parameters--e.g., period must be positive, a/R* must be greater than 1 to ensure that the orbit is outside the stellar surface, etc.).
- chainfile, probfile, accpfile: path for the output .npy format files containing the MCMC samples, the probability for each step, and the acceptance rate of each chain, respectively
- asciiout (optional): path to a file where the code will print out the MCMC chain in ascii format.
- nburnin (optional): if specified, the asciiout file will have the first nburnin steps cut off. If unspecified, the full chain will be printed. 
- plotfile (optional): if the --plotbest flag is included in the command line call, instead of performing an MCMC run the code will open the chainfile, find the best-fit parameters from the chain by cutting off the first nburnin steps (if nburnin is unspecified it will discard the first 1/5 of the chain), discard threads that still deviate from the mean by >5 sigma after nburnin steps (not 100% sure this is a good idea), and produce a plot with the specified filename. As of now at least .pdf, .eps, .jpg, .png formats should be supported. Note that for multi-planet systems only the first planet's lightcurve will be plotted unless you specify a .pdf filename (this is due to a limitation of matplotlib, which can only produce multipage plots in pdf format). It will also print these best-fit parameters and uncertainties (just calculated via standard deviation for now) to the command line. Also as of now this causes the program to shut down while calling emcee after the plot has been completed, which generates some error messages so you'll have to scroll up to be able to see the best-fit parameters.
- longflag (optional): omit or set to batman to use batman's binning method for long-cadence data, setting to any other value will use my binning method.
- filternumber# (optional): for multi-band datasets, specify which datasets are in the same filters. The # here is the same as in the photfile# parameters. Example: say you have photfile1 and photfile2, which are in V, and photfile3, which is in R. You would set filternumber1=1, filternumber2=1, and filternumber3=2.
- ewflag (optional): If fitting for eccentricity and ω, tells the code which combination of parameters to use as the fit parameters. Set to sesinw to use √e*sinω and √e*cosω (default); to ecsinw to use e*sinω and e*cosω; and to eomega to use e and ω.
-  rvtrend (optional): Fit for a linear trend in the RVs with a slope given by this parameter. The slope will be given in either km/s/day or m/s/day depending upon whether the input RVs are in km/s or m/s.
- photmodflag (optional): Specify which package to use for producing the photometric models. Should be set to either "batman" (default) or "jktebop"
- jktpath (optional): If photmodflag is set to "jktebop", you must set this parameter to the path to where jktebop can be found on your machine.
- vsini: starting value for vsini for Doppler tomographic or line profile fits
- linefile (optional): input file containing a single spectral line. Set to “tomfile” to tell the code to use the average line profile from a Doppler tomographic data set, otherwise supply an IDL or numpy save file.
- rvfile: path to file containing RV data. The file containing the RV data, specified in the rvfile parameter, should have the following format. Four space or tab-separated columns, containing: time RV error #; where # is an integer corresponding to the number of the dataset (i.e., if more than one spectrograph is used, 1 for the first spectrograph, 2 for the second, etc; the # in gamma# correspond to these indices; if only one spectrograph is used just have a 1 in every entry in this column). The time of each measurement should be in the same time standard as the photometric data. The units of RV and error are arbitrary, but should be the same as those of semiamp#. If the -b/--binary flag is specified, the RV input file should instead have six columns: time RV1 error1 RV2 error2 #, where RV1 and RV2 are the RVs of the two components.
- semiamp#: starting value for the RV semi-amplitude for planet # (same indices as for Per1, etc.)
- gamma#: (optional) starting value for the RV offset for dataset #. Assumed to be zero if not specified.
- fixgam: (optional) set to True to fix the RV offsets to the values given as gamma#, or False to include these values as fit parameters in the MCMC. Defaults to False if not specified.
- gpxpary: if Gaussian process regression is being used, add the hyperparameters as “gpxpary” where “x” denotes to which type of dataset the parameter belongs: “t” for tomographic, “p” for photometric, “r” for radial velocity; and “y” denotes the type of parameter. “amp” for amplitude of the kernel, “tau” for the scale, “tauz” for the scale along direction z for multi-dimensional data--i.e., for tomographic data, “taut” for time axis and “tauv” for velocity axis, “P” for the period for a periodic kernel. E.g., gppparamp for photometric amplitude, gptpartaut for tomographic scale in t axis.
- gpmodtypex: if Gaussian process regression is being used, add one or more parameters “gpmodtypex,” where “x” is defined above; these parameters should be a string denoting the type of kernel to use for this dataset. Full list of kernels supported by the george package that I use available at: http://dan.iel.fm/george/current/user/kernels/ Not all of these have been implemented in MISTTBORN yet. Ask Marshall if you want a specific kernel implemented. (Currently implemented: for photometry, Matern 3/2 (“Matern32”) and Cosine kernels; for tomographic data, Matern 3/2 (sum of kernels for time and velocity axes).)


There are a few restrictions on the format of the input file:
- The parameter names that are included are mostly arbitrary. A parameter name that doesn't match one that is used by the code will generally be ignored. However, for the parameters that can be specified for each planet (Per, epoch, rprs, bpar, aors, ecc, omega), no part of each of these strings should appear in another parameter name. So don't call a parameter "Permanent" or "eomega5" or anything like that.
- The parameters specified for each planet must use consistent planet numbers--1 for planet 1, 2 for planet 2, etc., but don't include parameters for more planets than are specified in the nplanets parameter, and planet numbers can't be skipped (i.e., don't specify parameters Per1, Per2, and Per4; Per3 must come next). Planet numbers should begin with 1, not 0.
- The order of the parameters for the different planets is arbitrary, except that currently bpar and rprs must be listed in the same order (i.e., if you specify bpar1 before bpar2 in the input file, rprs1 must also come before rprs2). I aim to fix this in the future.

STRUCTURE OF PERTURBATION FILE:
- The structure of the perturbation file is similar to that of the input file, with two tab-separated columns, the first with a parameter name and the second with the size of the input perturbation. The ordering of the parameters is mostly arbitrary, however, for all planet parameters, these must be listed in the same order as they are in the main input file--i.e., if bpar2 comes before bpar1 in the input file, it must also in the perturbation file.
- Currently all fit parameters must have perturbation sizes listed in this file (i.e., g1p, g2p, Per#, epoch#, rprs#, bpar#, either rhostar or aors#, and ecc# and omega# if these are specified). The parameter names in the perturbation file must exactly match those in the main input file.
- Note that the perturbations listed under "ecc#" and "omega#" are actually for sqrt(e)sin(omega) and sqrt(e)cos(omega), respectively. In the future I will fix this so that you will actually specify perturbations for sesinw# and secosw#.

STRUCTURE OF PRIOR FILE:
- The code now allows you to set priors on arbitrary fit parameters. The structure of this file should be the same as for the main input and perturbation files, with two tab-separated columns, the first with parameter names and the second with parameter values. Here the parameter names should be the name of the parameter you wish to set a prior upon (must exactly match the parameter name in the main input file), and the second column should be the width of that prior.
- Note that asymmetric priors are not currently supported, but I will implement these soon.
- Also currently priors may only be set on MCMC fit parameters; priors set on other parameters are ignored. At some point I want to be able to set priors on arbitrary combinations of MCMC fit parameters, but I'm still figuring out how to implement this.

STRUCTURE OF OUTPUT FILES:
- MISTTBORN normally produces three types of output files, as specified in the input file:
- chainfile: a numpy save file containing the MCMC chains. It is an array of size nwalkers x nsteps x nparameters. The names of the parameters are not specified in this file due to limitations of the numpy file format, but the order of the parameters is printed by MISTTBORN whenever the code terminates normally.
- probfile: a numpy save file containing the log likelihood of the fit at each MCMC step. It contains an array of size nwalkers x nsteps.
- accpfile: a numpy save file containing the fraction of proposed steps accepted for each parameter. It contains an array of size nparameters.
- Two other types of file can be produced if specified in the input file:
- plotfile: if the code is called with the --plotbest option, the code will produce plots of the best-fit solution in a PDF file with this filename. If --startnew is specified, the plots will show the model corresponding to the starting parameters instead of the best-fit solution.
- asciiout: if specified, in addition to the numpy save file containing the MCMC chains, the code will also produced an ascii-formatted table listing all of the parameters at each MCMC step.
