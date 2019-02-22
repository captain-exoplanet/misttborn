### MISTTBORNPLOTTER

[misttbornplotter.py](./misttbornplotter.py) is a code to produce publication-quality plots and tables from the MCMC chains output by misttborn.py, using the best-fit parameter values found by the MCMC.

FEATURES:

- Publication-quality plots of the photometric, radial velocity, and Doppler tomographic data and best-fit models.
- The photometric plots show the phase-folded transit light curves, with the light curve corresponding to each individual photometry files offset vertically. These use different colors for the data points corresponding to the different bands, roughly corresponding to the actual color of each band. If Gaussian process regression is used, this plot will show the detrended light curves.
- RV plots show the models corresponding to 50 random draws from the posteriors in order to illustrate the uncertainty on the RV solution.
- If Gaussian process regression is used for the photometric fits, the code will also produce a plot showing the full light curve  along with the GP fit.
- In addition to the MCMC jump parameters, the code also calculates and prints additional derived values (e.g., eccentricity, &c.) to the output LaTeX table. If the user provides the stellar absolute parameters (mass, radius, effective temperature) in the input file, the code will also calculate the absolute planetary parameters (mass, radius, equilibrium temperature, &c.).

CURRENT LIMITATION & KNOWN BUGS:

- Currently MISTTBORNplotter.py will not work properly for RV plots for multiplanet systems; it won’t subtract the best-fit RV models from the other planets from the data. I’ll fix this when this feature is needed.
- Currently the RV plotting capabilities can only handle up to five separate RV facilities.
- Trailing zeros at the end of a value will not be printed to the table due to a limitation of Python; i.e., if a value is 0.10 +/- 0.12, it will be printed as 0.1 +/- 0.12
- The number will be printed to one digit after the decimal place if the uncertainty is greater than 10, i.e., it will print 2106.6 +/- 15.7 rather than 2107 +/- 16
- For RV-only planets the semi-major axis needs to be calculated using Kepler’s laws rather than using a/R*; this has not yet been implemented. 
- The atmospheric scale height H is computed assuming a hydrogen-dominated atmosphere. This is fine for gas giants but will not be correct for smaller planets with heavier mean molecular weight atmospheres.
- The planetary equilibrium temperature calculation does not account for orbital eccentricity, i.e., this is the zero-albedo equilibrium temperature of a planet on a circular orbit at the same semimajor axis. Similarly, the duration of the “flat-bottomed” part of the transit T_23 does not totally properly account for eccentricity (except insofar as it is properly incorporated in the calculation of the full transit duration T_14).

DEPENDENCIES & REQUIRED PACKAGES:

- Same packages required by MISTTBORN
- [corner](https://corner.readthedocs.io/en/latest/) if corner plots are to be made with the --corner command-line flag.
- [uncertainties](https://pythonhosted.org/uncertainties/) if the code is to calculate the absolute planetary parameters.

SHORT INSTRUCTIONS:

In general, MISTTBORNplotter should be run with the same set of flags as you used for your MISTTBORN run. That is, if you ran MISTTBORN as:
```
python misttborn.py input.file -p -r -v
```
you should run MISTTBORNplotter as:
```
python misttbornplotter.py input.file -p -r -v
```
potentially with additional flags (described below) to add functionality for the plots and tables.

OUTPUTS:

The name of the system should be specified using the sysname parameter in the input file. For planet #, MISTTBORNplotter will produce the following output files:
- sysname_table#.tex: a file containing a LaTeX-formatted table containing an upper section labeled "Measured Parameters" containing the median and 1-sigma uncertainties on the MCMC jump parameters, and a lower section labeled "Derived Parameters" containing parameters derived from the MCMC parameters. Note that the table will only contain the table material itself; the user will need to supply the table header, &c. Also note that [tablecatter.py](https://github.com/captain-exoplanet/utilities/blob/master/tablecatter.py) can be used to concatenate multiple tables (for instance, for multiple planets, or for circular and eccentric runs for the same planet) into a single table with one column for input table.
- sysname_corner.pdf (optional): if the --corner flag is specified, a corner plot showing the correlations between all of the MCMC jump parameters.
- sysnameLC#.pdf (optional): if the -p --photometry flag is specified, a plot showing the phase-folded light curves for planet #.
- sysnameGPLC.pdf (optional): if the -g --gp and -p --photometry flags are specified, a plot showing the full light curve with the Gaussian process fit overplotted.
- sysnametransit#.pdf (optional): if the -g --gp and -p --photometry flags are specified, a series of plots showing each individual transit in the light curve along with the GP fit.
- sysnameRVphased#.pdf (optional): if the -r --rvs flag is specified, a plot showing the phase-folded RVs for planet #. Note that this currently will NOT work correctly for multi-planet systems!
- sysnameRVall#.pdf (optional): if the -r --rvs flag is specified, a plot showing the time series RVs for planet #. Note that this currently will NOT work correctly for multi-planet systems!
- sysnameDTdata#.pdf (optional): if the -t --tomography flag is specified, a plot showing the Doppler tomographic data in the form of time series line profile residuals from tomographic dataset #.
- sysnameDTmodel#.pdf (optional): if the -t --tomography flag is specified, a plot showing the best-fit Doppler tomographic model for tomographic dataset #. 
- sysnameDTresids#.pdf (optional): if the -t --tomography flag is specified, a plot showing the residuals to the best-fit Doppler tomographic model for tomographic dataset #. 

FULL LIST OF COMMAND-LINE FLAGS:

MISTTBORNplotter takes many of the same flags as MISTTBORN, along with additional flags to control the plots and tables output by the code:    
- -p --photometry
- -r --rvs
- -t --tomography
- -l --line (not currently implemented)
- -g --gp Use Gaussian process regression
- -v --verbose (no effect for misttbornplotter)
- -b --binary: fit a double-lined eclipsing binary.
- --dilution
- --skyline, --ttvs (not currently implemented)
- --plotresids: include the residuals to the transit, RV, and tomographic fits in the plots
- --earth: output planetary radii and masses will be in Earth units (default is Jupiter units)
- --ms: indicates that the input RV data are in units of m/s (default is km/s)
- --bold: have all of the values in the TeX table printed in bold type
- --corner: create a corner plot from the MCMC chains. Requires Dan Foreman-Mackey’s corner package to be installed.
- --tableonly: produce only a LaTeX output table and terminate the code before making any plots.
- --bw: have the output plots be in grayscale rather than color
- --dosecondary: if the --binary flag is used, make plots showing both the primary and secondary eclipses.
- --fullLC: Make a plot showing the full light-curve; this is really only useful for space-based data with continuous coverage.

INPUT FILE PARAMETERS:

MISTTBORNplotter should be used with the same input file as MISTTBORN. However, a number of additional parameters may be added to the input file to unlock additional capabilities (these are ignored by MISTTBORN itself).

- photlabel#: a label for the photometric dataset which will be printed next to the corresponding light curve in the photometry plot. Note that due to a limitation of the input format, any spaces which should appear in the label should be replaced with colons in the input file. E.g., if the label should read "DEMONEXT g-band," the photlabel# parameter should be "DEMONEXT:g-band."
- photname#: the band in which the data were obtained, which is used to color the light curve correspondingly. If this is not specified the light curve will be shown in black. The currently supported bands are: Kp CoRoT clear g r i z U B V R I J H K  3.8um 4.5um
- Mstar:    Stellar mass in MSun
- epMstar, emMstar:    positive and negative uncertainties on Mstar, respectively
- Rstar:    Stellar radius in RSun
- epRstar, emRstar:    positive and negative uncertainties on Mstar, respectively
- Teff:    Stellar effective temperature, in K
- eTeff:    Uncertainty on Teff; asymmetric uncertainties are not supported at the present time
- timestandard: String containing the name of the time standard used for the data; if unspecified, BJD will be assumed, unless the value of the first timestamp is less than 4000, in which case BKJD (i.e., BJD-2454833) will be assumed. This will be used as the units for the epochs listed in the output table, and axis labels for plots showing the full time-series of RVs or photometry.
- timeoffset: The number which has been subtracted from timestandard in order to produce the timestamps used in the input data, i.e., if the input data are in BJD-2450000, then timestandard should be set to BJD and timeoffset to 2450000. If unspecified, the code will check the data, and will assume 0 if the value of the first timestamp is greater than 2000000; 2400000 if the value of the first timestamp is greater than 50000; 2450000 if it is greater than 4000; and 2454833 (i.e., BKJD) otherwise.

If some or all of Mstar, Rstar, and Teff and associated uncertainties are specified in the input file, the code will calculate the corresponding absolute planetary parameters and print these to the LaTeX output file.

ACKNOWLEDGMENTS:

The code snippet for truncating a value to a specific number of significant numbers was obtained from [here](https://stackoverflow.com/questions/9415939/how-can-i-print-many-significant-figures-in-python) on StackOverflow.
