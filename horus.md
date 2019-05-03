### HORUS

NOTE: this documentation is under development and the actual horus.py code has not yet been uploaded to GitHub, although this will occur soon!

[horus.py](./horus.py) is a code to produce Doppler tomographic models--i.e., model of the perturbation to a rotationally-broadened stellar line profile produced by a transiting planet. It is also capable of modeling stellar line profiles in general, and producing photometric light curves (although as it produces these by numerical integration over the stellar disk this is much slower and potentially less accurate than analytical models like batman or JKTEBOP).

FEATURES:

- 
- Can take into account the effects of micro- and macroturbulent broadening, differential rotation, and/or gravity darkening.
- Can return an image of the stellar surface used to produce the line profile model rather than the line profile model itself.

CURRENT LIMITATIONS AND KNOWN BUGS:

- The macroturbulent broadening and gravity darkening have not been rigorously tested. Use these features at your own risk!
- For an eccentric orbit the code assumes that the orbital velocity of the planet is constant over the course of a single exposure. This assumption should not cause significant problems unless the planetary orbit is extremely eccentric and/or the exposure times are very long.

DEPENDENCIES & REQUIRED PACKAGES:

- Standard Python packages: numpy, math

SHORT INSTRUCTIONS:

A minimal use of the code to construct a model of a rotationally broadened line profile could look something like:
```
>>> import horus
>>> horusstruc = {'vsini': 50.0, 'width':5.0 'obs':'hires', 'vabsfine':np.linspace(-100,100,1), 'gamma1':0.5, 'gamma2':0.5, 'sysname':'test'}
>>> modelstruc = horus.model(horusstruc, res='medium', onespec=True, convol=True)
>>> model = modelstruc['baseline']
```
This will produce a model of a line profile with vsini of 50 km/s, broadened by an intrinsic stellar broadening of 5 km/s and the instrumental broadening of the Keck/HIRES standard California Planet Search set-up (R=60,000). 

FULL LIST OF INPUT OPTIONS:

- An input dictionary containing the following keys:
  - vsini (required): the stellar vsini, in km/s
  - width (required): the sigma-width of the assumed Gaussian line profile of each stellar surface element
  - gamma1 (required): linear limb-darkening coefficient
  - gamma2 (required): quadratic limb-darkening coefficient
  - vabsfine: an array containing the velocities at which the line profile will be calculated, in km/s
  - obs: either the name of a observatory or spectrograph (see full list of supported keywords below), or the instrumental resolving power (R).
  - The following keys if a Doppler tomographic model is to be created:
    - Pd: the planetary orbital period in days
    - lambda: the spin-orbit misalignment in degrees
    - b: the transit impact parameter
    - rplanet: the Rp/R* value for the transit
    - t: an array containing the time stamps at which the model should be calculated, in units of ...
    - times: an array containing the exposure times, in units of ...
    - a: scaled semi-major axis of the orbit, i.e., a/R*
    - dur (optional): duration of the transit, in days. If the amode keyword is set to 'dur', this will be used to calculate a/R*.
    - e (optional): the eccentricity of the planetary orbit
    - periarg (optional): the argument of periastron of the planetary orbit, in units of degrees
  - The following keys if macroturbulence is in use:
    - zeta: the macroturbulent velocity dispersion in km/s
  - The following keys if differential rotation is in use:
    - inc: the inclination of the stellar rotation axis with respect to the line of sight, in degrees
    - alpha: the differential rotation parameter
  - The following keys if gravity darkening is in use:
    - inc: the inclination of the stellar rotation axis with respect to the line of sight, in degrees
    - beta: if gravity darkening is in use, the gravity darkening parameter
    - Omega: if gravity darkening is in use, the stellar rotation rate in radians/s
    - logg: if gravity darkening is in use, the stellar logg (assumed to be at the pole)
    - Reqcm: if gravity darkening is in use, the stellar radius in Solar radii
    - f: if gravity darkening is in use, (not actually sure what this does...)
  - lineshifts: an array (for a time-series model) or a float (for a single line profile) specifying the central velocity of the line profile at each epoch, in km/s
- mode (string): either 'spec' to tell the code to produce a spectroscpic model, or 'phot' to produce a photometric model
- res (string): specifies the resolution of the Cartesian grid on the stellar surface. This can either be 'low' (50x50 grid), 'medium' (200x200 grid), or 'high' (600x600 grid). Alternately, the size of the grid can be specified using the resnum keyword.
- resnum (float): radius of the star in units of the Cartesian grid cells, i.e., to have the integration use an NxN grid, set resnum to N/2
- onespec (Boolean): will produce only a single spectroscopic model line profile rather than a time series
- convol (Boolean): if True, will convolve the spectroscopic model with an instrumental line profile
- macroturb (Boolean): if True, include the effects of macroturbulence (parameterized as radial-tangential anisotropic macroturbulence) in the model
- diffrot (Boolean): if True, include the effects of differential rotation in the model
- gravd (Boolean): if True, include the effects of gravity darkening in the model.
- lineshift (Boolean): allow the central velocity of each line profile to be different from zero
- image (string): tells the code to return an array containing an image of the star. If set to 'all', this will be a 2-d array showing the normalized intensity on each element of the unocculted stellar disk. If set to 'each', this will be a 3-d array containing the normalized intensity of each surface element at each timestep. If set to 'vels' this will be a 3-d array similar to 'each' except that it contains the radial velocity of the unobscured stellar surface elements at each timestep. Setting image to 'comb' is identical to 'all'. If set to 'n' (default) no image will be returned.
- path (Boolean):
- amode (string): either 'a' to specify a/R* directly for the model, or 'dur' to specify the transit duration and let the code calculate a/R* from the duration and other planetary parameters

OUTPUT FORMAT

The code will return a dictionary containing the following keys:
- if mode = 'spec':
  - profarr: an array containing the time-series line profiles
  - basearr: an array containing the time-series line profiles without the planetary perturbation, but including any RV shifts or other time-variable effects
  - baseline: an array containing the out-of-transit line profile
- if mode = 'phot':
  - timeflux: an array containing the integrated flux from the star at each exposure
- if path=True, the following:
  - z1:
  - z2:
  - patharr:
  - staraxis:
-if image is not 'n':
  - imarr:

CURRENTLY SUPPORTED OBSERVATORY KEYWORDS:
  
- LBT/PEPSI, R=120,000 mode: 'lbt' or 'pepsi'
- KECK/HIRES California Planet Search setup, R=50,000: 'keck' or 'hires'
- Subaru/HDS, R=80,000: 'subaru' or 'hds'
- HET/HRS medium resolution mode, R=30,000: 'het' or 'hrs'
- Gemini North/GRACES, 1-fiber mode R=67,500: 'geminin' or 'graces1'
- Gemini North/GRACES, 2-fiber mode R=40,000: 'geminin' or 'graces2'
- AAT/UCLES, R=70,000: 'aat' or 'ucles'
- TNG/HARPS-N, R=120,000: 'tng' or 'harpsn'
- NOT/FIES, R=47,000: 'not' or 'fies'
- McDonald 2.7m HJST/TS23 spectrograph, R=60,000: 'hjst' or 'ts23'
- FLWO 1.5m/TRES, R=44,000: 'tres'
- Gemini South, DCT, or HJST/IGRINS, R=40,000: 'igrins'
- KECK/LSI, R=20,000: 'keck-lsi' or 'lsi'
