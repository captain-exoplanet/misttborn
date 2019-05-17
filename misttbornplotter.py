#MISTTBORN: the MCMC Interface for Synthesis of Transits, Tomography, Binaries, and Others of a Relevant Nature
#This code (MISTTBORNplotter) produces publication-quality plots and analysis from the MCMC chains output by MISTTBORN.
#Written by Marshall C. Johnson. Thanks to Andrew Mann, Pa Chia Thao, Fei Dai, Elisabeth Newton, and Aaron Rizzuto for various contributions, bug reports, feature requests, etc.
#The horus Doppler tomographic modeling code is also written by Marshall C. Johnson.
#The batman transit modeling code is written by Laura Kreidberg.
#The emcee affine-invariant MCMC, and the george and celerite Gaussian process regression codes, are written by Daniel Foreman-Mackey.
#The RadVel radial velocity code is written by B.J. Fulton.
#The name of this code is in reference to the Mistborn novels by Brandon Sanderson. 
#Please report any bugs or other issues by email to johnson.7240@osu.edu, or on GitHub at https://github.com/captain-exoplanet/misttborn

#import packages necessary for any run

import numpy as np
import math
import emcee
from readcol import readcol
import argparse
import sys
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages
import multiprocessing
import os


parser=argparse.ArgumentParser()
parser.add_argument("infile", type=str, help="name of the input file")
parser.add_argument("-p", "--photometry", action="store_true", help="perform photometric analysis")
parser.add_argument("-r", "--rvs", action="store_true", help="perform radial velocity analysis")
parser.add_argument("-t", "--tomography", action="store_true", help="perform Doppler tomographic analysis")
parser.add_argument("-l", "--line", action="store_true", help="fit a rotationally broadened model to a single spectral line")
parser.add_argument("-v", "--verbose", action="store_true", help="print a short message every MCMC step")
parser.add_argument("-g", "--gp", action="store_true", help="enable Gaussian process regression")
parser.add_argument("-b", "--binary", action="store_true", help="fit a binary star rather than an exoplanet: two sets of RVs, primary and secondary eclipses")
parser.add_argument("--skyline", action="store_true", help="include a sky line in some or all of the tomographic data set")
parser.add_argument("--ttvs", action="store_true", help="account for TTVs in the photometric fit")
parser.add_argument("--plotresids", action="store_true", help="include residuals for the plots.")
parser.add_argument("--tableonly", action="store_true", help="Only make the LaTeX tables with no plots.")
parser.add_argument("--earth", action="store_true", help="Output radii and masses in Earth rather than Jupiter units.")
parser.add_argument("--ms", action="store_true", help="Tell the code that the input radial velocities are in m/s, rather than km/s (default).")
parser.add_argument("--bold", action="store_true", help="Print output values in bold.")
parser.add_argument("--corner", action="store_true", help="Make a corner plot from the MCMC chains. Requires the corner package to be installed.")
parser.add_argument("--bw", action="store_true", help="Make all of the plots in black and white")
parser.add_argument("--dilution", action="store_true", help="account for dilution due to another star in the aperture")
parser.add_argument("--dosecondary", action="store_true", help="make plots for the secondary of an eclipsing binary")
parser.add_argument("--bestprob", action="store_true", help="Plot the values for the best-fit model rather than the posterior median.")
parser.add_argument("--fullLC", action="store_true", help="Make a plot showing the full light-curve; this is really only useful for space-based data with continuous coverage.")
parser.add_argument("--rvlegend", action="store_true", help="Show a legend of which point is which dataset on the RV plots")
parser.add_argument("--makereport", action="store_true", help="Make a PDF report including the parameter table and all of the plots")

args=parser.parse_args()



infile=args.infile


import time as timemod
thewholestart=timemod.time()

if args.photometry:
    try:
        import batman
    except ImportError:
        print 'batman does not appear to be installed correctly.'
        print 'you can install it with "pip install batman-package"'
        print 'exiting now'
        sys.exit()
        
    print 'burning tin to perform photometry'

if args.rvs:
    try:
        import radvel
    except ImportError:
        print 'radvel does not appear to be installed correctly.'
        print 'you can install it with "pip install radvel"'
        print 'Defaulting to the assumption of circular orbits for RVs.'
        
    print 'burning pewter to perform RV analysis'

if args.binary:
    print 'burning steel to analyze a stellar binary'

if args.tomography or args.line:
    try:
        import horus
    except ImportError:
        print 'horus does not appear to be installed correctly.'
        print 'you can install it from [URL TBD]'
        print 'exiting now'
        sys.exit()
        
    if args.tomography: 
        print 'burning bronze to perform Doppler tomography'
        import dtutils
    if args.line: print 'burning iron to analyze a single line'




def inreader(infile):
    names, values = readcol(infile, twod=False)
    outstruc = dict(zip(names, values))
    outstruc['index']=names
    outstruc['invals']=values
    return outstruc

struc1=inreader(infile)

index=np.array(struc1['index'])
invals=np.array(struc1['invals'])

#Get the parameters for the chains
nplanets=np.int64(struc1['nplanets'])
nwalkers=np.int64(struc1['nwalkers'])
nsteps=np.int64(struc1['nsteps'])
nthreads=np.int64(struc1['nthreads'])
sysname=struc1['sysname'] 

if args.gp:

    if not any('gppackflag' in s for s in index):
        struc1['gppackflag'], index, invals = 'george', np.append(index,'gppackflag'), np.append(invals, 'george')

    
    if struc1['gppackflag'] == 'celerite':
        try:
            import celerite as gppack
        except ImportError:
            print 'celerite does not appear to be installed correctly.'
            print 'you can install it with "pip install celerite"'
            print '(it also requires the Eigen package)'
            print 'exiting now'
            sys.exit()

        if struc1['gpmodtypep'] == 'Haywood14QP':
            from celeriteHaywood14QP2 import CustomTerm as celeritekernel


    else:
        try:
            import george as gppack
        except ImportError:
            print 'george does not appear to be installed correctly.'
            print 'you can install it with "pip install george"'
            print '(it also requires the Eigen package)'
            print 'exiting now'
            sys.exit()


    print 'burning electrum to perform Gaussian process regression'



if any('Mstar' in s for s in index) or any('Rstar' in s for s in index) or any('Teff' in s for s in index): 
    args.relpars=True
    from uncertainties import ufloat
    from uncertainties import unumpy as unp
else:
    args.relpars=False    


#get the general input and output filenames
chainfile=struc1['chainfile']
probfile=struc1['probfile']
accpfile=struc1['accpfile']
#read in the perturbations, if any
if any('perturbfile' in s for s in index): 
    perturbfile=struc1['perturbfile']
    perturbstruc=inreader(perturbfile)
    perturbindex=np.array(perturbstruc['index'])
    perturbinvals=np.array(perturbstruc['invals'])
#read in the priors, if any
if any('priorfile' in s for s in index): 
    priorfile=struc1['priorfile']
    priorstruc=inreader(priorfile)
    priorindex=np.array(priorstruc['index'])
    priorinvals=np.array(priorstruc['invals'])
else:
    priorstruc={'none':'none'}

#These will be needed for any system
if args.photometry or args.tomography or args.rvs:
    Per=np.array(invals[[i for i, s in enumerate(index) if 'Per' in s]], dtype=np.float)
    epoch=np.array(invals[[i for i, s in enumerate(index) if 'epoch' in s]], dtype=np.float)

#get the eccentricity if it exists, otherwise fix to zero and don't fit
if any('ecc' in s for s in index):
    ecc=np.array(invals[[i for i, s in enumerate(index) if 'ecc' in s]], dtype=np.float)
    omega=np.array(invals[[i for i, s in enumerate(index) if 'omega' in s]], dtype=np.float)
    fitecc=True
else:
    ecc=np.zeros(nplanets)
    omega=np.zeros(nplanets)+90.
    fitecc=False

omega*=np.pi/180.0 #degrees to radians

#check to see if eccentricity standard--default is sqrt(e) sin or cos omega
if any('ewflag' in s for s in index): 
    ewflag=struc1['ewflag']
else:
    ewflag='sesinw'
#implement the standards
if ewflag == 'sesinw':
    eccpar=np.sqrt(ecc)*np.sin(omega)
    omegapar=np.sqrt(ecc)*np.cos(omega)
    enames=['sesinw','secosw']
    for i in range(0,nplanets):
        struc1['sesinw'+str(i+1)], struc1['secosw'+str(i+1)]=eccpar[i],omegapar[i]
if ewflag == 'ecsinw':
    eccpar=ecc*np.sin(omega)
    omegapar=ecc*np.cos(omega)
    enames=['ecsinw','eccosw']
if ewflag == 'eomega':
    eccpar=ecc
    omegapar=omega
    enames=['ecc','omega']



#parameters needed for photometry
if args.photometry:
    photfile=np.array(invals[[i for i, s in enumerate(index) if 'photfile' in s]], dtype=str)
    pndatasets=len(photfile)
    if any('g1p' in s for s in index): g1p=np.array(invals[[i for i, s in enumerate(index) if 'g1p' in s]], dtype=np.float)
    if any('g2p' in s for s in index): g2p=np.array(invals[[i for i, s in enumerate(index) if 'g2p' in s]], dtype=np.float)
    if any('q1p' in s for s in index): q1p=np.array(invals[[i for i, s in enumerate(index) if 'q1p' in s]], dtype=np.float)
    if any('q2p' in s for s in index): q2p=np.array(invals[[i for i, s in enumerate(index) if 'q2p' in s]], dtype=np.float)
    if any('filternumber' in s for s in index): 
        filternumber=np.array(invals[[i for i, s in enumerate(index) if 'filternumber' in s]], dtype=np.int)
    else:
        filternumber=np.ones(pndatasets,dtype=np.int)
        
    pnfilters=np.max(filternumber)
    struc1['pnfilters']=pnfilters
    struc1['pndatasets']=pndatasets

    if any('photlcflag' in s for s in index): 
        photlcflag=struc1['photlcflag']
    else:
        photlcflag='q'
    if photlcflag == 'q':
        try:
            q1p, q2p
        except NameError:
            q1p=(g1p+g2p)**2
            q2p=g1p/(2.0*(g1p+g2p))
            for i in range(0,pnfilters):
                index=np.append(index, ['q1p'+str(i+1), 'q2p'+str(i+1)],axis=0)
                invals=np.append(invals, [q1p[i], q2p[i]],axis=0)
                struc1['q1p'+str(i+1)], struc1['q2p'+str(i+1)] = q1p[i], q2p[i]
                if (any('g1p'+str(i+1) in s for s in priorindex)) & (any('g2p'+str(i+1) in s for s in priorindex)):
                    priorindex=np.append(priorindex,['q1p'+str(i+1),'q2p'+str(i+1)],axis=0)
                    sq1p=np.sqrt(2.0*(priorstruc['g1p'+str(i+1)]**2+priorstruc['g2p'+str(i+1)]**2))
                    sq2p=g1p[i]/(2.0*(g1p[i]+g2p[i]))*np.sqrt(priorstruc['g1p'+str(i+1)]**2/g1p[i]**2+(priorstruc['g1p'+str(i+1)]**2+priorstruc['g2p'+str(i+1)]**2)/(g1p[i]+g2p[i])**2)
                    priorinvals=np.append(priorinvals,[sq1p,sq2p],axis=0)
                    priorstruc['q1p'+str(i+1)], priorstruc['q2p'+str(i+1)] = sq1p, sq2p
                    priorstruc['index']=np.append(priorstruc['index'], ['q1p'+str(i+1),'q2p'+str(i+1)],axis=0)
                    priorstruc['invals']=np.append(priorstruc['invals'], [sq1p,sq2p],axis=0)
                
    #now read in the data
    for i in range(0, pndatasets):
        ptime1,pflux1,perror1,pexptime1=readcol(photfile[i],twod=False)
        goods=np.where((ptime1 != -1.) & (pflux1 != -1.))
        ptime1,pflux1,perror1,pexptime1=ptime1[goods],pflux1[goods],perror1[goods],pexptime1[goods]
    #check to see if using Kepler cadence and, if so, correct to exposure times
        if any('cadenceflag'+str(i+1) in s for s in index):
            pexptime1=np.array(pexptime1, dtype=float)
            if struc1['cadenceflag'+str(i+1)] == 'kepler':
                longcad=np.where(pexptime1 == 1)
                shortcad=np.where(pexptime1 == 0)
                pexptime1[longcad], pexptime1[shortcad] = 30., 1.
                pexptime1=pexptime1/(60.*24.)
            if struc1['cadenceflag'+str(i+1)] == 'corot':
                longcad=np.where(pexptime1 == 1)
                shortcad=np.where(pexptime1 == 0)
                pexptime1[longcad], pexptime1[shortcad] = 512., 32.
                pexptime1=pexptime1/(60.*60.*24.)
        if any('expunit'+str(i+1) in s for s in index):
            if (struc1['expunit'+str(i+1)] == 's') or (struc1['expunit'+str(i+1)] == 'seconds'): pexptime1=pexptime1/(60.*60.*24.)
            if (struc1['expunit'+str(i+1)] == 'm') or (struc1['expunit'+str(i+1)] == 'minutes'): pexptime1=pexptime1/(60.*24.)
            if (struc1['expunit'+str(i+1)] == 'h') or (struc1['expunit'+str(i+1)] == 'hours'): pexptime1=pexptime1/(24.)
            if (struc1['expunit'+str(i+1)] == 'd') or (struc1['expunit'+str(i+1)] == 'days'): pexptime1=pexptime1/(1.)
        else:
            struc1['expunit'+str(i+1)] = 'days'

        if i == 0:
            ptime, pflux, perror, pexptime = ptime1,pflux1,perror1,pexptime1
            pfilter=np.ones(len(ptime))
            pdataset=np.ones(len(ptime))
            if args.gp:
                if any('gppuse'+str(i+1) in s for s in index):
                    gppuse=np.zeros(len(ptime))+struc1['gppuse'+str(i+1)]
                else:
                    gppuse=np.ones(len(ptime))
        else:
            ptime, pflux, perror, pexptime = np.append(ptime,ptime1), np.append(pflux,pflux1), np.append(perror,perror1), np.append(pexptime,pexptime1)
            pfilter=np.append(pfilter, np.zeros(len(ptime1))+filternumber[i])
            pdataset=np.append(pdataset, np.zeros(len(ptime1))+i+1)
            if args.gp:
                if any('gppuse'+str(i+1) in s for s in index):
                    gppuse=np.append(gppuse,np.zeros(len(ptime1))+np.int(struc1['gppuse'+str(i+1)]))
                else:
                    gppuse=np.append(gppuse,np.ones(len(ptime1)))

    #flux ratio if doing EB
    if args.binary:
        fluxrat=np.array(invals[[i for i, s in enumerate(index) if 'fluxrat' in s]], dtype=np.float)
            
if args.photometry or args.tomography:
    if any('rhostar' in s for s in index): rhostar=np.float(struc1['rhostar'])
    if any('aors' in s for s in index): aors=np.array(invals[[i for i, s in enumerate(index) if 'aors' in s]], dtype=np.float) #NOTE: if args.binary, aors is actually (a/(R1+R2)), not (a/R*)!
    if any ('cosi' in s for s in index): cosi=np.array(invals[[i for i, s in enumerate(index) if 'cosi' in s]], dtype=np.float)
    rprs=np.array(invals[[i for i, s in enumerate(index) if 'rprs' in s]], dtype=np.float)
    bpar=np.array(invals[[i for i, s in enumerate(index) if 'bpar' in s]], dtype=np.float)

    if any('rhobaflag' in s for s in index):
        rhobaflag=struc1['rhobaflag']
    else:
        rhobaflag='rhostarb'

if args.tomography or args.line:
    if any('g1t' in s for s in index): g1t=np.float(struc1['g1t'])
    if any('g2t' in s for s in index): g2t=np.float(struc1['g2t'])
    if any('q1t' in s for s in index): q1t=np.float(struc1['q1t'])
    if any('q2t' in s for s in index): q2t=np.float(struc1['q2t'])
    if any('tomlcflag' in s for s in index): 
        tomlcflag=struc1['tomlcflag']
    else:
        tomlcflag='q'
    if tomlcflag == 'q':
        try:
            q1t, q2t
        except NameError:
            q1t=(g1t+g2t)**2
            q2t=g1t/(2.0*(g1t+g2t))
            index=np.append(index, ['q1t', 'q2t'],axis=0)
            invals=np.append(invals, [q1t, q2t],axis=0)
            struc1['q1t'], struc1['q2t'] = q1t, q2t
        
        if (any('g1t' in s for s in priorindex)) & (any('g2t' in s for s in priorindex)):
            priorindex=np.append(priorindex,['q1t','q2t'],axis=0)
            sq1t=np.sqrt(2.0*(priorstruc['g1t']**2+priorstruc['g2t']**2))
            sq2t=g1t/(2.0*(g1t+g2t))*np.sqrt(priorstruc['g1t']**2/g1t**2+(priorstruc['g1t']**2+priorstruc['g2t']**2)/(g1t+g2t)**2)
            priorinvals=np.append(priorinvals,[sq1t,sq2t],axis=0)
            priorstruc['q1t'], priorstruc['q2t'] = sq1t, sq2t

if args.tomography:
    tomfile=np.array(invals[[i for i, s in enumerate(index) if 'tomfile' in s]], dtype=np.str)
    llambda=np.array(invals[[i for i, s in enumerate(index) if 'lambda' in s]], dtype=np.float)

    if any('tomdrift' in s for s in index):
        tomdriftc=np.array(invals[[i for i, s in enumerate(index) if 'tomdriftc' in s]], dtype=np.float) #constant term
        tomdriftl=np.array(invals[[i for i, s in enumerate(index) if 'tomdriftl' in s]], dtype=np.float) #linear term

    #now read in the data
    ntomsets=len(tomfile)
    tomdict={}
    tomdict['ntomsets']=ntomsets
    tomdict['nexptot'], tomdict['nvabsfinemax'],tomdict['whichvabsfinemax'] =0, 0, 0
    for i in range (0,ntomsets):
        tomnlength=len(tomfile[i])
        lastthree=tomfile[i][tomnlength-3:tomnlength]
        if lastthree == 'sav':
            import idlsave
            try:
                datain=idlsave.read(struc1['tomfile'+str(i+1)])
            except IOError:
                print 'Your tomographic input file either does not exist or is in the wrong format.'
                print 'Please supply a file in IDL save format.'
                print 'Exiting now.'
                sys.exit()

    
            profarr = datain.profarr
            tomdict['avgprof'+str(i+1)] = datain.avgprof
            tomdict['avgproferr'+str(i+1)] = datain.avgproferr
            profarrerr = datain.profarrerr
            profarr = np.transpose(profarr) #idlsave reads in the arrays with the axes flipped
            tomdict['profarr'+str(i+1)]=profarr*(-1.0)
            tomdict['profarrerr'+str(i+1)] = np.transpose(profarrerr)
            tomdict['ttime'+str(i+1)]=datain.bjds
            tomdict['tnexp'+str(i+1)] = tomdict['ttime'+str(i+1)].size
            tomdict['texptime'+str(i+1)]=datain.exptimes
            tomdict['vabsfine'+str(i+1)] = datain.vabsfine
            tomdict['nexptot']+=tomdict['tnexp'+str(i+1)]
            if len(tomdict['vabsfine'+str(i+1)]) > tomdict['nvabsfinemax']: tomdict['nvabsfinemax'], tomdict['whichvabsfinemax'] = len(tomdict['vabsfine'+str(i+1)]), i+1

        if lastthree == 'pkl':
            import pickle

            try:
                datain=pickle.load(open(struc1['tomfile'+str(i+1)],"rb"))
            except IOError:
                print 'Your tomographic input file either does not exist or is in the wrong format.'
                print 'Please supply a file in pickle format.'
                print 'Exiting now.'
                sys.exit()

            if not 'george' in struc1['tomfile'+str(i+1)]:
                tomdict['profarr'+str(i+1)], tomdict['profarrerr'+str(i+1)], tomdict['avgprof'+str(i+1)], tomdict['avgproferr'+str(i+1)], tomdict['ttime'+str(i+1)], tomdict['texptime'+str(i+1)], tomdict['vabsfine'+str(i+1)] = np.array(datain['profarr'],dtype=float), np.array(datain['profarrerr'],dtype=float), np.array(datain['avgprof'],dtype=float), np.array(datain['avgproferr'],dtype=float), np.array(datain['ttime'],dtype=float), np.array(datain['texptime'],dtype=float), np.array(datain['vabsfine'],dtype=float)

                tomdict['profarr'+str(i+1)]*=(-1.)
            else:
                tomdict['profarr'+str(i+1)], tomdict['ttime'+str(i+1)], tomdict['vabsfine'+str(i+1)], tomdict['texptime'+str(i+1)], tomdict['avgprof'+str(i+1)] = np.array(datain[0], dtype=float), np.array(datain[2], dtype=float)-2400000., np.array(datain[3], dtype=float), np.array(datain[4], dtype=float), np.array(datain[5], dtype=float)
                if struc1['obs'+str(i+1)] == 'harpsn': Resolve=120000.0
                if struc1['obs'+str(i+1)] == 'tres' : Resolve=44000.0
                tomdict['profarr'+str(i+1)]/=1.1 
                tomdict['profarr'+str(i+1)]*=(-1.0)
                 #downsample!
                vabsfinetemp=np.arange(np.min(tomdict['vabsfine'+str(i+1)]), np.max(tomdict['vabsfine'+str(i+1)]),(2.9979e5/Resolve)/2.)
                ntemp=len(vabsfinetemp)
                profarrtemp=np.zeros((len(tomdict['texptime'+str(i+1)]),ntemp))
                for iter in range(0,len(tomdict['texptime'+str(i+1)])):
                    profarrtemp[iter,:]=np.interp(vabsfinetemp,tomdict['vabsfine'+str(i+1)],tomdict['profarr'+str(i+1)][iter,:])
                avgproftemp=np.interp(vabsfinetemp,tomdict['vabsfine'+str(i+1)],tomdict['avgprof'+str(i+1)])
                tomdict['profarr'+str(i+1)]=profarrtemp
                tomdict['vabsfine'+str(i+1)]=vabsfinetemp
                tomdict['avgprof'+str(i+1)]=avgproftemp
                outs=np.where(np.abs(tomdict['vabsfine'+str(i+1)] ) > np.float(struc1['vsini'])*1.1)
                tomdict['profarrerr'+str(i+1)]=tomdict['profarr'+str(i+1)]*0.0+np.std(tomdict['profarr'+str(i+1)][:,outs[0]])
                tomdict['avgproferr'+str(i+1)]=tomdict['avgprof'+str(i+1)]*0.0+np.std(tomdict['avgprof'+str(i+1)][outs[0]])
               
                
                
            

        
        #cut off the few pixels on the edges, which are often bad
        tomdict['nvabsfine'+str(i+1)]=len(tomdict['vabsfine'+str(i+1)])
        tomdict['profarr'+str(i+1)]=tomdict['profarr'+str(i+1)][:,3:tomdict['nvabsfine'+str(i+1)]-2]
        tomdict['profarrerr'+str(i+1)]=tomdict['profarrerr'+str(i+1)][:,3:tomdict['nvabsfine'+str(i+1)]-2]
        tomdict['vabsfine'+str(i+1)]=tomdict['vabsfine'+str(i+1)][3:tomdict['nvabsfine'+str(i+1)]-2]
        tomdict['avgprof'+str(i+1)]=tomdict['avgprof'+str(i+1)][3:tomdict['nvabsfine'+str(i+1)]-2]
        tomdict['avgproferr'+str(i+1)]=tomdict['avgproferr'+str(i+1)][3:tomdict['nvabsfine'+str(i+1)]-2]
        nvabsfine1=tomdict['nvabsfine'+str(i+1)]
        tomdict['nvabsfine'+str(i+1)]=len(tomdict['vabsfine'+str(i+1)])
        tomdict['whichplanet'+str(i+1)] = struc1['whichtomplanet'+str(i+1)]

        

    if any('tomflat' in s for s in index): 
        if struc1['tomflat'] == 'True':
            tomflat=True
        else:
            tomflat=False
    else:
        tomflat=False

    if any('fitflat' in s for s in index): 
        if struc1['fitflat'] == 'True':
            fitflat=True
        else:
            fitflat=False




    else:
        struc1['fitflat'] = 'False'
        index=np.append(index,'fitflat')
        invals=np.append(invals,'False')
        fitflat=False

    if not tomflat and fitflat:
        for j in range (0,ntomsets):
            for i in range (0,tnexp) : 
                tomdict['profarr[i, : ]'+str(j+1)]-=tomdict['avgprof'+str(j+1)]
                tomdict['profarrerr'+str(j+1)][i,:]=np.sqrt(tomdict['profarrerr'+str(j+1)][i,:]**2+tomdict['avgproferr'+str(j+1)]**2)

    if any('tomfftfile' in s for s in index):
        if ntomsets > 1:
            print 'FFT is not yet implemented for multi-tomographic dataset fits!!!'
            print 'exiting now.'
            sys.exit()
        fftin = idlsave.read(struc1['tomfftfile'])
        ffterrin = idlsave.read(struc1['tomffterrfile'])
        profarrerr = np.transpose(ffterrin.filterr)
        profarrerr=profarrerr[:,3:nvabsfine1-2]
        mask = np.transpose(fftin.mask)
        mask=mask[:,3:nvabsfine1-2]
        profarr = horus.fourierfilt(profarr, mask)
        dofft = True
    else:
        dofft = False

if args.line:

    if any('linecenter' in s for s in index):
        linecenter=struc1['linecenter']
    else:
        linecenter=0.0
 
        
        
    if struc1['linefile'] == 'tomfile':
        if args.tomography:
            lineprof = avgprof
            lineerr = avgproferr
            linevel = vabsfine
        else:
            import idlsave
            try:
                datain=idlsave.read(struc1['tomfile'])
            except IOError:
                print 'Your tomographic input file either does not exist or is in the wrong format.'
                print 'Please supply a file in IDL save format.'
                print 'Exiting now.'
                sys.exit()

            lineprof  = datain.avgprof
            lineerr = datain.avgproferr
            linevel = datain.vabsfine

    else:
        namelength=len(struc1['linefile'])
        if struc1['linefile'][namelength-4:namelength] == '.sav':
            if not args.tomography:
                import idlsave
                datain=idlsave.read(struc1['linefile'])
                lineprof  = datain.avgprof
                lineerr = datain.avgproferr
                linevel = datain.vabsfine


if args.rvs:
    semiamp=np.array(invals[[i for i, s in enumerate(index) if 'semiamp' in s]], dtype=np.float)
    #read in the data
    if not args.binary:
        rtime,rv,rverror,rdataset=readcol(struc1['rvfile'],twod=False)
    else:
        rtime, rv1, rverror1, rv2, rverror2, rdataset=readcol(struc1['rvfile'],twod=False)
        rv=np.append(rv1,rv2)
        rverror=np.append(rverror1, rverror2)
    rndatasets=np.max(rdataset)
    if any('gamma' in s for s in index): 
        gamma=np.array(invals[[i for i, s in enumerate(index) if 'gamma' in s]], dtype=np.float)
    else:
        gamma=np.zeros(rndatasets)
        for i in range (0,rndatasets):
            struc1['gamma'+str(i+1)]=gamma[i]
            invals=np.append(invals,0)
            index=np.append(index,'gamma'+str(i+1))

    if any('fixgam' in s for s in index): 
        fixgamma=struc1['fixgam']
    else:
        fixgamma='False'

    if any('rvtrend' in s for s in index):
        fittrend=True
        rvtrend=np.float(struc1['rvtrend'])
    else:
        fittrend=False

    #check for jitter and use if being fit
    if any('jitter' in s for s in index):
        jitter=np.array(invals[[i for i, s in enumerate(index) if 'jitter' in s]], dtype=np.float)
        args.fitjitter=True
    else:
        args.fitjitter=False
    
#copy in the TTV model parameters
if args.ttvs:
    

    
    dottvs=np.zeros(nplanets, dtype=bool)
    for i in range (0,nplanets):
        if any('dottv'+str(i+1) in s for s in index): 
            if struc1['dottv'+str(i+1)] == 'True':
                dottvs[i]=True
 

    ttvpars=np.array(invals[[i for i, s in enumerate(index) if (('ttv' in s) & ('par' in s))]], dtype=np.float)
    ttvparnames=np.array(index[[i for i, s in enumerate(index) if (('ttv' in s) & ('par' in s))]], dtype=np.str)
    nttvpars=len(ttvpars)
    modtype=np.array(invals[[i for i, s in enumerate(index) if ('ttvmodtype' in s)]], dtype=np.str)

if args.dilution:
    if any('dilution' in s for s in index): 
        dilution=np.array(invals[[i for i, s in enumerate(index) if 'dilution' in s]], dtype=np.float)
        dilutionnames=np.array(index[[i for i, s in enumerate(index) if 'dilution' in s]], dtype=np.str)
        ndilute=len(dilution)


#copy in the Gaussian process parameters
if args.gp:
    gppars=np.array(invals[[i for i, s in enumerate(index) if (('gp' in s) & ('par' in s))]], dtype=np.float)
    gpparnames=np.array(index[[i for i, s in enumerate(index) if (('gp' in s) & ('par' in s))]], dtype=np.str)
    ngppars=len(gppars)
    gpmodval=np.array(invals[[i for i, s in enumerate(index) if ('gpmodtype' in s)]], dtype=np.str)
    gpmodname=np.array(index[[i for i, s in enumerate(index) if ('gpmodtype' in s)]], dtype=np.str)
    gpmodtype={}
    for i in range (0,len(gpmodval)):
        gpmodtype[gpmodname[i]]=gpmodval


    

#set up the initial position for emcee and get all of the data that are needed
data={}

#photometric parameters
if args.photometry:
    inpos, inposindex, perturbs = np.array(epoch), np.array(index[[i for i, s in enumerate(index) if 'epoch' in s]]), np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'epoch' in s]], dtype=np.float)
    inpos, inposindex, perturbs = np.append(inpos, Per), np.append(inposindex, index[[i for i, s in enumerate(index) if 'Per' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'Per' in s]], dtype=np.float))
    inpos, inposindex, perturbs = np.append(inpos, rprs), np.append(inposindex, index[[i for i, s in enumerate(index) if 'rprs' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'rprs' in s]], dtype=np.float))


    if rhobaflag != 'aorscosi':
        inpos, inposindex, perturbs = np.append(inpos, bpar), np.append(inposindex, index[[i for i, s in enumerate(index) if 'bpar' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'bpar' in s]], dtype=np.float))
    else:
        inpos, inposindex, perturbs = np.append(inpos, cosi), np.append(inposindex, index[[i for i, s in enumerate(index) if 'cosi' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'cosi' in s]], dtype=np.float))

    if rhobaflag == 'rhostarb':
        inpos, inposindex, perturbs = np.append(inpos, rhostar), np.append(inposindex, 'rhostar'), np.append(perturbs, np.float(perturbstruc['rhostar']))
    if rhobaflag == 'aorsb' or rhobaflag == 'aorscosi':
        inpos, inposindex, perturbs = np.append(inpos, aors), np.append(inposindex, index[[i for i, s in enumerate(index) if 'aors' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'aors' in s]], dtype=np.float))

    if photlcflag == 'q':
        for i in range (0, pnfilters): inpos, inposindex, perturbs = np.append(inpos, [q1p[i], q2p[i]], axis=0), np.append(inposindex, ['q1p'+str(i+1), 'q2p'+str(i+1)], axis=0), np.append(perturbs, [perturbstruc['q1p'+str(i+1)], perturbstruc['q2p'+str(i+1)]], axis=0)
    if photlcflag == 'g':
        for i in range (0, pnfilters): inpos, inposindex, perturbs = np.append(inpos, [g1p[i], g2p[i]], axis=0), np.append(inposindex, ['g1p'+str(i+1), 'g2p'+str(i+1)], axis=0), np.append(perturbs, [perturbstruc['g1p'+str(i+1)], perturbstruc['g2p'+str(i+1)]], axis=0)

    if args.binary: inpos, inposindex, perturbs = np.append(inpos,fluxrat), np.append(inposindex, index[[i for i, s in enumerate(index) if 'fluxrat' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'fluxrat' in s]], dtype=np.float))

    data['ptime'], data['pflux'], data['perror'], data['pexptime'], data['pfilter'], data['pdataset']  =ptime,pflux,perror,pexptime, pfilter, pdataset

#tomographic parameters will go here

if args.line: 
    if any('linecenter' in s for s in index):
        if not args.photometry: 
            inpos, inposindex, perturbs = np.array(linecenter, dtype=np.float), np.array('linecenter'),  np.array(perturbstruc['linecenter'],dtype=np.float)
        else:
            inpos, inposindex, perturbs = np.append(inpos, linecenter), np.append(inposindex, 'linecenter'), np.append(perturbs, np.float(perturbstruc['linecenter']))
    data['lineprof'], data['lineerr'], data['linevel'] = lineprof, lineerr, linevel


if args.tomography or args.line:
    if not args.photometry and args.tomography:
        inpos, inposindex, perturbs = np.array(epoch), np.array(index[[i for i, s in enumerate(index) if 'epoch' in s]]), np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'epoch' in s]], dtype=np.float)
        inpos, inposindex, perturbs = np.append(inpos, Per), np.append(inposindex, index[[i for i, s in enumerate(index) if 'Per' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'Per' in s]], dtype=np.float))
        inpos, inposindex, perturbs = np.append(inpos, rprs), np.append(inposindex, index[[i for i, s in enumerate(index) if 'rprs' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'rprs' in s]], dtype=np.float))
        inpos, inposindex, perturbs = np.append(inpos, bpar), np.append(inposindex, index[[i for i, s in enumerate(index) if 'bpar' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'bpar' in s]], dtype=np.float))

        if rhobaflag == 'rhostarb':
            inpos, inposindex, perturbs = np.append(inpos, rhostar), np.append(inposindex, 'rhostar'), np.append(perturbs, np.float(perturbstruc['rhostar']))
        if rhobaflag == 'aorsb':
            inpos, inposindex, perturbs = np.append(inpos, aors), np.append(inposindex, index[[i for i, s in enumerate(index) if 'aors' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'aors' in s]], dtype=np.float))

    inpos, inposindex, perturbs = np.append(inpos, np.float(struc1['vsini'])), np.append(inposindex, 'vsini'), np.append(perturbs, np.float(perturbstruc['vsini']))
    if args.tomography: inpos, inposindex, perturbs = np.append(inpos, llambda), np.append(inposindex, index[[i for i, s in enumerate(index) if 'lambda' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'lambda' in s]], dtype=np.float))
    if tomlcflag == 'q':
        inpos, inposindex, perturbs = np.append(inpos, [q1t, q2t], axis=0), np.append(inposindex, ['q1t', 'q2t'], axis=0), np.append(perturbs, [perturbstruc['q1t'], perturbstruc['q2t']], axis=0)
    if tomlcflag == 'g':
        inpos, inposindex, perturbs = np.append(inpos, [g1t, g2t], axis=0), np.append(inposindex, ['g1t', 'g2t'], axis=0), np.append(perturbs, [perturbstruc['g1t'], perturbstruc['g2t']], axis=0)
    
    if any('fitintwidth' in s for s in index):
        if struc1['fitintwidth'] == 'True':
            inpos, inposindex, perturbs = np.append(inpos, np.float(struc1['intwidth'])), np.append(inposindex, 'intwidth'), np.append(perturbs, np.float(perturbstruc['intwidth']))

    if args.skyline:
        inpos, inposindex, perturbs = np.append(inpos, [np.float(struc1['skycen']), np.float(struc1['skydepth'])], axis=0), np.append(inposindex, ['skycen', 'skydepth'], axis=0), np.append(perturbs, [perturbstruc['skycen'], perturbstruc['skydepth']], axis=0)

    if any('tomdrift' in s for s in index):
        for i in range (0,len(tomdriftc)):
            inpos, inposindex, perturbs = np.append(inpos, [tomdriftc[i], tomdriftl[i]], axis=0), np.append(inposindex, ['tomdriftc'+str(i+1),'tomdriftl'+str(i+1)], axis=0), np.append(perturbs, [perturbstruc['tomdriftc'+str(i+1)],perturbstruc['tomdriftl'+str(i+1)]], axis=0)

    if args.tomography: 
        data['tomdict']=tomdict
        if dofft:
            data['mask']=mask

    
    
    
if args.rvs:
    if not args.tomography and not args.photometry:
        inpos, inposindex, perturbs = np.array(epoch), np.array(index[[i for i, s in enumerate(index) if 'epoch' in s]]), np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'epoch' in s]], dtype=np.float)
        inpos, inposindex, perturbs = np.append(inpos, Per), np.append(inposindex, index[[i for i, s in enumerate(index) if 'Per' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'Per' in s]], dtype=np.float))
    
    inpos, inposindex, perturbs = np.append(inpos, semiamp), np.append(inposindex, index[[i for i, s in enumerate(index) if 'semiamp' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'semiamp' in s]], dtype=np.float))
    if fixgamma != 'True': inpos, inposindex, perturbs = np.append(inpos, gamma), np.append(inposindex, index[[i for i, s in enumerate(index) if 'gamma' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'gamma' in s]], dtype=np.float))
    if fittrend: inpos, inposindex, perturbs = np.append(inpos, rvtrend), np.append(inposindex, 'rvtrend'), np.append(perturbs, perturbinvals[[i for i, s in enumerate(perturbindex) if 'rvtrend' in s]])
    if args.fitjitter: inpos, inposindex, perturbs = np.append(inpos, jitter), np.append(inposindex, index[[i for i, s in enumerate(index) if 'jitter' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'jitter' in s]], dtype=np.float))

    data['rtime'], data['rv'], data['rverror'], data['rdataset'] = rtime,rv,rverror,rdataset




#add eccentricity if it's being fit
if fitecc == True:
    for i in range (0, nplanets):
        inpos, inposindex, perturbs = np.append(inpos, [eccpar[i], omegapar[i]]), np.append(inposindex, [enames[0]+str(i+1), enames[1]+str(i+1)], axis=0), np.append(perturbs, [perturbstruc['ecc'+str(i+1)], perturbstruc['omega'+str(i+1)]])

#add ttvs if being fit
if args.ttvs:
    for i in range (0, nttvpars):
        inpos, inposindex, perturbs = np.append(inpos, ttvpars[i]), np.append(inposindex, ttvparnames[i]), np.append(perturbs, perturbstruc[ttvparnames[i]])


    data['ttvmodtype'], data['dottvs'] = modtype, dottvs

#add dilution if being included
if args.dilution:
    for i in range (0, ndilute):
        inpos, inposindex, perturbs = np.append(inpos, dilution[i]), np.append(inposindex, dilutionnames[i]), np.append(perturbs, perturbstruc[dilutionnames[i]])

if any ('perturbfac' in s for s in perturbindex):
    perturbs*=perturbstruc['perturbfac']

#add Gaussian process parameters if being fit
if args.gp:
    for i in range (0, ngppars):
        inpos, inposindex, perturbs = np.append(inpos, gppars[i]), np.append(inposindex, gpparnames[i]), np.append(perturbs, perturbstruc[gpparnames[i]])


    data['gpmodtype']= gpmodtype
    if args.photometry:
        data['gppuse'] = gppuse


#get the time standard information set up
if any('timestandard' in s for s in index):
    timestandard=struc1['timestandard']
else:
    timestandard='BJD'


if any('timeoffset' in s for s in index):
    timeoffset=np.float(struc1['timeoffset'])
else:
    if args.rvs:
        time0=data['rtime'][0]
    elif args.photometry:
        time0=data['ptime'][0]
    elif args.tomography:
        time0=data['ttime'][0]
    if time0 > 2400000:
        timeoffset=0.
    elif time0 > 50000:
        timeoffset=2400000.
    elif time0 > 4000:
        timeoffset=2450000.
    else:
        timeoffset=2454833.

exstruc={'timestandard':timestandard,'timeoffset':timeoffset}

ndim=len(inpos)
inpos=np.array(inpos,dtype=np.float)
inpos1=inpos*1.

if not any('photname' in s for s in struc1) and args.photometry:
    for i in range (0, pndatasets):
        struc1['photname'+str(i+1)]='blank'

try:
    chainin=np.load(chainfile)
except IOError:
    print 'There is no emcee chain file with the given name!'
    print 'You need to run MISTTBORN on this file first!'
    print 'Exiting now.'
    sys.exit()

probfile=struc1['probfile']
probin=np.load(probfile)

if any('nburnin' in s for s in index): 
    nburnin=int(struc1['nburnin'])
else:
    nburnin=nin/5



samples=chainin[:,nburnin:,:].reshape((-1,ndim))
probin=probin[:,nburnin:].reshape((-1))

if not args.tomography and args.photometry:
    if rhobaflag == 'aorsb' or rhobaflag == 'rhostarb':
        whereb=[i for i, s in enumerate(inposindex) if 'bpar' in s]
        for i in range (0,nplanets):
            samples[:,whereb[i]]=np.abs(samples[:,whereb[i]])


def nsf(num, n=1):
    #from StackOverflow: https://stackoverflow.com/questions/9415939/how-can-i-print-many-significant-figures-in-python
    """n-Significant Figures"""
    while n-1 < 0: n+=1
    numstr = ("{0:.%ie}" % (n-1)).format(num)
    return float(numstr)

def symdex(par,args,exstruc):
    if 'epoch' in par: 
        if exstruc['timeoffset'] == 0.:
            return '$T_0$ ('+exstruc['timestandard']+')'
        else:
            return '$T_0$('+exstruc['timestandard']+'-'+str(np.int(np.round(exstruc['timeoffset'])))+')'
    if 'Per' in par: return '$P$ (days)'
    if 'rprs' in par and args.binary: return '$R_2/R_1$'
    if 'rprs' in par: return '$R_P/R_{\star}$'
    if 'bpar' in par: return '$b$'
    if 'aors' in par and args.binary: return '$a/(R_1+R_2)$'
    if 'aors' in par: return '$a/R_{\star}$'
    if 'q1p' in par: return '$q_{1,'+par[len(par)-1]+'}$'
    if 'q2p' in par: return '$q_{2,'+par[len(par)-1]+'}$'
    if 'g1p' in par: return '$g_{1,'+par[len(par)-1]+'}$'
    if 'g2p' in par: return '$g_{2,'+par[len(par)-1]+'}$'
    if 'q1t' in par: return '$q_{1,\mathrm{tom}}$'
    if 'q2t' in par: return '$q_{2,\mathrm{tom}}$'
    if 'g1t' in par: return '$g_{1,\mathrm{tom}}$'
    if 'g2t' in par: return '$g_{2,\mathrm{tom}}$'
    if 'semiamp' in par and args.binary:
        if 'b' in par: 
            return '$K_2$ (km s$^{-1}$)'
        else:
            return '$K_1$ (km s$^{-1}$)'
    if 'semiamp' in par and args.ms: return '$K$ (m s$^{-1}$)'
    if 'semiamp' in par: return '$K$ (km s$^{-1}$)'
    if 'gamma' in par and args.ms: return '$\gamma_{'+par[len(par)-1]+'}$ (m s$^{-1}$)'
    if 'gamma' in par: return '$\gamma_{'+par[len(par)-1]+'}$ (km s$^{-1}$)'
    if 'jitter' in par and args.ms: return 'jitter$_{'+par[len(par)-1]+'}$ (m s$^{-1}$)'
    if 'jitter' in par: return 'jitter$_{'+par[len(par)-1]+'}$ (km s$^{-1}$)'
    if 'ecsinw' in par: return '$e\sin\omega$'
    if 'eccosw' in par: return '$e\cos\omega$'
    if 'sesinw' in par: return '$\sqrt{e}\sin\omega$'
    if 'secosw' in par: return '$\sqrt{e}\cos\omega$'
    if 'ecc' in par: return '$e$'
    if 'omega' in par: return '$\omega$ ($^{\circ}$)'
    if 'lambda' in par: return '$\lambda$ ($^{\circ}$)'
    if 'rhostar' in par: return '$\rho_{\star}$ ($\rho_{\odot}$)'
    if 'rvtrend' in par and args.ms: return '$\dot{\gamma}$ (m s$^{-1}$ day$^{-1}$)'
    if 'rvtrend' in par: return '$\dot{\gamma}$ (km s$^{-1}$ day$^{-1}$)'
    if 'fluxrat' in par: return '$f_2/f_1$' #pretty sure this is right...
    if 'cosi' in par: return '$\cos i$'
    if 'dilution' in par: return '$\Delta$ mag (mag)'
    if 'M1sin3i' in par: return '$M_1\sin^3 i$ ($M_{\odot}$)'
    if 'M2sin3i' in par: return '$M_2\sin^3 i$ ($M_{\odot}$)'
    if 'M1' in par: return '$M_1$ ($M_{\odot}$)'
    if 'M2' in par: return '$M_2$ ($M_{\odot}$)'
    if 'tomdrift' in par: return 'Fix this later!'
    if 'gpppartau' in par: return '$\\tau_{\mathrm{GP}}$ (day)'
    if 'gppparP' in par: return '$P_{\mathrm{GP}}$ (day)'
    if 'gppparGamma' in par: return '$\Gamma_{\mathrm{GP}}$'
    if 'gppparamp' in par: return '$A_{\mathrm{GP}}$'
    if 'vsini' in par: return '$v\sin i_{\star}$ (km s$^{-1}$)'
    if 'intwidth' in par: return '$v_{\mathrm{int}}$ (km s$^{-1}$)'
    if 'instellation' in par: return '$S$ ($S_{\oplus}$)'

def parstring(temp):
    if temp[1] == 0.: return str(temp[0])+' (fixed)'
    numnum1=np.array([int(np.abs(np.floor(np.log10(temp[1]))-1)),int(np.abs(np.floor(np.log10(temp[2]))-1))])
    numnum=np.max(numnum1)
    valstr, errpstr, errmstr = str(round(temp[0],numnum)), str(nsf(temp[1],2)), str(nsf(temp[2],2))
    if 'e' in valstr:
        valstr=valstr.replace('e-0','\\times10^{-')
        valstr=valstr.replace('e+0','\\times10^{+')
        valstr=valstr.replace('e-','\\times10^{-')
        valstr=valstr.replace('e+','\\times10^{+')
        valstr+='}'
    if 'e' in errpstr:
        errpstr=errpstr.replace('e-0','\\times10^{-')
        errpstr=errpstr.replace('e+0','\\times10^{+')
        errpstr=errpstr.replace('e-','\\times10^{-')
        errpstr=errpstr.replace('e+','\\times10^{+')
        errpstr+='}'
    if 'e' in errmstr:
        errmstr=errmstr.replace('e-0','\\times10^{-')
        errmstr=errmstr.replace('e+0','\\times10^{+')
        errmstr=errmstr.replace('e-','\\times10^{-')
        errmstr=errmstr.replace('e+','\\times10^{+')
        errmstr+='}'    
    if nsf(temp[1],2) == nsf(temp[2],2):
        return valstr+' \pm '+ errpstr
    else:
        if numnum1[0] == numnum1[1]:
            return valstr+'^{+'+errpstr+'}_{-'+errmstr+'}'
        else:
            diff=numnum1[1]-numnum1[0]
            if diff < 0:
                errmstr=str(nsf(temp[2],2+diff))
                if 'e' in errmstr:
                    errmstr=errmstr.replace('e-0','\\times10^{-')
                    errmstr=errmstr.replace('e+0','\\times10^{+')
                    errmstr=errmstr.replace('e-','\\times10^{-')
                    errmstr=errmstr.replace('e+','\\times10^{+')
                    errmstr+='}'    
                return valstr+'^{+'+errpstr+'}_{-'+errmstr+'}'
            else:
                errpstr=str(nsf(temp[1],2+diff))
                if 'e' in errpstr:
                    errpstr=errpstr.replace('e-0','\\times10^{-')
                    errpstr=errpstr.replace('e+0','\\times10^{+')
                    errpstr=errpstr.replace('e-','\\times10^{-')
                    errpstr=errpstr.replace('e+','\\times10^{+')
                    errpstr+='}'
                return valstr+'^{+'+errpstr+'}_{-'+errmstr+'}'


def convfactor(whichone,args): #return various conversion factors
    if whichone == 'rprs-to-rp' and args.earth: return 109.
    if whichone == 'rprs-to-rp': return 1./0.1028
    if whichone == 'Mp-to-Massrat' and args.earth: return 1./333000.
    if whichone == 'Mp-to-Massrat': return 1./1047.5656

if args.bold: 
    schar1='$\mathbf{'
    schar2='}$'
else:
    schar1='$'
    schar2='$'

if args.bestprob:
    best=np.where(probin == np.max(probin))
    best=best[0][0]
    

for k in range (0,nplanets):
    f=open(sysname+'_table'+str(k+1)+'.tex','w')
    f.write('\\begin{table} \n')
    f.write('\\caption{Parameters of '+sysname+'} \n')
    f.write('\\begin{tabular}{lc} \n')
    f.write('\\hline \n')
    f.write('\\hline \n')
    f.write('Parameter & Value \\\ \n')
    f.write('\\hline \n')
    f.write('Measured Parameters \\\ \n')

    for i in range (0,ndim):
        if str(k+1) in inposindex[i] and not 'gamma' in inposindex[i] and not 'q' in inposindex[i] and not 'jitter' in inposindex[i] and not 'dilution' in inposindex[i]:
            v=np.nanpercentile(samples[:,i], [16, 50, 84], axis=0)
            temp=[v[1], v[2]-v[1], v[1]-v[0]]
            if not args.bestprob:
                inpos[i]=temp[0]
            else:
                inpos[i]=samples[best,i]
            f.write(symdex(inposindex[i],args,exstruc)+' & '+schar1+parstring(temp)+schar2+' \\\ \n')

            if args.relpars:
                if 'rprs' in inposindex[i]: rprsp, rprsm = ufloat(temp[0],temp[1]), ufloat(temp[0],temp[2])
                if 'semiamp' in inposindex[i]: Kp, Km = ufloat(temp[0],temp[1]), ufloat(temp[0],temp[2])
                if 'aors' in inposindex[i]: aorsp, aorsm = ufloat(temp[0],temp[1]), ufloat(temp[0],temp[2])
                if 'Per' in inposindex[i]: Pdp, Pdm = ufloat(temp[0],temp[1]), ufloat(temp[0],temp[2])
        elif 'gamma' in inposindex[i] or (('g' in inposindex[i] or 'q' in inposindex[i]) and ('p' in inposindex[i] or 't' in inposindex[i])) or 'rvtrend' in inposindex[i] or 'jitter' in inposindex[i] or 'fluxrat' in inposindex[i] or 'dilution' in inposindex[i] or 'vsini' in inposindex[i] or 'intwidth' in inposindex[i]:
            v=np.nanpercentile(samples[:,i], [16, 50, 84], axis=0)
            temp=[v[1], v[2]-v[1], v[1]-v[0]]
            inpos[i]=temp[0]
            f.write(symdex(inposindex[i],args,exstruc)+' & '+schar1+parstring(temp)+schar2+' \\\ \n')
        
            
            

    f.write('\hline \n')
    f.write('Derived Parameters \\\ \n')


    bstr='bpar'+str(k+1)
    rstr='rprs'+str(k+1)
    pstr='Per'+str(k+1)
    epstr='epoch'+str(k+1)

    if any('sinw' in s for s in inposindex) or any('ecc' in s for s in inposindex):
        if any('sesinw' in s for s in inposindex):
            estr1='sesinw'+str(k+1)
            estr2='secosw'+str(k+1)
            ecc=samples[:,[i for i, s in enumerate(inposindex) if estr1 in s]]**2+samples[:,[i for i, s in enumerate(inposindex) if estr2 in s]]**2
            omega=np.arccos(samples[:,[i for i, s in enumerate(inposindex) if estr2 in s]]/np.sqrt(ecc))
            news=news=np.where(samples[:,[i for i, s in enumerate(inposindex) if estr1 in s]] < 0.)
            omega[news]=2.*np.pi-omega[news]
            if any(not np.isfinite(t) for t in omega):
                bads=np.where(np.isfinite(omega) == True)
                omega[bads]=np.pi/2.0
                temp=samples[:,[i for i, s in enumerate(inposindex) if estr1 in s]]
                ecc[bads]=temp[bads]**2
        if any('ecc' in s for s in inposindex):
            estr1='ecc'+str(k+1)
            estr2='omega'+str(k+1)
            ecc=samples[:,[i for i, s in enumerate(inposindex) if estr1 in s]]
            omega=samples[:,[i for i, s in enumerate(inposindex) if estr2 in s]]

        if any('ecsinw' in s for s in inposindex):
            estr1='ecsinw'+str(k+1)
            estr2='eccosw'+str(k+1)
            ecc=np.sqrt(samples[:,[i for i, s in enumerate(inposindex) if estr1 in s]]**2+samples[:,[i for i, s in enumerate(inposindex) if estr2 in s]]**2)
            omega=np.arccos(samples[:,[i for i, s in enumerate(inposindex) if estr2 in s]]/ecc)
            news=np.where(samples[:,[i for i, s in enumerate(inposindex) if estr1 in s]] < 0.)
            omega[news]=2.*np.pi-omega[news]
            if any(not np.isfinite(t) for t in omega):
                bads=np.where(np.isfinite(omega) == True)
                omega[bads]=np.pi/2.0
                temp=samples[:,[i for i, s in enumerate(inposindex) if estr1 in s]]
                ecc[bads]=temp[bads]

        omega*=180./np.pi #radians->degrees
        

    else:
        ecc=0.0
        omega=90.
            

    if any('rhostar' in s for s in inposindex) and any('bpar' in s for s in inposindex) and (args.photometry or args.tomography):
        aors=215.*samples[:,[i for i, s in enumerate(inposindex) if 'rhostar' in s]]**(1./3.)*(samples[:,[i for i, s in enumerate(inposindex) if pstr in s]]/365.25)**(2./3.)*((1.+ecc*np.sin(omega*np.pi/180.))/np.sqrt(1.-ecc**2))
        if any('ecsinw' in s for s in inposindex):
            sstr='ecsinw'+str(k+1)
            inc=np.arccos(samples[:,[i for i, s in enumerate(inposindex) if bstr in s]]/aors*(1.0+samples[:,[i for i, s in enumerate(inposindex) if sstr in s]])/(1.0-ecc**2))*180./np.pi
        elif any('sesinw' in s for s in inposindex):
            sstr='sesinw'+str(k+1)
            inc=np.arccos(samples[:,[i for i, s in enumerate(inposindex) if bstr in s]]/aors*(1.0+np.sqrt(ecc)*samples[:,[i for i, s in enumerate(inposindex) if sstr in s]])/(1.0-ecc**2))*180./np.pi
        else:
            inc=np.arccos(samples[:,[i for i, s in enumerate(inposindex) if bstr in s]]/aors*(1.0+ecc*np.sin(omega*np.pi/180.))/(1.0-ecc**2))*180./np.pi
        dur=samples[:,[i for i, s in enumerate(inposindex) if pstr in s]]/np.pi*1./aors*np.sqrt(1.0-ecc**2)/(1.0+ecc*np.cos(omega*np.pi/180.))*np.sqrt((1.0+samples[:,[i for i, s in enumerate(inposindex) if rstr in s]])**2-samples[:,[i for i, s in enumerate(inposindex) if bstr in s]]**2)
        v=np.nanpercentile(aors, [16, 50, 84])
        temp=[v[1], v[2]-v[1], v[1]-v[0]]
        f.write('$a/R_{\star}$ & '+schar1+parstring(temp)+schar2+' \\\ \n')
        if args.relpars: aorsp, aorsm = ufloat(temp[0],temp[1]), ufloat(temp[0],temp[2])
    if any('aors' in s for s in inposindex) and any('bpar' in s for s in inposindex)  and (args.photometry or args.tomography):
        astr='aors'+str(k+1)
        if any('ecsinw' in s for s in inposindex):
            sstr='ecsinw'+str(k+1)
            inc=np.arccos(samples[:,[i for i, s in enumerate(inposindex) if bstr in s]]/samples[:,[i for i, s in enumerate(inposindex) if astr in s]]*(1.0+samples[:,[i for i, s in enumerate(inposindex) if sstr in s]])/(1.0-ecc**2))*180./np.pi
        elif any('sesinw' in s for s in inposindex):
            sstr='sesinw'+str(k+1)
            inc=np.arccos(samples[:,[i for i, s in enumerate(inposindex) if bstr in s]]/samples[:,[i for i, s in enumerate(inposindex) if astr in s]]*(1.0+np.sqrt(ecc)*samples[:,[i for i, s in enumerate(inposindex) if sstr in s]])/(1.0-ecc**2))*180./np.pi
        else:
            inc=np.arccos(samples[:,[i for i, s in enumerate(inposindex) if bstr in s]]/samples[:,[i for i, s in enumerate(inposindex) if astr in s]]*(1.0+ecc*np.sin(omega*np.pi/180.))/(1.0-ecc**2))*180./np.pi


            
        if any('ecsinw' in s for s in inposindex):
            cstr='eccosw'+str(k+1)
            dur=samples[:,[i for i, s in enumerate(inposindex) if pstr in s]]/np.pi*np.arcsin(np.sqrt(((1.0+samples[:,[i for i, s in enumerate(inposindex) if rstr in s]])**2-samples[:,[i for i, s in enumerate(inposindex) if bstr in s]]**2)/(samples[:,[i for i, s in enumerate(inposindex) if astr in s]]**2-samples[:,[i for i, s in enumerate(inposindex) if bstr in s]]**2)))*(1.0+samples[:,[i for i, s in enumerate(inposindex) if cstr in s]])/np.sqrt(1.0-ecc**2) #empirically ignore e for the moment....... BLEH
        elif any('sesinw' in s for s in inposindex):
            cstr='secosw'+str(k+1)
            dur=samples[:,[i for i, s in enumerate(inposindex) if pstr in s]]/np.pi*np.arcsin(np.sqrt(((1.0+samples[:,[i for i, s in enumerate(inposindex) if rstr in s]])**2-samples[:,[i for i, s in enumerate(inposindex) if bstr in s]]**2)/(samples[:,[i for i, s in enumerate(inposindex) if astr in s]]**2-samples[:,[i for i, s in enumerate(inposindex) if bstr in s]]**2)))*(1.0+np.sqrt(ecc)*samples[:,[i for i, s in enumerate(inposindex) if cstr in s]])/np.sqrt(1.0-ecc**2)
        else:
            dur=samples[:,[i for i, s in enumerate(inposindex) if pstr in s]]/np.pi*np.arcsin(np.sqrt(((1.0+samples[:,[i for i, s in enumerate(inposindex) if rstr in s]])**2-samples[:,[i for i, s in enumerate(inposindex) if bstr in s]]**2)/(samples[:,[i for i, s in enumerate(inposindex) if astr in s]]**2-samples[:,[i for i, s in enumerate(inposindex) if bstr in s]]**2)))*(1.0+ecc*np.cos(omega*np.pi/180.))/np.sqrt(1.0-ecc**2)
        
        rhostar=(samples[:,[i for i, s in enumerate(inposindex) if astr in s]]/215.)**3./(samples[:,[i for i, s in enumerate(inposindex) if pstr in s]]/365.25)**2./((1.+ecc*np.sin(omega*np.pi/180.))/np.sqrt(1.-ecc**2))**3.
        v=np.nanpercentile(rhostar, [16, 50, 84])
        temp=[v[1], v[2]-v[1], v[1]-v[0]]
        f.write('$\\rho_{\star}$ ($\\rho_{\odot}$) & '+schar1+parstring(temp)+schar2+' \\\ \n')

    if any('cosi' in s for s in inposindex)  and (args.photometry or args.tomography):
        istr='cosi'
        inc=np.arccos(samples[:,[i for i, s in enumerate(inposindex) if istr in s]])*180./np.pi

    if args.photometry or args.tomography:
        v=np.nanpercentile(inc, [16, 50, 84])
        temp=[v[1], v[2]-v[1], v[1]-v[0]]
        f.write('$i$ ($^{\circ}$) & '+schar1+parstring(temp)+schar2+' \\\ \n')
        if args.relpars: incp, incm = ufloat(temp[0], temp[1]), ufloat(temp[0], temp[2])

        if not args.binary:
            v=np.nanpercentile((samples[:,[i for i, s in enumerate(inposindex) if rstr in s]]**2)*100., [16, 50, 84]) #to get into percent
            temp=[v[1], v[2]-v[1], v[1]-v[0]]
            f.write('$\delta$ (\%) & '+schar1+parstring(temp)+schar2+' \\\ \n')

            v=np.nanpercentile(dur, [16, 50, 84])
            temp=[v[1], v[2]-v[1], v[1]-v[0]]
            f.write('$T_{14}$ (days) & '+schar1+parstring(temp)+schar2+' \\\ \n')

            t23=np.arcsin(np.sin(dur*np.pi/samples[:,[i for i, s in enumerate(inposindex) if pstr in s]])*np.sqrt((1.-samples[:,[i for i, s in enumerate(inposindex) if rstr in s]])**2-samples[:,[i for i, s in enumerate(inposindex) if bstr in s]]**2)/np.sqrt((1.+samples[:,[i for i, s in enumerate(inposindex) if rstr in s]])**2-samples[:,[i for i, s in enumerate(inposindex) if bstr in s]]**2))*samples[:,[i for i, s in enumerate(inposindex) if pstr in s]]/np.pi #This doesn't account for eccentricity!!! (except as it influences the transit duration)

            v=np.nanpercentile(t23, [16, 50, 84])
            temp=[v[1], v[2]-v[1], v[1]-v[0]]
            f.write('$T_{23}$ (days) & '+schar1+parstring(temp)+schar2+' \\\ \n')

        
    if args.rvs or args.photometry: #time of periastron
        if args.rvs and not args.photometry:
            Etransit=0.0
            epperi=samples[:,[i for i, s in enumerate(inposindex) if epstr in s]]
        else:
            Etransit=2.*np.arctan(np.sqrt((1.-ecc)/(1.+ecc))*np.tan((np.pi/2.-omega*np.pi/180.)/2.))
            epperi=samples[:,[i for i, s in enumerate(inposindex) if epstr in s]]-samples[:,[i for i, s in enumerate(inposindex) if pstr in s]]/(2.*np.pi)*(Etransit-ecc*np.sin(Etransit))
        v=np.nanpercentile(epperi, [16, 50, 84])
        temp=[v[1], v[2]-v[1], v[1]-v[0]]
        if exstruc['timeoffset'] == 0.:
            f.write('$T_{\mathrm{peri}}$ ('+exstruc['timestandard']+') & '+schar1+parstring(temp)+schar2+' \\\ \n')
        else:
            f.write('$T_{\mathrm{peri}}$ ('+exstruc['timestandard']+'-'+str(np.int(np.round(exstruc['timeoffset'])))+') & '+schar1+parstring(temp)+schar2+' \\\ \n')
        


    if any('q1p' in s for s in inposindex):
        for j in range (0,pnfilters):
            q1str = 'q1p'+str(j+1)
            q2str = 'q2p'+str(j+1)
            g1 = 2.0*samples[:,[i for i, s in enumerate(inposindex) if q2str in s]]*np.sqrt(samples[:,[i for i, s in enumerate(inposindex) if q1str in s]])
            g2 = np.sqrt(samples[:,[i for i, s in enumerate(inposindex) if q1str in s]])*(1.0-2.0*samples[:,[i for i, s in enumerate(inposindex) if q2str in s]])
            v=np.nanpercentile(g1, [16, 50, 84])
            temp=[v[1], v[2]-v[1], v[1]-v[0]]
            f.write(symdex('g1p'+str(j+1),args,exstruc)+' & '+schar1+parstring(temp)+schar2+' \\\ \n')

            v=np.nanpercentile(g2, [16, 50, 84])
            temp=[v[1], v[2]-v[1], v[1]-v[0]]
            f.write(symdex('g2p'+str(j+1),args,exstruc)+' & '+schar1+parstring(temp)+schar2+' \\\ \n')

    if any('q1t' in s for s in inposindex):
        q1str = 'q1t'
        q2str = 'q2t'
        g1 = 2.0*samples[:,[i for i, s in enumerate(inposindex) if q2str in s]]*np.sqrt(samples[:,[i for i, s in enumerate(inposindex) if q1str in s]])
        g2 = np.sqrt(samples[:,[i for i, s in enumerate(inposindex) if q1str in s]])*(1.0-2.0*samples[:,[i for i, s in enumerate(inposindex) if q2str in s]])
        v=np.nanpercentile(g1, [16, 50, 84])
        temp=[v[1], v[2]-v[1], v[1]-v[0]]
        f.write(symdex('g1t',args,exstruc)+' & '+schar1+parstring(temp)+schar2+' \\\ \n')

        v=np.nanpercentile(g2, [16, 50, 84])
        temp=[v[1], v[2]-v[1], v[1]-v[0]]
        f.write(symdex('g2t',args,exstruc)+' & '+schar1+parstring(temp)+schar2+' \\\ \n')
        

    if args.relpars: #relative parameters

        if any('Mstar' in s for s in index): Mstarp, Mstarm = ufloat(struc1['Mstar'],struc1['epMstar']), ufloat(struc1['Mstar'],struc1['emMstar'])
        if any('Rstar' in s for s in index): Rstarp, Rstarm = ufloat(struc1['Rstar'],struc1['epRstar']), ufloat(struc1['Rstar'],struc1['emRstar'])
        if any('Teff' in s for s in index): Teff = ufloat(struc1['Teff'],struc1['eTeff'])

        if any('Rstar' in s for s in index) and args.photometry:
            Rpp, Rpm = rprsp*Rstarp*convfactor('rprs-to-rp',args), rprsm*Rstarm*convfactor('rprs-to-rp',args)
            temp=[Rpp.n,Rpp.s,Rpm.s]
            if args.earth:
                f.write('$R_P$ ($R_{\oplus}$) & '+schar1+parstring(temp)+schar2+' \\\ \n')
            else:
                f.write('$R_P$ ($R_J$) & '+schar1+parstring(temp)+schar2+' \\\ \n')
        
        if any('Mstar' in s for s in index) and args.rvs:
            Mpsinip, Mpsinim = Kp/28.4329*Mstarp**(2./3.)*(Pdp/365.25)**(1./3.), Km/28.4329*Mstarm**(2./3.)*(Pdm/365.25)**(1./3.) #assumes RVs in km/s
            if not args.ms:
                Mpsinip*=1000.
                Mpsinim*=1000.
            if args.earth: 
                Mpsinip*=317.8
                Mpsinim*=317.8
            
            temp=[Mpsinip.n,Mpsinip.s,Mpsinim.s]
            
            if args.earth:
                f.write('$M_P\sin i$ ($M_{\oplus}$) & '+schar1+parstring(temp)+schar2+' \\\ \n')
            else:
                f.write('$M_P\sin i$ ($M_J$) & '+schar1+parstring(temp)+schar2+' \\\ \n')
            if args.rvs:
                Mpp, Mpm = Mpsinip/unp.sin(incm*np.pi/180.), Mpsinim/unp.sin(incp*np.pi/180.)
                temp=[Mpp.n,Mpp.s,Mpm.s]
                
                if args.earth:
                    f.write('$M_P$ ($M_{\oplus}$) & '+schar1+parstring(temp)+schar2+' \\\ \n')
                else:
                    f.write('$M_P$ ($M_J$) & '+schar1+parstring(temp)+schar2+' \\\ \n')

                massratp, massratm = Mpp/Mstarm*convfactor('Mp-to-Massrat',args), Mpm/Mstarp*convfactor('Mp-to-Massrat',args)
                temp=[massratp.n,massratp.s,massratm.s]
                f.write('$M_P/M_{\star}$ & '+schar1+parstring(temp)+schar2+' \\\ \n')
            
        if any('Rstar' in s for s in index) and args.photometry:
            ap, am = aorsp*Rstarp/(1.49597e11/6.957e8), aorsm*Rstarm/(1.49597e11/6.957e8)
            temp=[ap.n,ap.s,am.s]
            f.write('$a$ (AU) & '+schar1+parstring(temp)+schar2+' \\\ \n')
        elif args.rvs:
            junk=1 #calculate using Kepler's law, implement later


        if args.photometry and args.rvs and any('Rstar' in s for s in index) and any('Mstar' in s for s in index):
            rhopp, rhopm = Mpp/Rpm**3*1.3261972343759805, Mpm/Rpp**3*1.3261972343759805
            temp=[rhopp.n,rhopp.s,rhopm.s]
            f.write('$\\rho_P$ (g cm$^{-3}$) & '+schar1+parstring(temp)+schar2+' \\\ \n')

            gp, gm = Mpp/Rpm**2*6.67269e-8*1.8982e30/(7.1492e9)**2, Mpm/Rpp**2*6.67269e-8*1.8982e30/(7.1492e9)**2
            v=[np.log10(gp.n-gm.s),np.log10(gp.n),np.log10(gp.n+gp.s)]
            temp=[v[1], v[2]-v[1], v[1]-v[0]]
            f.write('$\log g_P$ (cgs) & '+schar1+parstring(temp)+schar2+' \\\ \n')

        if any('Teff' in s for s in index):
            Teqp, Teqm = Teff*unp.sqrt(0.5/aorsm), Teff*unp.sqrt(0.5/aorsp)
            temp=[Teqp.n,Teqp.s,Teqm.s]
            f.write('$T_{\mathrm{eq}}$ (K) & '+schar1+parstring(temp)+schar2+' \\\ \n')

            Sp, Sm = (Teff/5772.)**4.*(aorsm/215.032)**(-2.), (Teff/5772.)**4.*(aorsp/215.032)**(-2.)
            temp=[Sp.n,Sp.s,Sm.s]
            f.write('$S$ ($S_{\oplus})$ & '+schar1+parstring(temp)+schar2+' \\\ \n')

        if args.photometry and args.rvs and any('Rstar' in s for s in index) and any('Mstar' in s for s in index) and any('Teff' in s for s in index):
            Hp, Hm = 1.38065e-16/(2.*1.007825*1.661e-24)*Teqp/gm/1e5, 1.38065e-16/(2.*1.007825*1.661e-24)*Teqm/gp/1e5 #atmospheric scale height
            temp=[Hp.n,Hp.s,Hm.s]
            f.write('$H$ (km) & '+schar1+parstring(temp)+schar2+' \\\ \n')

            fracp, fracm = 2.*Hp/(Rpm*69911.), 2.*Hm/(Rpp*69911.)
            temp=[fracp.n, fracp.s, fracm.s]
            f.write('$2H/R_P$ & '+schar1+parstring(temp)+schar2+' \\\ \n')

    if any('sinw' in s for s in inposindex):
        v=np.nanpercentile(ecc, [16, 50, 84])
        temp=[v[1], v[2]-v[1], v[1]-v[0]]
        f.write('$e$ & '+schar1+parstring(temp)+schar2+' \\\ \n')
        v=np.nanpercentile(omega, [16, 50, 84])
        temp=[v[1], v[2]-v[1], v[1]-v[0]]
        f.write('$\omega$ ($^{\circ}$) & '+schar1+parstring(temp)+schar2+' \\\ \n')
        

    if args.binary: #get the absolute parameters we can get from SB2s and/or EBs
        if args.rvs:
            GkmsMsun=6.6740831e-11/(1000.)**3*1.9891e30
            M1sin3i=samples[:,[i for i, s in enumerate(inposindex) if 'Per1' in s]]*24.*60.*60./(2.*np.pi*GkmsMsun)*(samples[:,[i for i, s in enumerate(inposindex) if 'semiamp1a' in s]]+samples[:,[i for i, s in enumerate(inposindex) if 'semiamp1b' in s]])**3/(1.+samples[:,[i for i, s in enumerate(inposindex) if 'semiamp1a' in s]]/samples[:,[i for i, s in enumerate(inposindex) if 'semiamp1b' in s]])
            M2sin3i=samples[:,[i for i, s in enumerate(inposindex) if 'Per1' in s]]*24.*60.*60./(2.*np.pi*GkmsMsun)*(samples[:,[i for i, s in enumerate(inposindex) if 'semiamp1a' in s]]+samples[:,[i for i, s in enumerate(inposindex) if 'semiamp1b' in s]])**3/(1.+samples[:,[i for i, s in enumerate(inposindex) if 'semiamp1b' in s]]/samples[:,[i for i, s in enumerate(inposindex) if 'semiamp1a' in s]])

            v=np.nanpercentile(M1sin3i, [16, 50, 84])
            temp=[v[1], v[2]-v[1], v[1]-v[0]]
            f.write(symdex('M1sin3i',args,exstruc)+' & '+schar1+parstring(temp)+schar2+' \\\ \n')

            v=np.nanpercentile(M2sin3i, [16, 50, 84])
            temp=[v[1], v[2]-v[1], v[1]-v[0]]
            f.write(symdex('M2sin3i',args,exstruc)+' & '+schar1+parstring(temp)+schar2+' \\\ \n')

            if args.photometry:
                M1=M1sin3i/(np.sin(inc*np.pi/180.))**3
                M2=M2sin3i/(np.sin(inc*np.pi/180.))**3

                v=np.nanpercentile(M1, [16, 50, 84])
                temp=[v[1], v[2]-v[1], v[1]-v[0]]
                f.write(symdex('M1',args,exstruc)+' & '+schar1+parstring(temp)+schar2+' \\\ \n')

                v=np.nanpercentile(M2, [16, 50, 84])
                temp=[v[1], v[2]-v[1], v[1]-v[0]]
                f.write(symdex('M2',args,exstruc)+' & '+schar1+parstring(temp)+schar2+' \\\ \n')

                semimajor=(samples[:,[i for i, s in enumerate(inposindex) if 'Per1' in s]]/365.25)**(2./3.)*(M1+M2)**(1./3.)
                v=np.nanpercentile(semimajor, [16, 50, 84])
                temp=[v[1], v[2]-v[1], v[1]-v[0]]
                f.write('$a$ (AU) & '+schar1+parstring(temp)+schar2+' \\\ \n')

                R1=semimajor*1.49597e11/6.95508e8/(samples[:,[i for i, s in enumerate(inposindex) if 'aors1' in s]]*(1.+samples[:,[i for i, s in enumerate(inposindex) if 'rprs1' in s]]))
                R2=R1*samples[:,[i for i, s in enumerate(inposindex) if 'rprs1' in s]]

                v=np.nanpercentile(R1, [16, 50, 84])
                temp=[v[1], v[2]-v[1], v[1]-v[0]]
                f.write('$R_1$ ($R_{\odot}$) & '+schar1+parstring(temp)+schar2+' \\\ \n')

                v=np.nanpercentile(R2, [16, 50, 84])
                temp=[v[1], v[2]-v[1], v[1]-v[0]]
                f.write('$R_2$ ($R_{\odot}$) & '+schar1+parstring(temp)+schar2+' \\\ \n')

                Teffrat=(samples[:,[i for i, s in enumerate(inposindex) if 'fluxrat' in s]])**(1./4.)
                v=np.nanpercentile(Teffrat, [16, 50, 84])
                temp=[v[1], v[2]-v[1], v[1]-v[0]]
                f.write('$T_{\mathrm{eff},2}/T_{\mathrm{eff},1}$) & '+schar1+parstring(temp)+schar2+' \\\ \n')

    
            
    
    f.write('\\hline \n')
    f.write('\\end{tabular} \n')
    f.write('\\end{table} \n')
    f.close()
    print 'The table for planet '+str(k+1)+' is complete at '+sysname+'_table'+str(k+1)+'.tex'

if args.tableonly: sys.exit()

if args.corner:
    try:
        import corner
    except ImportError:
        print 'corner does not appear to be installed correctly.'
        print 'you can install it with "pip install corner"'
        print 'exiting now'
        sys.exit()
    labels=np.zeros(ndim,dtype='object')
    for i in range (0,ndim):
        temp=np.mean(samples[:,i])
        if np.std(samples[:,i]) < 1e-10:
            samples[:,i]+=np.random.randn(len(samples[:,i]))*temp*1e-5
        labels[i]=symdex(inposindex[i],args,exstruc)
    corner.corner(samples,labels=labels,truths=inpos1,truth_color='red',quantiles=[0.16,0.5,0.84])
    pl.savefig(sysname+'_corner.pdf',format='pdf')
    pl.clf()
    print 'The corner plot is saved at ',sysname+'_corner.pdf'
    

theta=inpos
parstruc = dict(zip(inposindex, theta))

def filtercolor(filtname,args):
    if args.bw: return 'gray'
    if filtname == 'blank': return 'black'
    if filtname == 'Kp': return 'gray'
    if filtname == 'CoRoT': return 'gray'
    if filtname == 'clear': return 'gray'
    if filtname == 'g': return 'turquoise'
    if filtname == 'r': return 'red'
    if filtname == 'i': return 'indianred'
    if filtname == 'z': return 'mediumpurple'
    if filtname == 'U': return 'purple'
    if filtname == 'B': return 'blue'
    if filtname == 'V': return 'green'
    if filtname == 'R': return 'red'
    if filtname == 'I': return 'darkred'
    if filtname == 'J': return 'maroon'
    if filtname == 'H': return 'firebrick'
    if filtname == 'K': return 'brown'
    if filtname == '3.8um': return 'goldenrod'
    if filtname == '4.5um': return 'darkorange'

def resampler(modin,t,tprime,cadence):
    #t is the raw time, tprime is what we want to resample to
    #cadence has the same length as tprime
    nexp=len(tprime)
    nmodps=len(t)
    modout=np.zeros(nexp)
    for i in range (0,nexp):
        heres=np.where(np.abs(t-tprime[i]) <= cadence[i]/2.0)
        modout[i]=np.mean(modin[heres])
    return modout


if any('sinw' in s for s in inposindex) or any('ecc' in s for s in inposindex):
    if any('sesinw' in s for s in inposindex):
        ecc=theta[[i for i, s in enumerate(inposindex) if 'sesinw' in s]]**2+theta[[i for i, s in enumerate(inposindex) if 'secosw' in s]]**2
        omega=np.arccos(theta[[i for i, s in enumerate(inposindex) if 'secosw' in s]]/np.sqrt(ecc))
        if any(not np.isfinite(t) for t in omega):
            bads=np.where(np.isfinite(omega) == True)
            omega[bads]=np.pi/2.0
            temp=theta[[i for i, s in enumerate(inposindex) if 'sesinw' in s]]
            ecc[bads]=temp[bads]**2

    if any('ecc' in s for s in inposindex):
        ecc=theta[[i for i, s in enumerate(inposindex) if 'ecc' in s]]
        omega=theta[[i for i, s in enumerate(inposindex) if 'omega' in s]]

    if any('ecsinw' in s for s in inposindex):
        ecc=np.sqrt(theta[[i for i, s in enumerate(inposindex) if 'ecsinw' in s]]**2+theta[[i for i, s in enumerate(inposindex) if 'eccosw' in s]]**2)
        omega=np.arccos(theta[[i for i, s in enumerate(inposindex) if 'eccosw' in s]]/ecc)
        news=np.where(theta[[i for i, s in enumerate(inposindex) if 'ecsinw' in s]] < 0.)
        omega[news]=2.*np.pi-omega[news]
        if any(not np.isfinite(t) for t in omega):
            bads=np.where(np.isfinite(omega) == True)
            omega[bads]=np.pi/2.0
            temp=theta[[i for i, s in enumerate(inposindex) if 'ecsinw' in s]]
            ecc[bads]=temp[bads]

    omega*=180./np.pi #radians->degrees
    

else:
    ecc=np.zeros(nplanets)
    omega=np.zeros(nplanets)+90.


def jktwrangler(struc):
    if struc['photmodflag'] == 'jktebop':
        timestamp=str(multiprocessing.current_process().pid)
        f=open('temp'+timestamp+'.in','w')
        f.write('2 1 Task to do (from 2 to 9)   Integ. ring size (deg) \n')
        f.write(str(1./struc['aors'])+'  '+str(struc['rprs'])+' Sum of the radii           Ratio of the radii \n')
        f.write(str(struc['inc'])+'  '+str(-1.)+' Orbital inclination (deg)  Mass ratio of system \n')
        f.write(str(struc['ecc']+10.)+'  '+str(struc['omega'])+' ecosw or eccentricity      esinw or periastron long \n')
        f.write('1.0 1.0 Gravity darkening (star A) Grav darkening (star B) \n')
        f.write(str(1./struc['fluxrat'])+'  '+str(struc['dilution']*(-1.))+' Surface brightness ratio   Amount of third light \n')
        f.write('quad  quad  LD law type for star A     LD law type for star B \n')
        f.write(str(struc['g1'])+'  '+str(struc['g1'])+' LD star A (linear coeff)   LD star B (linear coeff) \n')
        f.write(str(struc['g2'])+'  '+str(struc['g2'])+' LD star A (nonlin coeff)   LD star B (nonlin coeff) \n')
        f.write('0.0  0.0  Reflection effect star A   Reflection effect star B \n')
        f.write('0.0  0.0  Phase of primary eclipse   Light scale factor (mag) \n')
        f.write('temp'+timestamp+'.out Output file name (continuous character string) \n')
        f.close()

        os.system('rm -f temp'+timestamp+'.out')

        os.system('./../../system/jktebop/jktebop/jktebop temp'+timestamp+'.in')

        phase,mag,l1,l2,l3=readcol('temp'+timestamp+'.out',twod=False)

        highs=np.where(phase > 0.5)
        phase[highs[0]]-=1.0

        os.system('rm -f temp'+timestamp+'.out')
        os.system('rm -f temp'+timestamp+'.in')

        mflux1=10.**(((-1.)*mag)/2.5)
        
        flux=resampler(mflux1,phase*struc['Per'],struc['t'],struc['exptime'])
            
    return flux


if args.photometry or args.tomography:
    if any('rhostar' in s for s in inposindex) and any('bpar' in s for s in inposindex):
        aors=215.*parstruc['rhostar']**(1./3.)*(theta[[i for i, s in enumerate(inposindex) if 'Per' in s]]/365.25)**(2./3.)*((1.+ecc*np.sin(omega*np.pi/180.))/np.sqrt(1.-ecc**2))
        cosi=theta[[i for i, s in enumerate(inposindex) if 'bpar' in s]]/aors*(1.0+ecc*np.sin(omega*np.pi/180.))/(1.0-ecc**2)
        inc=np.arccos(cosi)*180./np.pi
    if any('aors' in s for s in inposindex) and any('bpar' in s for s in inposindex):
        aors=theta[[i for i, s in enumerate(inposindex) if 'aors' in s]]
        cosi=theta[[i for i, s in enumerate(inposindex) if 'bpar' in s]]/aors*(1.0+ecc*np.sin(omega*np.pi/180.))/(1.0-ecc**2)
        inc=np.arccos(cosi)*180./np.pi
        #NOTE that for this option for multiplanet systems, there is currently no enforcement of rhostar being the same for all planets!!!

    if any('cosi' in s for s in inposindex) and any('aors' in s for s in inposindex):
        aors=theta[[i for i, s in enumerate(inposindex) if 'aors' in s]]
        cosi=theta[[i for i, s in enumerate(inposindex) if 'cosi' in s]]
        inc=np.arccos(cosi)*180./np.pi
    
    if not args.binary:
        dur=theta[[i for i, s in enumerate(inposindex) if 'Per' in s]]/np.pi*1./aors*np.sqrt(1.0-ecc**2)/(1.0+ecc*np.cos(omega*np.pi/180.))*np.sqrt((1.0+theta[[i for i, s in enumerate(inposindex) if 'rprs' in s]])**2-theta[[i for i, s in enumerate(inposindex) if 'bpar' in s]]**2)
    else:
        dur=np.array([4.0])



if args.photometry:
    pmodelp=np.zeros(len(data['ptime']))
    for i in range (0,nplanets):
        pp = PdfPages(sysname+'LC'+str(i+1)+'.pdf')
        phased = np.mod(data['ptime']-parstruc['epoch'+str(i+1)], parstruc['Per'+str(i+1)])
        ftransit=np.pi/2.-(omega[i])*np.pi/180. #true anomaly at transit 
        Etransit=2.*np.arctan(np.sqrt((1.-ecc[i])/(1.+ecc[i]))*np.tan(ftransit/2.)) #eccentric anomaly at secondary eclipse
        timesince=parstruc['Per'+str(i+1)]/(2.*np.pi)*(Etransit-ecc[i]*np.sin(Etransit))
        timesince*=24.
        highs = np.where(phased > parstruc['Per'+str(i+1)]/2.0)
        phased[highs]-=parstruc['Per'+str(i+1)]
        closes = np.where(np.abs(phased) <= (dur[i]/2.)*1.5)
        modelstruc={'Per':parstruc['Per'+str(i+1)], 'rprs':parstruc['rprs'+str(i+1)], 'aors':aors[i], 'inc':inc[i], 'ecc':ecc[i], 'omega':omega[i]}
        if any('photmodflag' in s for s in struc1):
            modelstruc['photmodflag']=struc1['photmodflag']
        else:
            modelstruc['photmodflag']='batman'
        if args.binary and modelstruc['photmodflag'] == 'batman': modelstruc['aors'] = aors[i]*(1.+parstruc['rprs'+str(i+1)]) #PRIMARY eclipse, so PRIMARY in background, R*=R1, Rp=R2
        oflux=np.array(data['pflux'])
        mresid=np.array(data['pflux'])
        if args.gp: mresidgp=np.array(data['pflux'])
        if pnfilters > 1:
            fig=pl.figure(figsize=(8.0,6.0+2.0*pnfilters))
        if args.plotresids: 
            import matplotlib.gridspec as gridspec
            gs=gridspec.GridSpec(2, 1, height_ratios=[4,1])
            ax1=pl.subplot(gs[0])
        if args.gp:
            if any('gpmodtypep' in s for s in data['gpmodtype']):
                if data['gpmodtype']['gpmodtypep'] == 'Matern32':
                    pkern=gppack.kernels.Matern32Kernel(parstruc['gpppartau']**2)*parstruc['gppparamp']**2
                elif data['gpmodtype']['gpmodtypep'] == 'Cosine':
                    pkern=gppack.kernels.CosineKernel(parstruc['gppparP'])*parstruc['gppparamp']**2
                elif data['gpmodtype']['gpmodtypep'] == 'ExpSine2':
                    pkern=gppack.kernels.ExpSine2Kernel(parstruc['gpppartau'],parstruc['gppparP'])*parstruc['gppparamp']**2
                elif data['gpmodtype']['gpmodtypep'] == 'Haywood14QP':
                    if struc1['gppackflag'] == 'celerite':
                        pkern=celeritekernel(np.log(parstruc['gppparamp']**2),np.log(parstruc['gppparGamma']),np.log(1./np.sqrt(2.)/parstruc['gpppartau']),np.log(parstruc['gppparP']*2.))
                    else:
                        pkern1=gppack.kernels.ExpSine2Kernel(parstruc['gppparGamma'],parstruc['gppparP'])
                        pkern2=gppack.kernels.ExpSquaredKernel(parstruc['gpppartau'])
                        pkern=pkern1*pkern2*parstruc['gppparamp']**2
                            
                        
                gp=gppack.GP(pkern)
                useforgp=np.where(data['gppuse'] == 1)
                notforgp=np.where(data['gppuse'] == 0)
                useforgp, notforgp = useforgp[0], notforgp[0]
                gp.compute(np.array(data['ptime'][useforgp]),np.array(data['perror'][useforgp]))

        for j in range (0,pndatasets):
            if any('q1p' in s for s in inposindex) and any('q1p' in s for s in inposindex):
                modelstruc['g1'], modelstruc['g2'] = 2.0*parstruc['q2p'+struc1['filternumber'+str(j+1)]]*np.sqrt(parstruc['q1p'+struc1['filternumber'+str(j+1)]]), np.sqrt(parstruc['q1p'+struc1['filternumber'+str(j+1)]])*(1.0-2.0*parstruc['q2p'+struc1['filternumber'+str(j+1)]])
            else:
                modelstruc['g1'], modelstruc['g2'] = parstruc['g1p'+struc1['filternumber'+str(j+1)]], parstruc['g2p'+struc1['filternumber'+str(j+1)]]

            

            closefilter=np.array(np.where((np.abs(phased) <= (dur[i]/2.)*1.5) & (data['pdataset'] == j+1)))
            goo=np.where(data['pdataset'] == j+1)
            if args.gp:
                if any('gpmodtypep' in s for s in data['gpmodtype']):
                    closedatasetg=np.where((np.abs(phased) <= (dur[i]/2.+np.max(data['pexptime'])*1.25)) & (data['pdataset'] == j+1) & (data['gppuse'] == 1))
                else:
                    closedatasetg=0


            modphase=np.linspace((dur[i]/2.)*(-1.5),(dur[i]/2.)*1.5,1000)
            flux=np.zeros(1000)

            if args.binary:
                modelstruc['fluxrat']=parstruc['fluxrat'+str(j+1)]


        
            if modelstruc['photmodflag'] == 'batman':
                params=batman.TransitParams()
                params.t0=0.0
                params.per=modelstruc['Per']
                params.rp=modelstruc['rprs']
                params.a=modelstruc['aors']
                params.inc=modelstruc['inc']
                params.ecc=modelstruc['ecc']
                params.w=modelstruc['omega']
                params.limb_dark='quadratic' #will have to de-hardcode this eventually...
                params.u=[modelstruc['g1'],modelstruc['g2']]
        
                tenminutes=10.0/(60.*24.)
                if np.max(data['pexptime'][closefilter]) >= tenminutes:
                    maxexp=np.mean(data['pexptime'][closefilter])
                    ttemp=np.arange(np.min(modphase)-maxexp*2., np.max(modphase)+maxexp*2., maxexp/100.)
                    ml=batman.TransitModel(params,ttemp,nthreads=1)
                    ftemp=ml.light_curve(params)
                    flux=resampler(ftemp, ttemp, modphase, np.zeros(1000)+maxexp)
                else:
                    ms=batman.TransitModel(params,modphase,nthreads=1)
                    flux=np.array(ms.light_curve(params))

                if args.binary:
                    params.w-=180.
                    fsecondary=np.pi/2.-omega[i]*np.pi/180.-np.pi #true anomaly at secondary eclipse
                    Esecondary=2.*np.arctan(np.sqrt((1.-ecc[i])/(1.+ecc[i]))*np.tan(fsecondary/2.)) #eccentric anomaly at secondary eclipse
                    timesince=parstruc['Per'+str(i+1)]/(2.*np.pi)*(Esecondary-ecc[i]*np.sin(Esecondary))
                    params.a=aors[i]*(1.+1./parstruc['rprs'+str(i+1)])
                    params.rp=1./parstruc['rprs'+str(i+1)]
                    params.t0+=timesince

                
                    if np.max(data['pexptime'][closefilter]) >= tenminutes:
                        fluxtemp=np.array(ml.light_curve(params))
                        flux2=resampler(fluxtemp, ttemp, modphase, np.zeros(1000)+maxexp)
                    else:
                        flux2=np.array(ms.light_curve(params))
                    flux+=1./parstruc['fluxrat'+str(j+1)]
                    flux/=1.+1./parstruc['fluxrat'+str(j+1)]
                    flux2+=parstruc['fluxrat'+str(j+1)]
                    flux2/=1.+parstruc['fluxrat'+str(j+1)]
                    flux-=1.
                    flux2-=1.
                    flux+=flux2+1.
                    params.w+=180.
                    params.a, params.rp, params.t0 = aors[i]*(1.+parstruc['rprs'+str(i+1)]), parstruc['rprs'+str(i+1)], 0.0
                if args.dilution:
                    if any('dilution'+str(j+1) in s for s in inposindex):
                        flux+= 10.**(-0.4*parstruc['dilution'+str(j+1)])
                        flux/= (1.+10.**(-0.4*parstruc['dilution'+str(j+1)]))


            else:
                tenminutes=10.0/(60.*24.)
                maxexp=np.mean(data['pexptime'][closefilter])
                modelstruc['t'], modelstruc['exptime'] = modphase, np.zeros(len(modphase))+maxexp
                if args.dilution: 
                    modelstruc['dilution'] = parstruc['dilution'+str(j+1)]
                else:
                    modelstruc['dilution'] = 0.0
                flux=jktwrangler(modelstruc)
                





            

            if args.plotresids or (args.gp and any('gpmodtypep' in s for s in data['gpmodtype'])):
                if np.min(data['pexptime'][closefilter[0]]) < tenminutes:
                    shorts=np.where(data['pexptime'][closefilter[0]] < tenminutes)
                    m2s=batman.TransitModel(params,phased[closefilter[0][shorts]],nthreads=1)
                    pmodel1=np.array(m2s.light_curve(params))-1.
                    mresid[closefilter[0][shorts]]-=pmodel1
                    if args.gp: 
                        mresidgp[closefilter[0][shorts]]-=pmodel1
                        pmodelp[closefilter[0][shorts]]+=pmodel1
                        

                if np.max(data['pexptime'][closefilter[0]]) >= tenminutes:
                    longs=np.array(np.where(data['pexptime'][closefilter[0]] >= tenminutes))
                    maxexp=np.max(data['pexptime'][closefilter[0]])
                    ttemp=np.arange(np.min(phased[closefilter[0][longs]])-maxexp*2., np.max(phased[closefilter[0][longs]])+maxexp*2., np.min(data['pexptime'][closefilter[0][longs]])/100.)
                    m2l=batman.TransitModel(params,ttemp,nthreads=1)
                    ftemp=m2l.light_curve(params)
                    pmodel1=resampler(ftemp, ttemp, phased[closefilter[0][longs[0]]], data['pexptime'][closefilter[0][longs[0]]])-1.
                    mresid[closefilter[0][longs]]-=pmodel1
                    if args.gp: 
                        mresidgp[closefilter[0][longs]]-=pmodel1
                        pmodelp[closefilter[0][longs]]+=pmodel1


            if not args.binary:
                offset=1.95*(parstruc['rprs'+str(i+1)]**2)
            else:
                offset=0.15
            chere=filtercolor(struc1['photname'+str(j+1)],args)

            if args.gp and len(closedatasetg[0]) > 1:
                mu, var = gp.predict(mresid[useforgp],t=data['ptime'][closedatasetg[0]], return_var=True)

                mresid[closedatasetg[0]]-=mu

                pl.plot(phased[closedatasetg]*24.,np.array(data['pflux'][closedatasetg], dtype=float)-mu-offset*j+1.,'.',color=chere)
            else:
                pl.plot(phased[closefilter]*24.,np.array(data['pflux'][closefilter], dtype=float)-offset*j,'.',color=chere)
                mresid[closefilter]-=1.
            
            if struc1['photname'+str(j+1)] == 'blank': chere='red'
            pl.plot(modphase*24.,flux-offset*j,color='black',linewidth=2)
            if any('photlabel'+str(j+1) in s for s in index): pl.text(np.min(phased[closes]*24.),1.002-offset*j,struc1['photlabel'+str(j+1)].replace(':',' '),fontsize=16,weight='medium') 
        
        if pndatasets > 1: pl.ylim([1.-offset*np.float(pndatasets+0.25),1.005])
        pl.ylabel('normalized flux', fontsize=14)
        if args.plotresids:
            pl.setp(ax1.get_xticklabels(), visible=False)
            ax2=pl.subplot(gs[1], sharex=ax1)
            for j in range (0,pndatasets):
                closefilter=np.where((np.abs(phased) <= (dur[i]/2.)*1.5) & (data['pdataset'] == j+1))
                pl.plot(phased[closefilter]*24.,mresid[closefilter],'.',color=filtercolor(struc1['photname'+str(j+1)],args))
            pl.plot([np.min(phased[closes]*24.),np.max(phased[closes]*24.)],[0,0],color='black',linewidth=2)
            pl.ylabel('normalized flux',fontsize=10)

        if not args.binary:
            pl.xlim([(-1.)*(dur[i]/2.)*1.5*24.,(dur[i]/2.)*1.5*24.])
        else:
            pl.xlim([-3.,3.])
        pl.xlabel('time from center of transit (hours)', fontsize=16)

        pl.subplots_adjust(hspace=0)
        pl.savefig(pp, format='pdf')
        pp.close()
        pl.clf()
        print 'The light curve plot for planet '+str(i+1)+' is complete at '+sysname+'LC'+str(i+1)+'.pdf'

        #make full GP light curve plot
    if args.gp and any('gpmodtypep' in s for s in data['gpmodtype']):
        pp = PdfPages(sysname+'GPLC.pdf')
        fig=pl.figure(figsize=(10,3.5))
        pl.plot(data['ptime'][useforgp],data['pflux'][useforgp],'.',color='gray')
        mu, var = gp.predict(mresidgp[useforgp], return_var=True)
        pl.plot(data['ptime'][useforgp],mu+pmodelp[useforgp],color='red')
        pl.xlabel(timestandard+'-'+str(int(np.round(timeoffset))), fontsize=16)
        pl.ylabel('normalized flux', fontsize=16)
        pl.xlim([np.min(data['ptime'][useforgp]),np.max(data['ptime'][useforgp])])
        pl.tight_layout()
            
        pl.savefig(pp, format='pdf')
        print 'The GP light curve plot is complete at '+sysname+'GPLC.pdf'
        pp.close()
        pl.clf()
        
        goodtransit=1
        epoch0=0
        while goodtransit == 1:
            pp = PdfPages(sysname+'transit'+str(epoch0)+'.pdf')
            fig=pl.figure(figsize=(5.0,3.5))
            center=parstruc['epoch1']+epoch0*parstruc['Per1']
            pl.plot((data['ptime'][useforgp]-center)*24.,data['pflux'][useforgp],'.',color='gray')
            pl.plot((data['ptime'][useforgp]-center)*24.,mu+pmodelp[useforgp],color='red')
            pl.xlim([dur*(-24.),dur*24.])
            pl.xlabel('hours from center of transit',fontsize=14)
            pl.ylabel('normalized flux',fontsize=14)
            pl.tight_layout()
            pl.savefig(pp, format='pdf')
            pp.close()
            pl.clf()
            print 'A plot of transit ',(epoch0+1),' is complete at ',sysname+'transit'+str(epoch0)+'.pdf'
            centerplus=parstruc['epoch1']+(epoch0+1)*parstruc['Per1']
            nnext=np.where(np.abs(data['ptime'][useforgp]-centerplus) <= dur)
            if len(nnext[0]) < 5: 
                goodtransit=0
            else:
                epoch0+=1


if args.bw:
    rvcolors=['black','black','black','black','black']
else:
    rvcolors=['red','blue','green','purple','orange']
rvsymbols=['o','^','s','v','D']


if args.tomography:
    ntomsets=data['tomdict']['ntomsets']
    dur=np.array(dur,dtype=np.float)
    ntexptohere=0
    if ntomsets > 1:
        profarrbig, profarrerrbig, modelbig, residsbig = np.zeros((data['tomdict']['nexptot'],len(tomdict['vabsfine'+str(tomdict['whichvabsfinemax'])]))), np.zeros((data['tomdict']['nexptot'],len(tomdict['vabsfine'+str(tomdict['whichvabsfinemax'])]))), np.zeros((data['tomdict']['nexptot'],len(tomdict['vabsfine'+str(tomdict['whichvabsfinemax'])]))), np.zeros((data['tomdict']['nexptot'],len(tomdict['vabsfine'+str(tomdict['whichvabsfinemax'])])))
        tphasebig, tphase2big = 0, 0
        vabsfinebig=data['tomdict']['vabsfine'+str(tomdict['whichvabsfinemax'])]
    for i in range(0,ntomsets):
        whichplanet=int(data['tomdict']['whichplanet'+str(i+1)])
        tomphase=np.mod(data['tomdict']['ttime'+str(i+1)]-parstruc['epoch'+str(whichplanet)], parstruc['Per'+str(whichplanet)])
        highphase=np.where(tomphase >= parstruc['Per'+str(whichplanet)]/2.)
        tomphase[highphase]-=parstruc['Per'+str(whichplanet)]
        tomphase*=24.0*60.0 #to minutes
        texptime=data['tomdict']['texptime'+str(i+1)]/60.0 #to minutes
        if not any('tomdrift' in s for s in inposindex):
            tomins=np.where(np.abs(tomphase) <= dur[whichplanet-1]*24.*60.*1.25)
            tomouts=np.where(np.abs(tomphase) > dur[whichplanet-1]*24.*60.*1.25)
        else:
            tomins=np.isfinite(tomphase)
            tomouts=0
        horusstruc = {'vsini': parstruc['vsini'], 'sysname': struc1['sysname'], 'obs': struc1['obs'+str(i+1)], 'vabsfine': data['tomdict']['vabsfine'+str(i+1)], 'Pd': parstruc['Per'+str(whichplanet)], 'lambda': parstruc['lambda'+str(whichplanet)], 'b': parstruc['bpar'+str(whichplanet)], 'rplanet': parstruc['rprs'+str(whichplanet)], 't': tomphase[tomins], 'times': texptime[tomins], 'e': ecc[whichplanet-1], 'periarg': omega[whichplanet-1]*np.pi/180., 'a': aors[whichplanet-1]}
        
        if args.line and any('linecenter' in s for s in inposindex): 
            horusstruc['vabsfine']+=parstruc['linecenter']
        if any('q1t' in s for s in inposindex) and any('q2t' in s for s in inposindex):
            horusstruc['gamma1'], horusstruc['gamma2'] = 2.0*parstruc['q2t']*np.sqrt(parstruc['q1t']), np.sqrt(parstruc['q1t'])*(1.0-2.0*parstruc['q2t'])
        else:
            horusstruc['gamma1'], horusstruc['gamma2'] = parstruc['g1t'], parstruc['g2t']
        if any('intwidth' in s for s in inposindex):
            horusstruc['width'] = parstruc['intwidth']
        elif any('intwidth' in s for s in index):
            horusstruc['width'] = np.float(struc1['intwidth'])
        else:
            horusstruc['width'] = 10.


        if any('tomdrift' in s for s in inposindex):
            lineshift=(data['tomdict']['ttime'+str(i+1)]-data['tomdict']['ttime'+str(i+1)][0])*parstruc['tomdriftl'+str(i+1)]+parstruc['tomdriftc'+str(i+1)]
            horusstruc['lineshifts'] = lineshift
            outstruc = horus.model(horusstruc, resnum=50,lineshifts='y')
        else:
            outstruc = horus.model(horusstruc, resnum=50)
        profarr1, baseline, basearr = outstruc['profarr'], outstruc['baseline'], outstruc['basearr']
        tnexp = len(data['tomdict']['ttime'+str(i+1)])
        nvabsfine=len(data['tomdict']['vabsfine'+str(i+1)])
        vabsfine=data['tomdict']['vabsfine'+str(i+1)]
        model=np.zeros((tnexp, nvabsfine))
        model[tomins,:]=profarr1
        if tomouts:
            for j in range (0, len(tomouts[0])): model[tomouts[0][j],:]=outstruc['basearr'][0,:]
        if args.skyline:
            skyline=horus.mkskyline(vabsfine,parstruc['skydepth'],parstruc['skycen'],struc1['obs'+str(i+1)])
            for j in range (np.int(struc1['skyfirst']), np.int(struc1['skylast'])+1):
                model[j,:]+=skyline

        vgood=np.where(np.abs(vabsfine) <= np.float(struc1['vsini'])*1.25)
        #GP stuff will go here when implement it
        profarrflat=np.zeros((tnexp,nvabsfine))
        modelflat=np.zeros((tnexp,nvabsfine))
        tphase, tphase2=np.mod((data['tomdict']['ttime'+str(i+1)]-parstruc['epoch'+str(whichplanet)]),parstruc['Per'+str(whichplanet)]),np.mod((data['tomdict']['ttime'+str(i+1)]+data['tomdict']['texptime'+str(i+1)]/(60.*60.*24.)-parstruc['epoch'+str(whichplanet)]),parstruc['Per'+str(whichplanet)])
        bads=np.where(tphase > parstruc['Per'+str(whichplanet)]/2.)
        tphase[bads[0]]-=parstruc['Per'+str(whichplanet)]
        bads=np.where(tphase2 > parstruc['Per'+str(whichplanet)]/2.)
        tphase2[bads[0]]-=parstruc['Per'+str(whichplanet)]
        tphase += dur[whichplanet-1]/2.
        tphase2 += dur[whichplanet-1]/2.
        tphase /= dur[whichplanet-1]
        tphase2 /= dur[whichplanet-1]
        for j in range (0, tnexp):
            profarrflat[j,:]=data['tomdict']['profarr'+str(i+1)][j,:]-outstruc['basearr'][0,:]*(-1.)
            modelflat[j,:]=model[j,:]+outstruc['basearr'][0,:]*(-1.)


        if ntomsets == 1: 
            stack1, stack2, stack3 = 1, 1, 0
        else:
            if j == 0: stack1, stack2, stack3 = 1, 1, 0
            if j > 0: stack1, stack2, stack3 = 3, 3, 2
            tphasebig, tphase2big = np.append(tphasebig, tphase), np.append(tphase2big, tphase2)
            if j == tomdict['whichvabsfinemax']:
                profarrbig[ntexptohere:ntexptohere+tnexp,:], profarrerrbig[ntexptohere:ntexptohere+tnexp,:], modelbig[ntexptohere:ntexptohere+tnexp,:], residsbig[ntexptohere:ntexptohere+tnexp,:] = profarrflat, data['tomdict']['profarrerr'+str(i+1)], modelflat, (data['tomdict']['profarr'+str(i+1)]-(model*(-1.)))*(-1.)
            else:
                for k in range (0,tnexp):
                    profarrbig[ntexptohere+k,:]=np.interp(vabsfinebig,vabsfine,profarrflat[k,:])
                    profarrerrbig[ntexptohere+k,:]=np.interp(vabsfinebig,vabsfine,data['tomdict']['profarrerr'+str(i+1)][k,:])
                    modelbig[ntexptohere+k,:]=np.interp(vabsfinebig,vabsfine,modelflat[k,:])
                    residsbig[ntexptohere+k,:]=np.interp(vabsfinebig,vabsfine,(data['tomdict']['profarr'+str(i+1)][k,:]-model[k,:]*(-1.))*(-1.))
        dtutils.mktslprplot(profarrflat*(-1.),data['tomdict']['profarrerr'+str(i+1)],data['tomdict']['vabsfine'+str(i+1)],tphase,tphase2,parstruc['vsini'],parstruc['bpar'+str(whichplanet)],parstruc['rprs'+str(whichplanet)],filename=sysname+'DTdata'+str(i+1)+'.pdf',stack=stack1)
        dtutils.mktslprplot(modelflat,data['tomdict']['profarrerr'+str(i+1)],data['tomdict']['vabsfine'+str(i+1)],tphase,tphase2,parstruc['vsini'],parstruc['bpar'+str(whichplanet)],parstruc['rprs'+str(whichplanet)],filename=sysname+'DTmodel'+str(i+1)+'.pdf',stack=stack2,weighted=False)
        dtutils.mktslprplot(data['tomdict']['profarr'+str(i+1)]*(-1.)-model,data['tomdict']['profarrerr'+str(i+1)],data['tomdict']['vabsfine'+str(i+1)],tphase,tphase2,parstruc['vsini'],parstruc['bpar'+str(whichplanet)],parstruc['rprs'+str(whichplanet)],filename=sysname+'DTresids'+str(i+1)+'.pdf',stack=stack3)
        ntexptohere+=tnexp


    if ntomsets > 1:
        stack1, stack2, stack3 = 3, 3, 2
        tphasebig, tphase2big = tphasebig[1:], tphase2big[1:]
        dtutils.mktslprplot(profarrbig*(-1.),profarrerrbig,vabsfinebig,tphasebig,tphase2big,parstruc['vsini'],parstruc['bpar'+str(whichplanet)],parstruc['rprs'+str(whichplanet)],filename=sysname+'DTdataall.pdf',stack=stack1)
        dtutils.mktslprplot(modelbig,profarrerrbig,vabsfinebig,tphasebig,tphase2big,parstruc['vsini'],parstruc['bpar'+str(whichplanet)],parstruc['rprs'+str(whichplanet)],filename=sysname+'DTmodelall.pdf',stack=stack2,weighted=False)
        dtutils.mktslprplot(residsbig,profarrerrbig,vabsfinebig,tphasebig,tphase2big,parstruc['vsini'],parstruc['bpar'+str(whichplanet)],parstruc['rprs'+str(whichplanet)],filename=sysname+'DTresidsall.pdf',stack=stack3)

if args.rvs:
    #draw 50 random chains
    pp = PdfPages(sysname+'RVphased'+str(i+1)+'.pdf')
    rands=np.random.random_integers(0,len(samples[:,0]),50)
    fig=pl.figure(figsize=(8.0,6.0))
    if args.plotresids: rresids=np.array(data['rv'])
    if args.rvlegend:
        rvlabels=np.array(invals[[i for i, s in enumerate(index) if 'rvlabel' in s]], dtype=str)
    else:
        rvlabels=np.zeros(rndatasets,dtype=str)
    for j in range (0,50):

        if any('sinw' in s for s in inposindex) or any('ecc' in s for s in inposindex):
            if any('sesinw' in s for s in inposindex):
                ecc1=samples[rands[j],[i for i, s in enumerate(inposindex) if 'sesinw' in s]]**2+samples[rands[j],[i for i, s in enumerate(inposindex) if 'secosw' in s]]**2
                omega1=np.arccos(samples[rands[j],[i for i, s in enumerate(inposindex) if 'secosw' in s]]/np.sqrt(ecc1))
                if any(not np.isfinite(t) for t in omega1):
                    bads=np.where(np.isfinite(omega1) == True)
                    omega1[bads]=np.pi/2.0
                    temp=samples[rands[j],[i for i, s in enumerate(inposindex) if 'sesinw' in s]]
                    ecc1[bads]=temp[bads]**2

            if any('ecc' in s for s in inposindex):
                ecc1=samples[rands[j],[i for i, s in enumerate(inposindex) if 'ecc' in s]]
                omega1=samples[rands[j],[i for i, s in enumerate(inposindex) if 'omega' in s]]

            if any('ecsinw' in s for s in inposindex):
                ecc1=np.sqrt(samples[rands[j],[i for i, s in enumerate(inposindex) if 'ecsinw' in s]]**2+samples[rands[j],[i for i, s in enumerate(inposindex) if 'eccosw' in s]]**2)
                omega1=np.arccos(samples[rands[j],[i for i, s in enumerate(inposindex) if 'eccosw' in s]]/ecc1)
                if any(not np.isfinite(t) for t in omega1):
                    bads=np.where(np.isfinite(omega1) == True)
                    omega1[bads]=np.pi/2.0
                    temp=samples[rands[j],[i for i, s in enumerate(inposindex) if 'ecsinw' in s]]
                    ecc1[bads]=temp[bads]

            omega1*=180./np.pi #radians->degrees

        else:
            ecc1=np.zeros(nplanets)
            omega1=np.zeros(nplanets)+90.



        for i in range (0,nplanets):
            
            if args.plotresids and j == 0: 
                if i == 0 and not args.photometry: import matplotlib.gridspec as gridspec
                gs=gridspec.GridSpec(2, 1, height_ratios=[4,1])
                ax1=pl.subplot(gs[0])

            P1 = np.float(samples[rands[j],np.where(inposindex == 'Per'+str(i+1))])

            if args.photometry:
                ftransit=np.pi/2.-omega1[i]*np.pi/180.
                Etransit=2.*np.arctan(np.sqrt((1.-ecc1[i])/(1.+ecc1[i]))*np.tan(ftransit/2.))
                timesince=P1/(2.*np.pi)*(Etransit-ecc1[i]*np.sin(Etransit))
            else:
                timesince=0.0

            
            if not args.binary:
                K1=np.float(samples[rands[j],np.where(inposindex == 'semiamp'+str(i+1))])
            else:
                K1, K2 = np.float(samples[rands[j],np.where(inposindex == 'semiamp'+str(i+1)+'a')]), np.float(samples[rands[j],np.where(inposindex == 'semiamp'+str(i+1)+'b')])
            rphase=np.linspace(0.,P1,1000)
            rmodel1=radvel.kepler.rv_drive(rphase,np.array([P1,np.float(timesince)*(-1.),ecc1[i],omega1[i]*np.pi/180.,K1]))
            pl.plot(rphase,rmodel1,color='gray',alpha=0.5)  
            pl.plot(rphase,rmodel1,color='gray',alpha=0.5) 
            if args.binary: 
                rmodel2=radvel.kepler.rv_drive(rphase,np.array([P1,np.float(timesince)*(-1.),ecc1[i],omega1[i]*np.pi/180.-np.pi,K2]))
                pl.plot(rphase,rmodel2,color='gray',alpha=0.5) 
            
            if j == 49:
                if args.photometry:
                    ftransit=np.pi/2.-omega[i]*np.pi/180.
                    Etransit=2.*np.arctan(np.sqrt((1.-ecc[i])/(1.+ecc[i]))*np.tan(ftransit/2.))
                    timesince=parstruc['Per'+str(i+1)]/(2.*np.pi)*(Etransit-ecc[i]*np.sin(Etransit))
                else:
                    timesince=0.0
                rphase=np.linspace(0.,parstruc['Per'+str(i+1)],1000)
                if not args.binary:
                    rmodel=radvel.kepler.rv_drive(rphase,np.array([parstruc['Per'+str(i+1)],timesince*(-1.),ecc[i],omega[i]*np.pi/180.,parstruc['semiamp'+str(i+1)]]))
                else:
                    rmodel=radvel.kepler.rv_drive(rphase,np.array([parstruc['Per'+str(i+1)],timesince*(-1.),ecc[i],omega[i]*np.pi/180.,parstruc['semiamp'+str(i+1)+'a']]))
                    rmodel2=radvel.kepler.rv_drive(rphase,np.array([parstruc['Per'+str(i+1)],timesince*(-1.),ecc[i],omega[i]*np.pi/180.-np.pi,parstruc['semiamp'+str(i+1)+'b']]))

                if args.plotresids: 
                    if not args.binary:
                        temp=radvel.kepler.rv_drive(data['rtime'],np.array([parstruc['Per'+str(i+1)],parstruc['epoch'+str(i+1)]-timesince,ecc[i],omega[i]*np.pi/180.,parstruc['semiamp'+str(i+1)]]))
                        rresids-=temp
                    else:
                        temp=radvel.kepler.rv_drive(data['rtime'],np.array([parstruc['Per'+str(i+1)],parstruc['epoch'+str(i+1)]-timesince,ecc[i],omega[i]*np.pi/180.,parstruc['semiamp'+str(i+1)+'a']]))
                        temp2=radvel.kepler.rv_drive(data['rtime'],np.array([parstruc['Per'+str(i+1)],parstruc['epoch'+str(i+1)]-timesince,ecc[i],omega[i]*np.pi/180.-np.pi,parstruc['semiamp'+str(i+1)+'b']]))
                        rresids-=np.append(temp,temp2)
                    
                    

                pl.plot(rphase,rmodel,color='black',linewidth=2)
                

                if args.fitjitter: jitters=np.zeros(len(data['rdataset']))
                if args.binary:
                    pl.plot(rphase,rmodel2,color='black',linewidth=2)
                    allrvtimes, allrvdatasets = np.append(data['rtime'],data['rtime']), np.append(data['rdataset'], data['rdataset'])
                else:
                    allrvtimes, allrvdatasets = data['rtime'], data['rdataset']
                for k in range (0,rndatasets):
                    thisdataset=np.where(allrvdatasets == k+1)
                    if args.fitjitter: 
                        jitters[thisdataset]=parstruc['jitter'+str(k+1)]
                        data['rverror'][thisdataset]=np.sqrt(data['rverror'][thisdataset]**2+jitters[thisdataset]**2)
                    if any('gamma' in s for s in inposindex):
                        if any('rvtrend' in s for s in inposindex):
                            pl.plot(np.mod(allrvtimes[thisdataset]-parstruc['epoch'+str(i+1)],parstruc['Per'+str(i+1)]),data['rv'][thisdataset]-parstruc['gamma'+str(k+1)]-(allrvtimes[thisdataset]-parstruc['epoch1'])*parstruc['rvtrend'],rvsymbols[k],color=rvcolors[k])
                            pl.errorbar(np.mod(allrvtimes[thisdataset]-parstruc['epoch'+str(i+1)],parstruc['Per'+str(i+1)]),data['rv'][thisdataset]-parstruc['gamma'+str(k+1)]-(allrvtimes[thisdataset]-parstruc['epoch1'])*parstruc['rvtrend'], yerr=data['rverror'][thisdataset],fmt='none',ecolor=rvcolors[k])
                        else:
                            pl.plot(np.mod(allrvtimes[thisdataset]-parstruc['epoch'+str(i+1)],parstruc['Per'+str(i+1)]),data['rv'][thisdataset]-parstruc['gamma'+str(k+1)],rvsymbols[k],color=rvcolors[k])
                            pl.errorbar(np.mod(allrvtimes[thisdataset]-parstruc['epoch'+str(i+1)],parstruc['Per'+str(i+1)]),data['rv'][thisdataset]-parstruc['gamma'+str(k+1)], yerr=data['rverror'][thisdataset],fmt='none',ecolor=rvcolors[k])
                    else:
                        if any('rvtrend' in s for s in inposindex):
                            pl.plot(np.mod(allrvtimes[thisdataset]-parstruc['epoch'+str(i+1)],parstruc['Per'+str(i+1)]),data['rv'][thisdataset]-struc1['gamma'+str(k+1)]-(allrvtimes[thisdataset]-parstruc['epoch1'])*parstruc['rvtrend'],rvsymbols[k],color=rvcolors[k])
                            pl.errorbar(np.mod(allrvtimes[thisdataset]-parstruc['epoch'+str(i+1)],parstruc['Per'+str(i+1)]),data['rv'][thisdataset]-struc1['gamma'+str(k+1)]-(allrvtimes[thisdataset]-parstruc['epoch1'])*parstruc['rvtrend'], yerr=data['rverror'][thisdataset],fmt='none',ecolor=rvcolors[k])
                        else:
                            pl.plot(np.mod(allrvtimes[thisdataset]-parstruc['epoch'+str(i+1)],parstruc['Per'+str(i+1)]),data['rv'][thisdataset]-struc1['gamma'+str(k+1)],rvsymbols[k],color=rvcolors[k])
                            pl.errorbar(np.mod(allrvtimes[thisdataset]-parstruc['epoch'+str(i+1)],parstruc['Per'+str(i+1)]),data['rv'][thisdataset]-struc1['gamma'+str(k+1)], yerr=data['rverror'][thisdataset],fmt='none',ecolor=rvcolors[k])


                if args.rvlegend:
                    pl.legend()

                if args.ms:
                    pl.ylabel('RV (m s$^{-1}$)',fontsize=16)
                else:
                    pl.ylabel('RV (km s$^{-1}$)',fontsize=16)

                if args.plotresids:
                    pl.setp(ax1.get_xticklabels(), visible=False)
                    ax2=pl.subplot(gs[1], sharex=ax1)

                    for k in range (0,rndatasets):
                        pl.plot([0,parstruc['Per'+str(i+1)]],[0,0],linewidth=2,color='black')
                        thisdataset=np.where(data['rdataset'] == k+1)
                        if any('gamma' in s for s in inposindex):
                            if any('rvtrend' in s for s in inposindex):
                                pl.plot(np.mod(allrvtimes[thisdataset]-parstruc['epoch1'],parstruc['Per1']),rresids[thisdataset]-parstruc['gamma'+str(k+1)]-(allrvtimes[thisdataset]-parstruc['epoch1'])*parstruc['rvtrend'],rvsymbols[k],color=rvcolors[k])
                                pl.errorbar(np.mod(allrvtimes[thisdataset]-parstruc['epoch1'],parstruc['Per1']),rresids[thisdataset]-parstruc['gamma'+str(k+1)]-(allrvtimes[thisdataset]-parstruc['epoch1'])*parstruc['rvtrend'], yerr=data['rverror'][thisdataset],fmt='none',ecolor=rvcolors[k])
                            else:
                                pl.plot(np.mod(allrvtimes[thisdataset]-parstruc['epoch1'],parstruc['Per1']),rresids[thisdataset]-parstruc['gamma'+str(k+1)],rvsymbols[k],color=rvcolors[k])
                                pl.errorbar(np.mod(allrvtimes[thisdataset]-parstruc['epoch1'],parstruc['Per1']),rresids[thisdataset]-parstruc['gamma'+str(k+1)], yerr=data['rverror'][thisdataset],fmt='none',ecolor=rvcolors[k])
                        else:
                            if any('rvtrend' in s for s in inposindex):
                                pl.plot(np.mod(allrvtimes[thisdataset]-parstruc['epoch1'],parstruc['Per1']),rresids[thisdataset]-struc1['gamma'+str(k+1)]-(allrvtimes[thisdataset]-parstruc['epoch1'])*parstruc['rvtrend'],rvsymbols[k],color=rvcolors[k])
                                pl.errorbar(np.mod(allrvtimes[thisdataset]-parstruc['epoch1'],parstruc['Per1']),rresids[thisdataset]-struc1['gamma'+str(k+1)]-(allrvtimes[thisdataset]-parstruc['epoch1'])*parstruc['rvtrend'], yerr=data['rverror'][thisdataset],fmt='none',ecolor=rvcolors[k])
                            else:
                                pl.plot(np.mod(allrvtimes[thisdataset]-parstruc['epoch1'],parstruc['Per1']),rresids[thisdataset]-struc1['gamma'+str(k+1)],rvsymbols[k],color=rvcolors[k])
                                pl.errorbar(np.mod(allrvtimes[thisdataset]-parstruc['epoch1'],parstruc['Per1']),rresids[thisdataset]-struc1['gamma'+str(k+1)], yerr=data['rverror'][thisdataset],fmt='none',ecolor=rvcolors[k])
                
                    if args.ms:
                        pl.ylabel('RV (m s$^{-1}$)',fontsize=16)
                    else:
                        pl.ylabel('RV (km s$^{-1}$)',fontsize=16)

            pl.xlim([0,parstruc['Per'+str(i+1)]])
            pl.xlabel('phase (days)',fontsize=16)
                

    pl.subplots_adjust(hspace=0)
    pl.savefig(pp, format='pdf')
    pp.close()
    pl.clf()
    print 'The phased RV plot for planet '+str(i+1)+' is complete at '+sysname+'RVphased'+str(i+1)+'.pdf'

    pp = PdfPages(sysname+'RVall'+str(i+1)+'.pdf')
    fig=pl.figure(figsize=(10,3.5))


    
    for i in range(0,nplanets):
        if args.photometry:
            ftransit=np.pi/2.-omega[i]*np.pi/180.
            Etransit=2.*np.arctan(np.sqrt((1.-ecc[i])/(1.+ecc[i]))*np.tan(ftransit/2.))
            timesince=parstruc['Per'+str(i+1)]/(2.*np.pi)*(Etransit-ecc[i]*np.sin(Etransit))
        else:
            timesince=0.0
        rbjds=np.linspace(np.min(data['rtime'])-10.,np.max(data['rtime'])+10.,10000)
        if not args.binary:
            rmodel=radvel.kepler.rv_drive(rbjds,np.array([parstruc['Per'+str(i+1)],parstruc['epoch'+str(i+1)]-timesince,ecc[i],omega[i]*np.pi/180.,parstruc['semiamp'+str(i+1)]]))
        else:
            rmodel=radvel.kepler.rv_drive(rbjds,np.array([parstruc['Per'+str(i+1)],parstruc['epoch'+str(i+1)]-timesince,ecc[i],omega[i]*np.pi/180.,parstruc['semiamp'+str(i+1)+'a']]))
            rmodel2=radvel.kepler.rv_drive(rbjds,np.array([parstruc['Per'+str(i+1)],parstruc['epoch'+str(i+1)]-timesince,ecc[i],omega[i]*np.pi/180.-np.pi,parstruc['semiamp'+str(i+1)+'b']]))
        if any('rvtrend' in s for s in inposindex):
            rmodel+=(rbjds-(parstruc['epoch1']))*parstruc['rvtrend']
            if args.binary: rmodel2+=(rbjds-(parstruc['epoch1']))*parstruc['rvtrend']
        pl.plot(rbjds,rmodel,color='black')
        for k in range (0,rndatasets):
            thisdataset=np.where(data['rdataset'] == k+1)
            if any('gamma' in s for s in inposindex):
                pl.plot(allrvtimes[thisdataset],data['rv'][thisdataset]-parstruc['gamma'+str(k+1)],rvsymbols[k],color=rvcolors[k])
                pl.errorbar(allrvtimes[thisdataset],data['rv'][thisdataset]-parstruc['gamma'+str(k+1)], yerr=data['rverror'][thisdataset],fmt='none',ecolor=rvcolors[k])

            else:
                pl.plot(allrvtimes[thisdataset],data['rv'][thisdataset]-struc1['gamma'+str(k+1)],rvsymbols[k],color=rvcolors[k])
                pl.errorbar(allrvtimes[thisdataset],data['rv'][thisdataset]-struc1['gamma'+str(k+1)], yerr=data['rverror'][thisdataset],fmt='none',ecolor=rvcolors[k])
            
        if args.rvlegend:
            pl.legend()
        if offset == 0.:
            pl.xlabel(timestandard,fontsize=14)
        else:
            pl.xlabel(timestandard+'-'+str(np.int(np.round(timeoffset))),fontsize=14)
        if args.ms:
            pl.ylabel('RV (m s$^{-1}$)',fontsize=14)
        else:
            pl.ylabel('RV (km s$^{-1}$)',fontsize=14)
        pl.xlim([np.min(rbjds),np.max(rbjds)])
        pl.tight_layout()

    pl.savefig(pp, format='pdf')
    pp.close()
    pl.clf()

    print 'The all RV plot for planet '+str(i+1)+' is complete at '+sysname+'RVall'+str(i+1)+'.pdf'

if args.makereport:
    
    f=open(sysname+'_report.tex','w')
    f.write('\\documentclass{article} \n')
    f.write('\\usepackage{graphicx} \n')
    f.write('\\usepackage{graphics} \n')
    f.write('\\begin{document} \n')
    for k in range (0,nplanets):
        f.write('\\input{'+sysname+'_table'+str(k+1)+'} \n')
    f.write('\\newpage \n')
    if not args.tableonly:
        if args.photometry:
            for k in range (0,nplanets):
                f.write('\\begin{figure} \n')
                f.write('\\centering \n')
                f.write('\\includegraphics[width=1.0\\textwidth]{'+sysname+'LC'+str(k+1)+'.pdf} \n')
                f.write('\\caption{Phased light curve of planet '+str(k+1)+'.} \n')
                f.write('\\end{figure} \n')
            if args.gp and any('gpmodtypep' in s for s in data['gpmodtype']):
                f.write('\\begin{figure} \n')
                f.write('\\centering \n')
                f.write('\\includegraphics[width=1.0\\textwidth]{'+sysname+'GPLC.pdf} \n')
                f.write('\\caption{Gaussian process fit to the full light curve.} \n')
                f.write('\\end{figure} \n')

        if args.rvs:
            for i in range (0,nplanets):
                f.write('\\begin{figure} \n')
                f.write('\\centering \n')
                f.write('\\includegraphics[width=1.0\\textwidth]{'+sysname+'RVphased'+str(i+1)+'.pdf} \n')
                f.write('\\caption{RV measurements phased to the orbit of planet '+str(i+1)+'.} \n')
                f.write('\\end{figure} \n')

            f.write('\\begin{figure} \n')
            f.write('\\centering \n')
            f.write('\\includegraphics[width=1.0\\textwidth]{'+sysname+'RVall1.pdf} \n')
            f.write('\\caption{All RV measurements along with the best-fit models for all planets.} \n')
            f.write('\\end{figure} \n')
            

        if args.tomography:
            for i in range(0,ntomsets):
                f.write('\\begin{figure} \n')
                f.write('\\centering \n')
                f.write('\\includegraphics[width=1.0\\textwidth]{'+sysname+'DTdata'+str(i+1)+'.pdf} \\\ \n')
                f.write('\\includegraphics[width=1.0\\textwidth]{'+sysname+'DTmodel'+str(i+1)+'.pdf} \\\ \n')
                f.write('\\includegraphics[width=1.0\\textwidth]{'+sysname+'DTresids'+str(i+1)+'.pdf} \\\ \n')
                f.write('\\caption{Doppler tomographic data from dataset '+str(i+1)+'. Top: data. Middle: model. Bottom: residuals.} \n')
                f.write('\\end{figure} \n')
            if ntomsets > 1:
                f.write('\\begin{figure} \n')
                f.write('\\centering \n')
                f.write('\\includegraphics[width=1.0\\textwidth]{'+sysname+'DTdataall.pdf} \\\ \n')
                f.write('\\includegraphics[width=1.0\\textwidth]{'+sysname+'DTmodelall.pdf} \\\ \n')
                f.write('\\includegraphics[width=1.0\\textwidth]{'+sysname+'DTresidsall.pdf} \\\ \n')
                f.write('\\caption{Combined Doppler tomographic data. Top: data. Middle: model. Bottom: residuals.} \n')
                f.write('\\end{figure} \n')

        if args.corner:
            f.write('\\begin{figure} \n')
            f.write('\\centering \n')
            f.write('\\includegraphics[width=1.0\\textwidth]{'+sysname+'_corner.pdf} \n')
            f.write('\\caption{Corner plot for the MCMC.} \n')
            f.write('\\end{figure} \n')
                    
                    

    f.write('\\end{document}')
    f.close()
        
            
    os.system('pdflatex '+sysname+'_report.tex')
    os.system('pdflatex '+sysname+'_report.tex')
        

    
                
        
            
