#MISTTBORN: the MCMC Interface for Synthesis of Transits, Tomography, Binaries, and Others of a Relevant Nature
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
import os
import multiprocessing
import time as timemod
thewholestart=timemod.time()

#add command line arguments
parser=argparse.ArgumentParser()
parser.add_argument("infile", type=str, help="name of the input file")
parser.add_argument("-p", "--photometry", action="store_true", help="perform photometric analysis")
parser.add_argument("-r", "--rvs", action="store_true", help="perform radial velocity analysis")
parser.add_argument("-t", "--tomography", action="store_true", help="perform Doppler tomographic analysis")
parser.add_argument("-l", "--line", action="store_true", help="fit a rotationally broadened model to a single spectral line")
parser.add_argument("-v", "--verbose", action="store_true", help="print a short message every MCMC step")
parser.add_argument("-g", "--gp", action="store_true", help="enable Gaussian process regression")
parser.add_argument("-b", "--binary", action="store_true", help="fit a binary star rather than an exoplanet: two sets of RVs, primary and secondary eclipses")
parser.add_argument("--startnew", action="store_true", help="start a new chain regardless of whether the given output files already exist")
parser.add_argument("--plotbest", action="store_true", help="plot the best-fit model from the input chain file. Will not run a full chain.")
parser.add_argument("--plotstep", action="store_true", help="plot the current model every step. Very slow, mostly useful for debugging. This will only work for 1 thread.")
parser.add_argument("--ploterrors", action="store_true", help="include error bars on the plot.")
parser.add_argument("--plotresids", action="store_true", help="include residuals for the plots.")
parser.add_argument("--bestprob", action="store_true", help="Plot the values for the best-fit model rather than the posterior median.")
parser.add_argument("--time", action="store_true", help="calculate and print the elapsed time for each model call")
parser.add_argument("--getprob", action="store_true", help="print the contributions to lnprob from each dataset and priors")
parser.add_argument("--fullcurve", action="store_true", help="make a model lightcurve that will cover the full transit; call only with --plotbest. WARNING: doesn't totally work right, use at own risk.")
parser.add_argument("--skyline", action="store_true", help="include a sky line in some or all of the tomographic data set")
parser.add_argument("--ttvs", action="store_true", help="account for TTVs in the photometric fit")
parser.add_argument("--dilution", action="store_true", help="account for dilution due to another star in the aperture")
parser.add_argument("--pt", action="store_true", help="Use emcee's parallel tempered ensemble sampler")
args=parser.parse_args()



infile=args.infile

#import packages needed for specific fits

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
        print 'exiting now'
        sys.exit()
        
    print 'burning pewter to perform RV analysis'

if args.binary:
    print 'burning steel to analyze a stellar binary'

if args.tomography or args.line:
    try:
        import horus
    except ImportError:
        print 'horus does not appear to be installed correctly.'
        print 'it is available from https://github.com/captain-exoplanet'
        print 'exiting now'
        sys.exit()
        
    if args.tomography: print 'burning bronze to perform Doppler tomography'
    if args.line: print 'burning iron to analyze a single line'




#function to read in the input file
def inreader(infile):
    names, values = readcol(infile, twod=False)
    outstruc = dict(zip(names, values))
    outstruc['index']=names
    outstruc['invals']=values
    return outstruc

#read in the input file and put the input values in a structure and corresponding arrays
struc1=inreader(infile)

index=np.array(struc1['index'])
invals=np.array(struc1['invals'])

#load Gaussian process packages, if relevant
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

#Get the parameters for the MCMC chains
nplanets=np.int64(struc1['nplanets'])
nwalkers=np.int64(struc1['nwalkers'])
nsteps=np.int64(struc1['nsteps'])
nthreads=np.int64(struc1['nthreads'])
sysname=struc1['sysname'] #not actually used later in the code, just for my own checks

#import matplotlib and set up parameters if plots are to be made later
if args.plotstep or args.plotbest:
    import matplotlib.pyplot as pl
    if args.plotstep:
        pl.ion()
        pl.figure(1)
    nthreads=1



#get the general input and output filenames
chainfile=struc1['chainfile']
probfile=struc1['probfile']
accpfile=struc1['accpfile']
#read in the perturbations to the initial MCMC state
if any('perturbfile' in s for s in index): 
    perturbfile=struc1['perturbfile']
    perturbstruc=inreader(perturbfile)
    perturbindex=np.array(perturbstruc['index'])
    perturbinvals=np.array(perturbstruc['invals'])
else:
    print 'You need to specify a perturbation file!'
    print 'exiting now'
    sys.exit()

#read in the priors, if any
if any('priorfile' in s for s in index): 
    priorfile=struc1['priorfile']
    priorstruc=inreader(priorfile)
    priorindex=np.array(priorstruc['index'])
    priorinvals=np.array(priorstruc['invals'])
else:
    priorstruc={'none':'none'}

#set up the MCMC starting values
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
    for i in range(0,nplanets):
        struc1['ecsinw'+str(i+1)], struc1['eccosw'+str(i+1)]=eccpar[i],omegapar[i]


if ewflag == 'eomega':
    eccpar=ecc
    omegapar=omega
    enames=['ecc','omega']



#parameters needed for photometry, mostly limb darkening and filters
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
                
    #now read in the photometric data
    for i in range(0, pndatasets):
        ptime1,pflux1,perror1,pexptime1=readcol(photfile[i],twod=False)
        goods=np.where((ptime1 != -1.) & (pflux1 != -1.))
        ptime1,pflux1,perror1,pexptime1=ptime1[goods],pflux1[goods],perror1[goods],pexptime1[goods]
    #check to see if using TESS, Kepler, or CoRoT cadence and, if so, correct to exposure times
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
            if struc1['cadenceflag'+str(i+1)] == 'tess':
                longcad=np.where(pexptime1 == 1)
                shortcad=np.where(pexptime1 == 0)
                pexptime1[longcad], pexptime1[shortcad] = 30., 2.
                pexptime1=pexptime1/(60.*24.)
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
        if any('binfflag' in s for s in index): 
            binfflag=struc1['binfflag']
        else:
            binfflag='rprsfluxr'

            
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

        
        
    if any('tomfile' in i for i in struc1['linefile']):
        lnum=np.float(struc1['linefile'][7])
        if args.tomography:
            lineprof = tomdict['avgprof'+str(lnum)]
            lineerr = tomdict['avgproferr'+str(lnum)]
            linevel = tomdict['vabsfine'+str(lnum)]
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

    
#add the dilution parameters, if present
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






#all of the necessary functions will go here

        
#resample a light curve    
def resampler(modin,t,tprime,cadence):
    #t is the raw time, tprime is what we want to resample to
    #cadence has the same length as tprime
    nexp=len(tprime)
    nmodps=len(t)
    modout=np.zeros(nexp)
    for i in range (0,nexp):
        heres=np.where(np.abs(t-tprime[i]) <= cadence[i]/2.0)
        modout[i]=np.mean(modin[heres[0]])
    return modout

#compute the photometric model
def photmodel(struc):
    if struc['photmodflag'] == 'batman':
        params=batman.TransitParams()
        params.t0=0.0
        if any('secondary' in s for s in struc.keys()):
            params.t0+=struc['secondary']
        params.per=struc['Per']
        params.rp=struc['rprs']
        params.a=struc['aors']
        params.inc=struc['inc']
        params.ecc=struc['ecc']
        params.w=struc['omega']
        params.limb_dark='quadratic' #will have to de-hardcode this eventually...
        params.u=[struc['g1'],struc['g2']]

        tenminutes=10.0/(60.*24.)
        nexp=len(struc['t'])
        flux=np.zeros(nexp)
        if any(t < tenminutes for t in struc['exptime']):
            shorts=np.where(struc['exptime'] < tenminutes)
            ms=batman.TransitModel(params,struc['t'][shorts],nthreads=1)
            flux[shorts]=ms.light_curve(params)
        if any(t >= tenminutes for t in struc['exptime']):
            longs=np.where(struc['exptime'] >= tenminutes)
            if struc['longflag'] == 'batman':
                ml=batman.TransitModel(params,struc['t'][longs],nthreads=1, supersample_factor=50, exp_time=np.mean(struc['exptime'][longs])) #for now, just use the mean of the long exposure time--fix later to be able to handle multiple exposure lengths
                flux[longs]=ml.light_curve(params)
            else:
                maxexp=np.max(struc['exptime'][longs])
                ttemp=np.arange(np.min(struc['t'][longs])-maxexp*2., np.max(struc['t'][longs])+maxexp*2., np.min(struc['exptime'][longs])/100.)
                ml=batman.TransitModel(params,ttemp,nthreads=1)
                ftemp=ml.light_curve(params)
                flux[longs]=resampler(ftemp, ttemp, struc['t'][longs], struc['exptime'][longs])

    elif struc['photmodflag'] == 'jktebop':
        jktpath=struc['jktpath']
        timestamp=str(multiprocessing.current_process().pid)+'.'+str(timemod.time()-1537000000.)
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

        os.system(jktpath+'jktebop temp'+timestamp+'.in')

        phase,mag,l1,l2,l3=readcol('temp'+timestamp+'.out',twod=False)
        os.system('rm -f temp'+timestamp+'.out')
        os.system('rm -f temp'+timestamp+'.in')
        highs=np.where(phase > 0.5)
        phase[highs[0]]-=1.0

        mflux1=10.**(((-1.)*mag)/2.5)
        
        flux=resampler(mflux1,phase*struc['Per'],struc['t'],struc['exptime'])
            
    return flux

def lnlike(theta, parstruc, data, nplanets, inposindex, instruc, args):
    if args.time: temptime=timemod.time()-1458836000.
    index=instruc['index']
    nfigs=1
    #parse out the eccentricity
    if any('sinw' in s for s in inposindex) or any('ecc' in s for s in inposindex):
        if any('sesinw' in s for s in inposindex):
            ecc=theta[[i for i, s in enumerate(inposindex) if 'sesinw' in s]]**2+theta[[i for i, s in enumerate(inposindex) if 'secosw' in s]]**2
            omega=np.arccos(theta[[i for i, s in enumerate(inposindex) if 'secosw' in s]]/np.sqrt(ecc))
            news=np.where(theta[[i for i, s in enumerate(inposindex) if 'sesinw' in s]] < 0.)
            omega[news]=2.*np.pi-omega[news]
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
    #will need to do this for esinw, ecosw. code up later.



    else:
        ecc=np.zeros(nplanets)
        omega=np.zeros(nplanets)+90.

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
        if any(t > 1.0 for t in np.abs(theta[[i for i, s in enumerate(inposindex) if 'bpar' in s]])-theta[[i for i, s in enumerate(inposindex) if 'rprs' in s]]): #bpar and rprs must be in the same planet order for this to work--fix later
            return -np.inf #handle if no transit
        if any('cosi' in s for s in inposindex) and any('aors' in s for s in inposindex):
            aors=theta[[i for i, s in enumerate(inposindex) if 'aors' in s]]
            cosi=theta[[i for i, s in enumerate(inposindex) if 'cosi' in s]]
            inc=np.arccos(cosi)*180./np.pi
            
        
        if not args.binary:
            dur=theta[[i for i, s in enumerate(inposindex) if 'Per' in s]]/np.pi*1./aors*np.sqrt(1.0-ecc**2)/(1.0+ecc*np.cos(omega*np.pi/180.))*np.sqrt((1.0+theta[[i for i, s in enumerate(inposindex) if 'rprs' in s]])**2-theta[[i for i, s in enumerate(inposindex) if 'bpar' in s]]**2)
        else:
            dur=theta[[i for i, s in enumerate(inposindex) if 'Per' in s]]*10. #hack for now!!!--just use all of the datapoints



    lnl=0.0
    #do photometric fit
    if args.photometry:
        nexpp=len(data['ptime'])
        model=np.zeros(nexpp)
        for i in range (0,nplanets):
            modelstruc={'Per':parstruc['Per'+str(i+1)], 'rprs':parstruc['rprs'+str(i+1)], 'aors':aors[i], 'inc':inc[i], 'ecc':ecc[i], 'omega':omega[i]}

            
            phased = np.mod(data['ptime']-parstruc['epoch'+str(i+1)], parstruc['Per'+str(i+1)])
            highs = np.where(phased > parstruc['Per'+str(i+1)]/2.0)
            phased[highs]-=parstruc['Per'+str(i+1)] #this is still in days
            maxexp=np.max(data['pexptime'])
            if not args.ttvs: 
                closes = np.where(np.abs(phased) <= (dur[i]/2.+maxexp)*1.5)
            elif not data['dottvs'][i]:
                closes = np.where(np.abs(phased) <= (dur[i]/2.+maxexp)*1.5)
            else:
                #get the expected ttv epochs from the model here
                
                inargs=np.array(theta[[k for k, s in enumerate(inposindex) if 'ttv'+str(i+1)+'par' in s]], dtype=np.float)
                inargdex=np.array(inposindex[[k for k, s in enumerate(inposindex) if 'ttv'+str(i+1)+'par' in s]], dtype=np.str)
                nargs=len(inargs)
                inargstruc={}
                for tparcount in range (0,nargs):
                    inargstruc[inargdex[tparcount]]=inargs[tparcount]
                
                halflength=dur[i]/2.+maxexp+np.sum(np.array(theta[[k for k, s in enumerate(inposindex) if 'ttv'+str(i+1)+'parA' in s]], dtype=np.float))
                ttvepochs=ttvmodel(data['ptime'], parstruc['Per'+str(i+1)], parstruc['epoch'+str(i+1)], data['ttvmodtype'][i], inargstruc, halflength, i+1)
                halflength=dur[i]/2.+maxexp+maxttv
                closes = np.where(np.abs(phased) <= (halflength)*1.1)
                newt=ttvshift(data['ptime'][closes], parstruc['Per'+str(i+1)], parstruc['epoch'+str(i+1)], ttvepochs, halflength)
                phased[closes] = np.mod(newt-parstruc['epoch'+str(i+1)], parstruc['Per'+str(i+1)]) #this still leaves un-shifted elements in phased, but I'm pretty sure that modifying closes below will permanently exclude these problem elements
                closes1 = np.where(np.abs(phased) <= (dur[i]/2.+maxexp)*1.1)
                closes = closes[closes1] #not totally sure this is right
                
            print dur[i],(dur[i]/2.+maxexp)*1.1
            
            if any('longflag' in s for s in instruc):
                modelstruc['longflag']=instruc['longflag']
            else:
                modelstruc['longflag']='batman'

            if any('photmodflag' in s for s in instruc):
                modelstruc['photmodflag']=instruc['photmodflag']
            else:
                modelstruc['photmodflag']='batman'

            if args.binary and modelstruc['photmodflag'] == 'batman': modelstruc['aors'] = aors[i]*(1.+parstruc['rprs'+str(i+1)]) #PRIMARY eclipse, so PRIMARY in background, R*=R1, Rp=R2
                
            for j in range (0,instruc['pnfilters']):
                if any('q1p' in s for s in inposindex) and any('q1p' in s for s in inposindex):
                    modelstruc['g1'], modelstruc['g2'] = 2.0*parstruc['q2p'+str(j+1)]*np.sqrt(parstruc['q1p'+str(j+1)]), np.sqrt(parstruc['q1p'+str(j+1)])*(1.0-2.0*parstruc['q2p'+str(j+1)])
                else:
                    modelstruc['g1'], modelstruc['g2'] = parstruc['g1p'+str(j+1)], parstruc['g2p'+str(j+1)]

                    
                closefilter=np.where((np.abs(phased) <= (dur[i]/2.+maxexp)*1.5) & (data['pfilter'] == j+1))
                modelstruc['t'], modelstruc['exptime'] = phased[closefilter], data['pexptime'][closefilter]
                if args.dilution: 
                    modelstruc['dilution'] = parstruc['dilution'+str(j+1)]
                else:
                    modelstruc['dilution'] = 0.0
                if args.binary:
                    if data['binfflag'] == 'mycomb':
                        fluxrat=parstruc['fluxrat'+str(j+1)]/parstruc['rprs'+str(i+1)]**2
                    else:
                        fluxrat=parstruc['fluxrat'+str(j+1)]
                    modelstruc['fluxrat']=fluxrat
                        
                if args.plotbest and args.fullcurve:
                    if np.sign(np.min(phased[closefilter])) == np.sign(np.max(phased[closefilter])):
                        modelstruc['t']=np.append(modelstruc['t'],np.linspace(0.,np.min(np.abs(phased[closefilter])),100))
                        modelstruc['exptime']=np.append(modelstruc['exptime'],np.zeros(100)+np.mean(modelstruc['exptime']))
                        phased=np.append(phased,np.linspace(0.,np.min(np.abs(phased[closefilter])),100))
                        buh=1
                    else:
                        buh=0
                                                  

                if args.time: print 'about to call the model ',temptime,len(phased[closefilter])
                if modelstruc['photmodflag'] == 'jktebop': modelstruc['jktpath']=instruc['jktpath']
                rawmodel=photmodel(modelstruc)
                if args.binary and modelstruc['photmodflag'] == 'batman':
                    modelstruc['omega']-=180.
                    fsecondary=np.pi/2.-omega[i]*np.pi/180.-np.pi #true anomaly at secondary eclipse
                    Esecondary=2.*np.arctan(np.sqrt((1.-ecc[i])/(1.+ecc[i]))*np.tan(fsecondary/2.)) #eccentric anomaly at secondary eclipse
                    timesince=parstruc['Per'+str(i+1)]/(2.*np.pi)*(Esecondary-ecc[i]*np.sin(Esecondary))
                    modelstruc['aors'], modelstruc['rprs'], modelstruc['secondary'] = aors[i]*(1.+1./parstruc['rprs'+str(i+1)]), 1./parstruc['rprs'+str(i+1)], timesince #SECONDARY eclipse, so SECONDARY in background, R*=R2, Rp=R1
                    rawmodel2=photmodel(modelstruc)
                    rawmodel+=1./fluxrat #this is the PRIMARY eclipse--so still see the SECONDARY
                    rawmodel/=1.+1./fluxrat
                    rawmodel2+=fluxrat #this is the SECONDARY eclipse--so still see the PRIMARY
                    rawmodel2/=1.+fluxrat
                    rawmodel-=1.
                    rawmodel2-=1.
                    rawmodel+=rawmodel2+1.
                    #reset to the primary
                    modelstruc['aors'], modelstruc['rprs'], modelstruc['secondary'] = aors[i]*(1.+parstruc['rprs'+str(i+1)]), parstruc['rprs'+str(i+1)], 0.0 #PRIMARY eclipse, so PRIMARY in background, R*=R1, Rp=R2
                    modelstruc['omega']+=180.
                if args.dilution and modelstruc['photmodflag'] != 'jktebop':
                    if any('dilution'+str(j+1) in s for s in inposindex) and modelstruc['photmodflag'] == 'batman':
                        rawmodel+= 10.**(-0.4*parstruc['dilution'+str(j+1)])
                        rawmodel/= (1.+10.**(-0.4*parstruc['dilution'+str(j+1)]))
                rawmodel-=1.0
                if args.time: print 'model called ',temptime
                if args.plotbest and args.fullcurve:
                    if buh == 1:
                        orlength=len(closefilter[0])
                        print orlength,'ooom'
                        model[closefilter]=rawmodel[0:orlength]
                        model=np.append(model,rawmodel[orlength:])
                        nexpp+=100
                        data['ptime']=np.append(data['ptime'],np.zeros(100)+np.mean(data['ptime'][closefilter]))
                        data['pflux']=np.append(data['pflux'],np.zeros(100))
                        data['perror']=np.append(data['perror'],np.zeros(100))
                        data['pdataset']=np.append(data['pdataset'],np.zeros(100)-1)
                        data['pfilter']=np.append(data['pfilter'],np.zeros(100)+j)
                else:
                    model[closefilter]+=rawmodel


                if args.gp:
                    if any('gpmodtypep' in s for s in data['gpmodtype']):
                        if data['gpmodtype']['gpmodtypep'] == 'Matern32':
                            pkern=gppack.kernels.Matern32Kernel(parstruc['gpppartau']**2)*parstruc['gppparamp']**2
                        elif data['gpmodtype']['gpmodtypep'] == 'Cosine':
                            pkern=gppack.kernels.CosineKernel(parstruc['gppparP'])*parstruc['gppparamp']**2
                        elif data['gpmodtype']['gpmodtypep'] == 'ExpSine2':
                            pkern=gppack.kernels.ExpSine2Kernel(parstruc['gpppartau'],parstruc['gppparP'])*parstruc['gppparamp']**2
                        elif data['gpmodtype']['gpmodtypep'] == 'Haywood14QP':
                            if instruc['gppackflag'] == 'celerite':
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
                
                if args.plotstep or args.plotbest:
                    pl.plot(np.array(data['ptime'], dtype=float),np.array(data['pflux'], dtype=float),'ro')
                    pl.plot(np.array(data['ptime'], dtype=float),model+1.)
                    pl.show()
                    pl.clf()
                    pl.plot(np.array(data['ptime'], dtype=float),np.array(data['pflux'], dtype=float)-(model+1.),'ro')
                    pl.show()
                    pl.clf()
                    setsfilter=np.array(list(set(data['pdataset'][np.where(data['pfilter'] == j+1)])))
                    nsetsfilter=len(setsfilter)
                    for k in range(0,nsetsfilter):
                        closedataset=np.where((np.abs(phased) <= (dur[i]/2.+maxexp)*1.25) & (data['pdataset'] == setsfilter[k]))
                        if len(closedataset[0]) > 0:
                            if args.plotstep:
                                fignum=1
                            else:
                                fignum=nfigs
                                nfigs+=1
                            pl.plot(phased[closedataset], np.array(data['pflux'][closedataset], dtype=float), 'ro')
                            pl.plot(phased[closedataset], model[closedataset]+1.0, 'bo')
                            if args.gp:
                                if any('gpmodtypep' in s for s in data['gpmodtype']):
                                    closedatasetg=np.where((np.abs(phased) <= (dur[i]/2.+maxexp)*1.25) & (data['pdataset'] == setsfilter[k]) & (data['gppuse'] == 1))
                                    mu, var = gp.predict(np.array(data['pflux'][useforgp])-(model[useforgp]+1.), return_var=True)
                                    print "after"
                                    mu = mu + (model[useforgp]+1.)
                                    var = var + (model[useforgp]+1.)
                                    pl.plot(phased[closedatasetg], mu[closedatasetg],'go')
                                    pl.draw()
                                    pl.clf()
                                    pl.plot(np.array(data['ptime'][useforgp]),np.array(data['pflux'][useforgp]),'ro')
                                    pl.plot(np.array(data['ptime'][useforgp]),mu,color='green')
                                    pl.draw()#pl.show()
                                    pl.clf()

                                    pl.plot(np.array(data['ptime'][useforgp]),np.array(data['pflux'][useforgp])-(mu-(model[useforgp]+1.)),'ro')
                                    pl.plot(np.array(data['ptime'][useforgp]),model[useforgp]+1.,color='blue')
                                    pl.draw()#pl.show()
                                    pl.clf()
                                    
        
                            if args.plotstep:
                                pl.draw()
                                pl.pause(0.01)
                                pl.clf()
                            if args.plotbest:
                                if args.ploterrors: pl.errorbar(phased[closedataset], np.array(data['pflux'][closedataset], dtype=float), yerr=np.array(data['perror'][closedataset], dtype=float),fmt='none',ecolor='red')
                                pl.xlabel('time from center of transit (days)')
                                pl.ylabel('normalized flux')
                                namelength=len(instruc['plotfile'])
                                if instruc['plotfile'][namelength-4:namelength] == '.pdf':
                                    pl.savefig(pp, format='pdf')
                                    pl.clf()
                                    if args.plotresids:
                                        pl.plot(phased[closedataset], np.array(data['pflux'][closedataset]-(model[closedataset]+1.0)), 'ro')
                                        if args.ploterrors: pl.errorbar(phased[closedataset], np.array(data['pflux'][closedataset]-(model[closedataset]+1.0)), yerr=np.array(data['perror'][closedataset], dtype=float),fmt='none',ecolor='red')
                                        pl.plot([np.min(phased[closedataset]),np.max(phased[closedataset])],[0.0,0.0],color='blue')
                                        if j == 0: pl.xlim([-0.1,0.1])
                                        pl.savefig(pp, format='pdf')
                                        pl.clf()
                                else:
                                    pl.savefig(instruc['plotfile'], format=instruc['plotfile'][namelength-3:namelength])
                                    print 'Plot complete. If this is a multiplanet system and you want' 
                                    print 'more than the first planet, you must use PDF format.'
 
                                if i == nplanets-1 and j == instruc['pnfilters']-1 and k == nsetsfilter-1 and (not args.rvs and not args.tomography or args.binary): 
                                    print 'Plots complete.'
                                    if (not args.rvs and not args.tomography and not args.line) or args.binary: pp.close()
 

        model+=1.0
        if args.plotbest: 
            f=open('photmodel.txt','w')
            for i in range (0,nexpp):
                f.write(str(data['ptime'][i])+', '+str(phased[i])+', '+str(data['pflux'][i])+', '+str(data['perror'][i])+', '+str(model[i])+', '+str(data['pflux'][i]-model[i])+', '+str(int(data['pdataset'][i]))+' \n')
            f.close()
            
        inv_sigma2 = 1.0/data['perror']**2
        if not args.gp:
            lnl+=np.sum(((data['pflux']-model)**2)*inv_sigma2 - np.log(inv_sigma2))
        else:
            if any('gpmodtypep' in s for s in data['gpmodtype']):
                if struc1['gppackflag'] == 'celerite':
                    lnl2=gp.log_likelihood(np.array(data['pflux'][useforgp])-model[useforgp])
                else:
                    lnl2=gp.lnlikelihood(np.array(data['pflux'][useforgp])-model[useforgp])
                lnl-=2.*lnl2   #gp.lnlike gives the actual lnlike, but needs to be *'d to add in with the other types      
                
                if len(notforgp) > 1:
                    lnl+=np.sum(((data['pflux'][notforgp]-model[notforgp])**2)*inv_sigma2[notforgp] - np.log(inv_sigma2[notforgp]))
            else:
                lnl+=np.sum(((data['pflux']-model)**2)*inv_sigma2 - np.log(inv_sigma2))
        if args.getprob:
            print 'The total photometric chisq is ',np.sum(((data['pflux']-model)**2)*inv_sigma2),' and chisq_red=',np.sum(((data['pflux']-model)**2)*inv_sigma2)/len(data['pflux'])
            for i in range(0,instruc['pndatasets']):
                heres=np.where(data['pdataset'] == i+1)
                print 'The contribution of photometric dataset ',i+1,' to lnprob is ',np.sum(((data['pflux'][heres]-model[heres])**2)*inv_sigma2[heres] - np.log(inv_sigma2[heres])), ' or chisq=',np.sum(((data['pflux'][heres]-model[heres])**2)*inv_sigma2[heres])/len(heres[0])


    if args.tomography:
        ntomsets=data['tomdict']['ntomsets']
        dur=np.array(dur,dtype=np.float)
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
            horusstruc = {'vsini': parstruc['vsini'], 'sysname': instruc['sysname'], 'obs': instruc['obs'+str(i+1)], 'vabsfine': data['tomdict']['vabsfine'+str(i+1)], 'Pd': parstruc['Per'+str(whichplanet)], 'lambda': parstruc['lambda'+str(whichplanet)], 'b': parstruc['bpar'+str(whichplanet)], 'rplanet': parstruc['rprs'+str(whichplanet)], 't': tomphase[tomins], 'times': texptime[tomins], 'e': ecc[whichplanet-1], 'periarg': omega[whichplanet-1]*np.pi/180., 'a': aors[whichplanet-1]}
            
            if args.line and any('linecenter' in s for s in inposindex): 
                if np.abs(parstruc['linecenter']) > np.max(np.abs(horusstruc['vabsfine'])): return -np.inf
                horusstruc['vabsfine']+=parstruc['linecenter']
            if any('q1t' in s for s in inposindex) and any('q2t' in s for s in inposindex):
                horusstruc['gamma1'], horusstruc['gamma2'] = 2.0*parstruc['q2t']*np.sqrt(parstruc['q1t']), np.sqrt(parstruc['q1t'])*(1.0-2.0*parstruc['q2t'])
            else:
                horusstruc['gamma1'], horusstruc['gamma2'] = parstruc['g1t'], parstruc['g2t']
            if any('intwidth' in s for s in inposindex):
                horusstruc['width'] = parstruc['intwidth']
            elif any('intwidth' in s for s in index):
                horusstruc['width'] = np.float(instruc['intwidth'])
            else:
                horusstruc['width'] = 10.

            if any('tomdrift' in s for s in inposindex):
                lineshift=(data['tomdict']['ttime'+str(i+1)]-data['tomdict']['ttime'+str(i+1)][0])*parstruc['tomdriftl'+str(i+1)]+parstruc['tomdriftc'+str(i+1)]
                if np.max(np.abs(lineshift) > np.max(np.abs(horusstruc['vabsfine']))): return -np.inf
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
                if  len(tomins[0]) == 0:
                    return -np.inf #protects against the model transit not happening during the observations
                for j in range (0, len(tomouts[0])): model[tomouts[0][j],:]=outstruc['basearr'][0,:]
            if args.skyline:
                skyline=horus.mkskyline(vabsfine,parstruc['skydepth'],parstruc['skycen'],instruc['obs'+str(i+1)])
                for j in range (np.int(instruc['skyfirst']), np.int(instruc['skylast'])+1):
                    model[j,:]+=skyline

            vgood=np.where(np.abs(vabsfine) <= np.float(instruc['vsini'])*1.25)

            if args.gp:
                lnl1=0.
                if any('gpmodtypet' in s for s in data['gpmodtype']):
                    if data['gpmodtype']['gpmodtypet'] == 'Matern32':
                        tkern=parstruc['gptparamp']*george.kernels.Matern32Kernel([parstruc['gptpartaut'],parstruc['gptpartauv']],ndim=2)
                    gp=george.GP(tkern)
                    bjdfloor=np.floor(data['tomdict']['ttime'+str(i+1)])
                    ntnights=len(np.unique(bjdfloor))
                    for night in range (0,ntnights):
                        thisnight=np.where(bjdfloor == np.unique(bjdfloor)[night])
                        ttemp, vtemp = np.meshgrid(data['tomdict']['ttime'+str(i+1)][thisnight], data['tomdict']['vabsfine'+str(i+1)], indexing="ij")
                        profarrinds=np.vstack((ttemp.flatten(), vtemp.flatten())).T
                        profarrforgp=data['tomdict']['profarr'+str(i+1)][thisnight,:].flatten()
                        profarrerrforgp=data['tomdict']['profarrerr'+str(i+1)][thisnight,:].flatten()
                        modelforgp=model[thisnight,:].flatten()
                        gp.compute(profarrinds,profarrerrforgp)

                        wherebad=np.array(np.where(np.isnan(modelforgp)))
                        if wherebad.size != 0:
                            return -np.inf
                        lnl1=-2.*gp.lnlikelihood(profarrforgp-modelforgp*(-1.))

            if args.plotstep:
                fignum=1
                for j in range (0, tnexp):
                    pl.plot(data['tomdict']['vabsfine'+str(i+1)],data['tomdict']['profarr'+str(i+1)][j,:]+0.1*j)
                    if instruc['fitflat'] != 'True':
                        pl.plot(data['tomdict']['vabsfine'+str(i+1)],model[j,:]*(-1.)+0.1*j)
                    else:
                        pl.plot(data['tomdict']['vabsfine'+str(i+1)],(model[j,:]-outstruc['basearr'][0,:])*(-1.)+0.1*j)
                pl.savefig('active_misttborn.eps',format='eps')
                pl.draw()
                pl.pause(1.)
                pl.clf()
            if args.plotbest:
                import dtutils
                fignum=1
                profarrflat=np.zeros((tnexp,nvabsfine))
                modelflat=np.zeros((tnexp,nvabsfine))
                for j in range (0, tnexp):
                    pl.plot(data['tomdict']['vabsfine'+str(i+1)],data['tomdict']['profarr'+str(i+1)][j,:]+0.1*j)
                    pl.plot(data['tomdict']['vabsfine'+str(i+1)],model[j,:]*(-1.)+0.1*j)
                    if instruc['fitflat'] == 'False': 
                        profarrflat[j,:]=data['tomdict']['profarr'+str(i+1)][j,:]-outstruc['basearr'][0,:]*(-1.)
                        modelflat[j,:]=model[j,:]+outstruc['basearr'][0,:]*(-1.)
                    else:
                        profarrflat[j,:]=model[j,:]
                pl.xlabel('velocity (km s$^{-1}$)')
                pl.ylabel('normalized profile + offset')
                pl.savefig(pp, format='pdf')
                pl.clf()
                pl.plot(data['tomdict']['vabsfine'+str(i+1)],model[0,:]*(-1.),color='red')
                pl.plot(data['tomdict']['vabsfine'+str(i+1)],basearr[0,:]*(-1.),color='blue')
                pl.savefig(pp, format='pdf')
                pl.clf()
                tphase, tphase2=np.mod((data['tomdict']['ttime'+str(i+1)]-parstruc['epoch'+str(whichplanet)]),parstruc['Per'+str(whichplanet)]),np.mod((data['tomdict']['ttime'+str(i+1)]+data['tomdict']['texptime'+str(i+1)]/(60.*60.*24.)-parstruc['epoch'+str(whichplanet)]),parstruc['Per'+str(whichplanet)])
                bads=np.where(tphase > parstruc['Per'+str(whichplanet)]/2.)
                tphase[bads[0]]-=parstruc['Per'+str(whichplanet)]
                bads=np.where(tphase2 > parstruc['Per'+str(whichplanet)]/2.)
                tphase2[bads[0]]-=parstruc['Per'+str(whichplanet)]
                tphase += dur[whichplanet-1]/2.
                tphase2 += dur[whichplanet-1]/2.
                tphase /= dur[whichplanet-1]
                tphase2 /= dur[whichplanet-1]
                dtutils.mktslprplot(profarrflat*(-1.),data['tomdict']['profarrerr'+str(i+1)],data['tomdict']['vabsfine'+str(i+1)],tphase,tphase2,parstruc['vsini'],parstruc['bpar'+str(whichplanet)],parstruc['rprs'+str(whichplanet)],filename='none')
                pl.savefig(pp, format='pdf')
                pl.clf()
                dtutils.mktslprplot(modelflat,data['tomdict']['profarrerr'+str(i+1)],data['tomdict']['vabsfine'+str(i+1)],tphase,tphase2,parstruc['vsini'],parstruc['bpar'+str(whichplanet)],parstruc['rprs'+str(whichplanet)],filename='none',weighted=False)
                pl.savefig(pp, format='pdf')
                pl.clf()
                dtutils.mktslprplot((data['tomdict']['profarr'+str(i+1)]-model*(-1.))*(-1.),data['tomdict']['profarrerr'+str(i+1)],data['tomdict']['vabsfine'+str(i+1)],tphase,tphase2,parstruc['vsini'],parstruc['bpar'+str(whichplanet)],parstruc['rprs'+str(whichplanet)],filename='none')
                pl.savefig(pp, format='pdf')
                pl.clf()
                if args.gp and any('gptpar' in s for s in gpparnames):
                    #compute the GP model
                    ttemp, vtemp2 = np.meshgrid(data['tomdict']['ttime'+str(i+1)], data['tomdict']['vabsfine'+str(i+1)], indexing="ij")
                    profarrinds2=np.vstack((ttemp.flatten(), vtemp2.flatten())).T
                    modelforgp2=model.flatten()
                    modelGP1d1=gp.sample_conditional(profarrforgp-modelforgp*(-1.), profarrinds2)
                    modelGP1d=modelGP1d1+modelforgp2*(-1.)
                    #now need to stack it back up into a 2d array
                    modelGP2d=np.zeros((tnexp,nvabsfine))
                    modelGP2d1=np.zeros((tnexp,nvabsfine))
                    for ecount in range(0,tnexp):
                        modelGP2d[ecount,:]=modelGP1d[ecount*nvabsfine:(ecount+1)*nvabsfine]
                        modelGP2d1[ecount,:]=modelGP1d1[ecount*nvabsfine:(ecount+1)*nvabsfine]
                        pl.plot(data['tomdict']['vabsfine'+str(i+1)],profarr[ecount,:]+0.1*ecount)
                        pl.plot(data['tomdict']['vabsfine'+str(i+1)],modelGP2d[ecount,:]+0.1*ecount)
                        pl.plot(data['tomdict']['vabsfine'+str(i+1)],model[ecount,:]*(-1.)+0.1*ecount)
                        modelGP2d[ecount,:]+=model[0,:]

                    pl.xlabel('velocity (km s$^{-1}$)')
                    pl.ylabel('normalized profile + offset')
                    pl.savefig(pp, format='pdf')
                    pl.clf()
                    pl.imshow(modelGP2d1)
                    pl.savefig(pp, format='pdf')
                    pl.clf()
                    pl.imshow(profarr-(model-modelGP2d1)*(-1.))
                    pl.savefig(pp, format='pdf')
                    pl.clf()
                if not args.rvs and not args.line and i == ntomsets-1: pp.close()
                f=open('lineprofile.dat','w')
                for j in range (0,nvabsfine):
                    f.write(str(outstruc['basearr'][0,j])+' \n')
                f.close()
                for j in range (0,tnexp):
                    f=open('lineprofile.'+str(i)+'.'+str(j)+'.dat','w')
                    print 'lineprofile.'+str(i)+'.'+str(j)+'.dat'
                    for k in range (0,nvabsfine):
                        f.write(str(model[j,k]-outstruc['basearr'][0,k])+' \n')
                    f.close
                
            if instruc['fitflat'] == 'True':
                for j in range (0, tnexp): model[j, : ]-=baseline
                if data['dofft']:
                    model1=horus.fourierfilt(model,data['mask'])
                    model=model1
            model*=(-1.)
            inv_sigma2 = 1.0/data['tomdict']['profarrerr'+str(i+1)][:,vgood]**2
            if not args.gp:
                lnl+=np.sum((data['tomdict']['profarr'+str(i+1)][:,vgood]-model[:,vgood])**2*inv_sigma2 - np.log(inv_sigma2))
            else:
                if any('gpmodtypet' in s for s in data['gpmodtype']):
                    lnl+=lnl1
                else:
                    lnl+=np.sum((data['tomdict']['profarr'+str(i+1)][:,vgood]-model[:,vgood])**2*inv_sigma2 - np.log(inv_sigma2))
            if args.getprob:
                print 'The tomographic chisq for dataset ',i,' is ',np.sum((data['tomdict']['profarr'+str(i+1)][:,vgood]-model[:,vgood])**2*inv_sigma2),' and chisq_red=',np.sum((data['tomdict']['profarr'+str(i+1)][:,vgood]-model[:,vgood])**2*inv_sigma2)/(len(vgood[0])*tnexp)
                print 'The contribution of tomography to lnprob is ',np.sum((data['tomdict']['profarr'+str(i+1)][:,vgood]-model[:,vgood])**2*inv_sigma2 - np.log(inv_sigma2))

    if args.line:
        if args.tomography:
            linemodel=basearr[0,:]
            
            
        else:
            horusstruc = {'vsini': parstruc['vsini'], 'sysname': instruc['sysname'], 'obs': instruc['obs1'], 'vabsfine': data['linevel']+parstruc['linecenter']}
            if np.abs(parstruc['linecenter']) > np.max(np.abs(data['linevel'])): return -np.inf
            if any('q1t' in s for s in inposindex) and any('q2t' in s for s in inposindex):
                horusstruc['gamma1'], horusstruc['gamma2'] = 2.0*parstruc['q2t']*np.sqrt(parstruc['q1t']), np.sqrt(parstruc['q1t'])*(1.0-2.0*parstruc['q2t'])
            else:
                horusstruc['gamma1'], horusstruc['gamma2'] = parstruc['g1t'], parstruc['g2t']
            if any('intwidth' in s for s in inposindex):
                horusstruc['width'] = parstruc['intwidth']
            elif any('intwidth' in s for s in index):
                horusstruc['width'] = np.float(instruc['intwidth'])
            else:
                horusstruc['width'] = 10.

            if horusstruc['obs'] != 'igrins':
                outstruc = horus.model(horusstruc, resnum=50.0, onespec='y',convol='y')
            else:
                outstruc = horus.model(horusstruc, resnum=50.0, onespec='y', convol='n')
            linemodel=outstruc['baseline']
            
            
        if args.plotbest or args.plotstep:
            if args.plotstep:
                fignum=1
            else:
                fignum=nfigs
                nfigs+=1
            pl.plot(np.array(data['linevel'], dtype=float), np.array(data['lineprof'], dtype=float), 'red')
            pl.plot(np.array(data['linevel'], dtype=float), linemodel, 'blue')
            if args.ploterrors: 
                pl.errorbar(np.array(data['linevel'], dtype=float), np.array(data['lineprof'], dtype=float), yerr=np.array(data['lineerr'], dtype=float),fmt='none',ecolor='red')
                     
                            
            if args.plotstep:
                pl.draw()
                pl.pause(0.01)
                pl.clf()
            if args.plotbest:
                pl.xlabel('velocity (km s$^{-1}$)')
                pl.ylabel('normalized flux')
                namelength=len(instruc['plotfile'])
                if instruc['plotfile'][namelength-4:namelength] == '.pdf':
                    pl.savefig(pp, format='pdf')
                    pl.clf()
                    if not args.rvs: pp.close()
                else:
                    pl.savefig(instruc['plotfile'], format=instruc['plotfile'][namelength-3:namelength])
                    print 'Plot complete. If this is a multiplanet system and you want' 
                    print 'more than the first planet, you must use PDF format.'

                f=open('lineprofile.dat','w')
                for j in range (0,len(data['linevel'])):
                    f.write(str(linemodel[j])+' \n')
                f.close()


        inv_sigma2 = 1.0/data['lineerr']**2
        lnl+=np.sum((data['lineprof']-linemodel)**2*inv_sigma2 - np.log(inv_sigma2))

        if args.getprob:
            print 'The total chisq of the line data is ',np.sum(((data['lineprof']-linemodel)**2)*inv_sigma2),' and chisq_red=',np.sum(((data['lineprof']-linemodel)**2)*inv_sigma2)/len(data['lineprof'])
            print 'The contribution of the line to lnprob is',np.sum((data['lineprof']-linemodel)**2*inv_sigma2 - np.log(inv_sigma2))

    if args.rvs:
        nrvs=len(data['rtime'])
        rmodel=np.zeros(nrvs)
        if args.binary: rmodel2=np.zeros(nrvs)
        for i in range(0,nplanets):
            if ecc[i] == 0.0: #just use a sine model for the RVs
                if not args.binary:
                    rmodel+=parstruc['semiamp'+str(i+1)]*np.sin((data['rtime']-parstruc['epoch'+str(i+1)]+parstruc['Per'+str(i+1)]/2.)*2.*np.pi/parstruc['Per'+str(i+1)]) #Per/2 is because transit will occur on \ part of sine curve, not / part
                else:
                    rmodel+=parstruc['semiamp'+str(i+1)+'a']*np.sin((data['rtime']-parstruc['epoch'+str(i+1)]+parstruc['Per'+str(i+1)]/2.)*2.*np.pi/parstruc['Per'+str(i+1)]) #Per/2 is because transit will occur on \ part of sine curve, not / part
                    rmodel2+=parstruc['semiamp'+str(i+1)+'b']*np.sin((data['rtime']-parstruc['epoch'+str(i+1)])*2.*np.pi/parstruc['Per'+str(i+1)])
                
                
                

            else:
                #need to convert transit epoch to epoch of periastron
                #equations from https://exoplanetarchive.ipac.caltech.edu/docs/transit_algorithms.html#epoch_periastron
                if args.photometry:
                    ftransit=np.pi/2.-omega[i]*np.pi/180. #true anomaly at transit
                    Etransit=2.*np.arctan(np.sqrt((1.-ecc[i])/(1.+ecc[i]))*np.tan(ftransit/2.)) #eccentric anomaly at transit
                    timesince=parstruc['Per'+str(i+1)]/(2.*np.pi)*(Etransit-ecc[i]*np.sin(Etransit)) #time since periastron to transit
                else:
                    timesince=0.0 #use epoch of periastron if only have RVs

                if not args.binary:
                    rmodel+=radvel.kepler.rv_drive(data['rtime'],np.array([parstruc['Per'+str(i+1)],parstruc['epoch'+str(i+1)]-timesince,ecc[i],omega[i]*np.pi/180.,parstruc['semiamp'+str(i+1)]]))
                else:
                    rmodel+=radvel.kepler.rv_drive(data['rtime'],np.array([parstruc['Per'+str(i+1)],parstruc['epoch'+str(i+1)]-timesince,ecc[i],omega[i]*np.pi/180.,parstruc['semiamp'+str(i+1)+'a']]))
                    rmodel2+=radvel.kepler.rv_drive(data['rtime'],np.array([parstruc['Per'+str(i+1)],parstruc['epoch'+str(i+1)]-timesince,ecc[i],omega[i]*np.pi/180.-np.pi,parstruc['semiamp'+str(i+1)+'b']]))

        #add any RV offsets
        rndatasets=np.max(data['rdataset'])
        if args.fitjitter: jitters=np.zeros(len(data['rdataset']))
        for j in range(0,rndatasets):
            thisdataset=np.where(data['rdataset'] == j+1)
            if any('gamma' in s for s in inposindex):
                rmodel[thisdataset]+=parstruc['gamma'+str(j+1)]
                if args.binary: rmodel2[thisdataset]+=parstruc['gamma'+str(j+1)]
            else:
                rmodel[thisdataset]+=instruc['gamma'+str(j+1)]
                if args.binary: rmodel2[thisdataset]+=instruc['gamma'+str(j+1)]
            if args.fitjitter:
                jitters[thisdataset]=parstruc['jitter'+str(j+1)]

        #add any RV trend
        if any('rvtrend' in s for s in inposindex):
            rmodel+=(data['rtime']-parstruc['epoch1'])*parstruc['rvtrend']
            if args.binary: rmodel2+=(data['rtime']-parstruc['epoch1'])*parstruc['rvtrend']
        
        if args.plotbest or args.plotstep and not args.binary:
            for j in range(0,rndatasets):
                    
                if args.binary:
                    alldatasets=np.append(data['rdataset'], data['rdataset']) 
                    thisdataset=np.where(data['rdataset'] == j+1)
                    thisdataset1=np.where(alldatasets == j+1)
                    allrvtimes=np.append(data['rtime'],data['rtime'])
                else:
                    thisdataset=np.where(data['rdataset'] == j+1)
                    thisdataset1=thisdataset
                    allrvtimes=data['rtime'][thisdataset]
                if any('gamma' in s for s in inposindex):
                    pl.plot(np.mod(allrvtimes[thisdataset1]-parstruc['epoch1'],parstruc['Per1']),data['rv'][thisdataset1]-parstruc['gamma'+str(j+1)],'ro')
                    if args.ploterrors: pl.errorbar(np.mod(allrvtimes[thisdataset1]-parstruc['epoch1'],parstruc['Per1']),data['rv'][thisdataset1]-parstruc['gamma'+str(j+1)], yerr=data['rverror'][thisdataset1],fmt='none',ecolor='red')
                    pl.plot(np.mod(data['rtime'][thisdataset]-parstruc['epoch1'],parstruc['Per1']),rmodel[thisdataset]-parstruc['gamma'+str(j+1)],'bo')
                    if args.binary: 
                        pl.plot(np.mod(data['rtime'][thisdataset]-parstruc['epoch1'],parstruc['Per1']),rmodel2[thisdataset]-parstruc['gamma'+str(j+1)],'go')
                        pl.plot(np.mod(allrvtimes[thisdataset1]-parstruc['epoch1'],parstruc['Per1']),data['rv'][thisdataset1]-np.append(rmodel[thisdataset],rmodel2[thisdataset]),'^',color='magenta')
                else:
                    pl.plot(np.mod(allrvtimes[thisdataset1]-parstruc['epoch1'],parstruc['Per1']),data['rv'][thisdataset1]-instruc['gamma'+str(j+1)],'ro')
                    if args.ploterrors: pl.errorbar(np.mod(allrvtimes[thisdataset1]-parstruc['epoch1'],parstruc['Per1']),data['rv'][thisdataset1]-instruc['gamma'+str(j+1)], yerr=data['rverror'][thisdataset1],fmt='none',ecolor='red')
                    pl.plot(np.mod(data['rtime'][thisdataset]-parstruc['epoch1'],parstruc['Per1']),rmodel[thisdataset]-instruc['gamma'+str(j+1)],'bo')
                    if args.binary: pl.plot(np.mod(data['rtime'][thisdataset]-parstruc['epoch1'],parstruc['Per1']),rmodel2[thisdataset]-parstruc['gamma'+str(j+1)],'go')
            if args.plotstep:
                pl.draw()
                pl.pause(0.1)
                pl.clf()
            if args.plotbest:
                pl.xlabel('orbital phase (days)')
                pl.ylabel('dRV')
                namelength=len(instruc['plotfile'])
                if instruc['plotfile'][namelength-4:namelength] == '.pdf':
                    pl.savefig(pp, format='pdf')
                    pl.clf()
                    if args.plotresids:
                        pl.plot(np.mod(data['rtime']-parstruc['epoch1'],parstruc['Per1']),data['rv']-rmodel,'ro')
                        if args.ploterrors:
                            pl.errorbar(np.mod(data['rtime']-parstruc['epoch1'],parstruc['Per1']),data['rv']-rmodel, yerr=data['rverror'],fmt='none',ecolor='red')
                        pl.plot([np.min(np.mod(data['rtime'][thisdataset]-parstruc['epoch1'],parstruc['Per1'])),np.max(np.mod(data['rtime'][thisdataset]-parstruc['epoch1'],parstruc['Per1']))],[0.0,0.0],color='blue')
                        pl.savefig(pp, format='pdf')
                    pp.close()
                else:
                    pl.savefig(instruc['plotfile'], format=instruc['plotfile'][namelength-3:namelength])
                    print 'Plot complete. If this is a multiplanet system and you want' 
                    print 'more than the first planet, you must use PDF format.'
                                    
                                    

        if args.fitjitter:
            inv_sigma2=1.0/(data['rverror']**2+jitters**2)
            inv_sigma22=1.0/(data['rverror']**2)
            
        else:
            inv_sigma2 = 1.0/data['rverror']**2
            inv_sigma22 = 1.0/data['rverror']**2
        if args.binary: rmodel=np.append(rmodel,rmodel2) 
        lnl+=np.sum((data['rv']-rmodel)**2*inv_sigma2 - np.log(inv_sigma2))

            
        if args.getprob:
                print 'The RV chisq is ',np.sum((data['rv']-rmodel)**2*inv_sigma2),' and chisq_red=',np.sum((data['rv']-rmodel)**2*inv_sigma2)/(len(data['rv']))
                print 'The contribution of RVs to lnprob is ',np.sum((data['rv']-rmodel)**2*inv_sigma2 - np.log(inv_sigma2))
        
    if args.getprob and args.rvs:
        print np.sum(np.log(inv_sigma2)),np.sum(np.log(inv_sigma22)),np.sum(np.log(1./inv_sigma2))
        print 'The BIC is ',np.log(len(data['rv']))*len(parstruc)+np.sum((data['rv']-rmodel)**2*inv_sigma2),np.log(len(data['rv']))*len(parstruc)+lnl
    return -0.5*lnl

#LNPRIOR
def lnprior(parstruc, priorstruc, instruc, theta, inposindex):
    #prevent parameters from going outside of physical bounds
    lp=0.0
    if any('sinw' in s for s in inposindex) or any('ecc' in s for s in inposindex):
        if any('sesinw' in s for s in inposindex):
            ecc=theta[[i for i, s in enumerate(inposindex) if 'sesinw' in s]]**2+theta[[i for i, s in enumerate(inposindex) if 'secosw' in s]]**2
            omega=np.arccos(theta[[i for i, s in enumerate(inposindex) if 'secosw' in s]]/np.sqrt(ecc))
            news=np.where(theta[[i for i, s in enumerate(inposindex) if 'sesinw' in s]] < 0.)
            omega[news]=2.*np.pi-omega[news]
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
            
            
    #will need to do this for esinw, ecosw. code up later.
    else:
        ecc=np.zeros(nplanets)
        omega=np.zeros(nplanets)+90.
    #will need to do this for esinw, ecosw. code up later.
    
    #reject unphysical values
    if any(t >= 1.0 for t in theta[[i for i, s in enumerate(inposindex) if 'rprs' in s]]) and not args.binary:
        return -np.inf #don't want to disallow radius ratio >1 if fitting EB

    #reject unphysical values of limb darkening when using Kipping sampling
    if any('q1' in s for s in inposindex):
        g1, g2 = 2.0*theta[[i for i, s in enumerate(inposindex) if 'q2' in s]]*np.sqrt(theta[[i for i, s in enumerate(inposindex) if 'q1' in s]]), np.sqrt(theta[[i for i, s in enumerate(inposindex) if 'q1' in s]])*(1.0-2.0*theta[[i for i, s in enumerate(inposindex) if 'q2' in s]])
        if np.min(g1) < 0. or np.min(g2) < 0. or np.max(g1) > 1. or np.max(g2) > 1.:
            return -np.inf
        

    if any(t <= 0.0 for t in theta[[i for i, s in enumerate(inposindex) if 'Per' in s]]) or any(t <= 0.0 for t in theta[[i for i, s in enumerate(inposindex) if 'rprs' in s]]) or any(np.abs(t) > 1.0 for t in theta[[i for i, s in enumerate(inposindex) if 'sesinw' in s]]) or any(np.abs(t) > 1.0 for t in theta[[i for i, s in enumerate(inposindex) if 'secosw' in s]]) or any(t <= 0 for t in theta[[i for i, s in enumerate(inposindex) if 'rhostar' in s]]) or any(t < 1. for t in theta[[i for i, s in enumerate(inposindex) if 'aors' in s]]) or any(t <= 0.0 for t in theta[[i for i, s in enumerate(inposindex) if 'q1' in s]]) or any(t <= 0.0 for t in theta[[i for i, s in enumerate(inposindex) if 'q2' in s]]) or any(t >= 1.0 for t in theta[[i for i, s in enumerate(inposindex) if 'q1' in s]]) or any(t >= 1.0 for t in theta[[i for i, s in enumerate(inposindex) if 'q2' in s]]) or any(t < 0.0 for t in ecc) or any(t >= 0.99 for t in ecc) or any(t <= 0 for t in theta[[i for i, s in enumerate(inposindex) if 'vsini' in s]]) or any(t <= 0 for t in theta[[i for i, s in enumerate(inposindex) if 'semiamp' in s]]) or any(t <= 0 for t in theta[[i for i, s in enumerate(inposindex) if 'intwidth' in s]]) or any(t <= 0 for t in theta[[i for i, s in enumerate(inposindex) if (('gp' in s) & ('partau' in s))]]) or any(t <= 0 for t in theta[[i for i, s in enumerate(inposindex) if (('gp' in s) & ('paramp' in s))]]) or any(t <= 0 for t in theta[[i for i, s in enumerate(inposindex) if (('gp' in s) & ('parGamma' in s))]]) or any(t <= 0 for t in theta[[i for i, s in enumerate(inposindex) if (('gp' in s) & ('parP' in s))]]) or any(t < 0. for t in theta[[i for i, s in enumerate(inposindex) if 'jitter' in s]]) or any(t <= 0. for t in theta[[i for i, s in enumerate(inposindex) if 'fluxrat' in s]]) or any(np.abs(t) > 1. for t in theta[[i for i, s in enumerate(inposindex) if 'cosi' in s]]): #not totally sure this will work... #for now, set ecc limit to 0.99 to avoid mysterious batman crashes
        return -np.inf
        
    #handle the user-given priors
    priorindex=priorstruc['index']
    npriors=len(priorindex)
    lp=0.0
    for i in range (0, npriors):
        if any(priorindex[i] in s for s in inposindex): 
            lp+=(parstruc[priorindex[i]]-np.float(instruc[priorindex[i]]))**2/np.float(priorstruc[priorindex[i]])**2
            if args.getprob:
                print 'The value of the prior for ',priorindex[i],' is ',(parstruc[priorindex[i]]-np.float(instruc[priorindex[i]]))**2/np.float(priorstruc[priorindex[i]])**2
        #doing it this way handles any entries in the prior file that don't correspond to fit variables
        #but will probably want to modify this if want to be able to put penalties on parameters that are derived from fit variables
        #I'll work on that later...


    lp*=(-0.5)

    if not np.isfinite(lp):
        return 0.0

    return lp

def lnprob(theta, data, nplanets, priorstruc, inposindex, instruc, args):
    data['count']+=1

    if args.time: 
        thisisthestart=timemod.time()-1457540000.
        print 'starting ',thisisthestart
    parstruc = dict(zip(inposindex, theta))

    lnl=0.

    if not any('none' in s for s in priorstruc) and not args.pt: 
        lp = lnprior(parstruc, priorstruc, instruc, theta, inposindex)
        if not np.isfinite(lp) or np.isnan(lp):
            if args.time: print 'ending, prior out of range ',thisisthestart
            return -np.inf
        lnl+=lp

    lnl+= lnlike(theta, parstruc, data, nplanets, inposindex, instruc, args)

    if np.isnan(lnl):
        print 'The likelihood returned a NaN',lp,lnl
        return -np.inf #take care of nan crashes, hopefully...

    if args.verbose:
        print data['count'], lnl

    if args.getprob and args.rvs:
        print 'The total value of lnprob is ',lnl
        print 'The BIC is ',np.log(len(data['rv'])+len(data['pflux']))*len(parstruc)-2.*lnl
        

    if args.time:
        thisistheend=timemod.time()
        print 'ending ',thisisthestart

    return lnl

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

    if args.binary: 
        data['binfflag']=binfflag
        if binfflag == 'rprsfluxr':
            inpos, inposindex, perturbs = np.append(inpos,fluxrat), np.append(inposindex, index[[i for i, s in enumerate(index) if 'fluxrat' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'fluxrat' in s]], dtype=np.float))
        if binfflag == 'mycomb':
            combs=rprs**2*fluxrat
            inpos, inposindex, perturbs = np.append(inpos,combs), np.append(inposindex, index[[i for i, s in enumerate(index) if 'fluxrat' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'fluxrat' in s]], dtype=np.float))
            
        

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


ndim=len(inpos)

inpos=np.array(inpos,dtype=np.float)

#see if there is already an old chain file to load
try:
    chainin=np.load(chainfile)
except IOError:
    if not args.pt:
        pos = [inpos + perturbs*np.random.randn(ndim) for i in range(nwalkers)]
    else:
        ntemps=np.int(struc1['ntemps'])
        pos = np.zeros((ntemps,nwalkers,ndim))
        for i in range (0,ntemps): 
            for j in range (0,nwalkers): 
                pos[i,j,:] = inpos + perturbs*np.random.randn(ndim)
    loaded=False
else:
    if not args.startnew:
        chainshape=chainin.shape
        nin=chainshape[1]
        pos=chainin[:,nin-1,:]
        probin=np.load(probfile)
        loaded=True
    else:
        if not args.pt:
            pos = [inpos + perturbs*np.random.randn(ndim) for i in range(nwalkers)]
        else:
            ntemps=np.int(struc1['ntemps'])
            pos = np.zeros((ntemps,nwalkers,ndim))
            for i in range (0,ntemps): 
                for j in range (0,nwalkers): 
                    pos[i,j,:] = inpos + perturbs*np.random.randn(ndim)
        loaded=False

#check for bad starting values
#still need to add GP parameters
pos=np.array(pos)
if not loaded:
    for i in range (0,nwalkers):
        for j in range (0,ndim):
            if not args.pt:
                while ('Per' in inposindex[j] and pos[i,j] <= 0.0) or ('rprs' in inposindex[j] and (pos[i,j] <= 0.0 or pos[i,j] >= 1.0)) or ('sesinw'  in inposindex[j] and np.abs(pos[i,j]) > 1.0) or ('secosw'  in inposindex[j] and np.abs(pos[i,j]) > 1.0) or ('rhostar' in inposindex[j] and pos[i,j] <= 0.0) or ('aors' in inposindex[j] and pos[i,j] < 1.0) or ('q1' in inposindex[j] and pos[i,j] <= 0.0) or ('q2' in inposindex[j] and pos[i,j] <= 0.0) or ('q1' in inposindex[j] and pos[i,j] >= 1.0) or ('q2' in inposindex[j] and pos[i,j] >= 1.0) or ('ecc' in inposindex[j] and 'cos' not in inposindex[j] and pos[i,j] >= 0.99) or ('ecc' in inposindex[j] and 'cos' not in inposindex[j] and pos[i,j] < 0.0) or ('vsini' in inposindex[j] and pos[i,j] <= 0.0) or ('semiamp' in inposindex[j] and pos[i,j] <= 0.0) or ('intwidth' in inposindex[j] and pos[i,j] <= 0.0) or ('jitter' in inposindex[j] and pos[i,j] <= 0.0) or ('cosi' in inposindex[j] and pos[i,j] < 0.)  or ('parP' in inposindex[j] and pos[i,j] <= 0.) or ('paramp'in  inposindex[j] and pos[i,j] <= 0.) or ('partau' in inposindex[j] and pos[i,j] <= 0.) or ('parGamma' in inposindex[j] and pos[i,j] < 0.):
                    print inposindex[j],pos[i,j],j,i
                    pos[i,j]=inpos[j]+perturbs[j]*np.random.randn(1)
            else: 
                for k in range (0,ntemps):
                    while ('Per' in inposindex[j] and pos[k,i,j] <= 0.0) or ('rprs' in inposindex[j] and (pos[k,i,j] <= 0.0 or pos[k,i,j] >= 1.0)) or ('sesinw'  in inposindex[j] and np.abs(pos[k,i,j]) > 1.0) or ('secosw'  in inposindex[j] and np.abs(pos[k,i,j]) > 1.0) or ('rhostar' in inposindex[j] and pos[k,i,j] <= 0.0) or ('aors' in inposindex[j] and pos[k,i,j] <= 0.0) or ('q1' in inposindex[j] and pos[k,i,j] <= 0.0) or ('q2' in inposindex[j] and pos[k,i,j] <= 0.0) or ('q1' in inposindex[j] and pos[k,i,j] >= 1.0) or ('q2' in inposindex[j] and pos[k,i,j] >= 1.0) or ('ecc' in inposindex[j] and 'cos' not in inposindex[j] and pos[k,i,j] >= 0.99) or ('ecc' in inposindex[j] and 'cos' not in inposindex[j] and pos[k,i,j] < 0.0) or ('vsini' in inposindex[j] and pos[k,i,j] <= 0.0) or ('semiamp' in inposindex[j] and pos[k,i,j] <= 0.0) or ('intwidth' in inposindex[j] and pos[k,i,j] <= 0.0) or ('jitter' in inposindex[j] and pos[k,i,j] <= 0.0) or ('cosi' in inposindex[j] and pos[k,i,j] < 0.) or ('parP' in inposindex[j] and pos[k,i,j] <= 0.) or ('paramp'in  inposindex[j] and pos[k,i,j] <= 0.) or ('partau' in inposindex[j] and pos[k,i,j] <= 0.) or ('parGamma' in inposindex[j] and pos[k,i,j] < 0.):
                        print inposindex[j],pos[k,i,j],j,i,k
                        pos[k,i,j]=inpos[j]+perturbs[j]*np.random.randn(1)
                


if args.plotbest or args.getprob:
    if not args.startnew:
    #get the best-fit parameters from the loaded chain and plot them
        inpos=np.zeros(ndim)
        if any('nburnin' in s for s in index): 
            nburnin=int(struc1['nburnin'])
        else:
            nburnin=nin/5
        samples=chainin[:,nburnin:,:].reshape((-1,ndim))
        if args.bestprob:
            best=np.where(probin == np.max(probin))
        for i in range (0,ndim):
            if args.bestprob:
                temp=[chainin[best[0][0],best[1][0],i], 0, 0]
            else:
                if 'bpar' in inposindex[i] and not args.tomography:
                    v=np.percentile(np.abs(samples[:,i]), [16, 50, 84], axis=0)
                else: 
                    v=np.percentile(samples[:,i], [16, 50, 84], axis=0)
                temp=[v[1], v[2]-v[1], v[1]-v[0]]
        
            print 'The best fit value of ',inposindex[i],' is ',temp[0],' + ', temp[1],' - ',temp[2]
            inpos[i]=temp[0]
        print 'for ',nin,' steps total and cutting off the first ',nburnin,' steps'
    else:
        for i in range (0,ndim):
            print 'The starting value of ',inposindex[i],' is ',inpos[i]
    namelength=len(struc1['plotfile'])
    if struc1['plotfile'][namelength-4:namelength] == '.pdf':
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(struc1['plotfile'])

    
    
data['count']=0


#set up and run the MCMC sampler
if not args.getprob and not args.plotbest:

    if not args.pt:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, nplanets, priorstruc, inposindex, struc1, args), threads=nthreads)

        sampler.run_mcmc(pos, nsteps)

    else:
        
        sampler=emcee.PTSampler(ntemps, nwalkers, ndim, lnprob, lnpriorPT, threads=nthreads,loglargs=(data, nplanets, priorstruc, inposindex, struc1, args),logpargs=(data, nplanets, priorstruc, inposindex, struc1, args))

        for p, lnprob, lnlike in sampler.sample(pos, iterations=nsteps/10):
            pass
        sampler.reset()

        for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob, lnlike0=lnlike, iterations=nsteps):
            pass
            #let's see if this works.....




    print 'The order of your parameters is,',inposindex

    if loaded == False:
        np.save(chainfile, np.array(sampler.chain))
        np.save(probfile, np.array(sampler.lnprobability))
        np.save(accpfile,np.array(sampler.acceptance_fraction))
        chain=np.array(sampler.chain)
    if loaded == True:
        chain=np.array(sampler.chain)
        prob=np.array(sampler.lnprobability)
        accp=np.array(sampler.acceptance_fraction)
        chain2=np.append(chainin,chain,axis=1)
        prob2=np.append(probin,prob,axis=1)
        accpin=np.load(accpfile)
        accpin*=nin #to get total number of acceptances
        accp*=nsteps #ditto
        accp2=accpin+accp
        accp2=accp2/(nin+nsteps) #to get back to fraction
        np.save(chainfile, chain2)
        np.save(probfile, prob2)
        np.save(accpfile, accp2)
        chain=chain2

    if any('asciiout' in s for s in index): 
        if any('nburnin' in s for s in index): 
            nburnin=struc1['nburnin']
        else:
            nburnin=0
        samples=chain[:,nburnin:,:].reshape((-1,ndim))
        f=open(struc1['asciiout'], 'w')
        for j in range (0,ndim):
            f.write(str(inposindex[j]))
            if j != ndim-1: f.write('\t')
        f.write('\n')
        for i in range (0,(nsteps-nburnin)*nwalkers):
            for j in range (0,ndim):
                f.write(str(samples[i,j]))
                if j != ndim-1: f.write('\t')
            f.write('\n')
        f.close()

else:
    lnprob1=lnprob(inpos, data, nplanets, priorstruc, inposindex, struc1, args)

temp=chain.shape
thewholeend=timemod.time()
print 'This run of MISTTBORN took a total time of ',thewholeend-thewholestart,' seconds, for ',nsteps,' steps and a total length of ',temp[1],' steps'
