'''

script to conduct the cosmological inference 

'''
import os, sys
import h5py
import numpy as np
import astropy.table as aTable

import torch
from sbi import utils as Ut
from sbi import inference as Inference

import emcee, zeus


fnde = sys.argv[1]
emze = sys.argv[2]
reset = (sys.argv[3] == 'True') 

if emze != 'emcee': raise ValueError

device = ("cuda" if torch.cuda.is_available() else "cpu")
#####################################################################
# load qphi(Omega | X) 
#####################################################################
qphi_omega_X = torch.load(fnde, map_location=torch.device(device))
qphi_omega_X._device = device

#####################################################################
# load NSA within conservative support 
#####################################################################
nsa = aTable.Table.read('/tigress/chhahn/cgpop/nsa_v0_1_2.fits')
absmag_nsa = np.array(nsa['ABSMAG'].data)[:,3:] # g, r, i, z
ivar_absmag_nsa = np.array(nsa['AMIVAR'].data)[:,3:]

colors_nsa = np.array([absmag_nsa[:,0] - absmag_nsa[:,1],
                         absmag_nsa[:,0] - absmag_nsa[:,2],
                         absmag_nsa[:,0] - absmag_nsa[:,3],
                         absmag_nsa[:,1] - absmag_nsa[:,2],
                         absmag_nsa[:,1] - absmag_nsa[:,3],
                         absmag_nsa[:,2] - absmag_nsa[:,3]]).T

# cuts on absmag uncertainties
cuts = ((nsa['Z'] < 0.05) & 
        np.all((ivar_absmag_nsa[:,:-1]**-0.5 > 0.02) & (ivar_absmag_nsa[:,:-1]**-0.5 < 0.022), axis=1) & 
        (ivar_absmag_nsa[:,-1]**-0.5 > 0.03) & (ivar_absmag_nsa[:,-1]**-0.5 < 0.04))
# cuts on absmag 
for i in range(4): 
    cuts = cuts & (absmag_nsa[:,i] < -18) & (absmag_nsa[:,i] > -22.)
# cuts on color 
cuts = cuts & (colors_nsa[:,0] > 0.264) & (colors_nsa[:,0] < 0.687)
cuts = cuts & (colors_nsa[:,1] > 0.423) & (colors_nsa[:,1] < 1.009)
cuts = cuts & (colors_nsa[:,2] > 0.538) & (colors_nsa[:,2] < 1.231)
cuts = cuts & (colors_nsa[:,3] > 0.151) & (colors_nsa[:,3] < 0.329)
cuts = cuts & (colors_nsa[:,4] > 0.263) & (colors_nsa[:,4] < 0.557)
cuts = cuts & (colors_nsa[:,5] > 0.097) & (colors_nsa[:,5] < 0.241)

# get Xs 
Xs = np.concatenate([absmag_nsa[cuts], ivar_absmag_nsa[cuts]**-0.5], axis=1)

#####################################################################
# sample posterior 
#####################################################################

def log_posterior(theta, Xs):
    if ((theta[0] < 0.15) | (theta[0] > 0.45) | (theta[1] < 0.65) | (theta[1] > 0.95) |
        (theta[2] < np.log10(0.25)) | (theta[2] > np.log10(4.)) |
        (theta[3] < np.log10(0.25)) | (theta[3] > np.log10(4.)) |
        (theta[4] < np.log10(0.5)) | (theta[4] > np.log10(2.)) |
        (theta[5] < np.log10(0.5)) | (theta[5] > np.log10(2.))):
        return -np.inf

    _theta = np.tile(theta, (Xs.shape[0],1))
    # this exlcudes the leaky correction, which I'm not sure how much that matters
    logprobs = qphi_omega_X.posterior_estimator.log_prob(
            torch.tensor(_theta.astype(np.float32)).to(device),
            context=torch.tensor(Xs.astype(np.float32)).to(device))

    return np.float(torch.sum(logprobs).cpu())

ndim, nwalkers = 6, 200

# initialize walkers
if reset: 
    p0 = []
    while len(p0) < nwalkers: 
        _p0 = 0.2 * np.random.randn(1, ndim) + np.array([0.3, 0.8, 0., 0., 0., 0.]) 
        if np.isfinite(log_posterior(_p0[0], Xs[::10])): 
            p0.append(_p0[0])

backend = emcee.backends.HDFBackend('/tigress/chhahn/cgpop/omega_mcmc.emcee.%s.h5' % os.path.basename(fnde).split('.pt')[0])
if reset: backend.reset(nwalkers, ndim)

# run mcmc
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[Xs], backend=backend)
if reset: _ = sampler.run_mcmc(p0, 2000, progress=True)
else: _ = sampler.run_mcmc(None, 2000, progress=True)

chains = sampler.get_chain()
np.save('/tigress/chhahn/cgpop/omega_mcmc.emcee.%s.npy' % os.path.basename(fnde).split('.pt')[0], chains) 

#if emze == 'emcee': 
#elif emze == 'zeus': 
#    cb = zeus.callbacks.SaveProgressCallback('/tigress/chhahn/cgpop/omega_mcmc.zeus.%s.h5' % os.path.basename(fnde).split('.pt')[0], ncheck=100)
#    sampler = zeus.EnsembleSampler(nwalkers, ndim, log_posterior, args=[Xs])
#    chains = sampler.run_mcmc(p0, 200, progress=True, callbacks=cb)
#
#    np.save('/tigress/chhahn/cgpop/omega_mcmc.zeus.%s.npy' % os.path.basename(fnde).split('.pt')[0], chains) 
