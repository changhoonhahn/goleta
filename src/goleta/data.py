'''

module to interface with simulation data


'''
import os 
import h5py 
import numpy as np 
from astropy.table import Table 


if os.environ['machine'] == 'della': 
    dat_dir = '/tigress/chhahn/goleta/'
else: 
    raise ValueError


def get_data(dataset, yname, xname, sim='tng', downsample=True): 
    ''' get data set from camels 
    '''
    if not downsample: raise NotImplementedError 

    fdata = os.path.join(dat_dir, 
            'camels_%s%s.%s.dat' % (sim, ['', '.down'][downsample], dataset))
    
    if not os.path.isfile(fdata): 
        with h5py.File(os.path.join(dat_dir, '%s.snap33.subfind.galaxies.LHC.hdf5' % sim), 'r') as f: 
            props = f['props'][...].T

        # Om, s8, Asn1, Aagn1, Asn2, Aagn2, Mg, Mstar, Mbh, Mtot, 
        # Vmax, Vdisp, Zg, Zs, SFR, J, Vel, Rstar, Rtot, Rvmax,
        # absmag U, B, V, K, g, r, i, z
        props[:,2] = np.log10(props[:,2]) # log Asn1
        props[:,3] = np.log10(props[:,3]) # log Aagn1
        props[:,4] = np.log10(props[:,4]) # log Asn2
        props[:,5] = np.log10(props[:,5]) # log Aagn2

        props[:,6] = np.log10(props[:,6]) # log Mgas
        props[:,7] = np.log10(props[:,7]) # log M*
        props[:,10] = np.log10(props[:,10]) # log Vmax
        props[:,13] = np.log10(props[:,13]) # log Zs 
        props[:,17] = np.log10(props[:,17]) # log R*

        icosmo = [0, 1, 2, 3, 4, 5] # cosmological/hydro parameters
        igals = [6, 7, 10, 13, 17] # intrinsic properties of galaixes (Mg, M*, Vmax, Z*, R*)
        iobs = [-4, -3, -2, -1] # g, r, i, z absolute magnitudes

        ikeep = [] 
        w_lhc = np.ones(props.shape[0])
        for lhc in np.unique(np.sum(props[:,icosmo], axis=1)):
            is_lhc = (np.sum(props[:,icosmo], axis=1) == lhc)
            
            N_lhc = np.sum(is_lhc)
            
            w_lhc[is_lhc] = 100./float(N_lhc)
            
            ikeep.append(np.random.choice(np.arange(props.shape[0])[is_lhc], size=100, replace=False))
        ikeep = np.concatenate(ikeep)

        # forward model 
        absmag_sigmas = np.random.uniform(0.019, 0.023, size=(props.shape[0], len(iobs)))
        absmag_sigmas[:,-1] = np.random.uniform(0.029, 0.041, size=props.shape[0])
        absmags = props[:,np.array(iobs)] + absmag_sigmas * np.random.normal(size=(props.shape[0], len(iobs)))
    
        # full data set 
        ishuffle = np.arange(props.shape[0])
        np.random.shuffle(ishuffle)
        N_train = int(0.9 * props.shape[0])

        hdr = ('name: Omega_m, sigma_8, log A_SN1, log A_AGN1, log A_SN2, log A_AGN2, '+ 
                'log M_g, log M_*, log V_max, log Z_*, log R_*, '+ 
                'g absmag, r absmag, i, absmag, z absmag, sigma_g, sigma_r, sigma_i, sigma_z, w_lhc')

        _output = np.concatenate([props[:,icosmo], props[:,igals], 
            absmags, absmag_sigmas, w_lhc[:,None]], axis=1)
        _fdata = os.path.join(dat_dir, 'camels_%s.all.dat' % sim)
        np.savetxt(_fdata, _output, header=hdr)

        _fdata = os.path.join(dat_dir, 'camels_%s.train.dat' % sim)
        np.savetxt(_fdata, _output[ishuffle[:N_train]], header=hdr)
        
        _fdata = os.path.join(dat_dir, 'camels_%s.test.dat' % sim)
        np.savetxt(_fdata, _output[ishuffle[N_train:]], header=hdr)

        # downsampled data set
        _output = np.concatenate([props[:,icosmo], props[:,igals], absmags, absmag_sigmas], axis=1)[ikeep]
        ishuffle = np.arange(_output.shape[0])
        np.random.shuffle(ishuffle)
        N_train = int(0.9 * _output.shape[0])

        hdr = ('name: Omega_m, sigma_8, log A_SN1, log A_AGN1, log A_SN2, log A_AGN2, '+ 
                'log M_g, log M_*, log V_max, log Z_*, log R_*, '+ 
                'g absmag, r absmag, i, absmag, z absmag, sigma_g, sigma_r, sigma_i, sigma_z')
        _fdata = os.path.join(dat_dir, 'camels_%s.down.all.dat' % sim)
        np.savetxt(_fdata, _output, header=hdr)

        _fdata = os.path.join(dat_dir, 'camels_%s.down.train.dat' % sim)
        np.savetxt(_fdata, _output[ishuffle[:N_train]], header=hdr)

        _fdata = os.path.join(dat_dir, 'camels_%s.down.test.dat' % sim)
        np.savetxt(_fdata, _output[ishuffle[N_train:]], header=hdr)

    data = np.loadtxt(fdata, skiprows=1)
    
    if yname == 'omega': 
        y = data[:,:6]
    elif yname == 'theta': 
        y = data[:,6:11]
    elif yname == 'xobs':
        y = data[:,11:19]
    else: 
        raise ValueError('specify yname') 

    if xname == 'omega': 
        x = data[:,:6]
    elif xname == 'theta': 
        x = data[:,6:11]
    elif xname == 'xobs':
        x = data[:,11:19]
    else: 
        raise ValueError('specify xname') 

    return y, x 

