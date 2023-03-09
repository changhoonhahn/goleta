'''

script to estimate the distribution: p(Omega | X) 


'''
import os
import h5py 
import numpy as np

import torch
from torch.utils.tensorboard.writer import SummaryWriter

import optuna 

from sbi import utils as Ut
from sbi import inference as Inference

device = ("cuda" if torch.cuda.is_available() else "cpu")

output_dir = '/tigress/chhahn/cgpop/ndes/'

seed = 12387
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

###################################################################
# load CAMELS training data 
###################################################################
dat_dir = '/tigress/chhahn/cgpop/'
data = np.loadtxt(os.path.join(dat_dir, 'camels_tng.omega_x.down.dat'), skiprows=1, unpack=True)

data_omega = data.T[:,:6]     # cosmological and hydrodynamical parameters
data_photo = data.T[:,6:]     # measured photometry and noise

###################################################################
# train normalizing flows
###################################################################
# set prior 
prior_low = [0.1, 0.6, np.log10(0.25), np.log10(0.25), np.log10(0.5), np.log10(0.5)]
prior_high = [0.5, 1.0, np.log10(4.0), np.log10(4.0), np.log10(2.0), np.log10(2.0)]
lower_bounds = torch.tensor(prior_low)
upper_bounds = torch.tensor(prior_high)

prior = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device=device)

# setup training data 
ishuffle = np.arange(data_omega.shape[0])
np.random.shuffle(ishuffle)

N_train = int(0.9 * data_omega.shape[0])
x_train = data_photo[ishuffle[:N_train]]
y_train = data_omega[ishuffle[:N_train]]

N_test = data_omega.shape[0] - N_train
x_test = data_photo[ishuffle[N_train:]]
y_test = data_omega[ishuffle[N_train:]]

# Optuna Parameters
n_trials    = 1000
study_name  = 'qphi.omega_x'
n_jobs     = 1
if not os.path.isdir(os.path.join(output_dir, study_name)): 
    os.system('mkdir %s' % os.path.join(output_dir, study_name))
storage    = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)
n_startup_trials = 20

n_blocks_min, n_blocks_max = 2, 10
n_transf_min, n_transf_max = 2, 10
n_hidden_min, n_hidden_max = 64, 512
n_lr_min, n_lr_max = 1e-4, 1e-2 
p_drop_min, p_drop_max = 0., 1.
clip_max_min, clip_max_max = 1., 5.


def Objective(trial):
    ''' objective function for optuna 
    '''
    # Generate the model                                         
    n_blocks = trial.suggest_int("n_blocks", n_blocks_min, n_blocks_max)
    n_transf = trial.suggest_int("n_transf", n_transf_min,  n_transf_max)
    n_hidden = trial.suggest_int("n_hidden", n_hidden_min, n_hidden_max, log=True)

    lr  = trial.suggest_float("lr", n_lr_min, n_lr_max, log=True) 

    p_drop = trial.suggest_float("p_drop", p_drop_min, p_drop_max)
    clip_max = trial.suggest_float("clip_max_norm", clip_max_min, clip_max_max) 

    neural_posterior = Ut.posterior_nn('maf', 
            hidden_features=n_hidden, 
            num_transforms=n_transf, 
            num_blocks=n_blocks, 
            dropout_probability=p_drop, 
            use_batch_norm=True)

    anpe = Inference.SNPE(prior=prior,
            density_estimator=neural_posterior,
            device=device, 
            summary_writer=SummaryWriter('%s/%s/%s.%i' % 
                (output_dir, study_name, study_name, trial.number)))

    anpe.append_simulations(
            torch.as_tensor(y_train.astype(np.float32)).to(device),
            torch.as_tensor(x_train.astype(np.float32)).to(device))

    p_theta_x_est = anpe.train(
            training_batch_size=50,
            learning_rate=lr, 
            clip_max_norm=clip_max, 
            show_train_summary=True)
        
    # save trained NPE  
    qphi    = anpe.build_posterior(p_theta_x_est)
    fqphi   = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, trial.number))
    torch.save(qphi, fqphi)

    # calculat ranks:  
    rank_thetas = []
    for i in range(x_test.shape[0]):
        # sample posterior p(theta | x_test_i)
        y_prime = qphi.sample((10000,),
                x=torch.as_tensor(x_test[i].astype(np.float32)).to(device),
                show_progress_bars=False)
        y_prime = np.array(y_prime.detach().cpu())

        # calculate percentile score and rank
        rank_theta = []
        for itheta in range(y_test.shape[1]):
            rank_theta.append(np.sum(y_prime[:,itheta] < y_test[i,itheta]))
        rank_thetas.append(rank_theta)
    rank_thetas = np.array(rank_thetas) 

    np.save(os.path.join(output_dir, study_name, '%s.%i.rank.npy' % (study_name, trial.number)), rank_thetas)
        
    best_valid_log_prob = anpe._summary['best_validation_log_prob'][0]
    return -1*best_valid_log_prob


sampler     = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
study       = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, directions=["minimize"], load_if_exists=True)

study.optimize(Objective, n_trials=n_trials, n_jobs=n_jobs)
print("  Number of finished trials: %i" % len(study.trials))
