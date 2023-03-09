'''

script to estimate the distribution: p(Omega | X) with the importance weights


'''
import os
import h5py 
import numpy as np

import copy
import optuna 
import torch
import torch.optim as optim

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
data = np.loadtxt(os.path.join(dat_dir, 'camels_tng.omega_x.dat'), skiprows=1, unpack=True)

data_omega = data.T.astype(np.float32)[:,:5]       # cosmological and hydrodynamical parameters
data_photo = data.T.astype(np.float32)[:,5:-1]     # measured photometry and noise
data_w_lhc = data.T.astype(np.float32)[:,-1][:,None]

###################################################################
# train normalizing flows
###################################################################
# prepare training data 
ishuffle = np.arange(data_omega.shape[0])
np.random.shuffle(ishuffle)

N_train = int(0.8 * data_omega.shape[0])
train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(data_omega[ishuffle[:N_train]]),
            torch.from_numpy(data_photo[ishuffle[:N_train]]),
            torch.from_numpy(data_w_lhc[ishuffle[:N_train]])),
        batch_size=128, shuffle=True)

N_valid = int(0.1 * data_omega.shape[0])
valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(data_omega[ishuffle[N_train:N_train+N_valid]]),
            torch.from_numpy(data_photo[ishuffle[N_train:N_train+N_valid]]),
            torch.from_numpy(data_w_lhc[ishuffle[N_train:N_train+N_valid]])),
        batch_size=128, shuffle=True)

# Optuna Parameters
n_trials    = 1000
study_name  = 'qphi.omega_x.weighted'
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
    # initialize
    npe = neural_posterior(torch.from_numpy(data_omega), torch.from_numpy(data_photo))
    npe.to(device)

    # train NDE 
    optimizer = optim.Adam(list(npe.parameters()), lr=lr)
    # set up scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=1000)

    best_valid_loss, best_epoch = np.inf, 0
    train_losses, valid_losses = [], []
    for epoch in range(1000):

        npe.train()

        train_loss = 0.
        for batch in train_loader:
            optimizer.zero_grad()
            
            omega, photo, w_lhc = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
    
            logprobs = npe.log_prob(omega, context=photo) 
            
            loss = -torch.sum(logprobs * w_lhc) 
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        train_loss /= len(train_loader.dataset)

        with torch.no_grad():
            valid_loss = 0.  
        
            for batch in valid_loader:
                omega, photo, w_lhc = (batch[0].to(device), batch[1].to(device), batch[2].to(device))

                logprobs = npe.log_prob(omega, context=photo) 
            
                loss = -torch.sum(logprobs * w_lhc) 

                valid_loss += loss.item()
            valid_loss /= len(valid_loader.dataset)

        if epoch % 10 == 0:
            print('Epoch %i Training Loss %.2e Validation Loss %.2e' % (epoch, train_loss, valid_loss))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_npe = copy.deepcopy(npe)

        if epoch > best_epoch + 20: break
        scheduler.step()

    # save trained NPE  
    fqphi   = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, trial.number))
    torch.save(qphi, fqphi)

    # calculat ranks:  
    rank_thetas = []
    for i in np.arange(x_test.shape[0]):
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
