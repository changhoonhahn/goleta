'''


convenient functions 


'''
import os 
import glob
import numpy as np 

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


if os.environ['machine'] == 'della': 
    dat_dir = '/tigress/chhahn/goleta/'
else: 
    raise ValueError


def read_best_ndes(study_name, n_ensemble=5, device='cpu', return_i_models=False): 
    fevents = glob.glob(os.path.join(dat_dir, 'ndes/%s/*/events*' % study_name))

    events, best_valid = [], []
    for fevent in fevents: 
        ea = EventAccumulator(fevent)
        ea.Reload()

        try: 
            best_valid.append(ea.Scalars('best_validation_log_prob')[0].value)
            events.append(fevent)
        except: 
            pass #print(fevent)

    best_valid = np.array(best_valid)
    print('%i models trained' % np.max([int(os.path.dirname(event).split('.')[-1]) 
        for event in events]))
    
    i_models = [int(os.path.dirname(events[i]).split('.')[-1]) for i 
            in np.argsort(best_valid)[-n_ensemble:][::-1]]
    print(i_models) 
    
    qphis = []
    for i_model in i_models: 
        fqphi = os.path.join(dat_dir, 'ndes/%s/%s.%i.pt' % (study_name, study_name, i_model))
        qphi = torch.load(fqphi, map_location=device)
        qphis.append(qphi)
    
    if not return_i_models: 
        return qphis
    else: 
        return qphis, i_models
