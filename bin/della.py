import os, sys 

def train_optuna(hr=2, gpu=False, mig=False): 
    ''' train p(Omega|X) 
    '''
    jname = 'p_omega_x'
    ofile = '/home/chhahn/projects/goleta/bin/o/_p_omega_x'
    while os.path.isfile(ofile): 
        jname += '_'
        ofile += '_'

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % jname,
        "#SBATCH --output=%s" % ofile, 
        "#SBATCH --nodes=1", 
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --export=ALL",
        "#SBATCH --mem=4G", 
        ['', "#SBATCH --gres=gpu:1"][gpu], 
        ['', '#SBATCH --partition=mig'][mig], 
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python /home/chhahn/projects/goleta/bin/p_omega_x.py", 
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None


def train_p_theta_omega(sim, hr=2, gpu=False, mig=False): 
    ''' train p(theta|Omega) 
    '''
    jname = 'p_theta_omega'
    ofile = '/home/chhahn/projects/goleta/bin/o/_p_theta_omega'
    while os.path.isfile(ofile): 
        jname += '_'
        ofile += '_'

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % jname,
        "#SBATCH --output=%s" % ofile, 
        "#SBATCH --nodes=1", 
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --export=ALL",
        "#SBATCH --mem=4G", 
        ['', "#SBATCH --gres=gpu:1"][gpu], 
        ['', '#SBATCH --partition=mig'][mig], 
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python /home/chhahn/projects/goleta/bin/p_theta_omega.py %s" % sim, 
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None


def cosmo_infer(ichain, iensemble='all', hr=2, gpu=True, reset=False, mig=False): 
    ''' train p(Omega|X) with weights
    '''

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J cosmo_infer.%s.%i" % (str(iensemble), ichain),
        "#SBATCH --output=o/_cosmo_infer.%s.%i" % (str(iensemble), ichain), 
        "#SBATCH --nodes=1", 
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --export=ALL",
        ['', "#SBATCH --gres=gpu:1"][gpu], 
        ['', '#SBATCH --partition=mig'][mig],
        "#SBATCH --mem=4G", 
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python /home/chhahn/projects/goleta/bin/cosmo_infer.py %i %s %s" % (ichain, str(reset), str(iensemble)),
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None




#for j in range(5):
#    for i in range(10):
#        cosmo_infer(i, iensemble=j, hr=4, gpu=True, reset=False, mig=False)

#for i in range(1, 10):
#    cosmo_infer(i, hr=2, gpu=True, reset=True, mig=True)

#for i in range(20): 
#    train_optuna(hr=24, gpu=True, mig=True)

#train_p_theta_omega('simba', hr=1, gpu=True, mig=True)
#train_p_theta_omega('tng', hr=1, gpu=True, mig=True)
for i in range(10): 
    train_p_theta_omega('simba', hr=24, gpu=True, mig=True)
    train_p_theta_omega('tng', hr=24, gpu=True, mig=True)

#train_weighted_optuna(hr=6)
#train_qphi_omega_theta_w_optuna('tng', hr=1)
