import os, sys 

def train_optuna(hr=2): 
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
        "#SBATCH --gres=gpu:1"
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


def train_weighted_optuna(hr=2): 
    ''' train p(Omega|X) with weights
    '''
    jname = 'p_omega_x_w'
    ofile = '/home/chhahn/projects/goleta/bin/o/_p_omega_x_w'
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
        "#SBATCH --gres=gpu:1"
        '#SBATCH --partition=mig',
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python /home/chhahn/projects/goleta/bin/p_omega_x_weighted.py",
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


def train_qphi_omega_theta_w_optuna(subgrid,hr=2): 
    ''' train p(Omega|theta) with weights
    '''
    jname = 'p_omega_theta_w'
    ofile = '/home/chhahn/projects/goleta/bin/o/_p_omega_theta_w'
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
        "#SBATCH --gres=gpu:1"
        '#SBATCH --partition=mig',
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python /home/chhahn/projects/goleta/bin/p_omega_theta_weighted.py %s" % subgrid,
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


def cosmo_infer(fnde, hr=2, gpu=True, samp='emcee', reset=False, mig=False): 
    ''' train p(Omega|X) with weights
    '''

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J cosmo_infer",
        "#SBATCH --output=o/_cosmo_infer.%s" % samp, 
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
        "python /home/chhahn/projects/goleta/bin/cosmo_infer.py %s %s %s" % (fnde, samp, str(reset)),
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

cosmo_infer('/tigress/chhahn/cgpop/ndes/qphi.omega_x/qphi.omega_x.16.pt', hr=1, gpu=True, samp='emcee', reset=False, mig=False)
#cosmo_infer('/tigress/chhahn/cgpop/ndes/qphi.omega_x/qphi.omega_x.16.pt', hr=4, gpu=True, samp='zeus')

#for i in range(5): 
#    train_weighted_optuna(hr=6)
#    train_optuna(hr=6)

#train_weighted_optuna(hr=6)
#train_qphi_omega_theta_w_optuna('tng', hr=1)
