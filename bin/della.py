import os, sys 

def train_optuna(hr=2): 
    ''' train p(Omega|X) 
    '''
    jname = 'p_omega_x'
    ofile = '/home/chhahn/projects/CGPop/bin/o/_p_omega_x'
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
        "python /home/chhahn/projects/CGPop/bin/p_omega_x.py", 
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
    ofile = '/home/chhahn/projects/CGPop/bin/o/_p_omega_x_w'
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
        "python /home/chhahn/projects/CGPop/bin/p_omega_x_weighted.py",
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


def cosmo_infer(fnde, hr=2, gpu=True): 
    ''' train p(Omega|X) with weights
    '''

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J cosmo_infer",
        "#SBATCH --output=o/_cosmo_infer", 
        "#SBATCH --nodes=1", 
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --export=ALL",
        "#SBATCH --mem=4G", 
        ['', "#SBATCH --gres=gpu:1"][gpu], 
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python /home/chhahn/projects/CGPop/bin/cosmo_infer.py %s" % fnde,
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

cosmo_infer('/tigress/chhahn/cgpop/ndes/qphi.omega_x/qphi.omega_x.16.pt', hr=1, gpu=True)

#for i in range(5): 
#    train_weighted_optuna(hr=6)
#    train_optuna(hr=6)
