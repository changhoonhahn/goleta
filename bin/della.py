import os, sys 

def train_optuna(): 
    ''' train p(Omega|X) 
    '''
    jname = 'p_omega_x'
    ofile = "o/_p_omega_x"
    while os.path.isfile(ofile): 
        jname += '_'
        ofile += '_'

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % jname,
        "#SBATCH --output=%s" % ofile, 
        "#SBATCH --nodes=1", 
        "#SBATCH --time=02:00:00",
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

for i in range(10): 
    train_optuna()
