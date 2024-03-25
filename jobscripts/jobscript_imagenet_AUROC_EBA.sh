#!/bin/bash
#SBATCH -J AOE-Imagenet  # please replace by short unique self speaking identifier
###SBATCH -N 1         # number of nodes, we only have one
#SBATCH --gres=gpu:v100:1     # type:number of GPU cards
#SBATCH --mem-per-cpu=4000    # main MB per task? max. 500GB/80=6GB
#SBATCH --ntasks-per-node 1   # bigger for mpi-tasks
#SBATCH --cpus-per-task 10    # 10 CPU-threads needed (physcores*2)
#SBATCH --time 120:00:00        # set 120 hour walltime
#
. /usr/local/bin/slurmProlog.sh  # output slurm settings, debugging
module load cuda     # latest cuda, or use cuda/10.0 cuda/10.1 cuda/11.2
echo "debug: CUDA_ROOT=$CUDA_ROOT"
# testenv=py36tf200    # venv for tensorflow (see lsvirtualenv)
testenv=aoe     #      for pyTorch
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=/usr/local/bin/virtualenv
export WORKON_HOME=/nfs1/$USER
source /nfs1/$USER/aoe/bin/activate
# workon $testenv #  "deactivate" to leave "workon ..."
cd /nfs1/$USER/
echo "pwd=$(pwd)"
python -V #moves to the main directory
srun python3 /nfs1/$USER/Adversarial-Outlier-Exposure/run.py --config-name ImageNet_AUROC_EBA.yaml >slurm-$SLURM_JOBID.pyout 2>&1
deactivate  # workoff :)
. /usr/local/bin/slurmEpilog.sh   # cleanup