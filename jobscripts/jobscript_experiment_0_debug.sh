#!/bin/bash
#SBATCH -J AOE-Imagenet  # please replace by short unique self speaking identifier
###SBATCH -N 1         # number of nodes, we only have one
#SBATCH --gres=gpu:v100:1     # type:number of GPU cards
#SBATCH --mem-per-cpu=4000    # main MB per task? max. 500GB/80=6GB
#SBATCH --ntasks-per-node 1   # bigger for mpi-tasks
#SBATCH --cpus-per-task 10    # 10 CPU-threads needed (physcores*2)
#SBATCH --time 120:00:00        # set 120 hour walltime
#
# Standard environment setup
. /usr/local/bin/slurmProlog.sh         # outputs SLURM settings, for debugging
module load cuda                         # load appropriate CUDA module
echo "debug: CUDA_ROOT=$CUDA_ROOT"
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=/usr/local/bin/virtualenv
export WORKON_HOME=/nfs1/$USER

source /nfs1/$USER/miniconda/miniconda/bin/activate
conda activate aoe
# workon $testenv #  "deactivate" to leave "workon ..."
cd /nfs1/$USER/AOE/Adversarial-Outlier-Exposure
echo "pwd=$(pwd)"
python -V #moves to the main directory
srun python3 run.py --config-name cifar_exp_0_debug.yaml >slurm-$SLURM_JOBID.pyout 2>&1
deactivate  # workoff :)
. /usr/local/bin/slurmEpilog.sh   # cleanup




