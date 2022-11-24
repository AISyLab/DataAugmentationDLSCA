#!/bin/bash -l

#####################
# job-array example #
#####################

#SBATCH --job-name=ascadr_search
#SBATCH --qos=short
#SBATCH --time=04:00:00

# 500MB memory per core
# this is a hard limit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=jobspernode:four:1
#SBATCH --mem-per-cpu=6000MB
#SBATCH --gres=gpu:turing:1

# you may not place bash commands before the last SBATCH directive
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.6.0.64
module list

cd /home/nfs/gperin/AISY_Framework
source env/bin/activate
cd /home/nfs/gperin/paper_1_data_augmentation_paper

srun python script_search.py $1 $2 $3 $4 $5 $6 $7 $8
exit 0


