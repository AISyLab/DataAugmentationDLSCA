#!/bin/bash -l

#####################
# job-array example #
#####################

#SBATCH --job-name=ascadr_augmentation
#SBATCH --qos=short
#SBATCH --time=04:00:00

# 500MB memory per core
# this is a hard limit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=jobspernode:four:1
#SBATCH --mem-per-cpu=3500MB
#SBATCH --gres=gpu:turing:1

# you may not place bash commands before the last SBATCH directive
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.6.0.64
module list

cd /home/nfs/gperin/AISY_Framework
source env/bin/activate
cd /home/nfs/gperin/paper_1_data_augmentation_paper

# srun python script_data_augmentation.py --dataset_name ascad-variable --leakage_model $1 --desync $2 --desync_level $3 --desync_level_augmentation $4 --file_id $5 --n_prof $6 --n_augmented $7 --data_augmentation_per_epoch $8 --augmented_traces_only $9
srun python script_data_augmentation.py --dataset_name ascad-variable --leakage_model $1 --gaussian_noise $2 --std $3 --std_augmentation $4 --file_id $5 --n_prof $6 --n_augmented $7 --data_augmentation_per_epoch $8 --augmented_traces_only $9
exit 0


