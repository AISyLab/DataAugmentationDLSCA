#!/bin/bash -l

#####################
# job-array example #
#####################

#SBATCH --job-name=ascadr_augmentation
#SBATCH --qos=short
#SBATCH --time=03:00:00

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

srun python script_data_augmentation.py --dataset_name ascad-variable --leakage_model ID --desync $1 --desync_level $2 --desync_level_augmentation $3 --file_id $4 --n_prof $5 --n_augmented $6 --data_augmentation_per_epoch $7 --augmented_traces_only $8
# srun python script_data_augmentation.py --dataset_name ascad-variable --leakage_model ID --desync $1 --gaussian_noise $1 --std $2 --std_augmentation $3 --file_id $4 --n_prof $5 --n_augmented $6 --data_augmentation_per_epoch $7 --augmented_traces_only $8
exit 0


