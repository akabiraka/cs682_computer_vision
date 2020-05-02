#!/usr/bin/sh
#SBATCH --job-name=3x_human_pose_entimation
#SBATCH --qos=csqos
##SBATCH --workdir=/scratch/akabir4/human_pose_estimation
#SBATCH --output=/scratch/akabir4/human_pose_estimation/output_models/log_3x-%N-%j.output
#SBATCH --error=/scratch/akabir4/human_pose_estimation/output_models/log_3x-%N-%j.error
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq
#SBATCH --mem=64G


##this is the first testing file.  the path should be started from the "--workdir"
python run.py
