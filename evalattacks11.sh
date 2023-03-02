#!/bin/bash -e
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --job-name=jobName
#SBATCH --output=allattacks/slurm_%j.out
#SBATCH --error=allattacks/slurm_%j.err
#SBATCH --chdir=/scratch/sca321/robustness/attacks/ameya/Transformer-attacks-master/latesttimm/pytorch-image-models
#SBATCH --mem-per-cpu=24G
#SBATCH --gres=gpu:1

module purge

singularity exec --nv \
        --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
        --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
	    --overlay /scratch/sca321/conda/simple2/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python attacks_optimized.py -o alloutputseps4 -mt $1 --gpu 0 -dpath $2  -it $3 -mp $4 -ni $5 -clip -lr $6 -ps $7 -si $8 -eps $9;"