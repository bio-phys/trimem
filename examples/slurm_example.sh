#!/bin/bash -l
#
#-------------------------------------------------------------
#running a shared memory (multithreaded) job over multiple CPUs
#-------------------------------------------------------------
#
#SBATCH --job-name=trlmp_example
#SBATCH --chdir=/nfs/scistore15/saricgrp/ *** directory where python simfile is located example.py
#SBATCH --exclude=zeta[243-263]




#
#Number of CPU cores to use within one node
#SBATCH --nodes 1
#SBATCH -c 16
#
#Define the number of hours the job should run.
#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=4:00:00
#
#Define the amount of RAM used by your job in GigaBytes
#In shared memory applications this is shared among multiple CPUs
#SBATCH --mem=16G

#
#Send emails when a job starts, it is finished or it exits
#SBATCH --mail-user= *** your email ***
#SBATCH --mail-type=ALL
#
#Pick whether you prefer requeue or not. If you use the --requeue
#option, the requeued job script will start from the beginning,
#potentially overwriting your previous progress, so be careful.
#For some people the --requeue option might be desired if their
#application will continue from the last state.
#Do not requeue the job in the case it fails.
#SBATCH --no-requeue
#
#Do not export the local environment to the compute nodes
#SBATCH --export=None
env > 'env_file.dat'
unset SLURM_EXPORT_ENV

#
#Set the number of threads to the SLURM internal variable
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


#
#load the respective software module you intend to use
# here one would need to load the conda environment for Trilmp, here called Trimenv
module purge
module load anaconda3/2023.04
source /mnt/nfs/clustersw/Debian/bullseye/anaconda3/2023.04/activate_anaconda3_2023.04.txt

conda activate Trimenv


export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
#
#run the respective binary through SLURM's srun
srun --cpu_bind=verbose python3 example.py