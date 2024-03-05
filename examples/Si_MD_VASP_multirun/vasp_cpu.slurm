#!/bin/bash
#SBATCH --nodes=1           # 1 node reserved 
#SBATCH --ntasks-per-node=8# 40 MPI tasks
#SBATCH --cpus-per-task=1   # 1 OpenMP thread/task 
#SBATCH --hint=nomultithread# Desabling hyperthreading
#SBATCH --job-name=VASP
#SBATCH --output=%x.o%j     # Output file %x is the jobname, %j the jobid
#SBATCH --error=%x.o%j      # Error file
#SBATCH --time=00:10:00     # Expected runtime HH:MM:SS (max 100h)
##
## Please, refer to comments below for
## more information about these 4 last options.
#SBATCH --account=def@cpu   # To specify cpu accounting: <account> = echo $IDRPROJ
##SBATCH --partition=<partition>   # To specify partition (see IDRIS web site for more info)
#SBATCH --qos=qos_cpu-dev  # Uncomment for job requiring less than 2 hours
##SBATCH --qos=qos_cpu-t4  # Uncomment for job requiring more than 20h (only one node)
     
# Cleans out the modules loaded in interactive and inherited by default
module purge
# Load the necessary modules
     
module load vasp/5.4.4-mpi-cuda
     
#Create directory on $SCRATCH
SCRATCHDIR=$SCRATCH/$SLURM_JOB_NAME.$SLURM_JOB_ID
RESULTDIR=$SLURM_SUBMIT_DIR/$SLURM_JOB_ID

# WARNIN : files are copied only when the job starts and not when it is submitted.
mkdir $SCRATCHDIR
cp $SLURM_SUBMIT_DIR/* $SCRATCHDIR/
cd $SCRATCHDIR/
srun vasp_std
echo "End of job $SLURM_JOB_ID. Copying files from scratch folder to folder :"
echo $RESULTDIR
mkdir $RESULTDIR
cp $SCRATCHDIR/* $RESULTDIR/
rm -rf $SCRATCHDIR

