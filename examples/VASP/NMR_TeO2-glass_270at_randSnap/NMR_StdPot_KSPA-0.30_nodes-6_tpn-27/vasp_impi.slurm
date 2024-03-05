#!/bin/bash
#SBATCH -o vasp.%j.out
#SBATCH --nodes 6
#SBATCH --ntasks-per-node 27
# Had to remove the following because it results in mpirun seg fault 
#SBATCH --cpus-per-task 1 
#SBATCH -t 0-99:59:00

module load mpi/intel/2018.5.274 compiler/intel/2018.5.274

export SJID="$SLURM_JOB_NAME"_"`echo $SLURM_JOB_ID | cut -d . -f 1`_`date '+%d%m%Y'`"
SCRDIR=/scratch/$USER/$SJID
mkdir -p $SCRDIR
cd $SCRDIR
for infile in INCAR POTCAR POSCAR KPOINTS
do
  if [ -f $SLURM_SUBMIT_DIR/$infile ]; then
    cp $SLURM_SUBMIT_DIR/$infile $SCRDIR/
  fi
done
mpirun vasp_std > $SLURM_JOB_NAME.out
mkdir $SLURM_SUBMIT_DIR/$SLURM_JOB_ID 
cp $SCRDIR/* $SLURM_SUBMIT_DIR/$SLURM_JOB_ID/
rm -rf $SCRDIR

