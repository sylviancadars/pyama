function jobNumber = submitJob_local()
%-------------------------------------------------------------
%This routine is to check if the submitted job is done or not
%One needs to do a little edit based on your own case.

%1   : whichCluster (default 0, 1: local submission, 2: remote submission)
%-------------------------------------------------------------

%Step 1: to prepare the job script which is required by your supercomputer
% CHANGE -t [WALLTIME] and --ntasks-per-node ACCORDING TO THE DESIRED NB OF CPUS REQUIRED

fp = fopen('myrun', 'w');    
fprintf(fp, '#!/bin/bash\n');
fprintf(fp, '#SBATCH -o out\n');
fprintf(fp, '#SBATCH -J USPEX\n');
fprintf(fp, '#SBATCH -t 00:20:00\n');
fprintf(fp, '#SBATCH --nodes 1\n');
fprintf(fp, '#SBATCH --ntasks-per-node 2\n');
fprintf(fp, '#SBATCH --cpus-per-task 1\n');
fprintf(fp, '\n');
fprintf(fp, 'module load mpi/intel/2018.5.274 compiler/intel/2018.5.274\n');
fprintf(fp, 'SCRDIR=\"/scratch/$USER/$SLURM_JOB_NAME\"_\"$SLURM_JOB_ID\"\n');
fprintf(fp, 'mkdir -p $SCRDIR\n');
fprintf(fp, 'cd $SCRDIR\n');
fprintf(fp, 'cp $SLURM_SUBMIT_DIR/* $SCRDIR/\n');
fprintf(fp, 'mpirun $HOME/bin/vasp_std > vasp.out\n');
fprintf(fp, 'cp ./* $SLURM_SUBMIT_DIR/ && rm -rf $SCRDIR\n');
fprintf(fp, '\n');

fclose(fp);

%Step 2: to submit the job with the command like qsub, bsub, llsubmit, .etc.
%It will output some message on the screen like '2350873.nano.cfn.bnl.local'
[a,b]=unix(['sbatch myrun']) ;
% disp(['Submission command \"sbatch myrun\" returns : ',b])

%Step 3: to get the jobID from the screen message
end_marker = findstr(b,'job') ;
jobNumber = b(end_marker(1)+4:end) ;
% disp(['Str variable jobNumber equals : ',jobNumber])

