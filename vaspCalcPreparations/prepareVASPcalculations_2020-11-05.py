# -*- coding: utf-8 -*-
"""
Script preparing a series of VASP calculations from all structures generated
with the supercell program located in a given folder. VASP input files other
than POSCAR (i.e. POTCAR, INCAR) are copied from a sampleFilesDir folder.

Created on Thu Nov  5 15:36:09 2020

@author: cadarp02
"""

import os
from shutil import copyfile
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar
# import fileinput
# import time

def longestCommonPrefix(strs):
    """
    :type strs: List[str]
    :rtype: str
    """
    if len(strs) == 0:
        return ""
    current = strs[0]
    for i in range(1,len(strs)):
        temp = ""
        if len(current) == 0:
            break
        for j in range(len(strs[i])):
            if j<len(current) and current[j] == strs[i][j]:
                temp+=current[j]
            else:
                break
        current = temp
    return current
    
# ************ MAIN *******************
# TODO : change SYSTEM = <description> in INCAR file

isDryRun = False

# inputStructuresDir =r'D:\cadarp02\Documents\Modeling\supercell\AsTe3\fromBetaAs2Te3\from1BetaAs2Te3Layer_s421\structures2'
# inputStructuresDir =r'D:\cadarp02\Documents\Modeling\supercell\AsTe3\fromBetaAs2Te3\fromBetaAs2Te3_s221\structures2'
inputStructuresDir =r'D:\cadarp02\Documents\Modeling\supercell\AsTe3\fromBetaAs2Te3\fromBetaAs2Te3_s221\structures3'

sampleFilesDir=r'D:\cadarp02\Documents\Modeling\VASP\sampleInputFiles\GO-PBE_2020-11-05'
calcDir = os.path.join(inputStructuresDir,'calcDir2')
calcPrefix = 'calc_'
vasprunfile = 'vasp_impi.slurm'
multirunScriptfileName = 'multirun.exe'
submissionCommand = 'sbatch ./' + vasprunfile
jobIdentifier = 'vasp'
maxNbOfJobs = 20
sleepTime = 120

if isDryRun == True :
    sleepTime = 2
    
if not os.path.exists(calcDir) :
    os.mkdir(calcDir)

listOfStructureFiles = []
fileIndex = -1
for file in os.listdir(inputStructuresDir) :
    if file.endswith(".cif"):
        listOfStructureFiles.append(os.path.join(file))

f = open(os.path.join(calcDir,multirunScriptfileName),'w')
f.write('#!/bin/bash\n')
copyfile(os.path.join(sampleFilesDir,'POTCAR'),os.path.join(calcDir,'POTCAR'))

commonInputPrefix = longestCommonPrefix(listOfStructureFiles)
for file in listOfStructureFiles :
    # Create calculation directories with chosen prefix and without .cif suffix
    currentCalcDir = os.path.join(calcDir,calcPrefix+file[len(commonInputPrefix):-4])
    if not os.path.exists(currentCalcDir) :
        os.mkdir(currentCalcDir)
    # copyfile(os.path.join(sampleFilesDir,'INCAR'),os.path.join(currentCalcDir,'INCAR'))
    # change INCAR file SYSTEM= line
    # for line in fileinput.input(os.path.join(currentCalcDir,'INCAR'),
    #                             inplace=True):
    #     if 'SYSTEM' in line :
    #         print('SYSTEM = {}'.format('test'), end='')
    f2 = open(os.path.join(currentCalcDir,'INCAR'),'w')
    with open(os.path.join(sampleFilesDir,'INCAR')) as f3:
        for line in f3:
            if 'SYSTEM' in line :
                f2.write('SYSTEM = '+file+'\n')
            else :
                f2.write(line)
    f2.close()
    
    # POTCAR will be copied into the current calculation file by the script before submission
    copyfile(os.path.join(sampleFilesDir,vasprunfile),os.path.join(currentCalcDir,vasprunfile))
    
    # Reading structure from cif file and creating POSCAR file
    structure = Structure.from_file(os.path.join(inputStructuresDir,file),primitive=False,sort=True)
    poscar = Poscar(structure)
    poscar.write_file(os.path.join(currentCalcDir,'POSCAR'))

    # script : move to current calculation folder and copy POTCAR file therein
    f.write('cd '+calcPrefix+file[len(commonInputPrefix):-4]+'\n')
    f.write('cp ../POTCAR .\n')

    # count number of vasp jobs in queue
    f.write('while [ $(qstat -u $USER | grep '+str(jobIdentifier)+' | wc -l) -ge '+str(maxNbOfJobs)+' ]; do\n')
    f.write('  sleep '+str(sleepTime)+'\n')
    f.write('done\n')
    
    f.write('echo "Ready to run job '+calcPrefix+file[len(commonInputPrefix):-4]+' with command : '+submissionCommand+'"\n')
    if isDryRun == False :
        f.write(submissionCommand+'\n')
    f.write('cd ..\n')
    
f.write('echo "All calculations have been submitted. End of script file '+multirunScriptfileName+'.\n')
f.close()

    # KPOINTS= can be avoided in which case VASP will use KSPACING tag INCAR
    # (which defaults to 0.5 A-1 which seems very large to me)

# EXAMPLE OF BASH SCRIPT TO HELP WRITE 
# while [ ! -f ./USPEX_IS_DONE ]; do
#    date >> log
#    USPEX -r -o >> log
#    # automatically clean log to avoid critical size due to Octave warnings
#    # This line should be removed for debugging
#    tail -n 5000 log > tmp.txt
#    rm -f log
#    cp tmp.txt log
#    rm -f tmp.txt

#    sleep 120
# done


# TODO : in multirunScriptfileName count number of running VASP jobs
# First copy the vasprun script in sampleFilesDir

