# -*- coding: utf-8 -*-
"""
Script preparing a series of VASP calculations from all structures generated
with the supercell program located in a given folder. VASP input files other
than POSCAR (i.e. POTCAR, INCAR) are copied from a sampleFilesDir folder.

A multirun.exe script will be created which should then be executed (with nohup) to submit calculations

Created on Thu Nov  5 15:36:09 2020

@author: cadarp02
"""

import os
from shutil import copyfile
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar, Incar, Potcar
import numpy as np
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
# Set to isDryRun to True to test the creation of all files without actually submitting jobs
isDryRun = False
limitNbOfJobs = False  # If true job should be submitted with nohup, which may cause problems with some job schedulers

# inputStructuresDir =r'/gpfs/home/scadars/supercell/AsTe3/DRX-models/AsTe3_C2m-5sites_As2.88_Te8.64/As3-Ec001-l0295/Te0-Ec0001-l1087/Te123'
# inputStructuresDir =r'/gpfs/home/scadars/supercell/AsTe3/DRX-models/AsTe3_C2m-5sites_As2.88_Te8.64/As3-Ec001-l0295/Te0-Ec0002-l1789/Te123'
# inputStructuresDir =r'/gpfs/home/scadars/supercell/AsTe3/DRX-models/AsTe3_C2m-5sites_As2.88_Te8.64/As3-Ec001-l0295/Te0-Ec0003-l0826/Te123'
# inputStructuresDir =r'/gpfs/home/scadars/supercell/AsTe3/DRX-models/AsTe3_C2m-5sites_As2.88_Te8.64/As3-Ec002-l0537/Te0-Ec0001-l0893/Te123'
# inputStructuresDir =r'/gpfs/home/scadars/supercell/AsTe3/DRX-models/AsTe3_C2m-5sites_As2.88_Te8.64/As3-Ec002-l0537/Te0-Ec0002-l1704/Te123'
inputStructuresDir =r'/gpfs/home/scadars/supercell/AsTe3/DRX-models/AsTe3_C2m-5sites_As2.88_Te8.64/As3-Ec002-l0537/Te0-Ec0003-l0926/Te123'

calcDir = os.path.join(inputStructuresDir,'calcDir')

sampleFilesDir=r'/gpfs/home/scadars/VASP/AsTe3/DRX-models/sampleInputFiles'
calcPrefix = 'calc_'
vasprunfile = 'vasp_impi.slurm'
multirunScriptfileName = 'multirun.exe'
submissionCommand = 'sbatch ./' + vasprunfile
jobIdentifier = 'vasp'
maxNbOfJobs = 20
sleepTime = 120
# Calculation Parameters (job- or system-dependent parameters are set automatically below)
# this includes ENCUT (based in ENMAX in POTCAR) and SYSTEM
encutToEnmaxRatio = 1.0
incarDict={}
incarDict['KSPACING'] = '0.40'

if isDryRun == True :
    sleepTime = 2
    
if not os.path.exists(calcDir) :
    os.mkdir(calcDir)

listOfStructureFiles = []
fileIndex = -1
for fileName in os.listdir(inputStructuresDir) :
    if fileName.endswith(".cif"):
        listOfStructureFiles.append(os.path.join(fileName))

f = open(os.path.join(calcDir,multirunScriptfileName),'w')
f.write('#!/bin/bash\n')

commonInputPrefix = longestCommonPrefix(listOfStructureFiles)
for file in listOfStructureFiles :
    # Create calculation directories with chosen prefix and without .cif suffix
    currentCalcDir = os.path.join(calcDir,calcPrefix+file[len(commonInputPrefix):-4])
    if not os.path.exists(currentCalcDir) :
        os.mkdir(currentCalcDir)
    
    # Reading structure from cif file and creating POSCAR file
    structure = Structure.from_file(os.path.join(inputStructuresDir,file),primitive=False,sort=True)
    poscar = Poscar(structure)
    poscar.write_file(os.path.join(currentCalcDir,'POSCAR'))

    # Generating POTCAR file and reading corresponding encut 
    potcar = Potcar(symbols=structure.symbol_set,functional='PBE_54')
    enmax = max([pot.enmax for pot in potcar])
    encut = np.ceil(encutToEnmaxRatio*enmax)
    potcar.write_file(os.path.join(currentCalcDir,'POTCAR'))
 
    incar = Incar.from_file(os.path.join(sampleFilesDir,'INCAR.sample'))
    # Change INCAR parameters
    incarDict["SYSTEM"] = fileName
    incarDict["ENCUT"] = str(encut)
    incar = Incar.from_dict(incar.as_dict())
    incar.update(incarDict)
    incar.write_file(os.path.join(currentCalcDir,'INCAR'))

    # KPOINTS file is not required as long as KSPACING is defined in INCAR file (recommended by VASP wiki)

    copyfile(os.path.join(sampleFilesDir,vasprunfile),os.path.join(currentCalcDir,vasprunfile))
    
    # script : move to current calculation folder
    f.write('cd '+calcPrefix+file[len(commonInputPrefix):-4]+'\n')

    # count number of vasp jobs in queue
    if limitNbOfJobs == True :
      f.write('while [ $(qstat -u $USER | grep '+str(jobIdentifier)+' | wc -l) -ge '+str(maxNbOfJobs)+' ]; do\n')
      f.write('  sleep '+str(sleepTime)+'\n')
      f.write('done\n')

    f.write('echo "Ready to run job '+calcPrefix+file[len(commonInputPrefix):-4]+' with command : '+submissionCommand+'"\n')
    if isDryRun == False :
        f.write(submissionCommand+'\n')
    f.write('cd ..\n')
    
f.write('echo \"All calculations have been submitted. End of script file '+multirunScriptfileName+'\".\n')
f.close()

print('Calculations and multi-submission script ',multirunScriptfileName, ' have been prepared in directory :\n',calcDir)
if isDryRun == False :
  print('Move entire folder to desired destination, and type chmod \'u+x ',
        multirunScriptfileName,'\' and \'./',multirunScriptfileName,'\'')
else :
  print('Verify input files and multi-run script and then re-run script with isDryRun = False.')

