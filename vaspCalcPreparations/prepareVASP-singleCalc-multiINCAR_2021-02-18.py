# -*- coding: utf-8 -*-
"""
Script preparing a series of VASP calculations from all structures generated
with the supercell program located in a given folder. VASP input files other
than POSCAR (i.e. POTCAR, INCAR) are copied from a sampleFilesDir folder.
Calculations are conducted sequentially from INCAR_X files.

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

# ************ MAIN *******************
# Set to isDryRun to True to test the creation of all files without actually submitting jobs
isDryRun = False
limitNbOfJobs = False  # If true job should be submitted with nohup, which may cause problems with some job schedulers

currentCalcDir = os.getcwd()
sampleFilesDir=r'/gpfs/home/scadars/VASP/AsTe3/DRX-models/sampleInputFiles'
vasprunfile = 'vasp_multiINCAR.slurm'
submissionCommand = 'sbatch ./' + vasprunfile
jobIdentifier = 'vasp'

# Calculation Parameters (job- or system-dependent parameters are set automatically below)
# this includes ENCUT (based in ENMAX in POTCAR) and SYSTEM
encutToEnmaxRatios = [ 1.0,   1.0,   1.0,   1.3,   1.3,   1.3,   1.3  ]
kSpacings =          [ 0.60,  0.60,  0.50,  0.40,  0.40,  0.30,  0.30 ]
eDiffs =             [ 3e-3,  2e-3,  1e-3,  5e-4,  2e-4,  1e-4,  1e-4 ]
eDiffgs =            [ -0.15, -0.15, -0.10, -0.05, -0.04, -0.03, 0    ]
isifs =              [ 4,     4,     4,     3,     3,     3,     0    ]
ibrions =            [ 2,     2,     1,     2,     2,     1,     -1   ] 
nsws =               [ 50,    200,   200,   50,    200,   300,   0    ]
precs =              ['low',  'low', 'Normal','Normal','Normal','Normal', 'High']
stepDescriptions = ['fixed-volume, coarse (1)',
                    'fixed-volume, coarse (2)',
                    'fixed-volume, medium',
                    'free-volume, accurate (1)',
                    'free-volume, accurate (2)',
                    'free-volume, very accurate',
		    'single-point energy, accurate']
incarDict={}

structure=Structure.from_file('POSCAR')

# Generating POTCAR file and reading corresponding encut 
potcar = Potcar(symbols=structure.symbol_set,functional='PBE_54')
enmax = max([pot.enmax for pot in potcar])
potcar.write_file(os.path.join(currentCalcDir,'POTCAR'))

incar = Incar.from_file(os.path.join(sampleFilesDir,'INCAR.sample'))
# Change INCAR parameters
for incarIndex,encutToEnmaxRatio in enumerate(encutToEnmaxRatios) :
  incarDict["SYSTEM"] = 'Step '+str(incarIndex)+' : '+stepDescriptions[incarIndex]
  incarDict["ENCUT"] = str(np.ceil(encutToEnmaxRatio*enmax))
  incarDict["KSPACING"] = str(kSpacings[incarIndex])
  incarDict["EDIFF"] = str(eDiffs[incarIndex])
  incarDict["EDIFFG"] = str(eDiffgs[incarIndex])
  incarDict["ISIF"] = str(isifs[incarIndex])
  incarDict["IBRION"] = str(ibrions[incarIndex])
  incarDict["NSW"] = str(nsws[incarIndex])
  incarDict["PREC"] = precs[incarIndex]
    
  incar = Incar.from_dict(incar.as_dict())
  incar.update(incarDict)
  incar.write_file(os.path.join(currentCalcDir,'INCAR_'+str(incarIndex)))

# KPOINTS file is not required as long as KSPACING is defined in INCAR file (recommended by VASP wiki)

