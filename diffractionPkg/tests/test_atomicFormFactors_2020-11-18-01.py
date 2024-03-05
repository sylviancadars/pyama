# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:18:40 2020

@author: cadarp02
"""

#************************ MAIN PROGRAM ***************************
import numpy as np
from pymatgen.core.structure import IStructure # non-mutable Structure
# import structureComparisonsPkg.distanceTools as dt
import diffractionPkg.atomicFormFactors as aff


#********************* TeO2-alpha ***********************
# inputFileName = r'D:\cadarp02\Documents\CristalStructures\TeO2-alpha_P41212_1968_COD-1537586.cif'
#********************* TeO2-gamma ***********************
# inputFileName = r'D:\cadarp02\Documents\CrystalStructures\TeO2-gamma_P212121_2000_COD-1520934.cif'

# TODO : debug issue with the structure below (problem associated with CIF file)
# inputFileName = r'D:\cadarp02\Documents\CrystalStructures\LaB6_Pm-3m_1986_icsd-612685.cif'

inputFileName = r'D:\cadarp02\Documents\CrystalStructures\LaB6_Pm-3m_2005_icsd-152466.cif'

structure = IStructure.from_file(inputFileName)

# User parameters
Qmax = 25
Qstep = 0.1
kmax = 20   # Length of the Fourier series expansion of FZ weighting functions
rMin = 0.1
rMax = 8    # 
rSteps = 512
showPlot = True

listOfSpecies = list(structure.symbol_set)
nbOfSpecies = len(listOfSpecies)

[ffc] = aff.read_coeffs_for_f0_form_factor(['B'])
f0, listOfSpecies, Q = aff.calculate_form_factors_f0_from_coeffs(
    [ffc],showPlots=True,Qmax=Qmax,Qstep=Qstep)
[dc] = aff.read_dispersion_corrections_f1_f2_from_file(['B'],showPlots=True)
f1,f2 = dc.calculate_f1_f2_from_E_eV(25000)

# listOfFormFactors,listOfSpecies = \
#     aff.get_list_of_formFactor_objects_from_list_of_species(listOfSpecies)

