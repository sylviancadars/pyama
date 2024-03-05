# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:37:22 2020

@author: cadarp02
"""
from pymatgen.core.structure import IStructure # non-mutable Structure
import diffractionPkg.totalXRayScattering as txrs
#TODO : move get_Q_vector function to some other more explicit module within
#       the diffractionPkg

#********************* TeO2-alpha ***********************
# inputFileName = r'D:\cadarp02\Documents\CristalStructures\TeO2-alpha_P41212_1968_COD-1537586.cif'
#********************* TeO2-gamma ***********************
# inputFileName = r'D:\cadarp02\Documents\CrystalStructures\TeO2-gamma_P212121_2000_COD-1520934.cif'

# TODO : debug issue with the structure below (problem associated with CIF file)
# inputFileName = r'D:\cadarp02\Documents\CrystalStructures\LaB6_Pm-3m_1986_icsd-612685.cif'

inputFileName = r'D:\cadarp02\Documents\CrystalStructures\LaB6_Pm-3m_2005_icsd-152466.cif'

structure = IStructure.from_file(inputFileName)

totalScattering,Q = txrs.calculate_total_xray_scattering(
    structure,kmax=8,Qmax=40,Qstep=0.01,rMin=0.1,rMax=7,rSteps=512,showPlot=True)

