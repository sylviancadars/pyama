# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:15:15 2020

@author: cadarp02
"""

import numpy as np
from pymatgen.core.periodic_table import Specie
from pymatgen.core.structure import IStructure # non-mutable Structure
import structureComparisonsPkg.distanceTools as dt

#********************* TeO2-alpha ***********************
# inputFileName = r'D:\cadarp02\Documents\CristalStructures\TeO2-alpha_P41212_1968_COD-1537586.cif'
#********************* TeO2-gamma ***********************
# inputFileName = r'D:\cadarp02\Documents\CrystalStructures\TeO2-gamma_P212121_2000_COD-1520934.cif'

# TODO : PROBLEM WITH THIS STRUCTURE : DEBUG
# inputFileName = r'D:\cadarp02\Documents\CrystalStructures\LaB6_Pm-3m_1986_icsd-612685.cif'

inputFileName = r'D:\cadarp02\Documents\CrystalStructures\LaB6_Pm-3m_2005_icsd-152466.cif'

structure = IStructure.from_file(inputFileName)
dmd = dt.distanceMatrixData(sigma=0.05)
g_ij = dmd.calculate_partial_RDF(structure,'La','B',showPlot=True)   
