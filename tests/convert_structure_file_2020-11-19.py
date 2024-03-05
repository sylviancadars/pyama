# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:41:41 2020

@author: cadarp02
"""

from pymatgen.io.xyz import XYZ
from pymatgen.core.structure import IStructure # non-mutable Structure

inputFileName = r'D:\cadarp02\Documents\CrystalStructures\LaB6_Pm-3m_2005_icsd-152466.cif'
outputFileName = r'D:\cadarp02\Documents\Programmes\molecularModeling\rings\examples\LaB6\LaB6.xyz'

structure = IStructure.from_file(inputFileName)
# structure.to(fmt='xyz',filename=outputFileName)

xyz = XYZ(structure)
xyz.write_file(outputFileName)

