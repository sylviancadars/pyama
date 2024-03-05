# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:42:28 2021

@author: cadarp02
"""

from ase.io import read
import os.path
from ase.visualize import view

dirName = r'D:\cadarp02\Documents\Programming\examples\VASP\NMR_TeO2-glass_270at_randSnap\NMR_StdPot_KSPA-0.30_nodes-6_tpn-27'

# fileName = os.path.join(dirName,'CONTCAR') # -> OK
# fileName = os.path.join(dirName,'vasprun.xml') # -> OK
fileName = os.path.join(dirName,'OUTCAR') # -> Not OK (because NMR calculation ?)

myAtoms = read(fileName)
print(myAtoms.cell.cellpar())
print(myAtoms.get_chemical_formula())

view(myAtoms,viewer='ase')


